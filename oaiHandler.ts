/**
 * Handler for OpenAI-compatible /oai routes with provider config injection.
 *
 * Endpoints supported:
 * - POST /oai/v1/chat/completions -> forwards to {OPENROUTER_BASE_URL}/chat/completions
 * - POST /oai/v1/responses        -> forwards to {OPENROUTER_BASE_URL}/responses
 *
 * Provider configuration injection precedence:
 * 1) body.provider (object used as-is; string key mapped via providerConfigs)
 * 2) header: X-OpenRouter-Provider or X-Provider (string key or JSON)
 * 3) env.OPENROUTER_DEFAULT_PROVIDER (string key)
 *
 * Streaming:
 * - If body.stream === true OR upstream returns text/event-stream, the response
 *   is piped back as SSE with appropriate headers.
 */

import type { Env } from "./env";
import { mapModel } from "./formatRequest";

export type ProviderConfig = {
    only?: string[];
    ignore?: string[];
    allow_fallbacks?: boolean;
    data_collection?: "allow" | "deny" | string;
    zdr?: boolean;
};

// Extend Env locally to allow optional default provider without modifying env.ts immediately
type ExtendedEnv = Env & {
    OPENROUTER_DEFAULT_PROVIDER?: string;
};

export const OAI_CHAT_PATH = "/oai/v1/chat/completions";
export const OAI_RESPONSES_PATH = "/oai/v1/responses";
export const OAI_MODELS_PATH = "/oai/v1/models";
// Central provider presets (expand as needed)
const providerConfigs: Record<string, ProviderConfig> = {
    "z-ai/glm-4.6:exacto": {
        only: ["z-ai"],
        ignore: ["deepinfra", "chutes", "novita"],
        allow_fallbacks: false,
        data_collection: "deny",
        zdr: true,
    },
    "minimax/minimax-m2": {
        only: ["minimax"],
        ignore: ["deepinfra", "chutes", "novita"],
        allow_fallbacks: false,
        data_collection: "deny",
        zdr: true,
    },
    "z-ai/glm-4.5-air:free": {
        only: ["z-ai"],
        ignore: ["deepinfra", "chutes", "novita"],
        allow_fallbacks: false,
    },
    "tngtech/deepseek-r1t2-chimera:free": {
        only: ["chutes"],
        ignore: ["deepinfra", "novita"],
        allow_fallbacks: false,
    },
};

/**
 * Resolve provider configuration from body, headers, or env.
 */
function resolveProviderConfig(
    body: any,
    headers: Headers,
    envDefault?: string,
): ProviderConfig | undefined {
    // 1) From body.provider
    if (body?.provider) {
        if (typeof body.provider === "string") {
            return providerConfigs[body.provider];
        }
        if (typeof body.provider === "object") {
            return body.provider as ProviderConfig;
        }
    }

    // 2) From header (string key or JSON)
    const headerVal =
        headers.get("X-OpenRouter-Provider") || headers.get("X-Provider");
    if (headerVal) {
        if (providerConfigs[headerVal]) return providerConfigs[headerVal];
        try {
            const parsed = JSON.parse(headerVal);
            if (parsed && typeof parsed === "object") {
                return parsed as ProviderConfig;
            }
        } catch {
            // not JSON - ignore
        }
    }

    // 3) From env default
    if (envDefault && providerConfigs[envDefault]) {
        return providerConfigs[envDefault];
    }

    return undefined;
}

/**
 * Build forward headers for OpenRouter request.
 */
function buildForwardHeaders(bearerToken?: string): Record<string, string> {
    const headers: Record<string, string> = {
        "Content-Type": "application/json",
    };
    if (bearerToken) {
        headers.Authorization = `Bearer ${bearerToken}`;
    }
    return headers;
}

/**
 * Determines if this request should be handled by the OAI proxy.
 */
export function shouldHandleOai(url: URL, method: string): boolean {
    const p = url.pathname;
    if (method === "POST") {
        return p === OAI_CHAT_PATH || p === OAI_RESPONSES_PATH;
    }
    if (method === "GET") {
        // Support listing models and retrieving a model by ID
        return p === OAI_MODELS_PATH || p.startsWith(`${OAI_MODELS_PATH}/`);
    }
    return false;
}

/**
 * Handle the OAI route if matched. Returns null if the route does not match.
 */
export async function handleOaiIfMatch(
    request: Request,
    env: ExtendedEnv,
): Promise<Response | null> {
    const url = new URL(request.url);
    if (!shouldHandleOai(url, request.method)) return null;

    // Handle GET /oai/v1/models and /oai/v1/models/:id by forwarding directly (preserving query)
    if (
        request.method === "GET" &&
        (url.pathname === OAI_MODELS_PATH ||
            url.pathname.startsWith(`${OAI_MODELS_PATH}/`))
    ) {
        const baseUrl =
            env.OPENROUTER_BASE_URL || "https://openrouter.ai/api/v1";
        const suffix = url.pathname.slice(OAI_MODELS_PATH.length); // includes leading slash if present
        const forwardUrl = `${baseUrl}/models${suffix}${url.search || ""}`;

        const bearerToken =
            request.headers.get("X-Api-Key") ||
            request.headers.get("Authorization")?.replace(/^Bearer\s+/i, "");

        const forwardHeaders = buildForwardHeaders(bearerToken || undefined);
        // Content-Type is not needed for GET
        if (forwardHeaders["Content-Type"])
            delete forwardHeaders["Content-Type"];

        const forwardResp = await fetch(forwardUrl, {
            method: "GET",
            headers: forwardHeaders,
        });

        if (!forwardResp.ok) {
            const text = await forwardResp.text();
            return new Response(text, { status: forwardResp.status });
        }

        // Pass upstream headers and body directly
        const responseHeaders = new Headers(forwardResp.headers);
        return new Response(forwardResp.body, {
            status: forwardResp.status,
            headers: responseHeaders,
        });
    }

    let incoming: any;
    try {
        incoming = await request.json();
    } catch {
        return new Response(
            JSON.stringify({ error: { message: "Invalid JSON body" } }),
            { status: 400, headers: { "Content-Type": "application/json" } },
        );
    }

    // Resolve provider configuration and inject if applicable
    const resolvedProvider = resolveProviderConfig(
        incoming,
        request.headers,
        env.OPENROUTER_DEFAULT_PROVIDER,
    );

    if (resolvedProvider) {
        if (!incoming.provider || typeof incoming.provider !== "object") {
            incoming.provider = resolvedProvider;
        }
        // If incoming.provider is a string key, replace with resolved object
        if (typeof incoming.provider === "string") {
            const mapped = providerConfigs[incoming.provider];
            if (mapped) incoming.provider = mapped;
        }
    }

    // If model maps to Chimera, strip tools and tool-related messages (Chimera doesn't support tools)
    try {
        const mappedModel =
            typeof incoming?.model === "string"
                ? mapModel(incoming.model)
                : undefined;
        const isChimera = mappedModel === "tngtech/deepseek-r1t2-chimera:free";
        if (isChimera) {
            // Remove top-level tools (if present)
            if (incoming.tools) delete incoming.tools;

            // Remove tool messages and tool_calls from assistant messages
            if (Array.isArray(incoming.messages)) {
                incoming.messages = incoming.messages
                    .map((message: any) => {
                        if (message && typeof message === "object") {
                            // Drop tool role messages entirely
                            if (message.role === "tool") return null;
                            // Remove tool_calls from assistant messages
                            if (
                                message.role === "assistant" &&
                                message.tool_calls
                            ) {
                                const { tool_calls, ...rest } = message;
                                return rest;
                            }
                        }
                        return message;
                    })
                    .filter(Boolean);
            }
        }
    } catch {
        // If mapping fails for any reason, forward the body unchanged
    }

    // Determine forward endpoint
    const baseUrl = env.OPENROUTER_BASE_URL || "https://openrouter.ai/api/v1";
    const endpoint =
        url.pathname === OAI_CHAT_PATH ? "/chat/completions" : "/responses";

    // Extract API key
    const bearerToken =
        request.headers.get("X-Api-Key") ||
        request.headers.get("Authorization")?.replace(/^Bearer\s+/i, "");

    // Forward request to OpenRouter
    const forwardResp = await fetch(`${baseUrl}${endpoint}`, {
        method: "POST",
        headers: buildForwardHeaders(bearerToken || undefined),
        body: JSON.stringify(incoming),
    });

    if (!forwardResp.ok) {
        const text = await forwardResp.text();
        return new Response(text, { status: forwardResp.status });
    }

    // Streaming passthrough: if request asked for streaming OR upstream responds with SSE
    const contentType = forwardResp.headers.get("content-type") || "";
    if (
        incoming?.stream === true ||
        contentType.includes("text/event-stream")
    ) {
        const headers = new Headers(forwardResp.headers);
        headers.set("Cache-Control", "no-cache");
        headers.set("Connection", "keep-alive");
        headers.set("Content-Type", "text/event-stream");
        return new Response(forwardResp.body, {
            status: forwardResp.status,
            headers,
        });
    }

    // Non-streaming: pass body & content-type
    const text = await forwardResp.text();
    return new Response(text, {
        status: forwardResp.status,
        headers: { "Content-Type": contentType || "application/json" },
    });
}
