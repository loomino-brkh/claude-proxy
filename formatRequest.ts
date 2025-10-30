// Type declarations for missing built-in types
interface Map<K, V> {
    size: number;
    set(key: K, value: V): this;
    get(key: K): V | undefined;
    has(key: K): boolean;
    delete(key: K): boolean;
    clear(): void;
    forEach(
        callbackfn: (value: V, key: K, map: Map<K, V>) => void,
        thisArg?: any,
    ): void;
}

interface Set<T> {
    size: number;
    add(value: T): this;
    has(value: T): boolean;
    delete(value: T): boolean;
    clear(): void;
    forEach(
        callbackfn: (value: T, value2: T, set: Set<T>) => void,
        thisArg?: any,
    ): void;
}

declare global {
    interface String {
        includes(searchString: string): boolean;
    }
}

/**
 * Represents a tool call in OpenAI format
 */
interface OpenAIToolCall {
    id: string;
    type: "function";
    function_def: {
        name: string;
        arguments: string;
    };
}

/**
 * Represents a message in OpenAI chat completion format
 */
interface OpenAIMessage {
    role: "user" | "assistant" | "system" | "tool";
    content?: string | null;
    tool_calls?: OpenAIToolCall[];
    tool_call_id?: string;
}

/**
 * Represents a tool schema definition
 */
interface ToolSchema {
    name: string;
    description: string;
    input_schema: Record<string, unknown>;
}

/**
 * Represents a system message content part
 */
interface SystemContentPart {
    type: "text";
    text: string;
    cache_control?: { type: "ephemeral" };
}

/**
 * Represents a system message
 */
interface SystemMessage {
    role: "system";
    content: SystemContentPart[];
}

/**
 * Base interface for message creation parameters
 */
interface MessageCreateParamsBase {
    model: string;
    messages: AnthropicMessage[];
    system?: string | Array<{ text: string }>;
    temperature?: number;
    tools?: ToolSchema[];
    stream?: boolean;
}

/**
 * Represents a content part in Anthropic format
 */
type AnthropicContentPart =
    | { type: "text"; text: string }
    | {
          type: "tool_use";
          id: string;
          name: string;
          input: Record<string, unknown>;
      }
    | {
          type: "tool_result";
          tool_use_id: string;
          content: string | Record<string, unknown>;
      };

/**
 * Represents a message in Anthropic format
 */
interface AnthropicMessage {
    role: string;
    content: string | AnthropicContentPart[];
}

/**
 * Safely stringifies JSON content with error handling
 * @param data - The data to stringify
 * @param context - Context description for error messages
 * @returns Stringified JSON representation
 * @throws {Error} When JSON stringification fails
 */
function safeJsonStringify(data: unknown, context: string): string {
    try {
        return JSON.stringify(data);
    } catch (error) {
        throw new Error(
            `Failed to stringify ${context}: ${error instanceof Error ? error.message : "Unknown error"}`,
        );
    }
}

/**
 * Validates an individual Anthropic message
 * @param message - The message to validate
 * @throws {Error} When message format is invalid
 */
function validateAnthropicMessage(message: AnthropicMessage): void {
    if (!message || typeof message !== "object") {
        throw new Error("Message must be a valid object");
    }

    if (!message.role || typeof message.role !== "string") {
        throw new Error("Message must have a valid role");
    }

    if (!message.content) {
        throw new Error("Message must have content");
    }

    // Validate content structure
    if (Array.isArray(message.content)) {
        message.content.forEach((part, index) => {
            if (!part || typeof part !== "object") {
                throw new Error(
                    `Content part at index ${index} must be a valid object`,
                );
            }

            if (!part.type || typeof part.type !== "string") {
                throw new Error(
                    `Content part at index ${index} must have a valid type`,
                );
            }

            // Validate specific content part types
            switch (part.type) {
                case "text":
                    if (typeof part.text !== "string") {
                        throw new Error(
                            `Text content part at index ${index} must have a string text field`,
                        );
                    }
                    break;
                case "tool_use":
                    if (!part.id || typeof part.id !== "string") {
                        throw new Error(
                            `Tool_use content part at index ${index} must have a string id field`,
                        );
                    }
                    if (!part.name || typeof part.name !== "string") {
                        throw new Error(
                            `Tool_use content part at index ${index} must have a string name field`,
                        );
                    }
                    if (!part.input || typeof part.input !== "object") {
                        throw new Error(
                            `Tool_use content part at index ${index} must have an object input field`,
                        );
                    }
                    break;
                case "tool_result":
                    if (
                        !part.tool_use_id ||
                        typeof part.tool_use_id !== "string"
                    ) {
                        throw new Error(
                            `Tool_result content part at index ${index} must have a string tool_use_id field`,
                        );
                    }
                    if (!part.content) {
                        throw new Error(
                            `Tool_result content part at index ${index} must have content`,
                        );
                    }
                    break;
                default:
                    throw new Error(
                        `Unknown content part type: ${(part as any).type}`,
                    );
            }
        });
    } else if (typeof message.content !== "string") {
        throw new Error(
            "Message content must be either a string or an array of content parts",
        );
    }
}

/**
 * Validates OpenAI format messages to ensure complete tool_calls/tool message pairing.
 * Requires tool messages to immediately follow assistant messages with tool_calls.
 * Enforces strict immediate following sequence between tool_calls and tool messages.
 */
function validateOpenAIToolCalls(messages: OpenAIMessage[]): OpenAIMessage[] {
    const validatedMessages: OpenAIMessage[] = [];

    for (let i = 0; i < messages.length; i++) {
        const currentMessage = { ...messages[i] };

        // Process assistant messages with tool_calls
        if (currentMessage.role === "assistant" && currentMessage.tool_calls) {
            const validToolCalls: OpenAIToolCall[] = [];
            const removedToolCallIds: string[] = [];

            // Collect all immediately following tool messages and create a Map for O(1) lookups
            const immediateToolMessages: OpenAIMessage[] = [];
            let j = i + 1;
            while (j < messages.length && messages[j].role === "tool") {
                immediateToolMessages.push(messages[j]);
                j++;
            }

            // Create a Map of tool_call_id to tool message for O(1) lookups
            const toolMessageMap = new Map<string, OpenAIMessage>();
            immediateToolMessages.forEach((toolMsg) => {
                if (toolMsg.tool_call_id) {
                    toolMessageMap.set(toolMsg.tool_call_id, toolMsg);
                }
            });

            // For each tool_call, check if there's an immediately following tool message using Map
            currentMessage.tool_calls.forEach((toolCall: OpenAIToolCall) => {
                if (toolMessageMap.has(toolCall.id)) {
                    validToolCalls.push(toolCall);
                } else {
                    removedToolCallIds.push(toolCall.id);
                }
            });

            // Update the assistant message
            if (validToolCalls.length > 0) {
                currentMessage.tool_calls = validToolCalls;
            } else {
                delete currentMessage.tool_calls;
            }

            // Only include message if it has content or valid tool_calls
            if (currentMessage.content || currentMessage.tool_calls) {
                validatedMessages.push(currentMessage);
            }
        }

        // Process tool messages
        else if (currentMessage.role === "tool") {
            let hasImmediateToolCall = false;

            // Check if the immediately preceding assistant message has matching tool_call
            if (i > 0) {
                const prevMessage = messages[i - 1];
                if (
                    prevMessage.role === "assistant" &&
                    prevMessage.tool_calls
                ) {
                    // Use Set for O(1) lookups
                    const toolCallIds = new Set(
                        prevMessage.tool_calls.map(
                            (toolCall: OpenAIToolCall) => toolCall.id,
                        ),
                    );
                    hasImmediateToolCall = currentMessage.tool_call_id
                        ? toolCallIds.has(currentMessage.tool_call_id)
                        : false;
                } else if (prevMessage.role === "tool") {
                    // Check for assistant message before the sequence of tool messages
                    for (let k = i - 1; k >= 0; k--) {
                        const candidate = messages[k];
                        if (!candidate) break;
                        if (candidate.role === "tool") continue;
                        if (
                            candidate.role === "assistant" &&
                            candidate.tool_calls
                        ) {
                            const toolCallIds = new Set(
                                candidate.tool_calls.map(
                                    (toolCall: OpenAIToolCall) => toolCall.id,
                                ),
                            );
                            hasImmediateToolCall = currentMessage.tool_call_id
                                ? toolCallIds.has(currentMessage.tool_call_id)
                                : false;
                        }
                        break;
                    }
                }
            }

            if (hasImmediateToolCall) {
                validatedMessages.push(currentMessage);
            }
        }

        // For all other message types, include as-is
        else {
            validatedMessages.push(currentMessage);
        }
    }

    return validatedMessages;
}

/**
 * Configuration mapping Anthropic model patterns to OpenRouter model IDs
 */
const MODEL_MAPPING = {
    haiku: "z-ai/glm-4.5-air:free",
    sonnet: "z-ai/glm-4.6:exacto",
    opus: "tngtech/deepseek-r1t2-chimera:free",
} as const;

/**
 * Model keywords to search for in Anthropic model names
 */
type ModelKeyword = keyof typeof MODEL_MAPPING;

/**
 * Maps Anthropic model names to OpenRouter model IDs using a configuration-driven approach
 * @param anthropicModel - The Anthropic model name (e.g., "claude-3-haiku", "claude-3-sonnet")
 * @returns The corresponding OpenRouter model ID or the original model if it's already an OpenRouter ID
 */
export function mapModel(anthropicModel: string): string {
    // If model already contains '/', it's an OpenRouter model ID - return as-is
    if (anthropicModel.includes("/")) {
        return anthropicModel;
    }

    // Search for model keywords in the Anthropic model name
    for (const [keyword, openRouterModel] of Object.entries(MODEL_MAPPING)) {
        if (anthropicModel.includes(keyword)) {
            return openRouterModel;
        }
    }

    // Return original model if no mapping found
    return anthropicModel;
}

/**
 * Converts Anthropic message format to OpenAI chat completion format
 * @param body - The message creation parameters in Anthropic format
 * @returns Formatted request body compatible with OpenAI API
 * @throws {Error} When input validation fails or JSON parsing encounters errors
 */
export function formatAnthropicToOpenAI(body: MessageCreateParamsBase): any {
    // Input validation
    if (!body) {
        throw new Error("Request body is required");
    }

    if (!body.model || typeof body.model !== "string") {
        throw new Error("Model is required and must be a string");
    }

    if (!Array.isArray(body.messages)) {
        throw new Error("Messages must be an array");
    }

    const { model, messages, system = [], temperature, tools, stream } = body;

    // Validate all messages
    messages.forEach((message, index) => {
        try {
            validateAnthropicMessage(message);
        } catch (error) {
            throw new Error(
                `Invalid message at index ${index}: ${error instanceof Error ? error.message : "Unknown error"}`,
            );
        }
    });

    const openAIMessages = Array.isArray(messages)
        ? messages.flatMap((anthropicMessage) => {
              const openAiMessagesFromThisAnthropicMessage: any[] = [];

              if (!Array.isArray(anthropicMessage.content)) {
                  if (typeof anthropicMessage.content === "string") {
                      openAiMessagesFromThisAnthropicMessage.push({
                          role: anthropicMessage.role,
                          content: anthropicMessage.content,
                      });
                  }
                  return openAiMessagesFromThisAnthropicMessage;
              }

              if (anthropicMessage.role === "assistant") {
                  const assistantMessage: any = {
                      role: "assistant",
                      content: null,
                  };
                  let textContent = "";
                  const toolCalls: any[] = [];

                  anthropicMessage.content.forEach(
                      (contentPart: AnthropicContentPart) => {
                          if (contentPart.type === "text") {
                              textContent +=
                                  (typeof contentPart.text === "string"
                                      ? contentPart.text
                                      : safeJsonStringify(
                                            contentPart.text,
                                            "text content",
                                        )) + "\n";
                          } else if (contentPart.type === "tool_use") {
                              toolCalls.push({
                                  id: contentPart.id,
                                  type: "function",
                                  function_def: {
                                      name: contentPart.name,
                                      arguments: safeJsonStringify(
                                          contentPart.input,
                                          `tool arguments for ${contentPart.name}`,
                                      ),
                                  },
                              });
                          }
                      },
                  );

                  const trimmedTextContent = textContent.trim();
                  if (trimmedTextContent.length > 0) {
                      assistantMessage.content = trimmedTextContent;
                  }
                  if (toolCalls.length > 0) {
                      assistantMessage.tool_calls = toolCalls;
                  }
                  if (
                      assistantMessage.content ||
                      (assistantMessage.tool_calls &&
                          assistantMessage.tool_calls.length > 0)
                  ) {
                      openAiMessagesFromThisAnthropicMessage.push(
                          assistantMessage,
                      );
                  }
              } else if (anthropicMessage.role === "user") {
                  let userTextMessageContent = "";
                  const subsequentToolMessages: any[] = [];

                  anthropicMessage.content.forEach(
                      (contentPart: AnthropicContentPart) => {
                          if (contentPart.type === "text") {
                              userTextMessageContent +=
                                  (typeof contentPart.text === "string"
                                      ? contentPart.text
                                      : safeJsonStringify(
                                            contentPart.text,
                                            "user text content",
                                        )) + "\n";
                          } else if (contentPart.type === "tool_result") {
                              subsequentToolMessages.push({
                                  role: "tool",
                                  tool_call_id: contentPart.tool_use_id,
                                  content:
                                      typeof contentPart.content === "string"
                                          ? contentPart.content
                                          : safeJsonStringify(
                                                contentPart.content,
                                                "tool result content",
                                            ),
                              });
                          }
                      },
                  );

                  const trimmedUserText = userTextMessageContent.trim();
                  if (trimmedUserText.length > 0) {
                      openAiMessagesFromThisAnthropicMessage.push({
                          role: "user",
                          content: trimmedUserText,
                      });
                  }
                  openAiMessagesFromThisAnthropicMessage.push(
                      ...subsequentToolMessages,
                  );
              }
              return openAiMessagesFromThisAnthropicMessage;
          })
        : [];

    const systemMessages = Array.isArray(system)
        ? system.map((item) => {
              const content: any = {
                  type: "text",
                  text: item.text,
              };
              if (model.includes("claude")) {
                  content.cache_control = { type: "ephemeral" };
              }
              return {
                  role: "system",
                  content: [content],
              };
          })
        : [
              {
                  role: "system",
                  content: [
                      {
                          type: "text",
                          text: system,
                          ...(model.includes("claude")
                              ? { cache_control: { type: "ephemeral" } }
                              : {}),
                      },
                  ],
              },
          ];

    const mappedModel = mapModel(model);
    const isChimera = mappedModel === "tngtech/deepseek-r1t2-chimera:free";
    const data: any = {
        model: mappedModel,
        messages: [...systemMessages, ...openAIMessages],
        temperature,
        stream,
    };

    const modelProviderConfigs = {
        "z-ai/glm-4.6:exacto": {
            only: ["z-ai"],
            ignore: ["deepinfra", "chutes", "novita"],
            allow_fallbacks: false,
            data_collection: "deny",
            zdr: true,
        },
        "z-ai/glm-4.5-air:free": {
            only: ["z-ai"],
            ignore: ["deepinfra", "chutes", "novita"],
            allow_fallbacks: false,
            // data_collection: "deny",
            // zdr: true,
        },
        "tngtech/deepseek-r1t2-chimera:free": {
            only: ["chutes"],
            ignore: ["deepinfra", "novita"],
            allow_fallbacks: false,
            // data_collection: "allow",
            // zdr: false,
        },
    };

    if (modelProviderConfigs[mappedModel]) {
        data.provider = modelProviderConfigs[mappedModel];
    }

    if (tools && !isChimera) {
        data.tools = tools.map((item: any) => ({
            type: "function",
            function: {
                name: item.name,
                description: item.description,
                parameters: item.input_schema,
            },
        }));
    }

    // Strip tool-related messages for Chimera
    const finalMessages = isChimera
        ? [...systemMessages, ...openAIMessages]
              .map((message) => {
                  // Remove tool_calls from assistant messages
                  if (message.role === "assistant" && message.tool_calls) {
                      const { tool_calls, ...rest } = message;
                      return rest;
                  }
                  // Remove tool messages entirely
                  if (message.role === "tool") return null;
                  return message;
              })
              .filter(Boolean)
        : [...systemMessages, ...validateOpenAIToolCalls(openAIMessages)];

    data.messages = finalMessages;

    return data;
}
