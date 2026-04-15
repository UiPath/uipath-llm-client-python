from enum import StrEnum


class ApiType(StrEnum):
    COMPLETIONS = "completions"
    EMBEDDINGS = "embeddings"


class RoutingMode(StrEnum):
    PASSTHROUGH = "passthrough"
    NORMALIZED = "normalized"


class VendorType(StrEnum):
    OPENAI = "openai"
    VERTEXAI = "vertexai"
    AWSBEDROCK = "awsbedrock"
    AZURE = "azure"
    ANTHROPIC = "anthropic"


class ApiFlavor(StrEnum):
    # Routing-level flavors (used in X-UiPath-LlmGateway-ApiFlavor header)
    CHAT_COMPLETIONS = "chat-completions"
    RESPONSES = "responses"
    GENERATE_CONTENT = "generate-content"
    CONVERSE = "converse"
    INVOKE = "invoke"
    ANTHROPIC_CLAUDE = "anthropic-claude"

    # BYOM discovery flavors (returned by discovery endpoint for BYOM models)
    OPENAI_CHAT_COMPLETIONS = "OpenAiChatCompletions"
    OPENAI_RESPONSES = "OpenAiResponses"
    OPENAI_EMBEDDINGS = "OpenAiEmbeddings"
    GEMINI_GENERATE_CONTENT = "GeminiGenerateContent"
    GEMINI_EMBEDDINGS = "GeminiEmbeddings"
    AWS_BEDROCK_INVOKE = "AwsBedrockInvoke"
    AWS_BEDROCK_CONVERSE = "AwsBedrockConverse"


API_FLAVOR_TO_VENDOR_TYPE: dict[ApiFlavor, VendorType] = {
    # Routing flavors
    ApiFlavor.CHAT_COMPLETIONS: VendorType.OPENAI,
    ApiFlavor.RESPONSES: VendorType.OPENAI,
    ApiFlavor.GENERATE_CONTENT: VendorType.VERTEXAI,
    ApiFlavor.ANTHROPIC_CLAUDE: VendorType.VERTEXAI,
    ApiFlavor.CONVERSE: VendorType.AWSBEDROCK,
    ApiFlavor.INVOKE: VendorType.AWSBEDROCK,
    # BYOM discovery flavors
    ApiFlavor.OPENAI_CHAT_COMPLETIONS: VendorType.OPENAI,
    ApiFlavor.OPENAI_RESPONSES: VendorType.OPENAI,
    ApiFlavor.OPENAI_EMBEDDINGS: VendorType.OPENAI,
    ApiFlavor.GEMINI_GENERATE_CONTENT: VendorType.VERTEXAI,
    ApiFlavor.GEMINI_EMBEDDINGS: VendorType.VERTEXAI,
    ApiFlavor.AWS_BEDROCK_INVOKE: VendorType.AWSBEDROCK,
    ApiFlavor.AWS_BEDROCK_CONVERSE: VendorType.AWSBEDROCK,
}


BYOM_TO_ROUTING_FLAVOR: dict[str, ApiFlavor] = {
    ApiFlavor.OPENAI_CHAT_COMPLETIONS: ApiFlavor.CHAT_COMPLETIONS,
    ApiFlavor.OPENAI_RESPONSES: ApiFlavor.RESPONSES,
    ApiFlavor.GEMINI_GENERATE_CONTENT: ApiFlavor.GENERATE_CONTENT,
    ApiFlavor.AWS_BEDROCK_INVOKE: ApiFlavor.INVOKE,
    ApiFlavor.AWS_BEDROCK_CONVERSE: ApiFlavor.CONVERSE,
}
