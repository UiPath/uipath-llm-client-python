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


class ModelFamily(StrEnum):
    OPENAI = "OpenAi"
    GOOGLE_GEMINI = "GoogleGemini"
    ANTHROPIC_CLAUDE = "AnthropicClaude"


class ApiFlavor(StrEnum):
    CHAT_COMPLETIONS = "chat-completions"
    RESPONSES = "responses"
    GENERATE_CONTENT = "generate-content"
    CONVERSE = "converse"
    INVOKE = "invoke"
    ANTHROPIC_CLAUDE = "anthropic-claude"


class ByomApiFlavor(StrEnum):
    """API flavors returned by the discovery endpoint for BYOM models."""

    OPENAI_CHAT_COMPLETIONS = "OpenAiChatCompletions"
    OPENAI_RESPONSES = "OpenAiResponses"
    OPENAI_EMBEDDINGS = "OpenAiEmbeddings"
    GEMINI_GENERATE_CONTENT = "GeminiGenerateContent"
    GEMINI_EMBEDDINGS = "GeminiEmbeddings"
    AWS_BEDROCK_INVOKE = "AwsBedrockInvoke"
    AWS_BEDROCK_CONVERSE = "AwsBedrockConverse"


API_FLAVOR_TO_VENDOR_TYPE: dict[str, VendorType] = {
    ApiFlavor.CHAT_COMPLETIONS: VendorType.OPENAI,
    ApiFlavor.RESPONSES: VendorType.OPENAI,
    ApiFlavor.GENERATE_CONTENT: VendorType.VERTEXAI,
    ApiFlavor.ANTHROPIC_CLAUDE: VendorType.VERTEXAI,
    ApiFlavor.CONVERSE: VendorType.AWSBEDROCK,
    ApiFlavor.INVOKE: VendorType.AWSBEDROCK,
    ByomApiFlavor.OPENAI_CHAT_COMPLETIONS: VendorType.OPENAI,
    ByomApiFlavor.OPENAI_RESPONSES: VendorType.OPENAI,
    ByomApiFlavor.OPENAI_EMBEDDINGS: VendorType.OPENAI,
    ByomApiFlavor.GEMINI_GENERATE_CONTENT: VendorType.VERTEXAI,
    ByomApiFlavor.GEMINI_EMBEDDINGS: VendorType.VERTEXAI,
    ByomApiFlavor.AWS_BEDROCK_INVOKE: VendorType.AWSBEDROCK,
    ByomApiFlavor.AWS_BEDROCK_CONVERSE: VendorType.AWSBEDROCK,
}


BYOM_TO_ROUTING_FLAVOR: dict[str, ApiFlavor] = {
    ByomApiFlavor.OPENAI_CHAT_COMPLETIONS: ApiFlavor.CHAT_COMPLETIONS,
    ByomApiFlavor.OPENAI_RESPONSES: ApiFlavor.RESPONSES,
    ByomApiFlavor.GEMINI_GENERATE_CONTENT: ApiFlavor.GENERATE_CONTENT,
    ByomApiFlavor.AWS_BEDROCK_INVOKE: ApiFlavor.INVOKE,
    ByomApiFlavor.AWS_BEDROCK_CONVERSE: ApiFlavor.CONVERSE,
}
