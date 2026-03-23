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
    CHAT_COMPLETIONS = "chat-completions"
    RESPONSES = "responses"
    GENERATE_CONTENT = "generate-content"
    CONVERSE = "converse"
    INVOKE = "invoke"
    ANTHROPIC_CLAUDE = "anthropic-claude"


_API_FLAVOR_TO_VENDOR_TYPE: dict[ApiFlavor, VendorType] = {
    ApiFlavor.CHAT_COMPLETIONS: VendorType.OPENAI,
    ApiFlavor.RESPONSES: VendorType.OPENAI,
    ApiFlavor.GENERATE_CONTENT: VendorType.VERTEXAI,
    ApiFlavor.ANTHROPIC_CLAUDE: VendorType.VERTEXAI,
    ApiFlavor.CONVERSE: VendorType.AWSBEDROCK,
    ApiFlavor.INVOKE: VendorType.AWSBEDROCK,
}
