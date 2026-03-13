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
