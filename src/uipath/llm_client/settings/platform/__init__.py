from typing_extensions import override

from uipath.llm_client.settings.platform.auth import PlatformAuth
from uipath.llm_client.settings.platform.settings import PlatformBaseSettings


class PlatformSettings(PlatformBaseSettings):
    """Settings for UiPath Platform (AgentHub/Orchestrator) client."""

    @override
    def build_auth_pipeline(self) -> PlatformAuth:
        """Build an httpx Auth pipeline for Platform authentication."""
        return PlatformAuth(settings=self)


__all__ = [
    "PlatformAuth",
    "PlatformBaseSettings",
    "PlatformSettings",
]
