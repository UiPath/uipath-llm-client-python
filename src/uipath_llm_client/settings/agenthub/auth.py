"""Authentication handler for UiPath AgentHub."""

from collections.abc import Generator

from httpx import Auth, Request, Response

from uipath_llm_client.settings.agenthub.settings import AgentHubBaseSettings
from uipath_llm_client.settings.utils import SingletonMeta


class AgentHubAuth(Auth, metaclass=SingletonMeta):
    """Bearer authentication handler with automatic token refresh.

    Singleton class that reuses the same token across all requests to minimize
    token generation overhead. Automatically refreshes the token on 401 responses.
    """

    def __init__(
        self,
        settings: AgentHubBaseSettings,
    ):
        """Initialize the auth handler.

        Args:
            settings: AgentHub settings containing authentication credentials.
        """
        self.settings = settings
        self.access_token = self._get_access_token()

    def _get_access_token(self) -> str:
        """Retrieve or refresh the access token."""
        self.settings.authenticate()
        assert self.settings.access_token is not None
        return self.settings.access_token.get_secret_value()

    def auth_flow(self, request: Request) -> Generator[Request, Response, None]:
        """HTTPX auth flow that handles token refresh on authentication failures."""
        request.headers["Authorization"] = f"Bearer {self.access_token}"
        response = yield request
        if response.status_code == 401:
            self.access_token = self._get_access_token()
            request.headers["Authorization"] = f"Bearer {self.access_token}"
            yield request
