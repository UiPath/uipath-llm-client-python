"""Authentication handler for UiPath AgentHub."""

from collections.abc import Generator

from httpx import Auth, Request, Response
from pydantic import SecretStr
from uipath.platform.identity import IdentityService

from uipath.llm_client.settings.platform.settings import PlatformBaseSettings
from uipath.llm_client.settings.utils import SingletonMeta


class PlatformAuth(Auth, metaclass=SingletonMeta):
    """Bearer authentication handler with automatic token refresh.

    Singleton class that reuses the same token across all requests to minimize
    token generation overhead. Automatically refreshes the token on 401 responses.
    """

    def __init__(
        self,
        settings: PlatformBaseSettings,
    ):
        """Initialize the auth handler.

        Args:
            settings: AgentHub settings containing authentication credentials.
        """
        self.settings = settings

    def get_access_token(self, refresh: bool = False) -> str:
        """Retrieve or refresh the access token."""
        assert self.settings.access_token is not None
        access_token = self.settings.access_token.get_secret_value()
        if not refresh:
            return access_token

        assert self.settings.base_url is not None
        assert self.settings.client_id is not None
        assert self.settings.refresh_token is not None

        identity_service = IdentityService(self.settings.base_url)
        new_token_data = identity_service.refresh_access_token(
            self.settings.refresh_token.get_secret_value(), self.settings.client_id
        )
        self.settings.access_token = SecretStr(new_token_data.access_token)
        self.settings.refresh_token = SecretStr(
            new_token_data.refresh_token or self.settings.refresh_token.get_secret_value()
        )
        return new_token_data.access_token

    def auth_flow(self, request: Request) -> Generator[Request, Response, None]:
        """HTTPX auth flow that handles token refresh on authentication failures."""
        request.headers["Authorization"] = f"Bearer {self.get_access_token()}"
        response = yield request
        if response.status_code == 401:
            request.headers["Authorization"] = f"Bearer {self.get_access_token(refresh=True)}"
            yield request
