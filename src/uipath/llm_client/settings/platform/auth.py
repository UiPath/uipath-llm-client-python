"""Authentication handler for UiPath Platform (AgentHub/Orchestrator)."""

from collections.abc import Generator

from httpx import Auth, Request, Response
from uipath.platform.identity import IdentityService

from uipath.llm_client.settings.platform.settings import PlatformBaseSettings
from uipath.llm_client.settings.utils import SingletonMeta


class PlatformAuth(Auth, metaclass=SingletonMeta):
    """Bearer authentication handler with automatic token refresh.

    Singleton class that stores access_token and refresh_token directly,
    reusing them across all requests. Automatically refreshes on 401 responses.
    """

    def __init__(
        self,
        settings: PlatformBaseSettings,
    ):
        self.base_url = settings.base_url
        self.client_id = settings.client_id
        self.access_token: str | None = (
            settings.access_token.get_secret_value() if settings.access_token else None
        )
        self.refresh_token: str | None = (
            settings.refresh_token.get_secret_value() if settings.refresh_token else None
        )

    def get_access_token(self, refresh: bool = False) -> str | None:
        """Retrieve or refresh the access token.

        Returns None on failure instead of raising, so the request proceeds
        and the client receives the actual error response from the server.
        """
        if not refresh:
            return self.access_token

        if self.base_url is None or self.client_id is None or self.refresh_token is None:
            return None

        try:
            identity_service = IdentityService(self.base_url)
            new_token_data = identity_service.refresh_access_token(
                refresh_token=self.refresh_token,
                client_id=self.client_id,
            )
        except Exception:
            return None

        self.access_token = new_token_data.access_token
        self.refresh_token = new_token_data.refresh_token or self.refresh_token
        return self.access_token

    def auth_flow(self, request: Request) -> Generator[Request, Response, None]:
        """HTTPX auth flow that handles token refresh on authentication failures.

        On 401, attempts to refresh the token. If refresh produces a new token,
        retries the request. Otherwise, returns the 401 response as-is so the
        client receives the actual error.
        """
        token = self.get_access_token()
        if token:
            request.headers["Authorization"] = f"Bearer {token}"
        response = yield request
        if response.status_code == 401:
            token = self.get_access_token(refresh=True)
            if token:
                request.headers["Authorization"] = f"Bearer {token}"
                yield request
