from collections.abc import Generator

from httpx import Auth, Client, Request, Response

from uipath.llm_client.settings.llmgateway.settings import LLMGatewayBaseSettings
from uipath.llm_client.settings.llmgateway.utils import LLMGatewayEndpoints
from uipath.llm_client.settings.utils import SingletonMeta


class LLMGatewayS2SAuth(Auth, metaclass=SingletonMeta):
    """Bearer authentication handler with automatic token refresh.

    Singleton class that reuses the same token across all requests to minimize
    token generation overhead. Automatically refreshes the token on 401 responses.

    Does not raise errors on token retrieval failures — the request is sent
    without a valid token and the downstream client handles the error response.
    """

    def __init__(
        self,
        settings: LLMGatewayBaseSettings,
    ):
        self.settings = settings
        if self.settings.access_token is None:
            self.access_token = self.get_llmgw_token()
        else:
            self.access_token = self.settings.access_token.get_secret_value()

    def get_llmgw_token(
        self,
    ) -> str | None:
        """Retrieve a new access token from the LLM Gateway identity endpoint.

        Returns None on failure instead of raising, so the request proceeds
        and the client receives the actual error response from the server.
        """
        if self.settings.client_id is None or self.settings.client_secret is None:

            return None
        url_get_token = f"{self.settings.base_url}/{LLMGatewayEndpoints.IDENTITY_ENDPOINT.value}"
        token_credentials = dict(
            client_id=self.settings.client_id.get_secret_value(),
            client_secret=self.settings.client_secret.get_secret_value(),
            grant_type="client_credentials",
        )
        try:
            with Client() as http_client:
                response = http_client.post(url_get_token, data=token_credentials)
                if response.is_error:
                    return None
                return response.json().get("access_token")
        except Exception:
            return None

    def auth_flow(self, request: Request) -> Generator[Request, Response, None]:
        """HTTPX auth flow that handles token refresh on authentication failures."""
        if self.access_token:
            request.headers["Authorization"] = f"Bearer {self.access_token}"
        response = yield request
        if response.status_code == 401:
            new_token = self.get_llmgw_token()
            if new_token:
                self.access_token = new_token
                request.headers["Authorization"] = f"Bearer {self.access_token}"
                yield request
