from collections.abc import Generator

from httpx import Auth, Client, HTTPStatusError, Request, Response

from uipath_llm_client.settings.llmgateway.settings import LLMGatewayBaseSettings
from uipath_llm_client.settings.llmgateway.utils import LLMGatewayEndpoints
from uipath_llm_client.settings.utils import SingletonMeta
from uipath_llm_client.utils.exceptions import UiPathAPIError


class LLMGatewayS2SAuth(Auth, metaclass=SingletonMeta):
    """Bearer authentication handler with automatic token refresh.

    Singleton class that reuses the same token across all requests to minimize
    token generation overhead. Automatically refreshes the token on 401 responses.
    """

    def __init__(
        self,
        settings: LLMGatewayBaseSettings,
    ):
        self.settings = settings
        if self.settings.access_token is None:
            self.access_token = self.get_llmgw_token_header()
        else:
            self.access_token = self.settings.access_token.get_secret_value()

    def get_llmgw_token_header(
        self,
    ) -> str:
        """Retrieve a new access token from the LLM Gateway identity endpoint."""
        url_get_token = f"{self.settings.base_url}/{LLMGatewayEndpoints.IDENTITY_ENDPOINT.value}"
        assert self.settings.client_id is not None
        assert self.settings.client_secret is not None
        token_credentials = dict(
            client_id=self.settings.client_id.get_secret_value(),
            client_secret=self.settings.client_secret.get_secret_value(),
            grant_type="client_credentials",
        )
        with Client() as http_client:
            response = http_client.post(url_get_token, data=token_credentials)
            try:
                response.raise_for_status()
            except HTTPStatusError as e:
                raise UiPathAPIError.from_response(e.response)
            llmgw_token_header = response.json().get("access_token")
            return llmgw_token_header

    def auth_flow(self, request: Request) -> Generator[Request, Response, None]:
        """HTTPX auth flow that handles token refresh on authentication failures."""
        request.headers["Authorization"] = f"Bearer {self.access_token}"
        response = yield request
        if response.status_code == 401:
            self.access_token = self.get_llmgw_token_header()
            request.headers["Authorization"] = f"Bearer {self.access_token}"
            yield request
