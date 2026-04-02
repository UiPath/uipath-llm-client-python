import os
import ssl
from typing import Any


def expand_path(path: str | None) -> str | None:
    """Expand environment variables and user home directory in path."""
    if not path:
        return path
    # Expand environment variables like $HOME
    path = os.path.expandvars(path)
    # Expand user home directory ~
    path = os.path.expanduser(path)
    return path


def create_ssl_context() -> ssl.SSLContext:
    """Create an SSL context using system certificates.

    Tries ``truststore`` first for native system certificate support.
    Falls back to ``certifi`` for bundled Mozilla CA certificates.

    Raises:
        ImportError: If neither ``truststore`` nor ``certifi`` is installed.
    """
    # Try truststore first (system certificates)
    try:
        import truststore

        return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    except ImportError:
        pass

    # Fallback to manual certificate configuration
    try:
        import certifi
    except ImportError:
        raise ImportError(
            "SSL certificate support requires either 'truststore' or 'certifi'. "
            "Install one with: pip install truststore  or  pip install certifi"
        )

    ssl_cert_file = expand_path(os.environ.get("SSL_CERT_FILE"))
    requests_ca_bundle = expand_path(os.environ.get("REQUESTS_CA_BUNDLE"))
    ssl_cert_dir = expand_path(os.environ.get("SSL_CERT_DIR"))

    return ssl.create_default_context(
        cafile=ssl_cert_file or requests_ca_bundle or certifi.where(),
        capath=ssl_cert_dir,
    )


def get_httpx_ssl_client_kwargs() -> dict[str, Any]:
    """Get standardized httpx client configuration."""
    client_kwargs: dict[str, Any] = {"follow_redirects": True}

    # Check environment variable to disable SSL verification
    disable_ssl_env = os.environ.get("UIPATH_DISABLE_SSL_VERIFY", "").lower()
    disable_ssl_from_env = disable_ssl_env in ("1", "true", "yes", "on")

    if disable_ssl_from_env:
        client_kwargs["verify"] = False
    else:
        # Use system certificates with truststore fallback
        client_kwargs["verify"] = create_ssl_context()

    # Auto-detect proxy from environment variables (httpx handles this automatically)
    # HTTP_PROXY, HTTPS_PROXY, NO_PROXY are read by httpx by default

    return client_kwargs
