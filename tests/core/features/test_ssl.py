"""Tests for SSL configuration utilities."""

import os
from unittest.mock import patch


class TestSSLConfig:
    """Tests for SSL configuration utilities."""

    def test_expand_path_empty(self):
        from uipath.llm_client.utils.ssl_config import expand_path

        assert expand_path("") == ""
        assert expand_path(None) is None

    def test_expand_path_tilde(self):
        from uipath.llm_client.utils.ssl_config import expand_path

        result = expand_path("~/test")
        assert result is not None
        assert "~" not in result
        assert result.endswith("/test")

    def test_expand_path_env_var(self):
        from uipath.llm_client.utils.ssl_config import expand_path

        with patch.dict(os.environ, {"MY_PATH": "/custom"}):
            result = expand_path("$MY_PATH/cert.pem")
            assert result == "/custom/cert.pem"

    def test_get_httpx_ssl_client_kwargs_default(self):
        from uipath.llm_client.utils.ssl_config import get_httpx_ssl_client_kwargs

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("UIPATH_DISABLE_SSL_VERIFY", None)
            kwargs = get_httpx_ssl_client_kwargs()
            assert kwargs["follow_redirects"] is True
            assert kwargs["verify"] is not False  # Should be an SSL context

    def test_get_httpx_ssl_client_kwargs_disable_ssl(self):
        from uipath.llm_client.utils.ssl_config import get_httpx_ssl_client_kwargs

        for val in ("1", "true", "yes", "on", "TRUE", "True"):
            with patch.dict(os.environ, {"UIPATH_DISABLE_SSL_VERIFY": val}):
                kwargs = get_httpx_ssl_client_kwargs()
                assert kwargs["verify"] is False, f"Failed for value: {val}"

    def test_get_httpx_ssl_client_kwargs_not_disabled(self):
        from uipath.llm_client.utils.ssl_config import get_httpx_ssl_client_kwargs

        for val in ("0", "false", "no", "off", ""):
            with patch.dict(os.environ, {"UIPATH_DISABLE_SSL_VERIFY": val}):
                kwargs = get_httpx_ssl_client_kwargs()
                assert kwargs["verify"] is not False, f"Should not disable for value: {val}"


class TestCreateSSLContext:
    """Tests for the create_ssl_context function."""

    def test_default_returns_ssl_context(self):
        import ssl

        from uipath.llm_client.utils.ssl_config import create_ssl_context

        ctx = create_ssl_context()
        assert isinstance(ctx, ssl.SSLContext)

    def test_custom_ssl_cert_file(self, tmp_path):
        import ssl

        cert_file = tmp_path / "ca-bundle.crt"
        cert_file.write_text("")

        with (
            patch.dict(os.environ, {"SSL_CERT_FILE": str(cert_file)}),
            patch("uipath.llm_client.utils.ssl_config.ssl.create_default_context") as mock_ctx,
        ):
            mock_ctx.return_value = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            # Force certifi fallback by hiding truststore
            with patch.dict("sys.modules", {"truststore": None}):
                import importlib

                import uipath.llm_client.utils.ssl_config as ssl_mod

                importlib.reload(ssl_mod)
                ssl_mod.create_ssl_context()
                mock_ctx.assert_called_once()
                call_kwargs = mock_ctx.call_args
                assert call_kwargs.kwargs.get("cafile") == str(cert_file) or (
                    call_kwargs.args and call_kwargs.args[0] is not None
                )

    def test_custom_ssl_cert_dir(self):
        import ssl

        with (
            patch.dict(os.environ, {"SSL_CERT_DIR": "/custom/certs"}, clear=False),
            patch("uipath.llm_client.utils.ssl_config.ssl.create_default_context") as mock_ctx,
        ):
            mock_ctx.return_value = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            with patch.dict("sys.modules", {"truststore": None}):
                import importlib

                import uipath.llm_client.utils.ssl_config as ssl_mod

                importlib.reload(ssl_mod)
                ssl_mod.create_ssl_context()
                mock_ctx.assert_called_once()
                assert mock_ctx.call_args.kwargs.get("capath") == "/custom/certs"

    def test_requests_ca_bundle(self, tmp_path):
        import ssl

        bundle = tmp_path / "bundle.pem"
        bundle.write_text("")

        env = {"REQUESTS_CA_BUNDLE": str(bundle)}
        # Remove SSL_CERT_FILE so REQUESTS_CA_BUNDLE is used
        with (
            patch.dict(os.environ, env, clear=False),
            patch("uipath.llm_client.utils.ssl_config.ssl.create_default_context") as mock_ctx,
        ):
            os.environ.pop("SSL_CERT_FILE", None)
            mock_ctx.return_value = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            with patch.dict("sys.modules", {"truststore": None}):
                import importlib

                import uipath.llm_client.utils.ssl_config as ssl_mod

                importlib.reload(ssl_mod)
                ssl_mod.create_ssl_context()
                mock_ctx.assert_called_once()
                assert mock_ctx.call_args.kwargs.get("cafile") == str(bundle)

    def test_fallback_to_certifi_when_truststore_unavailable(self):
        import ssl

        with (
            patch.dict("sys.modules", {"truststore": None}),
            patch("uipath.llm_client.utils.ssl_config.ssl.create_default_context") as mock_ctx,
        ):
            mock_ctx.return_value = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            import importlib

            import uipath.llm_client.utils.ssl_config as ssl_mod

            importlib.reload(ssl_mod)
            ctx = ssl_mod.create_ssl_context()
            assert isinstance(ctx, ssl.SSLContext)
            mock_ctx.assert_called_once()
