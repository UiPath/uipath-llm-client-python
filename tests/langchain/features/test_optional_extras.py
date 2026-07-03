"""The package must import without optional provider extras installed."""

import subprocess
import sys

_OPTIONAL_MODULES = [
    "anthropic",
    "google.genai",
    "langchain_anthropic",
    "langchain_aws",
    "langchain_azure_ai",
    "langchain_fireworks",
    "langchain_google_genai",
    "langchain_google_vertexai",
    "langchain_litellm",
    "langchain_openai",
    "litellm",
    "openai",
]

_IMPORT_WITH_BLOCKED_EXTRAS = """
import sys

blocked = set(sys.argv[1:])


class _BlockOptionalDeps:
    def find_spec(self, name, path=None, target=None):
        if any(name == module or name.startswith(module + ".") for module in blocked):
            raise ImportError(f"blocked optional dependency: {name}")
        return None


sys.meta_path.insert(0, _BlockOptionalDeps())

import uipath_langchain_client

providers = [
    name
    for name in sys.modules
    if name.startswith("uipath_langchain_client.clients.")
    and not name.startswith("uipath_langchain_client.clients.normalized")
]
assert not providers, providers
print("OK")
"""


def test_package_imports_without_optional_extras():
    result = subprocess.run(
        [sys.executable, "-c", _IMPORT_WITH_BLOCKED_EXTRAS, *_OPTIONAL_MODULES],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "OK"
