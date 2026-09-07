"""Point a downstream uv project at locally built uipath* wheels.

Appends ``tool.uv.override-dependencies`` entries for every ``uipath*`` wheel
found under ``$WHEELS_DIR`` (default: ``./wheels``) to the ``pyproject.toml``
given as ``argv[1]``. uv overrides bypass version specifiers, so a downstream
cap like ``uipath-langchain-client<1.19.0`` cannot mask the new code, and they
apply to project commands (``uv sync``/``uv add``) where the ``UV_OVERRIDE``
env var is silently ignored.

An override replaces the whole requirement, dropping any extras the downstream
graph asked for (e.g. ``uipath-langchain-client[openai]``), so EXTRAS pins the
union of extras each overridden package must keep providing.
"""

import glob
import os
import pathlib
import re
import sys

EXTRAS: dict[str, str] = {
    "uipath-langchain-client": "all",
}


def main() -> None:
    pyproject = pathlib.Path(sys.argv[1])
    wheels = pathlib.Path(os.environ.get("WHEELS_DIR", "wheels")).resolve()

    entries: list[str] = []
    for whl in sorted(glob.glob(str(wheels / "**" / "*.whl"), recursive=True)):
        # Wheel filename is ``{distribution}-{version}-...whl`` where the
        # distribution escapes hyphens to underscores (uipath_llm_client ->
        # uipath-llm-client).
        dist = pathlib.Path(whl).name.split("-", 1)[0].replace("_", "-")
        if not dist.startswith("uipath"):
            continue
        extra = f"[{EXTRAS[dist]}]" if dist in EXTRAS else ""
        entries.append(f'    "{dist}{extra} @ {pathlib.Path(whl).as_uri()}",')

    if not entries:
        raise SystemExit(f"no uipath wheels found under {wheels}")

    block = "override-dependencies = [\n" + "\n".join(entries) + "\n]\n"
    text = pyproject.read_text()
    if re.search(r"^\[tool\.uv\]$", text, flags=re.M):
        text = re.sub(r"^\[tool\.uv\]\n", "[tool.uv]\n" + block, text, count=1, flags=re.M)
    else:
        text = text.rstrip() + "\n\n[tool.uv]\n" + block
    pyproject.write_text(text)
    print(f"{pyproject}:\n{block}")


if __name__ == "__main__":
    main()
