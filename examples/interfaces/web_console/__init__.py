"""
web_console — a browser-based testing UI for the harness.

This is an **example**, not part of the SDK. It is bundled with the repo so
contributors can demo and smoke-test `WorldEnvironment` implementations
without writing any frontend code. Production users should build their own
UI against the public `TriggerSource` / `OutputChannel` interfaces.

Run::

    uv sync --group examples
    python -m examples.interfaces.web_console --worlds examples/interfaces/web_console/worlds.toml

Then open http://localhost:8765 in a browser.
"""
