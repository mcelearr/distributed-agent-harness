# Examples

Reference implementations that show how to extend the
`distributed-agent-harness` SDK. **None of this code is part of the
published package** — these examples are bundled with the repository so
contributors can demo the SDK and use them as a starting point for their
own integrations.

If you want to use one of these patterns in production, **copy the file
into your own project** rather than depending on `examples.*`. The
import paths under `examples/` are not part of the SDK's stability
contract and may change without notice.

## Layout

Examples are grouped by which SDK extension point they demonstrate:

```
examples/
├── interfaces/   ← TriggerSource / OutputChannel implementations
│   └── web_console/         a browser-based testing console
├── use_cases/    ← WorldEnvironment implementations (domain logic)
│   └── data_protection/     full GDPR engagement lifecycle
└── adapters/     ← NamespaceAdapter / EventLog / MessageBus backends
                    (reserved for future examples like a Kafka emulator)
```

## Running an example

The standalone `run.py` scripts have zero LLM dependency — they
exercise the world environment directly:

```bash
python -m examples.use_cases.data_protection.run
```

The web console requires an LLM key and the `examples` dependency group.
The key lives in a local `.env` file (git-ignored); `uv run --env-file`
loads it into the subprocess for one command:

```bash
uv sync --group examples
cp .env.example .env              # then edit .env and paste in your key
uv run --env-file .env python -m examples.interfaces.web_console
# → http://localhost:8765
```

## Tests

Each example owns its tests under its own `tests/` directory. The top-level
`pytest` configuration discovers them automatically:

```bash
pytest examples/use_cases/data_protection
pytest examples/interfaces/web_console
```
