# isotope-core — Milestone 7: Testing, CI & Documentation

## Objective

Harden the library with comprehensive test coverage (90%+), CI pipeline, and user-facing documentation.

## Prerequisites

- M1-M6 merged

## Deliverables

### 1. Coverage Gap Analysis & New Tests

Run `pytest --cov` and identify uncovered branches. Add tests to reach 90%+ line coverage across all source files.

**Priority areas to cover:**
- Edge cases in provider streaming (partial streams, empty responses, connection drops)
- Router circuit breaker state transitions (timing-dependent edge cases)
- Middleware error propagation paths
- Agent state management (concurrent abort+steer, abort during continue_)
- Context pruning with complex message histories (mixed pinned/unpinned, edge counts)
- Tool execution error paths (tool raises, before_hook blocks, after_hook modifies)
- Budget limit edge cases (exactly at limit, tokens arrive mid-budget)
- Import guards for optional deps (tiktoken, openai, anthropic)
- Serialization roundtrips for all types

### 2. CI Pipeline (`.github/workflows/ci.yml`)

```yaml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.11", "3.12", "3.13"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - run: uv sync --all-extras
      - run: ruff check .
      - run: mypy src/
      - run: pytest --cov=isotope_core --cov-report=term-missing --cov-fail-under=90
```

### 3. Documentation

#### 3a. README.md (rewrite)

Replace boilerplate README with proper documentation:
- Quick start (install, basic agent loop, stream events)
- Architecture overview (diagram or description)
- Feature list with code examples:
  - Tool registration
  - Context management (pruning, pinning)
  - Steering & follow-up
  - Multi-provider routing
  - Middleware
- API reference pointers
- Contributing guide

#### 3b. API Reference (`docs/API.md`)

Document all public APIs:
- `Agent` class — constructor, methods, properties
- `agent_loop()` function — parameters, event flow
- All event types with field descriptions
- Provider protocol
- Tool class
- Middleware protocol
- Context management functions
- Configuration dataclasses

#### 3c. Examples (`examples/`)

Create working examples:
- `examples/basic_agent.py` — minimal agent with a tool
- `examples/streaming.py` — consuming the event stream
- `examples/context_management.py` — pruning strategies
- `examples/middleware.py` — custom middleware
- `examples/multi_provider.py` — router with fallback

Examples should use mock providers (no real API keys needed to run).

### 4. Package Metadata

Verify `pyproject.toml`:
- Classifiers are correct
- All optional extras declared (`tiktoken`, `openai`, `anthropic`)
- Add `dev` extra with test dependencies (`pytest`, `pytest-cov`, `pytest-asyncio`, `ruff`, `mypy`)
- Verify `py.typed` marker exists for PEP 561
- Entry points / scripts if needed

### 5. Final Cleanup

- Remove any TODO/FIXME comments (or convert to GitHub issues)
- Ensure all `__init__.py` exports are complete and consistent
- Verify all public symbols have docstrings
- Fix the tiktoken `type: ignore[assignment]` to work with AND without tiktoken installed

## Technical Constraints

- Coverage must be ≥90% line coverage
- CI must pass on Python 3.11, 3.12, 3.13
- No new runtime dependencies
- Examples must run without real API keys (use mock providers)
- mypy strict + ruff clean (as always)

## Branch

`feat/isotope-core/dev-m7`

## Definition of Done

- [ ] Coverage ≥90% line coverage
- [ ] CI workflow passing on 3.11, 3.12, 3.13
- [ ] README rewritten with quick start + examples
- [ ] API.md with all public APIs documented
- [ ] 5 working examples in `examples/`
- [ ] pyproject.toml finalized (extras, classifiers, py.typed)
- [ ] All cleanup items resolved
- [ ] ruff + mypy clean, all tests pass
