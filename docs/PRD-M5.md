# isotope-core — Milestone 5: Multi-Provider & Model Routing

## Objective

Support dynamic model switching, fallback chains, and OpenAI-compatible proxy providers for maximum deployment flexibility.

## Prerequisites

- M1-M4 merged

## Deliverables

### 1. Router Provider (`src/isotope_core/providers/router.py`)

A provider that wraps other providers with routing logic:

```python
class RouterProvider:
    def __init__(
        self,
        primary: Provider,
        fallbacks: list[Provider] | None = None,
        health_check_interval: float = 60.0,  # seconds
        circuit_breaker_threshold: int = 3,    # consecutive failures to trip
        circuit_breaker_timeout: float = 120.0, # seconds before retry
    ): ...
```

**Features:**
- **Fallback chains** — if primary fails with retryable error, try fallbacks in order
- **Circuit breaker** — after N consecutive failures, skip a provider for a timeout period
  - States: `closed` (healthy), `open` (tripped, skip), `half_open` (testing)
  - On success in `half_open`, reset to `closed`
- **Health tracking** — track per-provider success/failure rates
- **Dynamic switching** — `router.set_primary(provider)` to switch primary mid-session

### 2. Proxy Provider (`src/isotope_core/providers/proxy.py`)

An OpenAI-compatible provider for any API that follows the OpenAI Chat Completions format:

```python
class ProxyProvider:
    def __init__(
        self,
        model: str,
        base_url: str,               # e.g., "http://localhost:11434/v1"
        api_key: str | None = None,   # Some proxies need auth
        default_headers: dict[str, str] | None = None,
        timeout: float = 120.0,
    ): ...
```

**Use cases:**
- LiteLLM proxy
- vLLM / TGI endpoints
- Ollama (with OpenAI-compat mode)
- Azure OpenAI
- Any OpenAI-compatible endpoint

**Implementation:** Reuse the OpenAI provider's message conversion and streaming logic, but with configurable base_url. Can subclass or compose with `OpenAIProvider`.

### 3. Dynamic API Key Resolution

Support dynamic API key refresh for OAuth/expiring tokens:

```python
class OpenAIProvider:
    def __init__(
        self,
        ...
        api_key_resolver: Callable[[], Awaitable[str]] | None = None,
    ): ...
```

- If `api_key_resolver` is provided, call it before each `stream()` to get the current key
- Useful for rotating credentials, OAuth token refresh, shared key pools

Similarly for `AnthropicProvider` and `ProxyProvider`.

### 4. Usage Aggregation

Track usage across provider switches:

```python
@dataclass
class AggregatedUsage:
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_read_tokens: int = 0
    total_cache_write_tokens: int = 0
    provider_usage: dict[str, Usage] = field(default_factory=dict)
    model_usage: dict[str, Usage] = field(default_factory=dict)
```

- `RouterProvider` accumulates usage across all providers
- Expose via `router.get_usage()` or through events

### 5. Provider Info

Add a way to identify providers:

```python
class Provider(Protocol):
    @property
    def model_name(self) -> str: ...
    @property
    def provider_name(self) -> str: ...
```

This helps with logging, usage tracking, and identifying which provider actually served a request in fallback scenarios.

### 6. Tests

- `tests/test_router.py`:
  - Primary succeeds — no fallback used
  - Primary fails — fallback used
  - Multiple fallbacks in order
  - Circuit breaker trips after threshold failures
  - Circuit breaker half-open → success → close
  - Circuit breaker half-open → failure → reopen
  - Dynamic primary switching
  - Usage aggregation across providers
- `tests/test_proxy.py`:
  - Basic streaming with custom base_url
  - Custom headers
  - Auth token support
  - Context/tool schema conversion (same as OpenAI tests)
- `tests/test_api_key_resolver.py`:
  - Dynamic key resolution called on each stream
  - Resolver error handling
  - Key rotation during session

## Technical Constraints

- Circuit breaker must be thread-safe (asyncio locks)
- Router must not swallow non-retryable errors
- Proxy provider reuses OpenAI message format conversion
- API key resolver is optional and backward-compatible
- Provider info properties have defaults (not breaking change)
- All type-safe — mypy strict

## Branch

`feat/isotope-core/dev-m5`

## Definition of Done

- [ ] Router provider with fallback chains works
- [ ] Circuit breaker with closed/open/half_open states
- [ ] Proxy provider works with custom base_url
- [ ] Dynamic API key resolution
- [ ] Usage aggregation across providers
- [ ] Provider info (model_name, provider_name) on all providers
- [ ] All tests pass, ruff/mypy clean
