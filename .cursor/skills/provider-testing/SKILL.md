---
name: provider-testing
description: Rules and guidelines for writing pytest tests for Providers in OM Cortex Runtime
---

# Provider Testing Guide

Rules, structure, and patterns for writing pytest tests for **Providers** in OM Cortex Runtime. Use when adding or reviewing Provider tests so that tests mock external/HW and run without hardware.

## Quick Checklist

- [ ] Test file: `tests/providers/test_{name}_provider.py`
- [ ] `reset_singleton` fixture (autouse) if Provider uses `@singleton`
- [ ] Fixtures for constructor params (e.g. `provider_params`)
- [ ] Mock external dependencies with `patch` / `MagicMock`
- [ ] Tests: initialization, singleton, start/stop, `data`, control/API methods, error paths
- [ ] No real hardware or network required to run

## 1. File location and naming

- **Path**: `tests/providers/test_{name}_provider.py`
- **Naming**: `test_*.py`; test functions/methods named `test_*`.

Examples: `test_pointcloud_provider.py`, `test_unitree_go2_provider.py`.

## 2. Fixtures

### 2.1 Singleton reset (REQUIRED for singleton Providers)

```python
@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton between tests."""
    MyProvider.reset()  # type: ignore
    yield
    MyProvider.reset()  # type: ignore
```

### 2.2 Constructor params (optional)

```python
@pytest.fixture
def provider_params():
    return {"channel": "", "timeout": 10.0}
```

### 2.3 Mock client (optional)

Create a `MagicMock()` and set return values for methods the Provider calls.

```python
@pytest.fixture
def mock_sport_client():
    client = MagicMock()
    client.SetTimeout.return_value = None
    client.Init.return_value = None
    client.StopMove.return_value = 0
    return client
```

## 3. Mocking external dependencies

Provider tests MUST NOT require real hardware or SDK. Patch the name **where it is used** (in the provider module).

```python
with patch("providers.unitree_go2_provider.SportClient", return_value=mock_sport_client):
    provider = UnitreeGo2Provider(**provider_params)
    provider.start()
assert provider._sport_client is mock_sport_client
mock_sport_client.Init.assert_called_once()
```

Use `with patch(...):` or `@patch` on the test function.

## 4. What to test

| Area | Verify |
|------|--------|
| **Initialization** | Constructor stores params; internal state correct (e.g. `_client` None, `_running` False). |
| **Singleton** | Two calls to `MyProvider(...)` return the same instance. |
| **Lifecycle** | `start()` sets state and init client; `stop()` clears state. Idempotent `start()` if applicable. |
| **Data** | `data` property before/after start and after stop. |
| **Control/API** | Methods call mocked client with correct args and return expected True/False. |
| **Error handling** | Client None → False; client raises or error code → False and/or logs. |
| **SDK/import** | Helper like `check_sdk()` tested with mocked imports or skip when SDK absent. |

## 5. Test structure

Group related tests in classes (optional):

```python
class TestUnitreeGo2ProviderInitialization:
    def test_default_initialization(self, provider_params):
        provider = UnitreeGo2Provider(**provider_params)
        assert provider._channel == ""
        assert provider._sport_client is None

    def test_singleton_pattern(self, provider_params):
        p1 = UnitreeGo2Provider(**provider_params)
        p2 = UnitreeGo2Provider(channel="other", timeout=3.0)
        assert p1 is p2
```

## 6. Common patterns

### 6.1 Fixture

- `autouse=True`: runs for every test (e.g. `reset_singleton`).
- Without autouse: injected only when test requests it by argument name.

### 6.2 @patch parameter order

Bottom decorator applied first → first parameter:

```python
@patch("module.A")  # second param
@patch("module.B")  # first param
def test_foo(mock_b, mock_a, config):
    ...
```

### 6.3 MagicMock

- `mock.Method.return_value = 0`: call returns 0.
- `mock.Method.side_effect = RuntimeError("fail")`: call raises.
- `mock.Method.side_effect = [0, RuntimeError()]`: first call returns 0, second raises.

### 6.4 Asserting calls

- `assert_called_once()`, `assert_called_once_with(arg1, arg2)`, `assert_not_called()`.

### 6.5 caplog

Assert log messages on error paths: `assert "not ready" in caplog.text`.

## 7. SW vs HW

These tests are SW-only (mocks, no hardware). For real HW verification use a separate flow (e.g. `@pytest.mark.hardware`, skip by default).

## 8. Running tests

```bash
uv run pytest tests/providers/test_unitree_go2_provider.py -v
```

If no pythonpath in pytest config: `PYTHONPATH=src uv run pytest ...`

## 9. Review checklist

- [ ] File under `tests/providers/test_*_provider.py`
- [ ] Singleton: `reset_singleton` (autouse)
- [ ] SDK/client mocked; init, singleton, start/stop, data, API, errors covered
- [ ] Tests pass without HW

## 10. Reference examples

- `tests/providers/test_unitree_go2_provider.py`
- `tests/providers/test_pointcloud_provider.py`

Implementation: `.cursor/skills/provider-implementation/SKILL.md`
