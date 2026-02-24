---
name: background-testing
description: Rules and guidelines for writing pytest tests for Backgrounds in OM Cortex Runtime
---

# Background Testing Guide

This document provides rules, structure, and patterns for writing pytest tests for **Backgrounds** in OM Cortex Runtime. Use this guide when adding or reviewing Background tests so that tests are consistent, mock the Provider used inside the Background, and run without hardware.

## Quick Checklist

Before submitting Background tests, verify:

- [ ] Test file: `tests/backgrounds/test_{name}_bg.py`
- [ ] Fixtures for config (e.g. `config`, `config_default` or `config_with_*`)
- [ ] Patch the **Provider** used inside the Background (e.g. `@patch("backgrounds.plugins.xxx_bg.XxxProvider")`)
- [ ] Tests: config init, Background init (Provider called with correct args, `start()` called), name, config access, `run()`, init failure
- [ ] No real hardware or Provider implementation required to run

---

## 1. File location and naming

- **Path**: `tests/backgrounds/test_{name}_bg.py`
- **Naming**: `test_*.py`; test functions named `test_*`.

**Examples**:
- `tests/backgrounds/test_pointcloud_bg.py` → PointCloudBg
- `tests/backgrounds/test_unitree_go2_bg.py` → UnitreeGo2Bg

---

## 2. Fixtures

Config fixtures matching the Background's Config class. No `reset_singleton` is needed for the Background class itself; the Provider is mocked so the real singleton is not used.

```python
@pytest.fixture
def config():
    """Default config."""
    return UnitreeGo2BgConfig()


@pytest.fixture
def config_with_ethernet():
    """Config with ethernet channel and custom timeout."""
    return UnitreeGo2BgConfig(unitree_ethernet="eth0", timeout=10.0)
```

---

## 3. Mocking the Provider

**REQUIRED**: Patch the **Provider** that the Background instantiates. Patch path must be the **module where the Provider is used** (the Background module), not the Provider module.

```python
@patch("backgrounds.plugins.unitree_go2_bg.UnitreeGo2Provider")
def test_background_initialization(mock_provider_class, config):
    mock_provider_instance = MagicMock()
    mock_provider_class.return_value = mock_provider_instance

    background = UnitreeGo2Bg(config=config)

    mock_provider_class.assert_called_once_with(channel="", timeout=1.0)
    assert background.unitree_go2_provider is mock_provider_instance
    mock_provider_instance.start.assert_called_once()
```

This avoids starting the real Provider (and any HW/SDK).

---

## 4. What to test

| Area | What to verify |
|------|----------------|
| **Config** | Config class default and custom values (e.g. `UnitreeGo2BgConfig()`, `UnitreeGo2BgConfig(unitree_ethernet="eth0", timeout=10.0)`). |
| **Initialization** | With Provider mocked: Background constructs Provider with arguments derived from config (channel, timeout, etc.); stores the instance; calls `provider.start()` once. |
| **Config derivation** | If config has optional/whitespace fields (e.g. ethernet), assert Provider is called with stripped/defaulted values. |
| **Name** | `background.name` equals the class name (e.g. `"UnitreeGo2Bg"`). |
| **Config access** | `background.config` is the same object as the injected config; attributes match. |
| **run()** | If `run()` only sleeps, patch `time.sleep` and assert it was called with the expected argument (e.g. `60`). |
| **Init failure** | When Provider constructor or init raises, Background `__init__` propagates the exception (e.g. `pytest.raises(RuntimeError)`). |

---

## 5. Test structure

Prefer one test per behavior; use `@patch` on the test function or use `with patch(...):` inside the test.

```python
def test_config_initialization():
    cfg = UnitreeGo2BgConfig()
    assert cfg.unitree_ethernet is None
    assert cfg.timeout == 1.0


@patch("backgrounds.plugins.unitree_go2_bg.UnitreeGo2Provider")
def test_background_initialization(mock_provider_class, config):
    mock_provider_class.return_value = MagicMock()
    background = UnitreeGo2Bg(config=config)
    mock_provider_class.assert_called_once_with(channel="", timeout=1.0)
    mock_provider_class.return_value.start.assert_called_once()


@patch("backgrounds.plugins.unitree_go2_bg.UnitreeGo2Provider")
@patch("backgrounds.plugins.unitree_go2_bg.time.sleep")
def test_run_sleeps(mock_sleep, mock_provider_class, config):
    mock_provider_class.return_value = MagicMock()
    background = UnitreeGo2Bg(config=config)
    background.run()
    mock_sleep.assert_called_once_with(60)
```

### 5.1 `@patch` parameter order

When stacking multiple `@patch` decorators, the **bottom** (closest to the function) decorator is applied first, and its mock is the **first** parameter:

```python
@patch("backgrounds.plugins.unitree_go2_bg.UnitreeGo2Provider")  # second param
@patch("backgrounds.plugins.unitree_go2_bg.time.sleep")         # first param
def test_run_sleeps(mock_sleep, mock_provider_class, config):
    ...
```

---

## 6. Common patterns (pytest / mock)

### 6.1 `MagicMock` return_value

- `mock_provider_class.return_value = MagicMock()`: When the Background does `Provider(...)`, it gets this instance instead of a real Provider.

### 6.2 Asserting calls

- `mock_provider_class.assert_called_once_with(channel="", timeout=1.0)`: Provider was constructed once with these arguments.
- `mock_provider_instance.start.assert_called_once()`: `start()` was called once on the instance.

### 6.3 Init failure

```python
@patch("backgrounds.plugins.unitree_go2_bg.UnitreeGo2Provider")
def test_init_raises_on_provider_failure(mock_provider_class, config):
    mock_provider_class.side_effect = RuntimeError("DDS init failed")

    with pytest.raises(RuntimeError, match="DDS init failed"):
        UnitreeGo2Bg(config=config)
```

---

## 7. Running Background tests

```bash
uv run pytest tests/backgrounds/test_unitree_go2_bg.py -v
```

---

## 8. Review checklist

- [ ] File under `tests/backgrounds/test_*_bg.py`
- [ ] Config fixture(s); Provider patched in Background module
- [ ] Config initialization and Background initialization (Provider args + `start()`) tested
- [ ] Name, config access, `run()` (e.g. sleep) and init failure tested
- [ ] Tests pass with `uv run pytest tests/backgrounds/test_*_bg.py -v` (no HW)

---

## 9. Reference examples

- `tests/backgrounds/test_unitree_go2_bg.py`
- `tests/backgrounds/test_pointcloud_bg.py`

Implementation guide: `.cursor/skills/background-implementation/SKILL.md`
