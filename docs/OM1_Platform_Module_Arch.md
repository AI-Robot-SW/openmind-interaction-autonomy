# OM1 Platform Module Architecture

This document describes the module structure and key files for each module in the OpenMind (OM1) platform.

## Overview

The OM1 platform is organized into 8 main modules:

1. **Runtime Core (SYS-CORE)**: Core runtime execution and configuration management
2. **Input System (PER-INPUT)**: Sensor inputs and data collection
3. **Fuser (AGT-FUSER)**: Input fusion and prompt generation
4. **LLM System (AGT-LLM)**: Language model integration
5. **Action System (NAV-ACTION)**: Action execution and orchestration
6. **Providers (SYS-PROVIDER)**: Hardware/service communication abstraction
7. **Simulators (SYS-SIM)**: Action visualization and debugging
8. **Backgrounds (SYS-BG)**: Background processes and provider initialization

---

## 1. Runtime Core (SYS-CORE)

**Purpose**: Core runtime execution, configuration loading, and system orchestration.

### Key Files

#### Single Mode Runtime
- **`src/runtime/single_mode/cortex.py`**
  - `CortexRuntime`: Main runtime class that orchestrates all subsystems
  - Manages execution loop, hot-reload, and component lifecycle
  - Coordinates: Fuser, ActionOrchestrator, InputOrchestrator, SimulatorOrchestrator, BackgroundOrchestrator

- **`src/runtime/single_mode/config.py`**
  - `RuntimeConfig`: Runtime configuration dataclass
  - `load_config()`: Loads and parses JSON5 configuration files
  - `add_meta()`: Adds metadata (API keys, robot IP, etc.) to component configs
  - Handles config file parsing and component instantiation

- **`src/runtime/single_mode/__init__.py`**
  - Module initialization

#### Multi Mode Runtime
- **`src/runtime/multi_mode/config.py`**
  - `ModeSystemConfig`: Multi-mode system configuration
  - `ModeConfig`: Individual mode configuration
  - `TransitionRule`: Mode transition rules
  - `load_mode_config()`: Loads multi-mode configuration

#### Common Runtime Files
- **`src/runtime/version.py`**
  - Version verification utilities

- **`src/runtime/robotics.py`**
  - Robot-specific initialization (e.g., Unitree SDK loading)

- **`src/runtime/logging.py`**
  - Logging configuration

### Responsibilities

- Config file loading and parsing (JSON5)
- Runtime execution loop management
- Hot-reload functionality
- Component lifecycle management
- Single/Multi mode selection

### Config Definition and Ownership

**Important**: Config file structure is managed by Runtime Core (SYS-CORE), but **each config section is owned and maintained by its respective module developer**.

- **Config Structure**: Runtime Core defines the overall config file structure and loading mechanism
- **Config Content**: Each module developer is responsible for defining and maintaining their own config sections:
  - **PER-INPUT**: Defines `agent_inputs` structure and Sensor config fields
  - **AGT-FUSER**: Defines `system_prompt_base`, `system_governance`, `system_prompt_examples` structure
  - **AGT-LLM**: Defines `cortex_llm` structure and LLM config fields
  - **NAV-ACTION**: Defines `agent_actions` structure and Action config fields
  - **SYS-BG**: Defines `backgrounds` structure and Background config fields
  - **SYS-SIM**: Defines `simulators` structure and Simulator config fields
  - **SYS-CORE**: Defines global runtime settings (`version`, `hertz`, `name`, `api_key`, `URID`, `unitree_ethernet`, `robot_ip`, mode system settings)

When adding new config fields or modifying existing ones, the responsible module developer should:
1. Update their module's config class (e.g., `SensorConfig`, `ActionConfig`, `LLMConfig`)
2. Document the new fields in their module's documentation
3. Update config examples if needed

### Config File Structure and Module Mapping

#### Single Mode Config

Single mode config files define all components at the root level. Each section maps to a specific module:

| Config Field | Module | Description | Example Values |
|-------------|--------|-------------|----------------|
| `version` | **SYS-CORE** | Runtime version (for compatibility checking) | `"v1.0.1"` |
| `hertz` | **SYS-CORE** | Execution frequency (Hz) for cortex loop | `10`, `1`, `0.01` |
| `name` | **SYS-CORE** | Config name identifier | `"bits_basic"`, `"turtle_speak"` |
| `api_key` | **SYS-CORE** | Global API key (passed to all components) | `"openmind_free"`, `"your_api_key"` |
| `URID` | **SYS-CORE** | Unitree Robot ID (for multi-robot deployments) | `"default"`, `"robot_001"` |
| `unitree_ethernet` | **SYS-CORE** | Ethernet adapter name for Unitree robots | `"en0"`, `"enP2p1s0"` |
| `robot_ip` | **SYS-CORE** | Robot IP address (optional, can use .env) | `"192.168.0.241"` |
| `system_prompt_base` | **AGT-FUSER** | Base system prompt for LLM | `"You are a smart, curious, and friendly dog..."` |
| `system_governance` | **AGT-FUSER** | Governance rules (e.g., Asimov's laws) | `"First Law: A robot cannot harm..."` |
| `system_prompt_examples` | **AGT-FUSER** | Example interactions for LLM | `"Here are some examples..."` |
| `agent_inputs` | **PER-INPUT** | Array of Sensor configurations | `[{"type": "Gps"}, {"type": "VLMGemini", "config": {...}}]` |
| `cortex_llm` | **AGT-LLM** | LLM configuration | `{"type": "OpenAILLM", "config": {...}}` |
| `agent_actions` | **NAV-ACTION** | Array of Action configurations | `[{"name": "speak", "connector": "elevenlabs_tts", ...}]` |
| `backgrounds` | **SYS-BG** | Array of Background configurations | `[{"type": "Gps"}, {"type": "UnitreeGo2State"}]` |
| `simulators` | **SYS-SIM** | Array of Simulator configurations (optional) | `[{"type": "WebSim", "config": {...}}]` |

**Example Single Mode Config**:
```json5
{
  "version": "v1.0.1",
  "hertz": 10,
  "name": "bits_basic",
  "unitree_ethernet": "en0",
  "api_key": "openmind_free",
  "system_prompt_base": "You are a smart, curious, and friendly dog...",
  "system_governance": "First Law: A robot cannot harm...",
  "system_prompt_examples": "Here are some examples...",
  "agent_inputs": [
    {
      "type": "VLM_COCO_Local",
      "config": {
        "camera_index": 0
      }
    },
    {
      "type": "UnitreeGo2Battery"
    }
  ],
  "cortex_llm": {
    "type": "OpenAILLM",
    "config": {
      "agent_name": "Bits",
      "history_length": 10
    }
  },
  "agent_actions": [
    {
      "name": "speak",
      "llm_label": "speak",
      "connector": "elevenlabs_tts",
      "config": {
        "voice_id": "TbMNBJ27fH2U0VgpSNko",
        "silence_rate": 0
      }
    }
  ],
  "backgrounds": [
    {
      "type": "UnitreeGo2State"
    }
  ]
}
```

#### Multi Mode Config

Multi mode config files define a mode system with multiple operational modes. Global settings apply to all modes, while mode-specific settings override or extend global settings.

**Global Level (Root)**:

| Config Field | Module | Description | Example Values |
|-------------|--------|-------------|----------------|
| `version` | **SYS-CORE** | Runtime version | `"v1.0.1"` |
| `default_mode` | **SYS-CORE** | Initial mode to activate | `"welcome"`, `"autonomous"` |
| `allow_manual_switching` | **SYS-CORE** | Allow manual mode switching | `true`, `false` |
| `mode_memory_enabled` | **SYS-CORE** | Enable mode memory/state persistence | `true`, `false` |
| `api_key` | **SYS-CORE** | Global API key (passed to all modes) | `"openmind_free"` |
| `URID` | **SYS-CORE** | Unitree Robot ID | `"default"` |
| `unitree_ethernet` | **SYS-CORE** | Ethernet adapter name | `"enP2p1s0"` |
| `robot_ip` | **SYS-CORE** | Robot IP address | `"192.168.0.241"` |
| `system_governance` | **AGT-FUSER** | Global governance rules (applied to all modes) | `"First Law: A robot cannot harm..."` |
| `system_prompt_examples` | **AGT-FUSER** | Global examples (applied to all modes) | `"Here are some examples..."` |
| `cortex_llm` | **AGT-LLM** | Global LLM config (used if mode doesn't override) | `{"type": "OpenAILLM", "config": {...}}` |
| `global_lifecycle_hooks` | **SYS-CORE** | Lifecycle hooks executed for all modes | `[{"hook_type": "on_startup", ...}]` |
| `modes` | **SYS-CORE** | Mode definitions (see Mode Level below) | `{"welcome": {...}, "slam": {...}}` |
| `transition_rules` | **SYS-CORE** | Rules for transitioning between modes | `[{"from_mode": "welcome", "to_mode": "slam", ...}]` |

**Mode Level** (within `modes.<mode_name>`):

| Config Field | Module | Description | Example Values |
|-------------|--------|-------------|----------------|
| `display_name` | **SYS-CORE** | Human-readable mode name | `"Welcome Mode"`, `"SLAM Exploration"` |
| `description` | **SYS-CORE** | Mode description | `"Initial greeting and user information gathering"` |
| `system_prompt_base` | **AGT-FUSER** | Mode-specific system prompt | `"You are Bits in exploration mode..."` |
| `hertz` | **SYS-CORE** | Mode-specific execution frequency | `1`, `0.01` |
| `timeout_seconds` | **SYS-CORE** | Mode timeout (optional) | `300.0` |
| `save_interactions` | **SYS-CORE** | Save interactions for this mode | `true`, `false` |
| `remember_locations` | **SYS-CORE** | Remember locations in this mode | `true`, `false` |
| `cortex_llm` | **AGT-LLM** | Mode-specific LLM (overrides global) | `{"type": "OpenAILLM", "config": {...}}` |
| `agent_inputs` | **PER-INPUT** | Mode-specific Sensor configurations | `[{"type": "Odom"}, {"type": "VLMVilaRTSP"}]` |
| `agent_actions` | **NAV-ACTION** | Mode-specific Action configurations | `[{"name": "speak", "connector": "elevenlabs_tts", ...}]` |
| `backgrounds` | **SYS-BG** | Mode-specific Background configurations | `[{"type": "UnitreeGo2State"}]` |
| `simulators` | **SYS-SIM** | Mode-specific Simulator configurations | `[{"type": "WebSim"}]` |
| `lifecycle_hooks` | **SYS-CORE** | Mode-specific lifecycle hooks | `[{"hook_type": "on_entry", "handler_type": "message", ...}]` |

**Transition Rules** (within `transition_rules`):

| Config Field | Module | Description | Example Values |
|-------------|--------|-------------|----------------|
| `from_mode` | **SYS-CORE** | Source mode name (or `"*"` for any mode) | `"welcome"`, `"*"` |
| `to_mode` | **SYS-CORE** | Target mode name | `"slam"`, `"navigation"` |
| `transition_type` | **SYS-CORE** | Transition trigger type | `"input_triggered"`, `"time_based"`, `"context_aware"` |
| `trigger_keywords` | **SYS-CORE** | Keywords that trigger transition | `["explore", "map", "navigate"]` |
| `priority` | **SYS-CORE** | Rule priority (higher = more important) | `1`, `5` |
| `cooldown_seconds` | **SYS-CORE** | Minimum time before rule can trigger again | `5.0`, `10.0` |
| `timeout_seconds` | **SYS-CORE** | For time-based transitions | `300.0` |
| `context_conditions` | **SYS-CORE** | Context conditions for context-aware transitions | `{"exploration_done": true}` |

**Example Multi Mode Config**:
```json5
{
  "version": "v1.0.1",
  "default_mode": "welcome",
  "allow_manual_switching": true,
  "mode_memory_enabled": true,
  "api_key": "openmind_free",
  "unitree_ethernet": "enP2p1s0",
  "system_governance": "First Law: A robot cannot harm...",
  "cortex_llm": {
    "type": "OpenAILLM",
    "config": {
      "agent_name": "Bits",
      "history_length": 10
    }
  },
  "modes": {
    "welcome": {
      "display_name": "Welcome Mode",
      "description": "Initial greeting and user information gathering",
      "system_prompt_base": "You are Bits, a friendly robotic dog...",
      "hertz": 0.01,
      "agent_inputs": [
        {"type": "Odom"},
        {"type": "GoogleASRRTSPInput"}
      ],
      "agent_actions": [
        {
          "name": "speak",
          "llm_label": "speak",
          "connector": "elevenlabs_tts",
          "config": {
            "voice_id": "TbMNBJ27fH2U0VgpSNko",
            "silence_rate": 0
          }
        }
      ],
      "backgrounds": [
        {"type": "UnitreeGo2State"}
      ],
      "lifecycle_hooks": [
        {
          "hook_type": "on_startup",
          "handler_type": "message",
          "handler_config": {
            "message": "Hello! I'm Bits, your robotic companion."
          }
        }
      ]
    },
    "slam": {
      "display_name": "SLAM Exploration",
      "description": "Autonomous navigation and mapping mode",
      "system_prompt_base": "You are Bits in exploration mode...",
      "hertz": 1,
      "agent_inputs": [
        {"type": "Odom"},
        {"type": "VLMVilaRTSP"}
      ],
      "agent_actions": [
        {
          "name": "speak",
          "llm_label": "speak",
          "connector": "elevenlabs_tts",
          "config": {
            "voice_id": "TbMNBJ27fH2U0VgpSNko",
            "silence_rate": 20
          }
        }
      ],
      "backgrounds": [
        {"type": "UnitreeGo2State"},
        {"type": "UnitreeGo2Navigation"}
      ]
    }
  },
  "transition_rules": [
    {
      "from_mode": "welcome",
      "to_mode": "slam",
      "transition_type": "input_triggered",
      "trigger_keywords": ["explore", "map", "navigate"],
      "priority": 3,
      "cooldown_seconds": 5.0
    }
  ]
}
```

### Config Field Details

#### Sensor Config (`agent_inputs[].config`)

Defined by **PER-INPUT** module developers. Each Sensor plugin defines its own config structure.

**Common Fields** (defined by Sensor base class):
- Inherited from `SensorConfig` base class

**Plugin-Specific Fields** (examples):
- `Gps`: No additional config fields (uses default)
- `VLMGemini`: `api_key`, `base_url`, `camera_index`, `stream_base_url`
- `RPLidar`: `half_width_robot`, `use_zenoh`, `relevant_distance_max`, `sensor_mounting_angle`, `angles_blanked`
- `Odom`: `use_zenoh`, `URID`, `unitree_ethernet`

#### LLM Config (`cortex_llm.config`)

Defined by **AGT-LLM** module developers. Each LLM plugin defines its own config structure.

**Common Fields**:
- `agent_name`: Name of the agent (used in prompts)
- `history_length`: Number of previous interactions to include in context

**Plugin-Specific Fields** (examples):
- `OpenAILLM`: `model` (e.g., `"gpt-4.1"`, `"gpt-4"`)
- `GeminiLLM`: `model`, `temperature`
- `DeepSeekLLM`: `model`, `temperature`

#### Action Config (`agent_actions[].config`)

Defined by **NAV-ACTION** module developers. Each Action connector defines its own config structure.

**Common Fields** (at action level):
- `name`: Action name (must match action directory name)
- `llm_label`: Label shown to LLM in function schema
- `connector`: Connector name (must match connector file name)
- `exclude_from_prompt`: Whether to exclude from LLM prompt (default: `false`)
- `implementation`: `"passthrough"` or custom implementation

**Connector-Specific Fields** (examples):
- `elevenlabs_tts`: `voice_id`, `silence_rate`
- `unitree_go2_nav`: `timeout_seconds`, `goal_tolerance`
- `zenoh`: `topic`, `message_type`

#### Background Config (`backgrounds[].config`)

Defined by **SYS-BG** module developers. Each Background plugin defines its own config structure.

**Common Fields**:
- Inherited from `BackgroundConfig` base class

**Plugin-Specific Fields** (examples):
- `Gps`: `serial_port`, `baud_rate`
- `UnitreeGo2State`: No additional config (uses default)
- `UnitreeGo2Navigation`: `nav2_config_path`, `use_sim_time`

#### Simulator Config (`simulators[].config`)

Defined by **SYS-SIM** module developers. Each Simulator plugin defines its own config structure.

**Common Fields**:
- Inherited from `SimulatorConfig` base class

**Plugin-Specific Fields** (examples):
- `WebSim`: `port`, `host`
- `ConsoleSim`: No additional config (uses default)

---

## 2. Input System (PER-INPUT)

**Purpose**: Sensor inputs that convert raw data into LLM-readable text format.

### Key Files

#### Base Classes
- **`src/inputs/base/__init__.py`**
  - `Sensor`: Base class for all input sensors
  - `SensorConfig`: Base configuration class
  - `Message`: Timestamped message container
  - Defines interface: `_raw_to_text()`, `raw_to_text()`, `formatted_latest_buffer()`, `listen()`

- **`src/inputs/base/loop.py`**
  - `FuserInput`: Polling-based input implementation
  - `_poll()`: Polls for new input events
  - `_listen_loop()`: Main polling loop

#### Orchestration
- **`src/inputs/orchestrator.py`**
  - `InputOrchestrator`: Manages multiple input sources concurrently
  - `listen()`: Starts listening to all inputs
  - `_listen_to_input()`: Processes events from a single input

#### Plugin Loading
- **`src/inputs/__init__.py`**
  - `load_input()`: Dynamically loads input plugins
  - `find_module_with_class()`: Finds input class in plugin modules

#### Example Plugins
- **`src/inputs/plugins/gps.py`**: GPS sensor input
- **`src/inputs/plugins/amcl_localization_input.py`**: AMCL localization status
- **`src/inputs/plugins/vlm_gemini.py`**: Vision Language Model input
- **`src/inputs/plugins/twitter.py`**: Twitter query input

### Structure

```
src/inputs/
├── base/
│   ├── __init__.py          # Sensor base class
│   └── loop.py              # FuserInput polling implementation
├── orchestrator.py          # Input orchestration
├── __init__.py              # Plugin loading
└── plugins/                 # Input plugin implementations
    ├── gps.py
    ├── amcl_localization_input.py
    ├── vlm_gemini.py
    └── ...
```

### Responsibilities

- Convert raw sensor data → LLM-readable text
- Poll/stream data from providers or direct hardware
- Format data for Fuser consumption
- Manage input buffers

### Provider Usage Patterns

Sensors can use providers in three ways:

#### Pattern 1: Direct Property Access (Most Common)
```python
class Gps(FuserInput[SensorConfig, Optional[dict]]):
    def __init__(self, config: SensorConfig):
        self.gps = GpsProvider()  # Singleton instance
    
    async def _poll(self) -> Optional[dict]:
        return self.gps.data  # Direct property access
```

#### Pattern 2: Method Call
```python
class UnitreeG1LocationsInput(FuserInput[...]):
    def __init__(self, config: ...):
        self.locations_provider = UnitreeG1LocationsProvider(...)
    
    async def _poll(self) -> Optional[str]:
        locations = self.locations_provider.get_all_locations()  # Method call
        return formatted_locations
```

#### Pattern 3: Callback Registration (Async Events)
```python
class VLMGemini(FuserInput[VLMGeminiConfig, Optional[str]]):
    def __init__(self, config: VLMGeminiConfig):
        self.vlm = VLMGeminiProvider(...)
        self.vlm.start()
        self.vlm.register_message_callback(self._handle_vlm_message)  # Callback
        self.message_buffer = Queue()
    
    def _handle_vlm_message(self, raw_message: ChatCompletion):
        """Provider calls this callback when data arrives"""
        content = raw_message.choices[0].message.content
        if content:
            self.message_buffer.put(content)
    
    async def _poll(self) -> Optional[str]:
        """Poll queue for new messages"""
        try:
            return self.message_buffer.get_nowait()
        except Empty:
            return None
```

---

## 3. Fuser (AGT-FUSER)

**Purpose**: Combines all inputs into a single formatted prompt for LLM processing.

### Key Files

- **`src/fuser/__init__.py`**
  - `Fuser`: Main fusion class
  - `fuse()`: Combines inputs, system prompts, and action descriptions into final prompt
  - Integrates: system prompts, input buffers, action descriptions, command prompts

### Responsibilities

- Combine all sensor inputs into single prompt
- Integrate system prompts (base, governance, examples)
- Add action descriptions to prompt
- Format final prompt for LLM

### Automatic Integration

- **No modification required** when adding new sensors
- New sensors are automatically included via `formatted_latest_buffer()`
- Fuser calls `sensor.formatted_latest_buffer()` for all sensors in the input list

### Code Example

```python
# src/fuser/__init__.py
class Fuser:
    def fuse(self, inputs: list[Sensor], finished_promises: list) -> str:
        # Collect all input buffers
        input_strings = [input.formatted_latest_buffer() for input in inputs]
        
        # Combine system prompts
        system_prompt = self.config.system_prompt_base + ...
        
        # Add action descriptions
        actions_fused = ""
        for action in self.config.agent_actions:
            actions_fused += describe_action(...)
        
        # Final fused prompt
        return f"{system_prompt}\n\nAVAILABLE INPUTS:\n{inputs_fused}..."
```

---

## 4. LLM System (AGT-LLM)

**Purpose**: Language model integration and response generation.

### Key Files

#### Base Classes
- **`src/llm/__init__.py`**
  - `LLM`: Base class for LLM implementations
  - `LLMConfig`: Base configuration class
  - `load_llm()`: Dynamically loads LLM plugins
  - `get_llm_class()`: Gets LLM class by name
  - `find_module_with_class()`: Finds LLM class in plugins

#### Plugin Loading
- **`src/llm/plugins/__init__.py`**
  - Plugin module initialization

#### Example Plugins
- **`src/llm/plugins/openai_llm.py`**: OpenAI LLM integration
- **`src/llm/plugins/gemini_llm.py`**: Google Gemini integration
- **`src/llm/plugins/deepseek_llm.py`**: DeepSeek LLM integration
- **`src/llm/plugins/multi_llm.py`**: Multi-LLM orchestration

#### Function Schemas
- **`src/llm/function_schemas.py`**
  - `generate_function_schemas_from_actions()`: Auto-generates function schemas from actions

### Structure

```
src/llm/
├── __init__.py              # LLM base class and loading
├── function_schemas.py      # Function schema generation
└── plugins/                 # LLM plugin implementations
    ├── openai_llm.py
    ├── gemini_llm.py
    ├── deepseek_llm.py
    └── ...
```

### Responsibilities

- LLM API communication
- Function calling schema generation
- Response parsing and validation
- Multi-LLM orchestration (if needed)

### Automatic Function Schema Generation

- **No modification required** when adding new actions
- New actions automatically generate function schemas via `generate_function_schemas_from_actions()`
- Function schemas are passed to LLM for function calling

---

## 5. Action System (NAV-ACTION)

**Purpose**: Action execution, business logic processing, and ROS2/hardware communication.

### Key Files

#### Base Classes
- **`src/actions/base.py`**
  - `ActionConnector`: Base class for action connectors
  - `ActionConfig`: Base configuration class
  - `AgentAction`: Action metadata container
  - `Interface`: Input/output interface definition
  - Defines interface: `connect()`, `tick()`

#### Orchestration
- **`src/actions/orchestrator.py`**
  - `ActionOrchestrator`: Manages action execution
  - `promise()`: Queues actions for execution
  - `_promise_action()`: Executes individual actions
  - `flush_promises()`: Processes completed actions
  - Manages thread pool for connector execution

#### Plugin Loading
- **`src/actions/__init__.py`**
  - `load_action()`: Dynamically loads action plugins
  - `describe_action()`: Generates action descriptions for prompts
  - `find_module_with_class()`: Finds action class in plugins

#### Example Actions
- **`src/actions/navigate_location/`**
  - `connector/unitree_go2_nav.py`: Navigation connector
  - `interface.py`: NavigateLocationInput interface
- **`src/actions/speak/`**
  - `connector/zenoh.py`: Zenoh-based speech connector
  - `connector/ros2.py`: ROS2-based speech connector
  - `interface.py`: SpeakInput interface
- **`src/actions/move/`**
  - `connector/ros2.py`: ROS2 move connector
  - `interface.py`: MoveInput interface

### Structure

```
src/actions/
├── base.py                  # Action base classes
├── orchestrator.py          # Action orchestration
├── __init__.py              # Plugin loading
└── <action_name>/           # Action implementations
    ├── connector/
    │   ├── ros2.py          # ROS2 connector
    │   └── zenoh.py         # Zenoh connector
    └── interface.py         # Input interface
```

### Responsibilities

- **Business Logic**: Input parsing, data transformation, validation, error handling
- **Provider Communication**: Calls providers for actual ROS2/hardware communication
- **Action Execution**: Executes LLM-generated actions

### Business Logic vs Provider Communication

Action Connectors handle business logic, while Providers handle actual communication:

**Business Logic (Action Connector)**:
- Input parsing: "go to kitchen" → "kitchen"
- Data transformation: location name → ROS2 PoseStamped message
- Validation: Check if location exists, validate coordinates
- Error handling: Handle missing locations, invalid inputs

**Provider Communication (Provider)**:
- ROS2/Zenoh topic publish/subscribe
- Hardware communication
- State management
- Message serialization/deserialization

**Example**:
```python
# src/actions/navigate_location/connector/unitree_go2_nav.py
class UnitreeGo2NavConnector(ActionConnector[...]):
    def __init__(self, config: ...):
        # Provider initialization (for ROS2 communication)
        self.navigation_provider = UnitreeGo2NavigationProvider()
    
    async def connect(self, input: NavigateLocationInput):
        # Business logic 1: Parse input
        label = input.action.lower().strip()
        for prefix in ["go to the ", "go to ", ...]:
            if label.startswith(prefix):
                label = label[len(prefix):].strip()
                break
        
        # Business logic 2: Lookup location
        loc = self.location_provider.get_location(label)
        if loc is None:
            # Business logic 3: Error handling
            logging.warning(f"Location '{label}' not found")
            return
        
        # Business logic 4: Transform data
        pose = loc.get("pose") or {}
        goal_pose = PoseStamped(...)  # Create ROS2 message
        
        # Provider call: Actual ROS2 communication
        self.navigation_provider.publish_goal_pose(goal_pose, label)
```

---

## 6. Providers (SYS-PROVIDER)

**Purpose**: Hardware/service communication abstraction layer.

### Key Files

#### Core Providers
- **`src/providers/io_provider.py`**
  - `IOProvider`: I/O data management, dynamic variables, input history

- **`src/providers/config_provider.py`**
  - `ConfigProvider`: Runtime configuration broadcasting via Zenoh

- **`src/providers/sleep_ticker_provider.py`**
  - `SleepTickerProvider`: Sleep timing control

#### ROS2/Zenoh Providers
- **`src/providers/zenoh_publisher_provider.py`**
  - `ZenohPublisherProvider`: Zenoh message publishing

- **`src/providers/ros2_publisher_provider.py`**
  - `ROS2PublisherProvider`: ROS2 message publishing

#### Navigation Providers
- **`src/providers/unitree_go2_navigation_provider.py`**
  - `UnitreeGo2NavigationProvider`: Nav2 goal publishing and status monitoring

- **`src/providers/unitree_go2_locations_provider.py`**
  - `UnitreeGo2LocationsProvider`: Location management

#### Sensor Providers
- **`src/providers/gps_provider.py`**: GPS data provider
- **`src/providers/odom_provider.py`**: Odometry data provider
- **`src/providers/unitree_go2_amcl_provider.py`**: AMCL localization provider

#### VLM Providers
- **`src/providers/vlm_gemini_provider.py`**: Gemini VLM provider
- **`src/providers/turtlebot4_camera_vlm_provider.py`**: Camera VLM provider

### Structure

```
src/providers/
├── io_provider.py                    # I/O management
├── config_provider.py                # Config broadcasting
├── zenoh_publisher_provider.py      # Zenoh publishing
├── ros2_publisher_provider.py       # ROS2 publishing
├── unitree_go2_navigation_provider.py
├── gps_provider.py
├── odom_provider.py
└── ...
```

### Responsibilities

- Hardware communication (GPS, cameras, sensors)
- ROS2/Zenoh topic publish/subscribe
- External service communication (TTS, ASR, VLM)
- Data processing and state management
- Singleton pattern for shared resources

### Provider Classification Criteria

Providers are classified based on **Single Responsibility Principle**:

#### 1. Hardware/Sensor Level
**One physical sensor = One Provider**

- `GpsProvider`: GPS sensor only
- `OdomProvider`: Odometry sensor only
- `RPLidarProvider`: RPLidar sensor only
- `RealsenseProvider`: RealSense camera only

**Reason**: Each sensor is independent hardware with different communication protocols and data formats.

#### 2. Robot Platform Level
**Robot-specific functionality = Separate Provider**

- `UnitreeGo2NavigationProvider`: Unitree Go2 specific navigation
- `UnitreeG1NavigationProvider`: Unitree G1 specific navigation
- `TurtleBot4CameraVLMProvider`: TurtleBot4 specific camera VLM

**Reason**: Different robots use different communication protocols (CycloneDDS vs Zenoh) and APIs.

#### 3. Functionality/Service Level
**One functionality/service = One Provider**

- `VLMGeminiProvider`: Gemini VLM service only
- `VLMOpenAIProvider`: OpenAI VLM service only
- `ElevenLabsTTSProvider`: ElevenLabs TTS service only
- `GPSNavProvider`: GPS Navigation functionality only
- `DWANavProvider`: DWA Navigation functionality only

**Reason**: Different services have different APIs, configurations, and data formats.

#### 4. Communication Protocol Level
**Communication method = Separate Provider (when needed)**

- `ZenohPublisherProvider`: Zenoh publish only
- `ZenohListenerProvider`: Zenoh subscribe only
- `ROS2PublisherProvider`: ROS2 publish only

**Reason**: Different communication protocols and message formats.

#### 5. System/Utility Level
**System-wide functionality = Separate Provider**

- `IOProvider`: I/O data management (singleton)
- `ConfigProvider`: Config broadcasting
- `ContextProvider`: Context management

**Reason**: System-wide shared resources, not tied to specific hardware/service.

### Singleton Pattern

Most providers use `@singleton` decorator to ensure only one instance exists:

```python
# src/providers/singleton.py
@singleton
class GpsProvider:
    # Only one instance exists across the entire system
    pass

@singleton
class OdomProvider:
    # Only one instance exists across the entire system
    pass
```

**Benefits**:
- Shared resources (ROS2/Zenoh sessions, hardware connections)
- Consistent state across Sensor/Action/Background usage
- Efficient resource management

### Provider Usage in Sensor vs Action

**Same Provider Instance, Different Usage**:

- **Sensor**: Reads data from provider (read-only)
  ```python
  # Sensor uses provider for reading
  class Gps(FuserInput[...]):
      async def _poll(self):
          return self.gps.data  # Read-only access
  ```

- **Action**: Writes data via provider (write/read)
  ```python
  # Action uses provider for writing
  class NavigateGPSConnector(ActionConnector[...]):
      async def connect(self, input):
          self.gps_nav_provider.publish_dxdy(lat, lon)  # Write access
  ```

- **Background**: Initializes and starts provider
  ```python
  # Background initializes provider
  class Gps(Background[...]):
      def __init__(self, config):
          self.gps_provider = GpsProvider(serial_port=port)  # Initialize
  ```

### Provider Implementation Pattern

```python
# src/providers/realsense_provider.py
@singleton
class RealsenseProvider:
    """
    RealSense camera provider.
    Handles ROS2 topic subscription for RealSense camera data.
    """
    
    def __init__(self, topic: str = "/camera/color/image_raw"):
        self.session = open_zenoh_session()
        self.topic = topic
        self.latest_image_data = None
        
        # Subscribe to ROS2 topic
        self.subscriber = self.session.declare_subscriber(
            self.topic,
            self._handle_image
        )
    
    def _handle_image(self, sample):
        """Callback when image data arrives"""
        self.latest_image_data = sample.payload
    
    @property
    def data(self):
        """Sensor reads from this property"""
        return self.latest_image_data
```

---

## 7. Simulators (SYS-SIM)

**Purpose**: Action visualization and debugging.

### Key Files

#### Base Classes
- **`src/simulators/base.py`**
  - `Simulator`: Base class for simulators
  - `SimulatorConfig`: Base configuration class

#### Orchestration
- **`src/simulators/orchestrator.py`**
  - `SimulatorOrchestrator`: Manages simulator execution
  - `sim()`: Visualizes actions

#### Plugin Loading
- **`src/simulators/__init__.py`**
  - `load_simulator()`: Dynamically loads simulator plugins

#### Example Simulators
- **`src/simulators/plugins/websim.py`**: Web-based simulator
- **`src/simulators/plugins/console_sim.py`**: Console-based simulator

### Responsibilities

- Visualize actions for debugging
- Display action execution status
- Optional: Real-time visualization

### Automatic Action Reception

- **No modification required** when adding new actions
- Simulator automatically receives all actions via `sim(actions)` method
- Optional: Add custom visualization for specific action types if needed

---

## 8. Backgrounds (SYS-BG)

**Purpose**: Background processes, provider initialization, and continuous services.

### Key Files

#### Base Classes
- **`src/backgrounds/base.py`**
  - `Background`: Base class for background components
  - `BackgroundConfig`: Base configuration class
  - Defines interface: `run()`

#### Orchestration
- **`src/backgrounds/orchestrator.py`**
  - `BackgroundOrchestrator`: Manages background processes
  - Runs backgrounds in separate threads

#### Plugin Loading
- **`src/backgrounds/__init__.py`**
  - `load_background()`: Dynamically loads background plugins

#### Example Backgrounds
- **`src/backgrounds/plugins/gps.py`**: GPS provider initialization
- **`src/backgrounds/plugins/odom.py`**: Odometry provider initialization
- **`src/backgrounds/plugins/unitree_go2_navigation.py`**: Navigation provider initialization

### Structure

```
src/backgrounds/
├── base.py                  # Background base class
├── orchestrator.py          # Background orchestration
├── __init__.py              # Plugin loading
└── plugins/                 # Background implementations
    ├── gps.py
    ├── odom.py
    └── ...
```

### Responsibilities

- Initialize and start providers
- Maintain continuous background processes
- External server communication
- Resource lifecycle management

### Background-Provider Relationship

#### Basic Principle: 1:1 Relationship

**One Background = One Provider** (most common case)

```python
# src/backgrounds/plugins/gps.py
class Gps(Background[GpsConfig]):
    def __init__(self, config: GpsConfig):
        super().__init__(config)
        # One Background initializes one Provider
        self.gps_provider = GpsProvider(serial_port=port)
        logging.info("GPS Provider initialized in background")
```

**Naming Convention**: `{ProviderName}` Background = initializes `{ProviderName}Provider`

#### Exception: Multiple Providers for Special Functionality

When a specific functionality requires multiple providers, one Background can initialize multiple providers:

```python
# src/backgrounds/plugins/rf_mapper.py
class RFmapper(Background[RFmapperConfig]):
    def __init__(self, config: RFmapperConfig):
        super().__init__(config)
        # Special case: Multiple providers for RF mapping functionality
        self.gps = GpsProvider()
        self.rtk = RtkProvider()
        self.odom = OdomProvider()
        # RFmapper collects GPS, RTK, and Odom data to send to Fabric network
```

**Reason**: RFmapper is a special functionality that requires data from multiple sources.

### Background Initialization Criteria

#### Criterion 1: Provider Unit (Default)

**One Provider = One Background**

| Provider | Background | Purpose |
|----------|-----------|---------|
| `GpsProvider` | `Gps` Background | Initialize GPS sensor |
| `OdomProvider` | `Odom` Background | Initialize Odometry sensor |
| `RealsenseProvider` | `Realsense` Background | Initialize RealSense camera |
| `PointCloudProvider` | `PointCloud` Background | Initialize PointCloud data source |
| `GPSNavProvider` | `GPSNavigation` Background | Initialize GPS Navigation |
| `DWANavProvider` | `DWANavigation` Background | Initialize DWA Navigation |

#### Criterion 2: Functionality Unit (Special Cases)

**Special functionality requiring multiple providers = One Background**

| Background | Providers | Purpose |
|-----------|-----------|---------|
| `RFmapper` | `GpsProvider`, `RtkProvider`, `OdomProvider` | Collect and send location data to Fabric network |

#### Criterion 3: Robot Platform Unit

**Robot-specific providers = Robot-specific Background**

| Provider | Background | Purpose |
|----------|-----------|---------|
| `UnitreeGo2NavigationProvider` | `UnitreeGo2Navigation` Background | Unitree Go2 specific navigation |
| `UnitreeGo2StateProvider` | `UnitreeGo2State` Background | Unitree Go2 specific state |

### Background Implementation Pattern

```python
# src/backgrounds/plugins/realsense.py
class Realsense(Background[RealsenseConfig]):
    """
    RealSense Background.
    Initializes and starts RealSense Provider.
    """
    
    def __init__(self, config: RealsenseConfig):
        super().__init__(config)
        
        # Initialize Provider (singleton, so same instance shared)
        self.realsense_provider = RealsenseProvider(
            camera_index=config.camera_index
        )
        
        # Start Provider (if needed)
        self.realsense_provider.start()
        
        logging.info("RealSense Provider initialized in background")
    
    def run(self) -> None:
        """
        Background process loop.
        Can be used for continuous monitoring or periodic tasks.
        """
        time.sleep(60)  # Default: sleep, override if needed
```

### Why Background Initializes Providers

1. **Lifecycle Management**: Background ensures provider is initialized before Sensor/Action use
2. **Resource Management**: Background can manage provider lifecycle (start/stop)
3. **Configuration**: Background can pass configuration to provider during initialization
4. **Separation of Concerns**: Background handles initialization, Sensor/Action handle usage

---

## Module Interaction Flow

```
┌─────────────────────────────────────────────────────────┐
│              Runtime Core (CortexRuntime)                │
│  ┌───────────────────────────────────────────────────┐  │
│  │  InputOrchestrator                                │  │
│  │    ↓                                               │  │
│  │  Sensors (Gps, Odom, RPLidar, VLM, ...)          │  │
│  │    ↓ (대부분의 Sensor)                              │  │
│  │  Providers (GpsProvider, OdomProvider, ...)      │  │
│  │    ↓ (데이터 읽기)                                  │  │
│  │  formatted_latest_buffer() → LLM Input Text       │  │
│  └───────────────────────────────────────────────────┘  │
│                          ↓                                │
│  ┌───────────────────────────────────────────────────┐  │
│  │  Fuser → fuse(inputs) → Final Prompt            │  │
│  └───────────────────────────────────────────────────┘  │
│                          ↓                                │
│  ┌───────────────────────────────────────────────────┐  │
│  │  LLM → ask(prompt) → Actions                       │  │
│  └───────────────────────────────────────────────────┘  │
│                          ↓                                │
│  ┌───────────────────────────────────────────────────┐  │
│  │  ActionOrchestrator → Connectors → Providers      │  │
│  └───────────────────────────────────────────────────┘  │
│                                                           │
│  BackgroundOrchestrator → Backgrounds → Providers        │
│  SimulatorOrchestrator → Simulators                      │
└─────────────────────────────────────────────────────────┘
```

## Config File Structure

### Single Mode Config

```json5
{
  "version": "v1.0.1",
  "hertz": 10,
  "name": "config_name",
  "system_prompt_base": "...",
  "system_governance": "...",
  "system_prompt_examples": "...",
  "agent_inputs": [
    {
      "type": "SensorName",
      "config": { /* SensorConfig fields */ }
    }
  ],
  "cortex_llm": {
    "type": "LLMName",
    "config": { /* LLMConfig fields */ }
  },
  "agent_actions": [
    {
      "name": "action_name",
      "llm_label": "action_label",
      "connector": "connector_name",
      "config": { /* ActionConfig fields */ }
    }
  ],
  "backgrounds": [
    {
      "type": "BackgroundName",
      "config": { /* BackgroundConfig fields */ }
    }
  ]
}
```

### Multi Mode Config

```json5
{
  "version": "v1.0.0",
  "name": "multi_mode_system",
  "default_mode": "autonomous",
  "system_governance": "...",
  "cortex_llm": { /* Global LLM */ },
  "modes": {
    "autonomous": {
      "system_prompt_base": "...",
      "agent_inputs": [...],
      "agent_actions": [...],
      "cortex_llm": { /* Mode-specific LLM */ }
    }
  },
  "transition_rules": [...]
}
```

---

## Summary

| Module | Key Files | Purpose |
|--------|-----------|---------|
| **SYS-CORE** | `runtime/single_mode/cortex.py`, `config.py` | Runtime execution and config management |
| **PER-INPUT** | `inputs/base/__init__.py`, `orchestrator.py`, `plugins/` | Sensor inputs and data collection |
| **AGT-FUSER** | `fuser/__init__.py` | Input fusion and prompt generation |
| **AGT-LLM** | `llm/__init__.py`, `plugins/` | Language model integration |
| **NAV-ACTION** | `actions/base.py`, `orchestrator.py`, `*/connector/` | Action execution and orchestration |
| **SYS-PROVIDER** | `providers/*.py` | Hardware/service communication abstraction |
| **SYS-SIM** | `simulators/base.py`, `orchestrator.py` | Action visualization and debugging |
| **SYS-BG** | `backgrounds/base.py`, `orchestrator.py`, `plugins/` | Background processes and provider initialization |

