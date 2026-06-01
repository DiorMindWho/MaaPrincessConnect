# MaaPrincessConnect Agent Guidelines

Welcome, Agent! This document defines the development rules, standards, and engineering best practices for **MaaPrincessConnect**, a Princess Connect Re:Dive automation assistant powered by MaaFramework.

---

## 🛠️ Technology Stack

1. **Automation Core**: MaaFramework (C++20 / JSON Pipeline)
2. **Agent / Custom Scripting**: Python (type-hinted, modern async patterns)
3. **Web / UI Tools**: Node.js & TypeScript (`@nekosu/maa-tools`)

---

## 📖 MaaFramework Core Concepts & Pipeline Architecture

`MaaPrincessConnect` is powered by **MaaFramework**, a low-code/highly-customizable black-box game automation testing framework using image recognition and C++20. Below is a summary of the core concepts, objects, and pipeline protocols from MaaFramework (`D:\word\MaaFramework\AGENTS.md`):

### 1. Core Terminology & Data Flow
* **Node (节点)**: A single block/object in the Pipeline JSON that defines recognition rules, actions to take, and successors (`next`).
* **Task (任务)**: A sequence of connected Nodes representing a logical automation workflow.
* **Entry (入口)**: The starting Node of a Task.
* **Pipeline (流水线)**: The sum of all Nodes in the pipeline directory.
* **Bundle**: A standard resource package folder containing `pipeline`, `model`, and `image` directories.
* **Resource (资源)**: Loaded/assembled runtime resource representation from one or more Bundles.
* **Agent**: A decoupling mechanism allowing custom scripts to communicate with MaaFramework core across processes.

### 2. Primary Framework Objects
* **MaaResource**: Handles the loading of Pipeline JSONs, template images, and ML models.
* **MaaController**: Handles device connections, screenshot captures, and inputs (ADB, Win32, PlayCover).
* **MaaTasker**: Executes a task pipeline after binding a Resource and a Controller.
* **MaaContext**: Created during runtime; provides control API, screenshot retrieval, and pipeline operations inside custom callbacks/scripts.

### 3. Pipeline Protocol Cheatsheet

#### Recognition Algorithms
* `DirectHit`: Instantly matches without image processing.
* `TemplateMatch`: Classical template matching (找图) using `template`, `threshold`, `roi`.
* `FeatureMatch`: Perspective-distortion resistant feature matching using `template`, `count`, `detector`.
* `ColorMatch`: Matches specified colors (找色) using `lower`, `upper`, `method`.
* `OCR`: Optical character recognition using `expected`, `model`.
* `NeuralNetworkClassify` / `NeuralNetworkDetect`: ML-based classification/object detection.
* `Custom`: Delegates recognition to custom scripts (e.g. `custom_recognition`).

#### Action Methods
* `DoNothing`: Empty action.
* `Click`: Left clicks/taps at `target`, with optional `target_offset`.
* `LongPress`: Long presses at `target` for `duration`.
* `Swipe`: Performs single-finger swipe from `begin` to `end` over `duration`.
* `MultiSwipe`: Multi-finger complex gestures.
* `ClickKey` / `InputText`: Key press or text input.
* `StartApp` / `StopApp`: Launch/terminate an application by package name.
* `Command` / `Shell`: Executes host OS commands or ADB shell commands (`cmd`).
* `Custom`: Triggers a custom action script (e.g. `custom_action`).

---

## 🔍 Sibling Reference Project (M9A)

* **M9A Project Reference**: 
  We also use the **M9A** project (`D:\word\MaaFramework\M9A`) as our primary reference. M9A is a similar automation assistant built on MaaFramework for a different game.
  * **When to Reference M9A**: If you have questions about best practices for scripting complex Python Agent logic, structural organization of custom action pipelines, or interface designs, check `D:\word\MaaFramework\M9A` for patterns and existing solutions.

---

## 🚀 Elite Software Engineering Principles (AI Instructions)

Always adhere to these industry-standard best practices to deliver clean, production-grade, and highly maintainable code:

### 1. Guard Clauses & Early Returns (降低嵌套)
* **Preconditions First**: Validate inputs, resource availability, and states at the very top of functions. Return or raise errors immediately.
* **Avoid Else Blocks**: If the `if` branch returns or throws, omit the `else` block to keep the primary logic at the top indentation level.

```python
# GOOD
def process_screenshot(ctx: MaaContext) -> bool:
    if not ctx.is_connected():
        return False
    
    screenshot = ctx.get_screenshot()
    if screenshot is None:
        return False

    return analyze_image(screenshot)
```

### 2. Defensive Programming & Error Boundaries (防御性编程)
* **No Swallowed Exceptions**: Never use bare `except:` or empty `catch` blocks. Log every exception with a trace and handle it cleanly.
* **Null/None Safety**: Always explicitly check for `None` or null values when calling external APIs, filesystem operations, or image lookups.
* **Graceful Degradation**: If an automation task fails, make sure the agent returns to a safe "home" state or recovers gracefully instead of crashing.

### 3. Maintain Documentation & Code Integrity
* **Keep Code Self-Documenting**: Use descriptive naming (e.g., `is_battle_finished` instead of `flag`).
* **Preserve Comments**: Do not delete unrelated documentation or comments.
* **Explain "Why", Not "What"**: Comments should explain non-obvious reasoning and business rules, not just repeat what the code does.

### 4. Modular & Declarative JSON Pipeline Design
* **Loose Coupling**: Keep tasks modular. A task should focus on one operation (e.g., `claim_daily_rewards`, `clear_dungeon`).
* **Use Reusable Nodes**: Define reusable base templates for recognition (`TemplateMatch`, `OCR`) to avoid duplicate declarations.
* **Schema Validation**: Always validate pipeline JSON files against `deps/tools/pipeline.schema.json` before testing.

### 5. Strict Type Hinting & Code Quality (Python)
* **Strict Type Hints**: Use `typing` hints (`List`, `Dict`, `Optional`, `Union`) on all function signatures.
* **Standard Conventions**: Adhere to PEP 8 style guidelines.

---

## 📐 Project Structure & Navigation

* `assets/resource/pipeline/`: Contains JSON files defining the tasks and image recognition nodes.
* `assets/resource/image/`: Templates for `TemplateMatch`.
* `agent/`: Custom Python agent logic for advanced scenarios.
* `docs/`: Project documentation.
* `assets/interface.json`: The generic user interface configuration for GUI clients.

---

## 🧪 Verification & Testing Workflow

Before finalizing any changes:
1. Run static checks and schema validation if available.
2. Manually test connections using simulated environments or real emulators.
3. Keep the working tree clean and make small, atomic, descriptive commits.
