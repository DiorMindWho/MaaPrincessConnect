# MaaPrincessConnect Agent Guidelines

Welcome, Agent! This document defines the development rules, standards, and engineering best practices for **MaaPrincessConnect**, a Princess Connect Re:Dive automation assistant powered by MaaFramework.

---

## 🛠️ Technology Stack

1. **Automation Core**: MaaFramework (C++20 / JSON Pipeline)
2. **Agent / Custom Scripting**: Python (type-hinted, modern async patterns)
3. **Web / UI Tools**: Node.js & TypeScript (`@nekosu/maa-tools`)

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
