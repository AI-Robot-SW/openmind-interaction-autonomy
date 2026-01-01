# Contributing Guide

Thank you for contributing to the OpenMind Interaction Autonomy project!

## Project Philosophy

- **Minimal Overhead, Clear Rules**: Operate with simple and clear rules
- **main-only Strategy**: Only one long-term branch (main)
- **Requirement-driven Development**: All features are linked to Requirement IDs
- **PR-based Collaboration**: All changes go through PRs
- **Version Management with Tags Only**: No release branches

## Branch Strategy

### Long-term Branches
- `main`: The only long-term branch

### Working Branches
- **Feature Branches**: `feat/<REQUIREMENT-ID>-short-description`
  - Examples: `feat/NAV-001-nav-execution`, `feat/AGT-002-action-router`
- **Non-feature Work**: `chore/<short-description>`
  - Examples: `chore/update-docs`, `chore/refactor-config-loader`

### Rules
- Use lowercase
- Use hyphens (-) (no spaces)
- Concise but meaningful names

## Requirement-driven Development

### Requirement ID Format
```
<MODULE>-<NNN>
```

### Module Prefixes
- **AGT**: LLM / Agent / Decision
- **NAV**: Navigation / Autonomy
- **PER**: Perception / Sensors
- **SYS**: Orchestration / Runtime / Integration

### Examples
- `NAV-001`: Navigation execution via action schema
- `AGT-002`: Action router implementation

### Principles
- 1 Requirement ID = 1 feature branch
- All feature changes are linked to Requirement IDs
- Requirements are managed from a single source of truth

## PR / Commit / Merge Rules

### Merge Method
- **Squash merge only**
- main history is always in single commit units

### PR / Commit Message Rules
Only two types are used:
- `feat: <summary>`
- `chore: <summary>`

Examples:
```
feat: NAV-001 navigation execution via action schema
chore: update PROCESS.md
```

**Important**: Since we use squash merge, PR title = commit message that remains in main

### PR Template

- **For `feat/` branches only**: Use the PR template to verify testing before integration
- The PR template includes a testing checklist to ensure features are tested before merging
- Template location: `.github/pull_request_template.md`
- For `chore/` branches, the template is optional

## Collaboration Structure

### Teams
- **core**: Architecture, integration, final responsibility
  - Repo Admin permissions
  - Has main merge permissions
- **kist-collaborators**: Feature development, implementation
  - Repo Write permissions
  - Can create/review/approve PRs
  - ❌ Cannot merge to main

### Branch Protection (main)
- Direct push ❌
- PR required
- Approval ≥ 1
- Squash merge only
- Only core can merge
- Deletion/force push to main prohibited

## Contribution Process

1. **Check or Create Issue**
   - Check existing issues
   - Create new issue if needed

2. **Create Branch**
   ```bash
   git checkout -b feat/NAV-001-short-description
   # or
   git checkout -b chore/update-docs
   ```

3. **Commit Changes**
   - Write clear commit messages
   - Commit in small units

4. **Create PR**
   - Push branch to remote
   - Create PR on GitHub
   - PR title: `feat: <summary>` or `chore: <summary>`
   - **For `feat/` branches only**: Use the PR template (`.github/pull_request_template.md`) to verify testing before integration
   - Include Requirement ID in PR description (if applicable)

5. **Wait for Review**
   - Wait for core team review
   - Address feedback

6. **Merge**
   - Core team performs squash merge

## Coding Style

### Python
- Follow PEP 8 style guide
- Type hints recommended
- Write docstrings

### ROS2
- Follow ROS2 standard package structure
- Write `package.xml` and `CMakeLists.txt` accurately

## Version Management

- **No release branches**
- **Use git tags only** (vX.Y.Z format)
- Tags are created only on main, by core only
- Examples: `v0.1.0`, `v0.2.0`
