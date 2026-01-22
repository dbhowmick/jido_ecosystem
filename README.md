# Jido Ecosystem

A learning repository for exploring the **Jido Ecosystem** - a collection of Elixir libraries for building autonomous multi-agent workflows.

The name "Jido" (自動) comes from Japanese meaning "automatic" or "self-moving".

## Purpose

This repository serves as a local workspace for:
- Understanding the Jido Ecosystem codebase
- Providing context to AI coding agents (Claude, Cursor, Copilot, etc.)
- Experimenting with the libraries in a unified environment

The actual Jido packages are cloned into `packages/` and excluded from version control, keeping this repo lightweight while providing full access to the source code.

## Quick Start

```bash
# Clone this repository
git clone git@github.com:dbhowmick/jido_ecosystem.git
cd jido_ecosystem

# Clone all Jido packages
./pull_all.sh

# Navigate to any package
cd packages/jido
mix deps.get
mix test
```

## Repository Structure

```
jido_ecosystem/
├── CLAUDE.md         # Detailed guidance for AI coding agents
├── README.md         # This file
├── pull_all.sh       # Script to clone/pull all packages
└── packages/         # Cloned Jido repositories (git-ignored)
    ├── jido/             # Core agent framework
    ├── jido_action/      # Composable, validated actions
    ├── jido_signal/      # CloudEvents-based messaging
    ├── jido_ai/          # AI/LLM integration
    ├── req_llm/          # HTTP client for 57+ LLM providers
    └── jido_workbench/   # Phoenix demo application
```

## Package Overview

| Package | Description | Key Concepts |
|---------|-------------|--------------|
| **jido** | Core agent framework | Immutable agents, `cmd/2` pattern, directives |
| **jido_action** | Action system | Validated actions, workflows, DAG execution |
| **jido_signal** | Messaging layer | CloudEvents, pub/sub, signal routing |
| **jido_ai** | AI integration | LLM models, prompts, reasoning strategies |
| **req_llm** | LLM HTTP client | 57+ providers, streaming, tool calling |
| **jido_workbench** | Demo app | Phoenix LiveView showcase |

## Dependency Graph

```
req_llm (standalone)
jido_action (standalone)
jido_signal (standalone)
       │
       ▼
jido (core) ─────► depends on jido_action, jido_signal
       │
       ▼
jido_ai ─────────► depends on jido, req_llm
       │
       ▼
jido_workbench ──► depends on jido, jido_ai, req_llm
```

## For AI Coding Agents

When working with this codebase, refer to **[CLAUDE.md](./CLAUDE.md)** for:
- Core architecture patterns (the `cmd/2` pattern)
- Actions vs Directives vs State Operations
- Code style conventions
- Common commands for each package
- Key modules and their purposes

### Key Architectural Concepts

**The cmd/2 Pattern** (Elm/Redux-inspired):
```elixir
{agent, directives} = MyAgent.cmd(agent, action)
```
- Agents are immutable data structures
- `cmd/2` is pure: same inputs produce same outputs
- Side effects are described as directives (executed by runtime)

**Three-Layer Separation**:
1. **Actions** - Transform state, may perform side effects
2. **Directives** - Describe external effects (Emit, Spawn, Schedule, etc.)
3. **State Operations** - Describe internal state changes (SetState, DeleteKeys, etc.)

## Common Commands

All packages use Mix with consistent patterns:

```bash
mix deps.get          # Fetch dependencies
mix test              # Run tests
mix quality           # Format + compile + dialyzer + credo (alias: mix q)
mix format            # Format code
mix dialyzer          # Type checking
mix credo             # Linting
```

## Prerequisites

- Elixir ~> 1.17
- Erlang/OTP 26+

## Using Local Dependencies

To develop across packages with local changes:

```bash
LOCAL_JIDO_DEPS=true mix deps.get
```

This switches `jido_ai` and `jido_workbench` to use local path dependencies instead of Hex packages.

## Links

- [agentjido GitHub Organization](https://github.com/agentjido)
- [jido on Hex.pm](https://hex.pm/packages/jido)
- [jido_ai on Hex.pm](https://hex.pm/packages/jido_ai)
- [req_llm on Hex.pm](https://hex.pm/packages/req_llm)
