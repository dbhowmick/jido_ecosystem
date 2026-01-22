# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a learning repository containing the **Jido Ecosystem** - a collection of Elixir libraries for building autonomous multi-agent workflows. The name "Jido" (自動) comes from Japanese meaning "automatic" or "self-moving".

## Repository Structure

```
jido_ecosystem/
├── CLAUDE.md       # This file - guidance for AI coding agents
├── README.md       # Project overview and quick start
├── pull_all.sh     # Script to clone/pull all repos into packages/
└── packages/       # Cloned Jido repositories (git-ignored)
    ├── jido/           # Core agent framework (immutable agents + cmd/2)
    ├── jido_action/    # Composable, validated actions with AI tool integration
    ├── jido_signal/    # CloudEvents-based messaging and routing
    ├── jido_ai/        # AI/LLM integration for agents
    ├── req_llm/        # HTTP client for 57+ LLM providers
    └── jido_workbench/ # Phoenix demo application
```

## Dependency Graph

```
req_llm (standalone LLM client)
jido_action (standalone actions)
jido_signal (standalone signals)
       ↓
jido (core) → depends on jido_action, jido_signal
       ↓
jido_ai → depends on jido, req_llm
       ↓
jido_workbench → depends on jido, jido_ai, req_llm
```

Use `LOCAL_JIDO_DEPS=true` to switch jido_ai and jido_workbench to local path dependencies.

## Common Commands

All repositories use mix with consistent patterns:

| Command | Purpose |
|---------|---------|
| `mix deps.get` | Fetch dependencies |
| `mix test` | Run tests (excludes flaky) |
| `mix test path/to/test.exs` | Run single test file |
| `mix test path/to/test.exs:42` | Run test at specific line |
| `mix quality` or `mix q` | Format + compile + dialyzer + credo |
| `mix format` | Format code |
| `mix dialyzer` | Type checking |
| `mix credo` | Linting |

### Repository-Specific Commands

**jido**: Coverage threshold 80%+

**jido_signal**: Coverage threshold 90%+

**jido_ai**:
- `mix test.unit` - Excludes integration/provider tests
- `mix test.integration` - Only integration tests
- `mix test.providers` - Only provider validation tests

**req_llm**:
- `LIVE=true mix test` - Run against real APIs (regenerates fixtures)
- `REQ_LLM_DEBUG=1 mix test` - Verbose fixture debugging
- `mix mc` or `mix req_llm.model_compat` - Show models with passing fixtures
- `mix mc "openai:gpt-4o"` - Validate specific model

**jido_workbench**:
- `mix setup` - deps.get + assets.setup + assets.build
- `mix phx.server` - Run Phoenix server

## Core Architecture

### The cmd/2 Pattern (Elm/Redux-inspired)

The fundamental operation in Jido:

```elixir
{agent, directives} = MyAgent.cmd(agent, action)
```

- Agents are **immutable data structures**
- `cmd/2` is **pure**: same inputs → same outputs
- State changes are data transformations
- Side effects are described as **directives** (executed by OTP runtime)

### Actions vs Directives vs State Operations

| Actions | Directives | State Operations |
|---------|------------|------------------|
| Transform state, may perform side effects | Describe external effects | Describe internal state changes |
| Executed by `cmd/2` | Runtime (AgentServer) interprets | Applied by strategy layer |
| Can call APIs, read files, etc. | Never modify agent state | Never leave the strategy |

### Core Directives

| Directive | Purpose |
|-----------|---------|
| `Emit` | Dispatch a signal via adapters |
| `Error` | Signal an error from cmd/2 |
| `Spawn` | Spawn generic BEAM child (fire-and-forget) |
| `SpawnAgent` | Spawn child Jido agent with hierarchy tracking |
| `StopChild` | Stop a tracked child agent |
| `Schedule` | Schedule a delayed message |
| `Stop` | Stop the agent process |

### State Operations (Jido.Agent.StateOp)

| StateOp | Purpose |
|---------|---------|
| `SetState` | Deep merge into state |
| `ReplaceState` | Replace state wholesale |
| `DeleteKeys` | Remove top-level keys |
| `SetPath` | Set value at nested path |
| `DeletePath` | Delete value at nested path |

## Code Style

- `snake_case` for functions/variables, `PascalCase` for modules
- Pattern match with function heads instead of conditionals
- Return tagged tuples: `{:ok, result}` or `{:error, reason}`
- Use `with` statements for chaining fallible operations
- `@moduledoc`, `@doc`, and `@spec` on all public functions
- Prefix test modules with namespace: `JidoTest.ModuleName`

### Schema Preference

**New code**: Use Zoi schemas for better validation and transformations:
```elixir
@schema Zoi.struct(__MODULE__, %{
  email: Zoi.string() |> Zoi.trim() |> Zoi.email(),
  age: Zoi.integer() |> Zoi.min(0)
}, coerce: true)
```

**Existing code**: NimbleOptions still supported.

### req_llm Specific

**No inline comments in function bodies** - code should be self-explanatory through clear naming.

## Git Commit Guidelines

Use **Conventional Commits** format:

```
<type>[optional scope]: <description>
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

Examples:
```
feat(agent): add new directive for child process management
fix(router): handle subscriber exit during dispatch
test(plan): add DAG cycle detection tests
chore(deps): bump jido_action to 2.0.0-rc
```

**Important**: Do NOT include "Co-Authored-By" lines in commit messages.

## Key Modules by Repository

**jido**:
- `Jido.Agent` - Immutable agent struct with cmd/2
- `Jido.AgentServer` - GenServer wrapper for production
- `Jido.Agent.Directive` - Effect descriptors
- `Jido.Agent.Strategy` - Execution strategies (Direct, FSM)
- `Jido.Skill` - Reusable behavior modules

**jido_action**:
- `Jido.Action` - Core behavior for validated actions
- `Jido.Exec` - Execution engine (sync/async, retries)
- `Jido.Instruction` - Workflow composition
- `Jido.Plan` - DAG execution with dependencies
- `Jido.Tools.*` - 25+ pre-built tools

**jido_signal**:
- `Jido.Signal` - CloudEvents v1.0.2 struct
- `Jido.Signal.Bus` - GenServer pub/sub
- `Jido.Signal.Router` - Trie-based pattern matching (O(k) complexity)
- `Jido.Signal.Dispatch` - Adapters (pid, pubsub, http, logger, etc.)

**req_llm**:
- `ReqLLM` - Main API (generate_text, stream_text, generate_object)
- `ReqLLM.Provider` - Provider behavior (57+ implementations)
- `ReqLLM.Context` / `ReqLLM.Message` - Conversation structures
- `ReqLLM.Tool` - Function calling definitions

**jido_ai**:
- `Jido.AI.Model` - LLM model configuration
- `Jido.AI.Prompt` - Template-based prompts (EEx, Liquid)
- `Jido.AI.Runners.*` - Reasoning strategies (CoT, ReAct, ToT, GEPA)
- `Jido.AI.Conversation` - Stateful multi-turn conversations

## Environment Variables

- `LOCAL_JIDO_DEPS=true` - Use local paths instead of hex packages
- `LIVE=true` - Run req_llm tests against real APIs
- `REQ_LLM_DEBUG=1` - Verbose fixture debugging

## Prerequisites

- Elixir ~> 1.17
- Erlang/OTP 26+
