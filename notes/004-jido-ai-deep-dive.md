# 004 - Jido AI Deep Dive

> A comprehensive guide to building complex AI agents with Jido AI, from basic LLM calls to advanced patterns like dynamic tool loading and anti-hallucination strategies.

This note covers the complete Jido AI architecture, progressing from simple usage to building production-ready AI agents. It includes strategies, skills, state machines, directives, signals, and advanced patterns for tool search and dynamic tool management.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Level 1: Simple LLM Calls](#level-1-simple-llm-calls)
3. [Level 2: Creating Tools (Actions)](#level-2-creating-tools-actions)
4. [Level 3: ReAct Agents](#level-3-react-agents)
5. [Level 4: The ReAct Loop](#level-4-the-react-loop)
6. [Level 5: Strategies](#level-5-strategies)
7. [Level 6: State Machines](#level-6-state-machines)
8. [Level 7: Directives and Signals](#level-7-directives-and-signals)
9. [Level 8: Skills Framework](#level-8-skills-framework)
10. [Level 9: Dynamic System Prompts](#level-9-dynamic-system-prompts)
11. [Level 10: Tool Search Systems](#level-10-tool-search-systems)
12. [Level 11: Anti-Hallucination Patterns](#level-11-anti-hallucination-patterns)
13. [Level 12: Dynamic Tool Loading](#level-12-dynamic-tool-loading)

---

## Architecture Overview

Jido AI is the **AI integration layer** for the Jido ecosystem, providing LLM orchestration capabilities built on ReqLLM (supporting 57+ LLM providers).

### Core Architecture Layers

```
┌─────────────────────────────────────────────────────────────────┐
│                         JIDO.AI                                 │
│                                                                 │
│  Agent Layer → Strategy Layer → State Machine Layer → Directive │
│                      ↓                                    ↓     │
│                Tool Layer                            Signal     │
│                      ↓                               Layer      │
│                Skill Framework                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Key Design Principles

| Principle | Description |
|-----------|-------------|
| **Pure State Machines** | Same input → same output, no side effects |
| **Directive Pattern** | Side effects are declarative descriptions, not direct executions |
| **Signal-Driven** | Components communicate via typed signals (CloudEvents-based) |
| **Composability** | Skills and strategies are composable building blocks |

### Core Modules

| Module | Purpose |
|--------|---------|
| `Jido.AI` | Facade module for AI interactions |
| `Jido.AI.Directive` | LLM directives for agent runtime |
| `Jido.AI.Signal` | Custom signal types for LLM events |
| `Jido.AI.Strategies.*` | Reasoning strategies (ReAct, CoT, ToT, etc.) |
| `Jido.AI.Skills.*` | Modular capabilities (LLM, Planning, Reasoning) |
| `Jido.AI.Tools.Registry` | Unified tool storage and lookup |
| `Jido.AI.ToolAdapter` | Converts Actions to ReqLLM tools |

---

## Level 1: Simple LLM Calls

The simplest way to use Jido AI is direct text generation:

```elixir
# Generate text - just like calling an API
{:ok, response} = Jido.AI.generate_text("anthropic:claude-haiku-4-5", "What is 2 + 2?")
# => "2 + 2 equals 4"

# Generate structured output with a schema
schema = Zoi.object(%{name: Zoi.string(), age: Zoi.integer()})
{:ok, person} = Jido.AI.generate_object("openai:gpt-4", "Generate a person named Alice aged 30", schema)
# => %{name: "Alice", age: 30}
```

**Model identifiers** follow the pattern `provider:model_name`:
- `anthropic:claude-haiku-4-5`
- `openai:gpt-4`
- `google:gemini-pro`

---

## Level 2: Creating Tools (Actions)

Tools in Jido AI are `Jido.Action` modules:

```elixir
defmodule MyApp.Tools.Calculator do
  use Jido.Action,
    name: "calculator",
    description: "Performs basic arithmetic operations"

  # Schema defines and validates input parameters
  @schema Zoi.struct(__MODULE__, %{
    a: Zoi.number(description: "First number") |> Zoi.required(),
    b: Zoi.number(description: "Second number") |> Zoi.required(),
    operation: Zoi.string(description: "Operation: add, subtract, multiply, divide")
      |> Zoi.default("add")
  }, coerce: true)

  # The run function executes when the LLM calls this tool
  def run(%{a: a, b: b, operation: op}, _context) do
    result = case op do
      "add" -> a + b
      "subtract" -> a - b
      "multiply" -> a * b
      "divide" when b != 0 -> a / b
      "divide" -> {:error, "Division by zero"}
    end

    {:ok, %{result: result}}
  end
end
```

**Key components:**
- `name` - How the LLM refers to this tool
- `description` - Helps the LLM understand when to use it
- `@schema` - Defines and validates input parameters (using Zoi)
- `run/2` - The actual execution logic

---

## Level 3: ReAct Agents

The ReAct (Reason-Act) pattern is the most common agent architecture:

```elixir
defmodule MyApp.MathAgent do
  use Jido.AI.ReActAgent,
    name: "math_agent",
    description: "An agent that solves math problems",
    tools: [MyApp.Tools.Calculator],
    system_prompt: "You are a helpful math assistant. Use the calculator tool.",
    model: :fast  # Uses the :fast alias (claude-haiku)
end
```

**Using the agent:**

```elixir
# Start the agent
{:ok, pid} = Jido.AgentServer.start_link(MyApp.MathAgent, %{})

# Send a query via signal
signal = %Jido.Signal{
  type: "react.user_query",
  data: %{query: "What is 15 multiplied by 7?"}
}

Jido.AgentServer.call(pid, signal)
```

---

## Level 4: The ReAct Loop

The ReAct agent follows this state machine:

```
┌─────────────────────────────────────────────────────────┐
│                    ReAct Loop                           │
│                                                         │
│   [idle] ──start──> [awaiting_llm]                     │
│                          │                              │
│            ┌─────────────┴─────────────┐               │
│            │                           │               │
│            ▼                           ▼               │
│    (tool_calls)                  (final_answer)        │
│            │                           │               │
│            ▼                           ▼               │
│   [awaiting_tool]              [completed]             │
│            │                                           │
│            │ (all tools done)                          │
│            └──────────> [awaiting_llm] (loop back)     │
└─────────────────────────────────────────────────────────┘
```

**Example trace:**

```
User: "What is (5 + 3) * 2?"

Iteration 1:
  - LLM thinks: "I need to first add 5 + 3"
  - LLM calls: calculator(a: 5, b: 3, operation: "add")
  - Tool returns: {result: 8}

Iteration 2:
  - LLM thinks: "Now I need to multiply 8 by 2"
  - LLM calls: calculator(a: 8, b: 2, operation: "multiply")
  - Tool returns: {result: 16}

Iteration 3:
  - LLM thinks: "I have the answer"
  - LLM returns: "The result of (5 + 3) * 2 is 16"
  - Status: completed
```

---

## Level 5: Strategies

Strategies implement different reasoning patterns:

| Strategy | Use Case | Description |
|----------|----------|-------------|
| **ReAct** | Multi-step with tools | Reason-Act loop |
| **Chain-of-Thought** | Sequential reasoning | Step-by-step thinking |
| **Tree-of-Thoughts** | Exploration | Branching with evaluation |
| **Graph-of-Thoughts** | Complex reasoning | Graph-based with synthesis |
| **TRM** | Supervised reasoning | Thought-Refine-Merge |
| **Adaptive** | Auto-selection | Picks best strategy for task |

### Strategy Configuration

```elixir
defmodule MyApp.CustomAgent do
  use Jido.Agent,
    name: "custom_agent",
    strategy: {
      Jido.AI.Strategies.ReAct,
      tools: [MyApp.Tools.Calculator, MyApp.Tools.Search],
      system_prompt: "You are a research assistant...",
      model: "anthropic:claude-3-5-sonnet-20241022",
      max_iterations: 15
    }
end
```

### Adaptive Strategy

Automatically selects the best approach:

```elixir
defmodule MyApp.SmartAgent do
  use Jido.Agent,
    name: "smart_agent",
    strategy: {
      Jido.AI.Strategies.Adaptive,
      strategies: [:react, :cot, :tot, :got],
      tools: [Calculator, Search],
      model: :capable
    }
end
```

The Adaptive strategy analyzes queries:
- "Use the calculator to..." → ReAct (tool keywords)
- "Think step by step about..." → Chain-of-Thought
- "Explore different ways to..." → Tree-of-Thoughts

---

## Level 6: State Machines

Each strategy uses a **pure functional state machine** (Fsmx-based):

```elixir
defmodule Jido.AI.ReAct.Machine do
  use Fsmx.Struct,
    state_field: :status,
    transitions: %{
      "idle" => ["awaiting_llm"],
      "awaiting_llm" => ["awaiting_tool", "completed", "error"],
      "awaiting_tool" => ["awaiting_llm", "completed", "error"],
      "completed" => ["awaiting_llm"],
      "error" => ["awaiting_llm"]
    }

  defstruct status: "idle",
            iteration: 0,
            conversation: [],
            pending_tool_calls: [],
            result: nil,
            current_llm_call_id: nil,
            termination_reason: nil

  # Pure state transition - returns machine + directives
  def update(machine, msg, env \\ %{})

  def update(%__MODULE__{status: "idle"} = machine, {:start, query, call_id}, env) do
    system_prompt = Map.fetch!(env, :system_prompt)
    conversation = [system_message(system_prompt), user_message(query)]

    with_transition(machine, "awaiting_llm", fn machine ->
      machine = %{machine | iteration: 1, conversation: conversation}
      {machine, [{:call_llm_stream, call_id, conversation}]}
    end)
  end
end
```

**Key insight:** The machine is **pure** - it returns directives describing what to do, rather than doing it.

---

## Level 7: Directives and Signals

### Directives

Declarative side effect descriptions executed by the AgentServer runtime:

| Directive | Purpose |
|-----------|---------|
| `ReqLLMStream` | Stream LLM response with tools |
| `ReqLLMGenerate` | Non-streaming text generation |
| `ReqLLMEmbed` | Generate embeddings |
| `ToolExec` | Execute a tool (Jido.Action) |

```elixir
# Example directive creation
directive = Directive.ReqLLMStream.new!(%{
  id: "call_001",
  model: "anthropic:claude-haiku-4-5",
  context: conversation,
  tools: reqllm_tools
})
```

### Signals

Typed events for communication (CloudEvents-based):

| Signal | Type | Purpose |
|--------|------|---------|
| `ReqLLMResult` | `reqllm.result` | LLM call completed |
| `ReqLLMPartial` | `reqllm.partial` | Streaming token chunk |
| `ToolResult` | `ai.tool_result` | Tool execution completed |

### Signal Routes

Strategies declare which signals they handle:

```elixir
def signal_routes(_ctx) do
  [
    {"react.user_query", {:strategy_cmd, :react_start}},
    {"reqllm.result", {:strategy_cmd, :react_llm_result}},
    {"ai.tool_result", {:strategy_cmd, :react_tool_result}},
    {"reqllm.partial", {:strategy_cmd, :react_llm_partial}}
  ]
end
```

---

## Level 8: Skills Framework

Skills are **modular, self-contained capabilities** attached to agents:

```
┌─────────────────────────────────────────────────────────────────┐
│                        AGENT                                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │   LLM    │  │ Planning │  │Reasoning │  │  Tool    │       │
│  │  Skill   │  │  Skill   │  │  Skill   │  │ Calling  │       │
│  │          │  │          │  │          │  │  Skill   │       │
│  │ • Chat   │  │ • Plan   │  │ • Analyze│  │ • Call   │       │
│  │ • Embed  │  │ • Decomp │  │ • Infer  │  │ • Execute│       │
│  │ • Complt │  │ • Prior  │  │ • Explain│  │ • List   │       │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │
└─────────────────────────────────────────────────────────────────┘
```

### Available Skills

| Skill | Actions | Purpose |
|-------|---------|---------|
| **LLM** | Chat, Complete, Embed | Text generation and embeddings |
| **Planning** | Plan, Decompose, Prioritize | Task planning and decomposition |
| **Reasoning** | Analyze, Explain, Infer | Logical reasoning |
| **ToolCalling** | CallWithTools, ExecuteTool, ListTools | LLM tool calling |
| **Streaming** | StartStream, ProcessTokens, EndStream | Streaming responses |

### Skill Anatomy

```elixir
defmodule Jido.AI.Skills.LLM do
  use Jido.Skill,
    name: "llm",
    state_key: :llm,
    actions: [
      Jido.AI.Skills.LLM.Actions.Chat,
      Jido.AI.Skills.LLM.Actions.Complete,
      Jido.AI.Skills.LLM.Actions.Embed
    ],
    description: "Provides LLM chat, completion, and embedding capabilities",
    category: "ai",
    tags: ["llm", "chat", "completion", "embeddings"],
    vsn: "1.0.0"

  # Lifecycle callbacks
  @impl Jido.Skill
  def mount(_agent, config) do
    initial_state = %{
      default_model: Map.get(config, :default_model, :fast),
      default_max_tokens: Map.get(config, :default_max_tokens, 1024),
      default_temperature: Map.get(config, :default_temperature, 0.7)
    }
    {:ok, initial_state}
  end

  @impl Jido.Skill
  def router(_config) do
    [
      {"llm.chat", Jido.AI.Skills.LLM.Actions.Chat},
      {"llm.complete", Jido.AI.Skills.LLM.Actions.Complete},
      {"llm.embed", Jido.AI.Skills.LLM.Actions.Embed}
    ]
  end

  @impl Jido.Skill
  def handle_signal(_signal, _context), do: {:ok, :continue}

  @impl Jido.Skill
  def transform_result(_action, result, _context), do: result
end
```

### Skill Lifecycle Callbacks

| Callback | Purpose |
|----------|---------|
| `mount/2` | Initialize skill state when attached to agent |
| `router/1` | Map signal types to action modules |
| `handle_signal/2` | Pre-process signals before routing |
| `transform_result/3` | Post-process action results |

### Using Skills

**Direct action usage:**

```elixir
{:ok, result} = Jido.Exec.run(Jido.AI.Skills.LLM.Actions.Chat, %{
  model: :capable,
  prompt: "Explain GenServers in Elixir",
  system_prompt: "You are an expert Elixir teacher.",
  temperature: 0.5
})
```

**Mounting on agents:**

```elixir
defmodule MyApp.FullAgent do
  use Jido.Agent,
    name: "full_agent",
    skills: [
      {Jido.AI.Skills.LLM, [default_model: :fast]},
      {Jido.AI.Skills.Planning, []},
      {Jido.AI.Skills.Reasoning, []},
      {Jido.AI.Skills.ToolCalling, [auto_execute: true, max_turns: 10]}
    ],
    strategy: {Jido.AI.Strategies.ReAct, tools: [Calculator], model: :capable}
end
```

**Signal-based invocation:**

```elixir
signal = %Jido.Signal{
  type: "planning.decompose",
  data: %{
    goal: "Build a web application with authentication",
    max_depth: 3
  }
}
{:ok, result} = Jido.AgentServer.call(pid, signal)
```

### Creating Custom Skills

```elixir
defmodule MyApp.Skills.CodeReview do
  use Jido.Skill,
    name: "code_review",
    state_key: :code_review,
    actions: [
      MyApp.Skills.CodeReview.Actions.ReviewCode,
      MyApp.Skills.CodeReview.Actions.SuggestRefactor
    ],
    description: "AI-powered code review",
    category: "development",
    vsn: "1.0.0"

  @impl Jido.Skill
  def mount(_agent, config) do
    {:ok, %{
      default_model: Map.get(config, :default_model, :capable),
      strict_mode: Map.get(config, :strict_mode, false)
    }}
  end

  @impl Jido.Skill
  def router(_config) do
    [
      {"code_review.review", MyApp.Skills.CodeReview.Actions.ReviewCode},
      {"code_review.refactor", MyApp.Skills.CodeReview.Actions.SuggestRefactor}
    ]
  end
end
```

---

## Level 9: Dynamic System Prompts

### The Problem

The system prompt is currently **static** in the ReAct strategy - set once at initialization and not modified during iterations.

### Options for Dynamic Prompts

#### Option 1: Inject Context via User Message

```elixir
iteration_context = """
[Context for this request]
- Current iteration: #{iteration}
- Previous tools used: #{tools_used}

User query: #{actual_query}
"""

signal = %Jido.Signal{
  type: "react.user_query",
  data: %{query: iteration_context}
}
```

#### Option 2: Custom Strategy with Prompt Function

```elixir
defmodule MyApp.Strategies.DynamicReAct do
  use Jido.Agent.Strategy

  @impl true
  def init(%Agent{} = agent, ctx) do
    opts = ctx[:strategy_opts] || []

    # Store the system prompt generator function
    system_prompt_fn = Keyword.get(opts, :system_prompt_fn, &default_system_prompt/1)

    # ... rest of init
  end

  defp default_system_prompt(context) do
    base = "You are a helpful AI assistant."

    iteration_hint = case context.iteration do
      1 -> "This is the start. Think carefully about what tools you need."
      n when n > 5 -> "We've been at this for #{n} iterations. Try to conclude."
      _ -> "Continue reasoning step by step."
    end

    "#{base}\n\n#{iteration_hint}"
  end
end
```

**Usage:**

```elixir
defmodule MyApp.ResearchAgent do
  use Jido.Agent,
    name: "research_agent",
    strategy: {
      MyApp.Strategies.DynamicReAct,
      tools: [Search, Calculator],
      system_prompt_fn: fn context ->
        phase_prompt = case context.iteration do
          1 -> "PHASE 1 - EXPLORATION: Gather initial information"
          n when n in 2..4 -> "PHASE 2 - DEEP DIVE: Focus on details"
          n when n in 5..8 -> "PHASE 3 - SYNTHESIS: Connect information"
          _ -> "PHASE 4 - CONCLUSION: Summarize findings"
        end
        "You are a research assistant.\n\n#{phase_prompt}"
      end
    }
end
```

| Approach | Complexity | Flexibility |
|----------|------------|-------------|
| Inject via user message | Low | Medium |
| Custom strategy with prompt function | Medium | High |
| Custom machine | High | Maximum |

---

## Level 10: Tool Search Systems

### The Problem

With many tools (50+), loading all tool definitions into context is expensive:
- Each tool adds ~200-500 tokens
- 50 tools = 10,000-25,000 tokens just for definitions
- LLM takes longer and may get confused

### Solution: Meta-Tool Approach

Create a `tool_search` tool that returns matching tool definitions:

```elixir
defmodule MyApp.Tools.ToolSearch do
  use Jido.Action,
    name: "search_tools",
    description: """
    Search for available tools by keyword.
    Call this FIRST to discover what tools you can use.
    """,
    schema: Zoi.object(%{
      query: Zoi.string(description: "Keywords describing what you want to do"),
      category: Zoi.string() |> Zoi.optional(),
      limit: Zoi.integer() |> Zoi.default(5)
    })

  @tool_metadata %{
    "calculator" => %{
      description: "Perform arithmetic operations",
      category: "math",
      keywords: ["math", "calculate", "add", "multiply"]
    },
    "web_search" => %{
      description: "Search the web for information",
      category: "search",
      keywords: ["search", "web", "find", "lookup"]
    }
    # ... more tools
  }

  @impl Jido.Action
  def run(%{query: query, limit: limit} = params, _context) do
    results = @tool_metadata
      |> Enum.filter(fn {_name, meta} ->
        keyword_match?(meta, query) && category_match?(meta, params[:category])
      end)
      |> Enum.sort_by(fn {_name, meta} -> -relevance_score(meta, query) end)
      |> Enum.take(limit)
      |> Enum.map(fn {name, meta} ->
        %{name: name, description: meta.description, category: meta.category}
      end)

    {:ok, %{tools: results, total_found: length(results)}}
  end
end
```

---

## Level 11: Anti-Hallucination Patterns

### The Problem

Describing tools in text can lead to hallucination:
- LLM invents tool names that don't exist
- LLM guesses parameter names incorrectly
- LLM confuses text descriptions with actual callable tools

### Solution 1: Validated invoke_tool Wrapper

```elixir
defmodule MyApp.Tools.InvokeTool do
  use Jido.Action,
    name: "invoke_tool",
    description: "Execute a tool by name. The tool MUST exist in the registry.",
    schema: Zoi.object(%{
      tool_name: Zoi.string(description: "Exact name of the tool"),
      arguments: Zoi.map() |> Zoi.default(%{}),
      get_schema: Zoi.boolean() |> Zoi.default(false)
    })

  alias Jido.AI.Tools.{Registry, Executor}

  @impl Jido.Action
  def run(%{tool_name: name, get_schema: true}, _context) do
    case Registry.get(name) do
      {:ok, {:action, module}} ->
        {:ok, %{
          tool_name: name,
          description: module.description(),
          parameters: Jido.Action.Schema.to_json_schema(module.schema())
        }}
      {:error, :not_found} ->
        {:error, %{
          error: "Tool '#{name}' does not exist",
          similar_tools: find_similar_tools(name)
        }}
    end
  end

  def run(%{tool_name: name, arguments: args}, context) do
    case Registry.get(name) do
      {:ok, {:action, module}} ->
        case validate_arguments(module, args) do
          :ok -> Executor.execute(name, args, context)
          {:error, errors} -> {:error, %{validation_errors: errors}}
        end
      {:error, :not_found} ->
        {:error, %{error: "Tool '#{name}' not found"}}
    end
  end
end
```

### Solution 2: Two-Phase Strategy with Real Tool Loading

After search, **actually load the ReqLLM.Tool definitions** into the next LLM call:

```elixir
defp handle_search_complete(agent, state, {:ok, %{tools: found_tools}}) do
  tool_names = Enum.map(found_tools, & &1.name)

  # Load ACTUAL tool definitions from registry
  loaded_tools = Enum.flat_map(tool_names, fn name ->
    case Registry.get(name) do
      {:ok, {:action, module}} -> [ToolAdapter.from_action(module)]
      _ -> []
    end
  end)

  # Next LLM call includes REAL tools - no hallucination possible!
  directive = Directive.ReqLLMStream.new!(%{
    id: call_id,
    model: config.model,
    context: conversation,
    tools: loaded_tools  # ← Real ReqLLM.Tool structs!
  })

  {agent, [directive]}
end
```

### Anti-Hallucination Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ANTI-HALLUCINATION FLOW                         │
│                                                                     │
│  Phase 1: SEARCH                                                    │
│  ┌─────────────────────┐                                           │
│  │  LLM + search_tools │◄── Only search tool available             │
│  └──────────┬──────────┘                                           │
│             ▼                                                       │
│  Phase 2: LOAD REAL TOOLS                                          │
│  ┌─────────────────────┐                                           │
│  │ Registry.get(name)  │◄── Validate names exist                   │
│  └──────────┬──────────┘                                           │
│             ▼                                                       │
│  ┌─────────────────────┐                                           │
│  │ ToolAdapter.from_   │◄── Load REAL tool definitions             │
│  │ action(module)      │    with actual schemas                    │
│  └──────────┬──────────┘                                           │
│             ▼                                                       │
│  Phase 3: EXECUTE                                                   │
│  ┌─────────────────────┐                                           │
│  │  LLM + real tools   │◄── Real schemas prevent hallucination     │
│  └──────────┬──────────┘                                           │
│             ▼                                                       │
│  ┌─────────────────────┐                                           │
│  │ Executor validates  │◄── Double-check before execution          │
│  └─────────────────────┘                                           │
│                                                                     │
│  ✓ No hallucination possible at any step                           │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Level 12: Dynamic Tool Loading

### The Cleanest Solution

The ReAct strategy **already supports dynamic tool registration**:

```elixir
# These signals already exist in ReAct strategy
{"react.register_tool", {:strategy_cmd, :react_register_tool}}
{"react.unregister_tool", {:strategy_cmd, :react_unregister_tool}}
```

### Implementation

```elixir
defmodule MyApp.Tools.ToolSearch do
  use Jido.Action,
    name: "search_tools",
    description: "Search and LOAD tools. Found tools become immediately available."

  alias Jido.AI.Tools.Registry

  @impl Jido.Action
  def run(%{query: query, limit: limit}, context) do
    matching_tools = search_tools(query, limit)

    # Get the agent pid and dynamically register tools
    agent_pid = context[:agent_pid]

    registered = if agent_pid do
      Enum.map(matching_tools, fn tool_info ->
        signal = %Jido.Signal{
          type: "react.register_tool",
          data: %{tool_module: tool_info.module}
        }
        Jido.AgentServer.cast(agent_pid, signal)
        tool_info.name
      end)
    else
      Enum.map(matching_tools, fn tool_info ->
        Registry.register(tool_info.module)
        tool_info.name
      end)
    end

    {:ok, %{
      loaded_tools: Enum.map(matching_tools, &%{name: &1.name, description: &1.description}),
      message: "#{length(registered)} tools are now available"
    }}
  end
end
```

### Dynamic Tool Loading Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                     DYNAMIC TOOL LOADING                            │
│                                                                     │
│  Initial State:                                                     │
│  ┌──────────────────────────────────┐                              │
│  │ Strategy Tools: [search_tools]   │                              │
│  └──────────────────────────────────┘                              │
│                                                                     │
│  User: "Calculate the area of a circle"                            │
│         │                                                          │
│         ▼                                                          │
│  ┌──────────────────────────────────┐                              │
│  │ LLM: search_tools(query="math")  │                              │
│  └──────────────────────────────────┘                              │
│         │                                                          │
│         ▼                                                          │
│  ┌──────────────────────────────────┐                              │
│  │ ToolSearch.run():                │                              │
│  │  1. Find: [calculator]           │                              │
│  │  2. Send react.register_tool     │◄── Dynamic registration!     │
│  └──────────────────────────────────┘                              │
│         │                                                          │
│         ▼                                                          │
│  ┌──────────────────────────────────┐                              │
│  │ Strategy State Updated:          │                              │
│  │ tools: [search_tools, calculator]│◄── Now has real tool!        │
│  └──────────────────────────────────┘                              │
│         │                                                          │
│         ▼                                                          │
│  ┌──────────────────────────────────┐                              │
│  │ Next LLM Call includes real      │◄── Real tool definitions!    │
│  │ calculator tool definition       │                              │
│  └──────────────────────────────────┘                              │
└─────────────────────────────────────────────────────────────────────┘
```

### Complete Agent Setup

```elixir
defmodule MyApp.DynamicToolAgent do
  use Jido.Agent,
    name: "dynamic_tool_agent",
    strategy: {
      Jido.AI.Strategies.ReAct,
      # Start with ONLY the search tool
      tools: [MyApp.Tools.ToolSearch],
      system_prompt: """
      You have access to a large library of tools, but they must be loaded first.

      WORKFLOW:
      1. Call `search_tools` with keywords for what you need
      2. The matching tools will be AUTOMATICALLY LOADED
      3. You can then use them directly by name
      """,
      model: :capable,
      max_iterations: 15,
      use_registry: true
    }
end

# Register all tools at application startup
defmodule MyApp.Application do
  def start(_type, _args) do
    Jido.AI.Tools.Registry.register(MyApp.Tools.Calculator)
    Jido.AI.Tools.Registry.register(MyApp.Tools.WebSearch)
    Jido.AI.Tools.Registry.register(MyApp.Tools.FileRead)
    # ... 50+ more tools (not loaded into LLM context!)

    Supervisor.start_link(children, strategy: :one_for_one)
  end
end
```

---

## Key Takeaways

- **Jido AI** provides a complete framework for building AI agents with LLM integration
- **Strategies** implement different reasoning patterns (ReAct, CoT, ToT, Adaptive)
- **State machines** are pure - they return directives, not execute side effects
- **Directives** describe effects; **Signals** enable communication between components
- **Skills** provide modular, composable capabilities that can be mounted on agents
- **Dynamic system prompts** can be achieved through custom strategies with prompt functions
- **Tool search** reduces context usage by loading only needed tools
- **Anti-hallucination** requires loading real `ReqLLM.Tool` definitions, not text descriptions
- **Dynamic tool registration** is already supported via `react.register_tool` signals
- The **Registry + dynamic loading** pattern is the cleanest architecture for large tool libraries

---

## Next Topics to Explore

- [ ] Building custom strategies from scratch
- [ ] Implementing embedding-based semantic tool search
- [ ] Multi-agent coordination and hierarchies
- [ ] Streaming responses and partial updates
- [ ] Error handling and retry patterns
- [ ] Testing strategies for AI agents
- [ ] Production deployment considerations
- [ ] Telemetry and observability for AI agents
