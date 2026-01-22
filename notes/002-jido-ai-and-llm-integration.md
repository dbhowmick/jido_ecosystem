# 002 - Jido AI and LLM Integration

> Adding intelligence to Jido agents through LLM integration

Jido AI transforms regular Jido agents into **intelligent reasoning agents** by providing unified LLM access, advanced reasoning strategies, and seamless integration with the core Jido patterns.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      YOUR APPLICATION                        │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────┐   │
│  │                    jido_ai                           │   │
│  │  Models, Prompts, Conversations, Reasoning Runners   │   │
│  └──────────────────────┬──────────────────────────────┘   │
│                         │                                   │
│  ┌──────────────────────▼──────────────────────────────┐   │
│  │                    req_llm                           │   │
│  │         Unified access to 57+ LLM providers          │   │
│  │    (OpenAI, Anthropic, Google, Groq, xAI, etc.)     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              jido (core) + jido_action + jido_signal │   │
│  │         Agents, Actions, Skills, Signals, Plans      │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

**Key insight**: jido_ai doesn't replace core Jido patterns - it enhances them. Agents still use `cmd/2`, Skills, and Signals, but can now think, reason, and call LLMs.

---

## ReqLLM - The Universal LLM Client

ReqLLM provides **unified HTTP access to 57+ LLM providers** with a consistent API.

### Core API

```elixir
# Same code works with any provider
{:ok, response} = ReqLLM.generate_text("anthropic:claude-3-5-sonnet", "Hello!")
{:ok, response} = ReqLLM.generate_text("openai:gpt-4o", "Hello!")
{:ok, response} = ReqLLM.generate_text("google:gemini-1.5-pro", "Hello!")
```

| Function | Purpose |
|----------|---------|
| `generate_text/3` | Text generation (chat completions) |
| `stream_text/3` | Real-time streaming responses |
| `generate_object/4` | Structured output (JSON schema) |
| `generate_image/2` | Image generation (DALL-E, etc.) |
| `embed/3` | Text embeddings |

### Key Data Structures

```elixir
# Context = conversation history
ctx = ReqLLM.Context.new([
  ReqLLM.Message.system("You are helpful"),
  ReqLLM.Message.user("What is Elixir?")
])

# Tool = function the LLM can call
tool = ReqLLM.tool(
  name: "get_weather",
  description: "Get weather for a location",
  parameter_schema: [location: [type: :string, required: true]],
  callback: {WeatherService, :fetch}
)

# Response = what comes back
{:ok, response} = ReqLLM.generate_text(model, ctx, tools: [tool])
response.message.content  # The text response
response.usage            # Token counts and costs
```

### Supported Providers

- **Anthropic** - Claude 3.5 Sonnet, Opus, Haiku (with web search, prompt caching, thinking)
- **OpenAI** - GPT-4o, GPT-4 Turbo, o1/o3/o4 reasoning models, DALL-E
- **Google** - Gemini 1.5 Pro/Flash, Vertex AI
- **Groq** - Fast inference with Llama models
- **OpenRouter** - Gateway to 100+ models
- **xAI** - Grok models with reasoning
- **AWS Bedrock** - Managed inference service
- **And 50+ more...**

### Multi-Turn Conversation

```elixir
ctx = ReqLLM.Context.new([
  ReqLLM.Message.system("You are helpful"),
  ReqLLM.Message.user("What's the capital of France?")
])

{:ok, response1} = ReqLLM.generate_text("anthropic:claude-3-5-sonnet", ctx)

# Continue with response context (includes new assistant message)
ctx2 = response1.context
ctx2 = ReqLLM.Context.append(ctx2, ReqLLM.Message.user("And what's its population?"))

{:ok, response2} = ReqLLM.generate_text("anthropic:claude-3-5-sonnet", ctx2)
```

### Streaming

```elixir
{:ok, response} = ReqLLM.stream_text("anthropic:claude-3-5-sonnet", "Tell a story")

# Real-time token streaming
response.stream
|> ReqLLM.StreamResponse.tokens()
|> Stream.each(&IO.write/1)
|> Stream.run()

# Concurrent metadata collection
usage = ReqLLM.StreamResponse.usage(response)
```

### Structured Output

```elixir
schema = [
  name: [type: :string, required: true],
  age: [type: :pos_integer, required: true],
  email: [type: :string, required: true]
]

{:ok, response} = ReqLLM.generate_object(
  "anthropic:claude-3-5-sonnet",
  "Generate a person",
  schema
)

person = response.object  #=> %{name: "John Doe", age: 30, email: "john@example.com"}
```

---

## Jido.AI.Model - LLM Configuration

Models wrap ReqLLM with validation and normalization:

```elixir
# All these formats work
{:ok, model} = Jido.AI.Model.from({:anthropic, [model: "claude-3-5-sonnet"]})
{:ok, model} = Jido.AI.Model.from("openai:gpt-4o")
{:ok, model} = Jido.AI.Model.from(%{provider: :google, model: "gemini-1.5-pro"})

# Model stores configuration
model.provider      # :anthropic
model.model         # "claude-3-5-sonnet"
model.temperature   # 0.7
model.max_tokens    # 4096
```

---

## Jido.AI.Prompt - Template-Based Prompts

Prompts are **immutable, versioned message templates** with EEx and Liquid support.

### Basic Usage

```elixir
# Simple prompt
prompt = Jido.AI.Prompt.new(:user, "What is Elixir?")

# With EEx templating
prompt = Jido.AI.Prompt.new(:user, "Explain <%= @topic %> in simple terms",
  engine: :eex,
  params: %{topic: "recursion"}
)

# Multi-message prompt with mixed engines
prompt = Jido.AI.Prompt.new(%{
  messages: [
    %{role: :system, content: "You are a <%= @role %> assistant", engine: :eex},
    %{role: :user, content: "Help me with {{ task }}", engine: :liquid}
  ],
  params: %{role: "helpful", task: "coding"}
})

# Render resolves templates
messages = Jido.AI.Prompt.render(prompt)
# => [%{role: :system, content: "You are a helpful assistant"},
#     %{role: :user, content: "Help me with coding"}]
```

### Versioning

Prompts track history for experimentation and rollback:

```elixir
v1 = Jido.AI.Prompt.new(:user, "Initial question")
v2 = Jido.AI.Prompt.new_version(v1, fn p ->
  Jido.AI.Prompt.add_message(p, :assistant, "Response")
end)
v3 = Jido.AI.Prompt.new_version(v2, fn p ->
  Jido.AI.Prompt.add_message(p, :user, "Follow-up")
end)

# Can rewind to any version
{:ok, original} = Jido.AI.Prompt.get_version(v3, 1)
```

---

## ChatCompletion Action

The main action for calling LLMs:

```elixir
alias Jido.AI.Actions.ReqLlm.ChatCompletion

{:ok, result} = ChatCompletion.run(%{
  model: model,
  prompt: prompt,
  temperature: 0.7,
  max_tokens: 1000
}, %{})

result.content  # "Elixir is a functional programming language..."
```

### With Tools (Function Calling)

Jido Actions automatically become LLM tools:

```elixir
{:ok, result} = ChatCompletion.run(%{
  model: model,
  prompt: Jido.AI.Prompt.new(:user, "What's the weather in Paris?"),
  tools: [MyApp.Actions.GetWeather, MyApp.Actions.SearchWeb]
}, %{})

result.tool_results  # Results from executed tools
```

---

## Conversation Manager - Stateful Sessions

For multi-turn conversations with persistent state:

```elixir
alias Jido.AI.Conversation.Manager

# Create a session
{:ok, conv_id} = Manager.create(model,
  system_prompt: "You are a helpful coding assistant"
)

# Turn 1
:ok = Manager.add_message(conv_id, :user, "How do I write a GenServer?")
{:ok, messages} = Manager.get_messages_for_llm(conv_id)
{:ok, response} = ChatCompletion.run(%{model: model, prompt: messages}, %{})
:ok = Manager.add_message(conv_id, :assistant, response.content)

# Turn 2 (context is preserved)
:ok = Manager.add_message(conv_id, :user, "Can you show me an example?")
{:ok, messages} = Manager.get_messages_for_llm(conv_id)
# messages now contains the full conversation history

# Cleanup
:ok = Manager.delete(conv_id)
```

---

## Jido.AI.Skill - The handle_signal Pattern

The AI.Skill mounts AI capabilities on agents using a **signal interception pattern**.

### How It Works

```
Signal arrives with minimal data
         │
         ▼
┌─────────────────────────────────────────┐
│  Skill.handle_signal(signal, skill_opts) │
│                                         │
│  1. Extract model from skill_opts       │
│  2. Extract tools from skill_opts       │
│  3. Render prompt template with data    │
│  4. REPLACE signal.data with full config│
└─────────────────────────────────────────┘
         │
         ▼
Signal now contains model, prompt, tools
         │
         ▼
Action.run(params, context)
  └── params = signal.data (with injected config!)
```

### The Code Path

**Step 1: User sends signal with minimal data**

```elixir
# User code - just sends a message
{:ok, result} = Jido.AI.Agent.chat_response(agent_pid, "What is Elixir?")

# Internally creates signal with ONLY the message
signal = %Signal{
  type: "jido.ai.chat.response",
  data: %{message: "What is Elixir?"}  # ← Just the user's message
}
```

**Step 2: Skill intercepts and injects config**

```elixir
# In Jido.AI.Skill (handle_signal/2)
def handle_signal(%Signal{type: "jido.ai.chat.response"} = signal, skill_opts) do
  # Extract config from skill_opts (what you configured when mounting)
  base_prompt = Keyword.get(skill_opts, :prompt)
  model = Keyword.get(skill_opts, :model)

  # Render prompt template with signal data
  rendered_prompt = render_prompt(base_prompt, signal.data)

  # BUILD THE FULL PARAMS
  chat_response_params = %{
    model: model,              # ← From skill config!
    prompt: rendered_prompt    # ← From skill config + user message!
  }

  # REPLACE signal.data with full params
  {:ok, %{signal | data: chat_response_params}}
end
```

**Step 3: Action receives full config**

```elixir
# ChatResponse action receives params from the modified signal.data
def run(params, context) do
  # params NOW contains:
  # %{
  #   model: %Jido.AI.Model{provider: :anthropic, model: "claude-3-5-sonnet"},
  #   prompt: %Jido.AI.Prompt{messages: [...]}
  # }

  # Action just uses what it receives - doesn't know about the Skill
  ChatCompletion.run(%{model: params.model, prompt: params.prompt}, context)
end
```

### Complete Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│  Agent Definition                                                   │
│                                                                     │
│  skills: [                                                         │
│    {Jido.AI.Skill,                                                 │
│      model: {:anthropic, [model: "claude-3-5-sonnet"]},  ← CONFIG  │
│      prompt: "You are helpful. User says: <%= @message %>",         │
│      tools: [SearchWeb, Calculator]                                │
│    }                                                                │
│  ]                                                                  │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              │ skill_opts stored
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Signal Arrives: "jido.ai.chat.response"                           │
│  data: %{message: "What is Elixir?"}                               │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  AgentServer calls: Skill.handle_signal(signal, skill_opts)        │
│                                                                     │
│  skill_opts contains:                                              │
│    model: {:anthropic, [model: "claude-3-5-sonnet"]}               │
│    prompt: "You are helpful. User says: <%= @message %>"           │
│    tools: [SearchWeb, Calculator]                                  │
│                                                                     │
│  TRANSFORMS signal.data to:                                        │
│    %{                                                              │
│      model: %Model{provider: :anthropic, ...},                     │
│      prompt: %Prompt{content: "You are helpful. User says: What..."}│
│      tools: [SearchWeb, Calculator]                                │
│    }                                                                │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Router matches "jido.ai.chat.response" → ChatResponse action      │
│                                                                     │
│  ChatResponse.run(params, context)                                 │
│    params = signal.data (with injected config!)                    │
│                                                                     │
│  Action uses params.model, params.prompt, params.tools             │
│  Delegates to ChatCompletion → ReqLLM → Anthropic API              │
└─────────────────────────────────────────────────────────────────────┘
```

### Router Definition

```elixir
# In Jido.AI.Skill
def router(_opts \\ []) do
  [
    {"jido.ai.chat.response", %Instruction{action: Jido.AI.Actions.Internal.ChatResponse}},
    {"jido.ai.tool.response", %Instruction{action: Jido.AI.Actions.ReqLlm.ToolResponse}},
    {"jido.ai.boolean.response", %Instruction{action: Jido.AI.Actions.Internal.BooleanResponse}}
  ]
end
```

The router maps signal types to action modules. The **params come from handle_signal**, not from the router.

### Correct Usage Pattern

```elixir
defmodule MyApp.AssistantAgent do
  use Jido.Agent,
    name: "assistant",
    schema: [
      conversation_history: [type: {:list, :map}, default: []]
    ],
    skills: [
      {Jido.AI.Skill,
        model: {:anthropic, [model: "claude-3-5-sonnet"]},
        prompt: """
        You are a helpful assistant.

        User message: <%= @message %>
        """,
        tools: [MyApp.Actions.SearchWeb, MyApp.Actions.Calculator]
      }
    ]

  alias Jido.Agent.Directive
  alias Jido.Signal

  # Handle user chat request
  def cmd(agent, {:chat, user_message}) do
    # Create signal with JUST the user's message
    # The Skill's handle_signal will inject model, prompt template, and tools
    signal = Signal.new!("jido.ai.chat.response",
      %{message: user_message},  # ← Only pass the variable parts
      source: "/chat"
    )

    # Emit to self - Skill will intercept, inject config, and route to action
    {agent, [Directive.emit(signal, dispatch: {:pid, target: self()})]}
  end

  # Handle response from AI action
  def cmd(agent, {:ai_complete, response}) do
    updated_agent = update_in(
      agent.state.conversation_history,
      &[%{role: :assistant, content: response.content} | &1]
    )

    {updated_agent, []}
  end
end
```

---

## Advanced Reasoning Runners

Runners are execution strategies that enhance how agents process instructions with different reasoning patterns.

### Chain-of-Thought (CoT)

Makes the LLM "think step by step" before answering.

```
Without CoT: "The answer is 42"
With CoT:    "Let me think through this...
              Step 1: First, I need to consider X
              Step 2: Then, Y affects Z
              Step 3: Therefore, the answer is 42"
```

| Metric | Value |
|--------|-------|
| Accuracy improvement | +8-15% |
| Token increase | 3-4x |
| Latency increase | ~2-3s |

```elixir
defmodule ReasoningAgent do
  use Jido.Agent,
    name: "reasoning_agent",
    runner: Jido.AI.Runner.ChainOfThought,
    actions: [AnalyzeData, MakeDecision]
end
```

### ReAct (Reasoning + Acting)

Interleaves thinking with tool use in a loop:

```
Thought: I need to find the weather in Paris
Action: get_weather(location: "Paris")
Observation: 22°C, sunny
Thought: Now I can answer the user
Action: respond("The weather in Paris is 22°C and sunny")
```

| Metric | Value |
|--------|-------|
| Accuracy improvement | +27.4% (HotpotQA) |
| Cost | 10-20x (depends on steps) |
| Best for | Multi-step tasks requiring external data |

```elixir
defmodule ResearchAgent do
  use Jido.Agent,
    runner: Jido.AI.Runner.ReAct,
    actions: [SearchWeb, ReadDocument, Summarize]
end
```

### Tree-of-Thoughts (ToT)

Explores multiple reasoning paths and picks the best:

```
                    Problem
                       │
         ┌─────────────┼─────────────┐
         │             │             │
      Path A        Path B        Path C
      (score: 0.7)  (score: 0.9)  (score: 0.3)
         │             │             │
         ×          Continue         ×
                       │
                   Solution
```

| Metric | Value |
|--------|-------|
| Accuracy improvement | +74% |
| Cost | High (multiple LLM calls) |
| Best for | Complex problems with multiple approaches |

### Self-Consistency

Generates multiple answers and votes on the best:

```elixir
# Ask the same question 5 times
answers = ["Paris", "Paris", "Paris", "Lyon", "Paris"]
# Vote: "Paris" wins (4/5)
final_answer = "Paris"
```

| Metric | Value |
|--------|-------|
| Accuracy improvement | +17.9% |
| Cost | Multiple LLM calls |
| Best for | Reducing randomness |

### GEPA (Genetic Evolutionary Prompt Optimization)

Evolves prompts over generations to find optimal phrasing:

```
Generation 1: "Explain X" → 70% accuracy
Generation 2: "Clearly explain X step by step" → 82% accuracy
Generation 3: "As an expert, explain X with examples" → 91% accuracy
```

| Metric | Value |
|--------|-------|
| Accuracy improvement | +10-19% |
| Cost | Multiple generations/iterations |
| Best for | Optimizing prompts for specific tasks |

---

## Complete Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  User Request: "Research AI trends and write a report"      │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│  Jido Agent (with AI.Skill)                                 │
│  ├─ State: conversation history, research data              │
│  ├─ Skills: [AI.Skill, ResearchSkill, WritingSkill]        │
│  └─ Runner: ReAct (reasoning + acting)                      │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│  Runner (ReAct)                                             │
│                                                             │
│  Thought: I need to search for AI trends                    │
│  Action: SearchWeb.run(%{query: "AI trends 2024"})         │
│  Observation: [list of articles...]                         │
│                                                             │
│  Thought: I should read the top articles                    │
│  Action: ReadArticle.run(%{url: "..."})                    │
│  Observation: [article content...]                          │
│                                                             │
│  Thought: Now I can write the report                        │
│  Action: WriteReport.run(%{data: research_data})           │
│  Observation: [report content...]                           │
│                                                             │
│  Thought: Task complete                                     │
│  Final Answer: "Here's your report on AI trends..."        │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│  ChatCompletion Action                                      │
│  ├─ Converts Prompt to messages                            │
│  ├─ Converts Actions to tool schemas                        │
│  └─ Calls ReqLLM                                           │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│  ReqLLM                                                     │
│  ├─ Detects provider (Anthropic)                           │
│  ├─ Formats request for Claude API                         │
│  ├─ Handles streaming, tool calls                          │
│  └─ Returns normalized Response                            │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
                    Claude API (Anthropic)
```

---

## Real-World Example: AI Research Assistant

### Define Tools as Jido Actions

```elixir
defmodule MyApp.Actions.SearchWeb do
  use Jido.Action,
    name: "search_web",
    description: "Search the web for information",
    schema: [query: [type: :string, required: true]]

  def run(%{query: query}, _ctx) do
    results = WebSearch.search(query)
    {:ok, %{results: results}}
  end
end

defmodule MyApp.Actions.ReadUrl do
  use Jido.Action,
    name: "read_url",
    description: "Read content from a URL",
    schema: [url: [type: :string, required: true]]

  def run(%{url: url}, _ctx) do
    {:ok, content} = HttpClient.get(url)
    {:ok, %{content: content}}
  end
end
```

### Define the AI Agent (Signal-Driven)

```elixir
defmodule MyApp.ResearchAgent do
  use Jido.Agent,
    name: "research_agent",
    description: "AI-powered research assistant",
    schema: [
      research_notes: [type: {:list, :map}, default: []],
      current_task: [type: :string, default: nil]
    ],
    skills: [
      {Jido.AI.Skill,
        model: {:anthropic, [model: "claude-3-5-sonnet"]},
        prompt: """
        You are a research assistant. Help the user with their research task.

        Task: <%= @task %>

        Use the available tools to search and read information, then provide
        a comprehensive summary.
        """,
        tools: [
          MyApp.Actions.SearchWeb,
          MyApp.Actions.ReadUrl
        ]
      }
    ]

  alias Jido.Agent.Directive
  alias Jido.Signal

  # Handle research request - emit signal for AI.Skill
  def cmd(agent, {:research, topic}) do
    updated_agent = put_in(agent.state.current_task, topic)

    # Create signal with task data
    # AI.Skill will inject model, render prompt template, and add tools
    signal = Signal.new!("jido.ai.tool.response",
      %{task: topic},
      source: "/research"
    )

    {updated_agent, [Directive.emit(signal, dispatch: {:pid, target: self()})]}
  end

  # Handle AI response
  def cmd(agent, {:ai_complete, result}) do
    research_entry = %{
      topic: agent.state.current_task,
      summary: result.content,
      sources: result.tool_results,
      timestamp: DateTime.utc_now()
    }

    updated_agent = agent
      |> update_in([Access.key(:state), :research_notes], &[research_entry | &1])
      |> put_in([Access.key(:state), :current_task], nil)

    signal = Signal.new!("research.completed", research_entry, source: "/research")

    {updated_agent, [Directive.emit(signal)]}
  end
end
```

### Usage

```elixir
# Start the agent
{:ok, pid} = Jido.AgentServer.start_link(MyApp.ResearchAgent, id: "researcher_1")

# Send research request
signal = Jido.Signal.new!("research.start", %{topic: "Elixir GenServers"})
Jido.AgentServer.apply(pid, signal)
```

---

## Two Valid Approaches

### Approach 1: Use Skills (Signal-Driven)

Skills encapsulate configuration. Send signals, skill handles the rest:

```elixir
# Agent defines skill with config
skills: [
  {Jido.AI.Skill, model: my_model, tools: my_tools}
]

# In cmd/2, emit signal - skill's config is used
signal = Signal.new!("jido.ai.chat.response", %{message: "..."})
{agent, [Directive.emit(signal, dispatch: {:pid, target: self()})]}
```

**Pros:** Clean abstraction, config in one place, skill manages state
**Cons:** More indirect, signal-based flow

### Approach 2: Direct Action Calls (No Skill)

Don't configure a skill. Call actions directly:

```elixir
# No AI.Skill in skills list
skills: []

# In cmd/2, call action directly with explicit config
{:ok, result} = ChatCompletion.run(%{
  model: my_model,
  prompt: my_prompt,
  tools: my_tools
}, context)
```

**Pros:** Explicit, easy to follow, full control
**Cons:** Config repeated everywhere, no skill state management

### What NOT To Do

Configure a skill but then bypass it:

```elixir
# BAD: Configures skill...
skills: [{Jido.AI.Skill, model: model_a, tools: tools_a}]

# BAD: ...but calls action directly with different config!
ChatCompletion.run(%{model: model_b, tools: tools_b}, ctx)
```

This is confusing and defeats the purpose of skills.

---

## Key Architectural Insights

1. **handle_signal/2 is the magic** - Intercepts signals and injects skill config into `signal.data`

2. **Actions are config-agnostic** - They expect `params` with `model`, `prompt`, etc. They don't know about Skills.

3. **Signal data is transformed** - User sends `%{message: "..."}`, action receives `%{model: ..., prompt: ..., tools: ...}`

4. **Prompt templating** - Skill's prompt uses EEx templates like `<%= @message %>` filled with signal data

5. **Clean separation**:
   - **Skill**: Owns configuration, intercepts signals, renders templates
   - **Action**: Receives full params, executes LLM call
   - **Router**: Maps signal types to action modules

---

## Configuration

### API Key Management

Precedence order:
1. Per-request `:api_key` option
2. Application config: `config :req_llm, :anthropic_api_key, "..."`
3. Environment: `ANTHROPIC_API_KEY` (auto-loaded via dotenvy)

```elixir
# Set API key programmatically
Jido.AI.Keyring.set_session_value(:anthropic_api_key, "sk-ant-...")

# Get with defaults
key = Jido.AI.Keyring.get(:anthropic_api_key, "default")
```

### Common Generation Options

- `:temperature` - 0.0 to 2.0
- `:max_tokens` - Limit response length
- `:top_p` - Nucleus sampling
- `:tools` - List of tool definitions
- `:tool_choice` - `:auto | :required | {:function, "name"}`
- `:system_prompt` - Prepended system message

---

## Quick Reference

| Component | Purpose | Key File |
|-----------|---------|----------|
| ReqLLM | Unified LLM client | `req_llm/lib/req_llm.ex` |
| Model | LLM configuration | `jido_ai/lib/jido_ai/model.ex` |
| Prompt | Versioned templates | `jido_ai/lib/jido_ai/prompt.ex` |
| ChatCompletion | Main LLM action | `jido_ai/lib/jido_ai/actions/req_llm/chat_completion.ex` |
| Conversation.Manager | Stateful sessions | `jido_ai/lib/jido_ai/conversation/manager.ex` |
| AI.Skill | Mount AI on agents | `jido_ai/lib/jido_ai/skill.ex` |
| ChainOfThought | CoT runner | `jido_ai/lib/jido_ai/runner/chain_of_thought.ex` |
| ReAct | ReAct runner | `jido_ai/lib/jido_ai/runner/react.ex` |
| TreeOfThoughts | ToT runner | `jido_ai/lib/jido_ai/runner/tree_of_thoughts.ex` |

---

## Design Philosophy

1. **Provider Agnostic** - Switch LLMs without code changes
2. **Composable** - Mix reasoning strategies, tools, and agents
3. **Transparent** - Prompts and reasoning visible in agent state
4. **Extensible** - Create custom runners, actions, tools
5. **Production Ready** - Cost tracking, streaming, error handling
6. **Skill as Decorator** - AI.Skill enriches signals before they reach actions

---

## Next Topics to Explore

- [ ] Custom reasoning runners
- [ ] Multi-agent AI workflows
- [ ] Context window management
- [ ] Cost optimization strategies
- [ ] jido_workbench: Phoenix demo application
