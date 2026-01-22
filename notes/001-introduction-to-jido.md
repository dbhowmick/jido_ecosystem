# 001 - Introduction to Jido

> **Jido** (自動) - Japanese for "automatic" or "self-moving"

Jido is an Elixir framework for building **autonomous multi-agent systems**. It provides architecture for creating intelligent, self-managing processes that can make decisions independently, communicate through structured events, and compose complex workflows from simple pieces.

---

## The Three Foundational Packages

```
┌─────────────────────────────────────────────────────────────┐
│                    JIDO ECOSYSTEM                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ jido_action │  │ jido_signal │  │    jido     │         │
│  │             │  │             │  │   (core)    │         │
│  │ "Vocabulary"│  │  "Nervous   │  │  "Brain"    │         │
│  │             │  │   System"   │  │             │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 1. Jido Action - "The Vocabulary"

**Actions** are validated, composable units of work. They're like smart functions that know how to validate inputs/outputs, describe themselves to AI systems, handle errors, and chain into workflows.

```elixir
defmodule SendEmail do
  use Jido.Action,
    name: "send_email",
    schema: [
      to: [type: :string, required: true],
      subject: [type: :string, required: true],
      body: [type: :string, required: true]
    ]

  def run(params, _context) do
    # Send the email...
    {:ok, %{sent: true, message_id: "123"}}
  end
end
```

**Why Actions over plain functions?**
- Automatic input/output validation
- Retry logic with exponential backoff
- Timeout protection
- AI tool compatibility (self-describing for LLMs)
- Composability into workflows (Plans)

### 2. Jido Signal - "The Nervous System"

**Signals** are CloudEvents-compliant messages that agents use to communicate. They enable event-driven architecture with decoupled agent communication.

```elixir
# Create a signal
{:ok, signal} = Jido.Signal.new(
  "order.created",
  %{order_id: "123", total: 99.99},
  source: "/checkout"
)

# Subscribe to patterns
Jido.Signal.Bus.subscribe(:bus, "order.*",
  dispatch: {:pid, target: handler_pid}
)
```

**Key Features:**
- Standard format (CloudEvents spec)
- Pattern-based routing (`"order.*"`, `"payment.**"`)
- History and replay capabilities
- Multiple delivery targets
- Dead letter queues for failures

### 3. Jido Core - "The Brain"

**Agents** are immutable decision-makers that follow the fundamental `cmd/2` pattern:

```elixir
{new_agent, directives} = Agent.cmd(agent, action)
```

**The cmd/2 Pattern (Elm/Redux-inspired):**
- Takes current agent state + an action to perform
- Returns new agent state + effects to execute
- **Pure function**: same inputs → same outputs
- State changes are data transformations
- Side effects are described as **directives** (executed by runtime)

---

## Core Concepts

### The cmd/2 Pattern

This is the **single most important pattern** in Jido:

```elixir
{agent, directives} = MyAgent.cmd(agent, action)
```

- **Agents are immutable** — the returned agent is a complete new value
- **cmd/2 is pure** — given identical inputs, always produces identical outputs
- **State changes are complete** — returned agent is fully updated
- **Directives are external only** — never modify agent state; describe effects

### Directives: Describing vs. Doing

Agents **describe** what should happen, they don't **do** it. Directives are pure data structures:

| Directive | Purpose |
|-----------|---------|
| `%Emit{signal: ...}` | "Please send this signal" |
| `%SpawnAgent{...}` | "Please start a child agent" |
| `%Schedule{...}` | "Please send me a message later" |
| `%Stop{...}` | "Please stop this agent" |
| `%StopChild{...}` | "Please stop a child agent" |

The **AgentServer** runtime interprets and executes directives. This separation means:
- `cmd/2` stays pure and testable
- Easy to test: just check returned directives
- Time-travel debugging possible
- State changes complete before directives run

### Skills: Reusable Agent Capabilities

**Skills** are composable capabilities you attach to agents. They bundle:
- A set of actions the agent can perform
- State schema (nested under `state_key`)
- Signal routing rules
- Lifecycle hooks

```elixir
defmodule ChatSkill do
  use Jido.Skill,
    name: "chat",
    state_key: :chat,
    actions: [SendMessage, GetHistory],
    schema: [
      messages: [type: {:list, :map}, default: []],
      model: [type: :string, default: "gpt-4"]
    ]

  def router(_config) do
    [
      {"chat.send", SendMessage},
      {"chat.history", GetHistory}
    ]
  end
end

# Use in agents
defmodule SupportAgent do
  use Jido.Agent,
    skills: [ChatSkill, TicketSkill]
end
```

### Plans: Workflow Orchestration

**Plans** organize actions into DAGs (directed acyclic graphs) with dependencies:

```elixir
plan = Jido.Plan.new()
  |> Jido.Plan.add(:fetch, FetchOrderAction)
  |> Jido.Plan.add(:validate, ValidateAction, depends_on: :fetch)
  |> Jido.Plan.add(:save, SaveAction, depends_on: :validate)
```

Plans analyze dependencies and run steps in parallel when possible.

---

## Skills vs Plans

These are **orthogonal concepts** that serve different purposes:

| **Skills** | **Plans** |
|------------|-----------|
| What an agent **can do** | How to **orchestrate** work |
| Attach capabilities to agents | Organize actions into workflows |
| Bundle related actions + state | Define execution order + dependencies |
| Agent-level concept | Workflow-level concept |

**Relationship:**
- Skills provide actions, Plans use them
- An agent's `cmd/2` might execute a Plan internally
- Skills can define internal Plans for complex operations

```
┌─────────────────────────────────────────┐
│              Agent                      │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐   │
│  │ Skill A │ │ Skill B │ │ Skill C │   │  ← Skills give capabilities
│  │ actions │ │ actions │ │ actions │   │
│  └─────────┘ └─────────┘ └─────────┘   │
│                                         │
│  cmd/2 can orchestrate using:          │
│  ┌─────────────────────────────────┐   │
│  │  Plan (DAG of actions)          │   │  ← Plans sequence work
│  │  A ──→ B ──→ C                  │   │
│  │    ↘       ↗                    │   │
│  │      D ────                     │   │
│  └─────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

---

## Two Paradigms for Using Actions

### Paradigm 1: Signal-Driven (via Skills)

Skills define **reactive** behavior — "when this signal arrives, run this action":

```elixir
# Skill defines routing
def router(_config) do
  [{"payment.process", ProcessPayment}]
end

# Usage: Send signal, skill routes automatically
signal = Signal.new!("payment.process", %{amount: 99.99})
Jido.AgentServer.apply(agent_pid, signal)
```

### Paradigm 2: Plan-Driven (Direct Actions)

Plans define **explicit orchestration** — "run these actions in this order":

```elixir
plan = Plan.new()
  |> Plan.add(:pay, ProcessPayment)
  |> Plan.add(:ship, CreateShipment, depends_on: :pay)

Jido.Exec.run(Jido.Tools.ActionPlan, %{plan: plan})
```

### When to Use Each

| Approach | Use When |
|----------|----------|
| **Signal-driven (Skills)** | Agent responding to external events, reactive behavior |
| **Plan-driven (Direct)** | Explicit workflows, cross-agent orchestration, ad-hoc composition |

---

## Three-Layer Architecture

```
┌─────────────────────────────────────────────┐
│ Agent (Pure)                                │
│ - cmd/2: pure decision logic                │
│ - Returns {updated_agent, directives}       │
│ - Immutable                                 │
│ - No side effects                           │
└────────────────┬────────────────────────────┘
                 │ directives
                 │
┌────────────────▼────────────────────────────┐
│ AgentServer (Runtime)                       │
│ - GenServer wrapper per agent               │
│ - Routes signals to actions                 │
│ - Drains directive queue                    │
│ - Tracks children + parent hierarchy        │
│ - Manages process lifecycle                 │
└─────────────────────────────────────────────┘
```

**Signal Flow:**
```
Signal → AgentServer.call/cast
      → route signal to action
      → Agent.cmd/2 (with strategy)
      → {updated_agent, directives}
      → Queue directives
      → Drain loop: execute each directive
```

---

## Real-World Example: Order Processing System

This example demonstrates all concepts working together in an e-commerce order processing workflow.

### The Scenario

When a customer places an order:
1. Validate the order
2. Process payment
3. Update inventory (parallel)
4. Arrange shipping (parallel)
5. Send confirmation email

### Step 1: Define the Actions

```elixir
defmodule MyShop.Actions.ValidateOrder do
  use Jido.Action,
    name: "validate_order",
    description: "Validates order details and stock availability",
    schema: [
      order_id: [type: :string, required: true],
      items: [type: {:list, :map}, required: true],
      customer_id: [type: :string, required: true]
    ]

  def run(%{order_id: order_id, items: items}, _ctx) do
    case check_stock_availability(items) do
      :ok -> {:ok, %{order_id: order_id, validated: true, items: items}}
      {:error, out_of_stock} -> {:error, "Items out of stock: #{inspect(out_of_stock)}"}
    end
  end

  defp check_stock_availability(_items), do: :ok
end

defmodule MyShop.Actions.ProcessPayment do
  use Jido.Action,
    name: "process_payment",
    description: "Charges customer payment method",
    schema: [
      order_id: [type: :string, required: true],
      amount: [type: :float, required: true],
      payment_method: [type: :string, required: true]
    ]

  def run(%{order_id: order_id, amount: amount, payment_method: method}, _ctx) do
    case charge_payment(method, amount) do
      {:ok, transaction_id} ->
        {:ok, %{order_id: order_id, transaction_id: transaction_id, charged: amount}}
      {:error, reason} ->
        {:error, "Payment failed: #{reason}"}
    end
  end

  defp charge_payment(_method, _amount), do: {:ok, "txn_" <> Ecto.UUID.generate()}
end

defmodule MyShop.Actions.UpdateInventory do
  use Jido.Action,
    name: "update_inventory",
    description: "Decrements stock for ordered items",
    schema: [items: [type: {:list, :map}, required: true]]

  def run(%{items: items}, _ctx) do
    Enum.each(items, &decrement_stock/1)
    {:ok, %{inventory_updated: true, items_count: length(items)}}
  end

  defp decrement_stock(_item), do: :ok
end

defmodule MyShop.Actions.CreateShipment do
  use Jido.Action,
    name: "create_shipment",
    description: "Creates shipping label and schedules pickup",
    schema: [
      order_id: [type: :string, required: true],
      address: [type: :map, required: true],
      items: [type: {:list, :map}, required: true]
    ]

  def run(%{order_id: order_id}, _ctx) do
    tracking_number = "TRK" <> String.slice(Ecto.UUID.generate(), 0..7)
    {:ok, %{
      order_id: order_id,
      tracking_number: tracking_number,
      estimated_delivery: Date.add(Date.utc_today(), 3)
    }}
  end
end

defmodule MyShop.Actions.SendConfirmation do
  use Jido.Action,
    name: "send_confirmation",
    description: "Sends order confirmation email to customer",
    schema: [
      order_id: [type: :string, required: true],
      email: [type: :string, required: true],
      tracking_number: [type: :string, required: false]
    ]

  def run(%{order_id: _order_id, email: email}, _ctx) do
    {:ok, %{email_sent: true, recipient: email}}
  end
end
```

### Step 2: Define Skills

```elixir
defmodule MyShop.Skills.PaymentSkill do
  use Jido.Skill,
    name: "payment",
    state_key: :payment,
    actions: [MyShop.Actions.ProcessPayment],
    schema: [
      transactions: [type: {:list, :map}, default: []],
      total_processed: [type: :float, default: 0.0]
    ]

  def router(_config) do
    [{"payment.process", MyShop.Actions.ProcessPayment}]
  end
end

defmodule MyShop.Skills.InventorySkill do
  use Jido.Skill,
    name: "inventory",
    state_key: :inventory,
    actions: [MyShop.Actions.UpdateInventory],
    schema: [updates_count: [type: :integer, default: 0]]

  def router(_config) do
    [{"inventory.update", MyShop.Actions.UpdateInventory}]
  end
end

defmodule MyShop.Skills.ShippingSkill do
  use Jido.Skill,
    name: "shipping",
    state_key: :shipping,
    actions: [MyShop.Actions.CreateShipment],
    schema: [shipments: [type: {:list, :map}, default: []]]

  def router(_config) do
    [{"shipping.create", MyShop.Actions.CreateShipment}]
  end
end

defmodule MyShop.Skills.NotificationSkill do
  use Jido.Skill,
    name: "notification",
    state_key: :notifications,
    actions: [MyShop.Actions.SendConfirmation],
    schema: [sent_count: [type: :integer, default: 0]]

  def router(_config) do
    [{"notification.send", MyShop.Actions.SendConfirmation}]
  end
end
```

### Step 3: Define the Order Agent (Signal-Driven Approach)

This approach respects the Skill abstraction by using signals to trigger actions:

```elixir
defmodule MyShop.OrderAgent do
  use Jido.Agent,
    name: "order_agent",
    description: "Processes customer orders end-to-end",
    schema: [
      orders: [type: {:list, :map}, default: []],
      current_order: [type: :map, default: nil],
      status: [type: :atom, default: :idle]
    ],
    skills: [
      MyShop.Skills.PaymentSkill,
      MyShop.Skills.InventorySkill,
      MyShop.Skills.ShippingSkill,
      MyShop.Skills.NotificationSkill
    ]

  alias Jido.Agent.Directive
  alias Jido.Signal

  # Step 1: Order submitted - start the workflow
  def cmd(agent, {:submit_order, order_data}) do
    updated_agent = agent
      |> put_in([Access.key(:state), :current_order], order_data)
      |> put_in([Access.key(:state), :status], :validating)

    # Emit signal to trigger validation
    signal = Signal.new!("order.validate", order_data, source: "/orders")

    {updated_agent, [Directive.emit(signal, dispatch: {:pid, target: self()})]}
  end

  # Step 2: Validation complete - trigger payment
  def cmd(agent, {:validation_complete, _result}) do
    order = agent.state.current_order

    signal = Signal.new!("payment.process", %{
      order_id: order.order_id,
      amount: order.total,
      payment_method: order.payment_method
    }, source: "/orders")

    updated_agent = put_in(agent.state.status, :processing_payment)

    {updated_agent, [Directive.emit(signal, dispatch: {:pid, target: self()})]}
  end

  # Step 3: Payment complete - trigger inventory & shipping (parallel)
  def cmd(agent, {:payment_complete, %{transaction_id: txn_id}}) do
    order = agent.state.current_order

    inventory_signal = Signal.new!("inventory.update", %{
      items: order.items
    }, source: "/orders")

    shipping_signal = Signal.new!("shipping.create", %{
      order_id: order.order_id,
      address: order.shipping_address,
      items: order.items
    }, source: "/orders")

    updated_agent = agent
      |> put_in([Access.key(:state), :current_order, :transaction_id], txn_id)
      |> put_in([Access.key(:state), :status], :fulfilling)

    {updated_agent, [
      Directive.emit(inventory_signal, dispatch: {:pid, target: self()}),
      Directive.emit(shipping_signal, dispatch: {:pid, target: self()})
    ]}
  end

  # Step 4: All fulfillment complete - send notification
  def cmd(agent, {:fulfillment_complete, %{tracking_number: tracking}}) do
    order = agent.state.current_order

    signal = Signal.new!("notification.send", %{
      order_id: order.order_id,
      email: order.customer_email,
      tracking_number: tracking
    }, source: "/orders")

    updated_agent = put_in(agent.state.status, :notifying)

    {updated_agent, [Directive.emit(signal, dispatch: {:pid, target: self()})]}
  end

  # Step 5: Order complete
  def cmd(agent, {:notification_sent, _result}) do
    order = agent.state.current_order
    completed_order = Map.put(order, :status, :completed)

    updated_agent = agent
      |> put_in([Access.key(:state), :orders], [completed_order | agent.state.orders])
      |> put_in([Access.key(:state), :current_order], nil)
      |> put_in([Access.key(:state), :status], :idle)

    # Emit completion signal for other systems
    signal = Signal.new!("order.completed", completed_order, source: "/orders")

    {updated_agent, [Directive.emit(signal)]}
  end
end
```

### Step 4: Set Up Signal Bus and Routing

```elixir
defmodule MyShop.Application do
  use Application

  def start(_type, _args) do
    children = [
      {Jido.Signal.Bus, name: :shop_bus},
      {Jido.AgentServer,
        agent: MyShop.OrderAgent,
        id: "order_processor_1",
        name: {:global, :order_processor}
      }
    ]

    Supervisor.start_link(children, [strategy: :one_for_one])
  end
end

defmodule MyShop.EventHandlers do
  def setup_subscriptions do
    # Analytics listens to all order events
    Jido.Signal.Bus.subscribe(:shop_bus, "order.**",
      dispatch: {:pid, target: AnalyticsWorker}
    )

    # Warehouse listens for completed orders
    Jido.Signal.Bus.subscribe(:shop_bus, "order.completed",
      dispatch: {:pid, target: WarehouseWorker}
    )

    # Customer service listens for failures
    Jido.Signal.Bus.subscribe(:shop_bus, "order.failed",
      dispatch: [
        {:pid, target: CustomerServiceAgent},
        {:logger, level: :error}
      ]
    )

    # Audit log everything
    Jido.Signal.Bus.subscribe(:shop_bus, "**",
      dispatch: {:logger, level: :info, structured: true}
    )
  end
end
```

### Step 5: Using the System

```elixir
# Customer submits an order via API
order_data = %{
  order_id: "ORD-2024-001",
  customer_id: "CUST-123",
  customer_email: "alice@example.com",
  items: [
    %{sku: "WIDGET-01", quantity: 2, price: 29.99},
    %{sku: "GADGET-05", quantity: 1, price: 49.99}
  ],
  total: 109.97,
  payment_method: "card_visa_4242",
  shipping_address: %{
    street: "123 Main St",
    city: "Portland",
    state: "OR",
    zip: "97201"
  }
}

# Create and send signal
signal = Jido.Signal.new!("order.submit", order_data, source: "/api/checkout")
Jido.AgentServer.apply({:global, :order_processor}, signal)
```

### The Complete Flow Visualized

```
Customer                    OrderAgent                    Other Systems
   │                            │                              │
   │ POST /checkout             │                              │
   ├──────────────────────────►│                              │
   │                            │                              │
   │    Signal: order.submit    │                              │
   │ ──────────────────────────►│                              │
   │                            │                              │
   │                     cmd/2 executes                        │
   │                     ┌─────────────┐                       │
   │                     │  Validate   │ ← Signal: order.validate
   │                     └──────┬──────┘   routes via skill
   │                            │                              │
   │                     ┌──────▼──────┐                       │
   │                     │   Payment   │ ← Signal: payment.process
   │                     └──────┬──────┘   routes via PaymentSkill
   │                            │                              │
   │                     ┌──────┴──────┐   Parallel signals    │
   │               ┌─────▼─────┐ ┌─────▼─────┐                │
   │               │ Inventory │ │ Shipping  │                │
   │               │   Skill   │ │   Skill   │                │
   │               └─────┬─────┘ └─────┬─────┘                │
   │                     └──────┬──────┘                       │
   │                     ┌──────▼──────┐                       │
   │                     │ Notification│ ← Signal: notification.send
   │                     │    Skill    │                       │
   │                     └──────┬──────┘                       │
   │                            │                              │
   │                     Signal: order.completed               │
   │                            ├─────────────────────────────►│ Analytics
   │                            ├─────────────────────────────►│ Warehouse
   │                            ├─────────────────────────────►│ Logger
   │                            │                              │
   │◄───────────────────────────┤                              │
   │   Response: Order confirmed│                              │
```

### Plan Execution Visualization

```
    ┌──────────┐
    │ validate │
    └────┬─────┘
         │
    ┌────▼─────┐
    │   pay    │
    └────┬─────┘
         │
    ┌────┴────┐
    │         │
 ┌──▼──┐  ┌──▼──┐      ← Parallel execution!
 │ inv │  │ship │
 └──┬──┘  └──┬──┘
    │        │
    └───┬────┘
        │
   ┌────▼────┐
   │ notify  │
   └─────────┘
```

---

## Key Design Principles

1. **Pure Functional Core** — Agents are immutable data; decisions are pure functions
2. **Effect Separation** — Directives describe effects; agents never execute them
3. **Pluggable Strategies** — Different execution models via strategy modules
4. **Composable Capabilities** — Skills attach reusable behaviors
5. **Declarative Hierarchy** — Parent-child relationships via directives
6. **Protocol-Based Extension** — Custom directives via `DirectiveExec` protocol

---

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

---

## Quick Reference

| Concept | What It Is | Key File |
|---------|------------|----------|
| Action | Validated unit of work | `jido_action/lib/jido_action.ex` |
| Signal | CloudEvents message | `jido_signal/lib/jido_signal.ex` |
| Bus | Pub/sub message hub | `jido_signal/lib/jido_signal/bus.ex` |
| Router | Pattern-based routing | `jido_signal/lib/jido_signal/router.ex` |
| Agent | Immutable decision-maker | `jido/lib/jido/agent.ex` |
| AgentServer | Runtime wrapper | `jido/lib/jido/agent_server.ex` |
| Directive | Effect descriptor | `jido/lib/jido/agent/directive.ex` |
| Skill | Reusable capability | `jido/lib/jido/skill.ex` |
| Plan | DAG workflow | `jido_action/lib/jido_plan.ex` |
| Instruction | Action + params wrapper | `jido_action/lib/jido_instruction.ex` |

---

## Next Topics to Explore

- [ ] jido_ai: AI/LLM integration for agents
- [ ] req_llm: HTTP client for 57+ LLM providers
- [ ] Strategies: Different execution models (Direct, FSM)
- [ ] jido_workbench: Phoenix demo application
