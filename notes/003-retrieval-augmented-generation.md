# 003 - Retrieval Augmented Generation (RAG)

> Enhancing LLM responses with external knowledge through embeddings and document retrieval

Retrieval Augmented Generation (RAG) is a technique that improves LLM outputs by grounding them in external documents. Instead of relying solely on the model's training data, RAG retrieves relevant information at query time and injects it into the prompt context.

---

## Why RAG?

| Problem | How RAG Solves It |
|---------|-------------------|
| Knowledge cutoff | Retrieve up-to-date documents |
| Hallucinations | Ground responses in real sources |
| Domain-specific knowledge | Inject company/project docs |
| Source attribution | Track which documents informed the response |

---

## RAG Architecture in Jido

The Jido ecosystem provides RAG capabilities across multiple layers:

```
┌─────────────────────────────────────────────────────────────────┐
│                      RAG-Enabled Agent                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Jido.AI.Features.RAG                         │   │
│  │   Document formatting, citation extraction, provider-     │   │
│  │   specific RAG options (Cohere, Google, Anthropic)        │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│  ┌──────────────────────────▼───────────────────────────────┐   │
│  │              ReqLLM.Embedding                             │   │
│  │   Generate embeddings for semantic search                 │   │
│  │   (OpenAI, Google, Azure embedding models)                │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│  ┌──────────────────────────▼───────────────────────────────┐   │
│  │         Vector Store (external)                           │   │
│  │   Store and query embeddings (Pinecone, Qdrant, etc.)    │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## ReqLLM.Embedding - Generating Embeddings

ReqLLM provides embedding generation for semantic search:

```elixir
# Single text embedding
{:ok, embedding} = ReqLLM.embed("openai:text-embedding-3-small", "Hello world")
#=> {:ok, [0.1, -0.2, 0.3, ...]}  # 1536-dimensional vector

# Batch embedding (more efficient)
{:ok, embeddings} = ReqLLM.embed(
  "openai:text-embedding-3-small",
  ["Document one", "Document two", "Document three"]
)
#=> {:ok, [[0.1, -0.2, ...], [0.3, 0.4, ...], [0.5, 0.6, ...]]}
```

### Supported Embedding Models

| Provider | Models | Dimensions |
|----------|--------|------------|
| OpenAI | `text-embedding-3-small` | 1536 |
| OpenAI | `text-embedding-3-large` | 3072 |
| OpenAI | `text-embedding-ada-002` | 1536 |
| Google | `gemini-embedding-001` | varies |
| Azure | OpenAI embedding models | varies |

### Embedding Options

```elixir
ReqLLM.Embedding.embed(
  "openai:text-embedding-3-small",
  "Query text",
  dimensions: 512,           # Reduce dimensions (for supported models)
  encoding_format: "float",  # "float" or "base64"
  user: "user-123"           # For tracking
)
```

---

## Cosine Similarity - Finding Relevant Documents

After embedding documents and queries, use cosine similarity to find matches:

```elixir
defmodule KnowledgeBase do
  @doc """
  Compute cosine similarity between two embedding vectors.
  Returns a value between -1 and 1, where 1 is most similar.
  """
  def cosine_similarity(v1, v2) do
    dot_product =
      Enum.zip(v1, v2)
      |> Enum.map(fn {a, b} -> a * b end)
      |> Enum.sum()

    mag1 = Enum.map(v1, &(&1 * &1)) |> Enum.sum() |> :math.sqrt()
    mag2 = Enum.map(v2, &(&1 * &1)) |> Enum.sum() |> :math.sqrt()

    dot_product / (mag1 * mag2)
  end

  @doc """
  Find the most relevant document for a query.
  """
  def find_relevant(query_embedding, documents_with_embeddings, top_k \\ 3) do
    documents_with_embeddings
    |> Enum.map(fn doc ->
      similarity = cosine_similarity(query_embedding, doc.embedding)
      {doc, similarity}
    end)
    |> Enum.sort_by(fn {_doc, sim} -> sim end, :desc)
    |> Enum.take(top_k)
  end
end
```

---

## Complete RAG Pipeline

Here's a full RAG implementation using ReqLLM tools:

```elixir
defmodule SimpleAgent.WithRAG do
  use GenServer
  import ReqLLM.Context

  alias ReqLLM.{Context, Tool, ToolCall}

  defstruct [:model, :context, :tools, :knowledge_docs]

  def start_link(opts), do: GenServer.start_link(__MODULE__, opts)
  def ask(pid, text), do: GenServer.call(pid, {:ask, text}, 60_000)

  @impl true
  def init(opts) do
    embedding_model = Keyword.get(opts, :embedding_model, "openai:text-embedding-3-small")
    model = Keyword.fetch!(opts, :model)

    # Step 1: Precompute embeddings for knowledge base
    knowledge_docs = precompute_embeddings(embedding_model)

    # Step 2: Create knowledge retrieval tool
    knowledge_tool = create_knowledge_tool(embedding_model, knowledge_docs)

    system_prompt = """
    You are a helpful assistant with access to an internal knowledge base.
    Use the 'knowledge' tool when questions relate to the knowledge base topics.
    """

    context = Context.new([system(system_prompt)])

    {:ok, %__MODULE__{
      model: model,
      context: context,
      tools: [knowledge_tool],
      knowledge_docs: knowledge_docs
    }}
  end

  defp precompute_embeddings(model) do
    docs = [
      %{id: 1, text: "Jido is an Elixir framework for autonomous agents."},
      %{id: 2, text: "Actions in Jido are composable, validated operations."},
      %{id: 3, text: "Signals provide event-driven communication between agents."}
    ]

    Enum.map(docs, fn doc ->
      {:ok, embedding} = ReqLLM.embed(model, doc.text)
      Map.put(doc, :embedding, embedding)
    end)
  end

  defp create_knowledge_tool(model, docs) do
    ReqLLM.Tool.new!(
      name: "knowledge",
      description: "Search the internal knowledge base for relevant information",
      parameter_schema: [
        query: [type: :string, required: true, doc: "Search query"]
      ],
      callback: fn args ->
        query = args[:query] || args["query"]

        # Embed the query
        {:ok, query_embedding} = ReqLLM.embed(model, query)

        # Find most similar document
        {best_doc, similarity} =
          docs
          |> Enum.map(fn doc ->
            sim = KnowledgeBase.cosine_similarity(query_embedding, doc.embedding)
            {doc, sim}
          end)
          |> Enum.max_by(fn {_doc, sim} -> sim end)

        {:ok, best_doc.text}
      end
    )
  end

  @impl true
  def handle_call({:ask, user_text}, _from, state) do
    ctx = Context.append(state.context, user(user_text))

    # First LLM call - may decide to use knowledge tool
    {:ok, response} = ReqLLM.generate_text(state.model, ctx.messages, tools: state.tools)
    tool_calls = ReqLLM.Response.tool_calls(response)

    ctx2 = Context.append(ctx, response.message)

    # Execute any tool calls
    ctx3 =
      Enum.reduce(tool_calls, ctx2, fn call, acc_ctx ->
        tool = List.first(state.tools)
        args = ToolCall.args_map(call)

        case ReqLLM.Tool.execute(tool, args) do
          {:ok, result} ->
            Context.append(acc_ctx, Context.tool_result(call.id, result))
          {:error, reason} ->
            Context.append(acc_ctx, Context.tool_result(call.id, "Error: #{inspect(reason)}"))
        end
      end)

    # Final LLM call with retrieved context
    {:ok, final_response} = ReqLLM.generate_text(state.model, ctx3.messages, tools: [])
    text = ReqLLM.Response.text(final_response)

    {:reply, {:ok, text}, %{state | context: Context.append(ctx3, final_response.message)}}
  end
end
```

---

## Jido.AI.Features.RAG - Provider-Native RAG

For providers with native RAG support (Cohere, Google, Anthropic), Jido provides specialized formatting:

```elixir
alias Jido.AI.Features.RAG

# Check if model supports RAG
RAG.supports?(model)  #=> true for Cohere, Google, Anthropic

# Prepare documents for provider-specific format
documents = [
  %{content: "Elixir is a functional language...", title: "Elixir Intro", url: "https://..."},
  %{content: "GenServers manage state...", title: "OTP Basics"}
]

{:ok, formatted} = RAG.prepare_documents(documents, :cohere)
#=> {:ok, [%{"text" => "...", "title" => "...", "id" => "doc_1"}]}

{:ok, formatted} = RAG.prepare_documents(documents, :google)
#=> {:ok, [%{"inline_data" => %{"content" => "...", "mime_type" => "text/plain"}}]}

{:ok, formatted} = RAG.prepare_documents(documents, :anthropic)
#=> {:ok, "\n[1] Elixir Intro\n...\n\n[2] OTP Basics\n..."}
```

### Building RAG-Enhanced Requests

```elixir
# Build options with RAG parameters
base_opts = %{temperature: 0.7}

{:ok, enhanced_opts} = RAG.build_rag_options(documents, base_opts, :cohere)
#=> {:ok, %{temperature: 0.7, documents: [...]}}

# For Anthropic, documents are injected into system prompt
{:ok, enhanced_opts} = RAG.build_rag_options(documents, base_opts, :anthropic)
#=> {:ok, %{temperature: 0.7, system: "...\n\nReference Documents:..."}}
```

### Extracting Citations

```elixir
# After getting a response with RAG
{:ok, citations} = RAG.extract_citations(response, :cohere)
#=> {:ok, [%{text: "...", document_index: 0, start: 10, end: 50}]}
```

### Document Validation

The RAG module enforces limits:

| Limit | Value |
|-------|-------|
| Max documents | 100 |
| Max document size | 500,000 characters |
| Min content length | 1 character |

---

## Chunking Large Documents

For documents that exceed context windows, use `Jido.AI.Prompt.Splitter`:

```elixir
alias Jido.AI.Prompt.Splitter

# Create a splitter for a large document
splitter = Splitter.new(large_document_text, model)

# Get chunks that fit within context, accounting for other data
{chunk1, splitter} = Splitter.next_chunk(splitter, "System prompt and other context")
{chunk2, splitter} = Splitter.next_chunk(splitter, "Accumulated summary so far")
# ... continue until {:done, splitter}
```

The splitter:
1. Tokenizes the input using the model's tokenizer
2. Accounts for "bespoke" data (system prompts, accumulated context)
3. Returns chunks that fit within remaining token budget

---

## Context Window Management

Use `Jido.AI.ContextWindow` to manage context limits:

```elixir
alias Jido.AI.ContextWindow

# Check if content fits
{:ok, info} = ContextWindow.check_fit(prompt, model)
#=> %{tokens: 245, limit: 128000, fits: true, available: 127755}

# Truncate with strategy if needed
{:ok, truncated} = ContextWindow.ensure_fit(prompt, model,
  strategy: :keep_recent,    # Keep N most recent messages
  count: 20,                 # Number to keep
  reserve_completion: 2000   # Reserve tokens for response
)
```

### Truncation Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `:keep_recent` | Keep last N messages | Chat history |
| `:keep_bookends` | Keep system + last N | Preserve instructions |
| `:sliding_window` | Sliding window with overlap | Long documents |
| `:smart_truncate` | Intelligent context preservation | Complex conversations |

---

## RAG Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│  1. INDEXING PHASE (Offline)                                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Documents          Chunking           Embedding         Vector Store│
│  ┌─────────┐       ┌─────────┐        ┌─────────┐       ┌─────────┐ │
│  │ Doc 1   │──────▶│ Chunk 1 │───────▶│ [0.1,   │──────▶│ Store   │ │
│  │ Doc 2   │       │ Chunk 2 │        │  0.3,   │       │ vectors │ │
│  │ Doc 3   │       │ Chunk 3 │        │  ...]   │       │ + text  │ │
│  └─────────┘       └─────────┘        └─────────┘       └─────────┘ │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  2. QUERY PHASE (Online)                                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  User Query         Embed Query        Search            Retrieve    │
│  ┌─────────┐       ┌─────────┐        ┌─────────┐       ┌─────────┐ │
│  │"What is │──────▶│ [0.2,   │───────▶│ Cosine  │──────▶│ Top-k   │ │
│  │ Jido?"  │       │  0.4,   │        │ Search  │       │ chunks  │ │
│  └─────────┘       │  ...]   │        └─────────┘       └─────────┘ │
│                    └─────────┘                                │      │
│                                                               │      │
│                                                               ▼      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  3. GENERATION                                               │    │
│  │                                                              │    │
│  │  Augmented Prompt:                                          │    │
│  │  "Given these documents: [retrieved chunks]                 │    │
│  │   Answer: What is Jido?"                                    │    │
│  │                          │                                   │    │
│  │                          ▼                                   │    │
│  │                    ┌─────────┐                               │    │
│  │                    │   LLM   │                               │    │
│  │                    └────┬────┘                               │    │
│  │                         │                                    │    │
│  │                         ▼                                    │    │
│  │              Grounded Response                               │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Best Practices

### 1. Chunk Size Selection

```elixir
# Smaller chunks = more precise retrieval, less context
# Larger chunks = more context, may include irrelevant info

# Recommended: 200-500 tokens per chunk with 10-20% overlap
```

### 2. Embedding Model Selection

| Use Case | Recommended Model |
|----------|-------------------|
| General purpose | `text-embedding-3-small` |
| High accuracy | `text-embedding-3-large` |
| Cost-sensitive | `text-embedding-ada-002` |

### 3. Top-K Retrieval

```elixir
# Start with k=3-5, adjust based on:
# - Context window size
# - Document relevance distribution
# - Response quality requirements
```

### 4. Hybrid Search

Combine semantic search with keyword matching:

```elixir
def hybrid_search(query, docs, embedding_model) do
  # Semantic search
  {:ok, query_embedding} = ReqLLM.embed(embedding_model, query)
  semantic_results = find_relevant(query_embedding, docs, 5)

  # Keyword search (simple example)
  keywords = String.split(query, " ")
  keyword_results = Enum.filter(docs, fn doc ->
    Enum.any?(keywords, &String.contains?(doc.text, &1))
  end)

  # Combine and deduplicate
  merge_results(semantic_results, keyword_results)
end
```

---

## Integration with Jido Agents

RAG can be integrated as a Jido Action or Tool:

```elixir
defmodule MyApp.Actions.RAGSearch do
  use Jido.Action,
    name: "rag_search",
    description: "Search knowledge base and return relevant documents",
    schema: [
      query: [type: :string, required: true],
      top_k: [type: :integer, default: 3]
    ]

  def run(%{query: query, top_k: k}, ctx) do
    # Get pre-computed document embeddings from context
    docs = ctx[:knowledge_base]
    embedding_model = ctx[:embedding_model]

    {:ok, query_embedding} = ReqLLM.embed(embedding_model, query)

    results =
      docs
      |> find_relevant(query_embedding, k)
      |> Enum.map(fn {doc, score} ->
        %{text: doc.text, score: score, id: doc.id}
      end)

    {:ok, %{results: results}}
  end
end
```

---

## Key Takeaways

- **Embeddings** convert text to vectors for semantic similarity search
- **ReqLLM.Embedding** supports multiple providers (OpenAI, Google, Azure)
- **Cosine similarity** finds relevant documents by comparing vectors
- **Jido.AI.Features.RAG** provides provider-native RAG formatting
- **Chunking** splits large documents to fit context windows
- **Top-K retrieval** balances relevance with context budget
- **Tool-based RAG** integrates naturally with ReqLLM's tool calling

---

## Next Topics to Explore

- [ ] Vector database integration (Pinecone, Qdrant, Milvus)
- [ ] Advanced chunking strategies (semantic, hierarchical)
- [ ] Re-ranking retrieved documents
- [ ] Multi-modal RAG (images, code)
- [ ] RAG evaluation metrics (precision, recall, RAGAS)
