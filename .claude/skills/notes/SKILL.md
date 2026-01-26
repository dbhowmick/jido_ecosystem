---
name: notes
description: Generate structured learning notes about Jido ecosystem concepts. Can summarize current discussion or research the codebase to create comprehensive notes.
user-invocable: true
allowed-tools: Read, Write, Bash, Glob, Grep
argument-hint: [topic-name]
---

# Notes Skill

Generate structured learning notes about the Jido ecosystem.

## Instructions

When the user invokes `/notes [topic]`, follow these steps:

### Step 1: Determine Next Note Number

1. List all files in the `notes/` directory
2. Find the highest existing number (e.g., `002-jido-ai-and-llm-integration.md` has number 002)
3. Increment by 1 for the new note (e.g., next would be 003)
4. Pad to 3 digits with leading zeros

### Step 2: Generate Filename

1. Convert the topic to kebab-case (lowercase, spaces to hyphens)
2. Remove special characters
3. Format: `XXX-topic-name.md` where XXX is the 3-digit number

### Step 3: Gather Content

Depending on the topic, you may need to:

- **Summarize current discussion**: If the topic was discussed earlier in the conversation, use that context
- **Research codebase**: Use Glob and Grep to find relevant code in `packages/` directory
- **Combine both**: Use both conversation context and codebase research

For codebase research, look in:
- `packages/jido/` - Core agent framework
- `packages/jido_action/` - Actions and workflows
- `packages/jido_signal/` - Signal/event system
- `packages/jido_ai/` - AI/LLM integration
- `packages/req_llm/` - LLM client library
- `packages/jido_workbench/` - Phoenix demo app

### Step 4: Create Note Content

Follow this structure (based on existing notes):

```markdown
# XXX - Topic Name

> One-line summary of the topic

Brief introduction paragraph explaining what this topic is about and why it matters in the Jido ecosystem.

---

## Section 1

Explanation of the first major concept...

### Subsection (if needed)

More detailed explanation with code examples:

```elixir
# Code example from codebase or discussion
defmodule Example do
  # ...
end
```

---

## Section 2

Use ASCII diagrams where helpful:

```
┌─────────────────┐
│   Component A   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Component B   │
└─────────────────┘
```

Use tables for comparisons:

| Feature | Description |
|---------|-------------|
| Item 1  | Details     |
| Item 2  | Details     |

---

## Key Takeaways

- Bullet point summary of important concepts
- Another key insight
- etc.

---

## Next Topics to Explore

- [ ] Related topic 1
- [ ] Related topic 2
- [ ] Related topic 3
```

### Step 5: Write the Note

1. Use the Write tool to save the note to `notes/XXX-topic-name.md`
2. Confirm to the user with the filename and a brief summary of what was covered

## Content Guidelines

- **Code examples**: Include practical, working code from the codebase or discussion
- **ASCII diagrams**: Use box-drawing characters for architecture diagrams
- **Tables**: Use for comparisons, quick references, API summaries
- **Blockquote summary**: Always start with a `>` blockquote one-liner
- **Horizontal rules**: Use `---` between major sections
- **Checklists**: End with "Next Topics to Explore" using `- [ ]` format
- **Cross-references**: Reference other notes when relevant (e.g., "See 001-introduction-to-jido.md")

## Example Invocation

User: `/notes Strategies`

Result: Creates `notes/003-strategies.md` with comprehensive notes about Jido execution strategies (Direct, FSM), including code examples and diagrams.
