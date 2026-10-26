# Lumi Feature Wishlist

Last updated: 2026-10-22

Maintained by Breakpoint through the Lumi wishlist chat. Append new entries with the next sequential ID; keep existing IDs stable and update their status as decisions change. Wishlist entries record intent and do not authorize implementation.

## LUMI-001 — Easter Egg: Who Is WuGPT?

**Status:** Planned — no implementation authorized yet.

**Trigger:** “Who is WuGPT?”

**Planned response:**

> WuGPT is Qwen3.8:27B, Pow's local AI coding assistant.
>
> The very first prompt Pow gave WuGPT was: ‘Who was the main villain in the movie Spider-Man: Brand New Day?’
>
> Qwen confidently answered: ‘Dr. Wu.’
>
> And thus, WuGPT was born.
>
> Nobody knows where Dr. Wu came from. Not even WuGPT. 💀

**Implementation:** Include alongside Lumi's origin-story Easter egg in a future identity-training experiment, after Mission 008.

## LUMI-002 — Lumi and ChoaGPT Memory Architecture

**Status:** Planned — future ChoaGPT fork. No implementation authorized yet.

**Goal:** Give Lumi conventional LLM context handling while providing ChoaGPT with persistent, long-term memory.

### Lumi — Standard Context Memory

- Uses the model's native context window.
- Maintains recent conversation history within the available context.
- Older messages are truncated when the context limit is reached.
- No automatic persistent memory across sessions.
- Native context length may increase through future model improvements.

### ChoaGPT — Persistent Memory

- Forked from a conversationally capable Lumi.
- Retains the same native context mechanism as Lumi.
- Adds persistent memory stored locally on disk.
- Uses SQLite for structured memory storage and semantic retrieval for relevant memories.
- Automatically extracts and stores important facts, events, relationships, and preferences from conversations.
- Retrieves relevant memories and injects them into the active context at runtime.
- Supports memory correction, deletion, and source tracking.
- Preserves memories across application restarts and model upgrades.

### Architecture

```text
           Conversational Lumi
                   |
          +--------+--------+
          |                 |
        LUMI             CHOAGPT
          |                 |
    Native context     Native context
          |                 |
          |          Persistent memory
          |          SQLite + retrieval
          |                 |
          |          Choa personality
          |                 |
          +--------+--------+
                   |
             LM Studio
            MLX runtime
                   |
             GUI client
```

### Implementation Principles

- Persistent memory belongs to ChoaGPT's application layer, not the model weights.
- Reuse LM Studio as the inference runtime where compatible.
- Expose memory functionality through local API middleware for a compatible GUI client, such as Bionic.
- No Python command-line chat client required.
- Keep Lumi's training and development independent of ChoaGPT's accumulated memories.
- Both models should benefit from improvements to their shared model lineage.

### Implementation Timing

After Lumi achieves basic conversational and instruction-following capability. Build persistent storage and retrieval first, validate cross-session recall, then add automatic memory extraction and consolidation.

**Success criterion:** ChoaGPT accurately recalls relevant information from previous conversations after restarting the application, without requiring the user to repeat it.
