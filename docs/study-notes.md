# Study Notes — nostreambot

Running log while reading the reference implementation. Raw and unpolished by design — this is the input to `architecture-notes.md`, not a substitute for it.

**Belongs in Bottega** (the study repo), not Sibyl. Move it there on Aug 15 when the split happens.

**Format:** number each entry so it can be referenced later. Tag with `#cost` `#agents` `#eval` `#architecture` `#research` `#port` `#skip` `#question`. Lines starting with `→` are Claude's annotations, not mine.

---

## Log

### 1. Cost rules in `AGENTS.md` depend on compliance, not structure
`AGENTS.md` · `#cost` `#agents`

The rules around cost are hard to guarantee are always followed. Better: set up **fixtures for test purposes that make no real LLM calls**, or wire up **free models for testing** (Groq-type) so the cheap path is the default rather than the disciplined choice.

→ This is the same shape as the Lumi lesson: *don't rely on remembering, make the system enforce it.* Lumi solved exactly this with F7 fixture mode (fully runnable against contract examples, no API keys) plus Groq free tier and local Ollama as the $0 lane. Prior art you already built and validated — worth porting rather than reinventing. `#port`

→ Relevant to the $50 cap: an accidental live run during development is a real budget risk, and the current design has no structural guard against one.

---

### 2. `AGENTS.md` is too large to load every time
`AGENTS.md` · `#agents`

Burns too many tokens when it has to be read on every agent invocation.

→ Standard fix is progressive disclosure: a thin root file carrying only the always-true rules and hard prohibitions, with pointers to detail docs the agent loads on demand. Your `rialto/CLAUDE.md` is a good reference for the terse-and-directive register — it fits on one screen and is all constraints, no explanation.

---

<!-- Next entry: copy the block below

### N. Title
`path` · `#tag`

Note.

-->

---

## Open questions

*(Things not yet understood — resolve while reading or bring back from the trip.)*

- 

---

## Decisions for Sibyl

*(Port / skip / change. Feeds the Aug 15 restructure.)*

| # | Thing | Decision | Why |
|---|---|---|---|
| | | | |

---

## Quick reference

- Reading order and the 8 comprehension questions: `bottega:docs/planning/nostreambot-reading-guide.md`
- What's already built vs. mine: same file, "What's already built vs. what's actually yours"
- Why any of this exists: [`vision.md`](./vision.md)
