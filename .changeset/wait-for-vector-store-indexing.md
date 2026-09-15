---
'@openai/guardrails': patch
---

Wait for attached files to finish indexing before `createOpenAIVectorStoreFromPath` returns a vector store ID. Failed or cancelled indexing now rejects with the affected file and indexing status instead of returning a store that is not ready.
