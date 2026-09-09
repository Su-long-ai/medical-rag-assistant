# Architecture

The application is deliberately split into deterministic policy, retrieval/tool adapters and LLM synthesis.

```text
upload -> validation -> opaque local file name -> extraction/OCR -> chunking -> Chroma

question
   |
   v
query_policy.py  -- deterministic route decision
   | document        | web              | graph             | direct
   v                 v                  v                   v
Chroma retrieval   Tavily adapter   read-only Neo4j     no external context
   \_____________________|_________________|_________________/
                         |
                         v
               SmartMedicalAgent
             evidence-aware synthesis
                         |
                         v
                  SSE response events
```

## Why deterministic routing?

The previous implementation asked an agent to infer whether document context existed before retrieval had occurred. As a result, ordinary questions could bypass an uploaded corpus unless the user explicitly wrote phrases such as “according to the document.” The new policy makes the routing decision inspectable and testable: when indexed documents exist, ordinary informational questions use document retrieval unless the query is explicitly time-sensitive.

## Why no generated Cypher?

The earlier graph integration enabled `allow_dangerous_requests=True` for an LLM-generated Cypher chain. The public version now exposes only a parameterized, read-only entity lookup. Deployments should still use a Neo4j account with read-only privileges.

## Streaming semantics

The API uses Server-Sent Events (SSE) for incremental delivery of status, metadata and answer chunks. It does **not** claim provider-level token streaming.
