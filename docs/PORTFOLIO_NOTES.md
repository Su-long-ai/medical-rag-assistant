# Portfolio notes

Evidence-grounded resume bullets for this repository:

- Built a FastAPI document-RAG backend supporting PDF/DOCX/TXT ingestion, bounded OCR fallback, Chroma retrieval and SSE response delivery.
- Refactored query orchestration into a deterministic, testable routing layer that prioritizes uploaded-document retrieval, explicit time-sensitive web context and read-only graph lookup without relying on a free-form agent loop.
- Hardened the public API by replacing path-bearing upload metadata with opaque document IDs, removing hard-coded confidence fields, bounding upload/PDF processing, and disabling destructive/debug endpoints by default.
- Added offline unit tests for routing, upload policy, bounded session memory, tool isolation and public response-schema privacy boundaries.

Do not claim clinical accuracy, diagnostic performance, retrieval benchmark superiority or human-subject validation: this repository does not contain evidence for those claims.
