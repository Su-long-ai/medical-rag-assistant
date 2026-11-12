# Scoped RAG reliability collaboration — inquiry guide

This is a public **portfolio project**, not a clinical product or a record of an enterprise deployment. The purpose of this page is to make it easier for another engineer or a prospective collaborator to evaluate the project's relevant capabilities and discuss a narrowly scoped, testable problem.

## Problems this demo can help discuss

- A question that should retrieve a passage, but returns irrelevant or missing evidence.
- An answer that sounds plausible but cannot be traced to the uploaded source.
- Document upload and routing mistakes, such as ignoring indexed documents.
- Reproducible Python regression tests for retrieval routes, response metadata, and error handling.

The existing repository demonstrates FastAPI, bounded document ingestion, Chroma retrieval, deterministic routing, optional read-only graph lookup, SSE events, and offline tests. It does **not** prove real customer reliability, comparative benchmark superiority, clinical correctness, or production security.

## Example limited engagement (subject to mutual agreement)

1. **Inputs:** A small, non-sensitive sample dataset (or synthetic test documents), 5–15 representative questions, and expected source passages where possible.
2. **Baseline:** Reproduce current retrieval and citation behavior, separating missing evidence, wrong passage selection, unsupported answer, and application integration issues.
3. **Alternatives:** Check current library/platform configuration and simple fixes before recommending new infrastructure.
4. **Output:** A short evidence-based findings report, reproducible failing tests or minimal change suggestions, and clear limits of confidence.
5. **Next decision:** Discuss whether implementation or continued monitoring makes sense based on measured incremental value.

The exact scope, schedule, price, access controls, ownership, and support terms would need agreement **before** work begins. There is no claim that this repository includes a ready-made paid service or production SLA.

## How to start a conversation

If you have an **open-source, non-sensitive** question, [open a GitHub issue](https://github.com/Su-long-ai/medical-rag-assistant/issues/new) with:

- the behavior you observe and the result you expected;
- minimal reproduction steps using public or synthetic data;
- the affected component (ingestion, routing, retrieval, citations, API, etc.);
- whether you want an engineering discussion, a bounded review, or a contribution.

For non-public business inquiries, first use a contact method explicitly made available by the repository owner's GitHub profile; **do not place proprietary documents, personal information, API keys, medical records, or confidential business data in public GitHub issues**.

This page is an invitation to discuss a narrowly defined engineering problem, **not** a guarantee of availability, price, delivery, or a request to upload confidential material.
