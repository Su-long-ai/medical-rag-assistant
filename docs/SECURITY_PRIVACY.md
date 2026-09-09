# Security and privacy notes

This repository is a local-development portfolio project, not a hardened healthcare deployment.

The refactor applies several safer defaults:

- credentials are read from environment variables only;
- original upload names are not used as filesystem paths;
- stored upload filenames are opaque document IDs;
- public responses do not expose internal file paths or extracted document text;
- server errors are logged internally but returned to clients as generic error messages;
- destructive/debug endpoints are disabled unless `ENABLE_ADMIN_ENDPOINTS=true`;
- graph and web tools are opt-in;
- Neo4j access uses a parameterized read-only lookup rather than generated Cypher;
- upload size and PDF page limits bound common resource-exhaustion cases;
- logs do not print document previews.

Before any internet-facing deployment, add authentication/authorization, per-user data isolation, rate limiting, encrypted storage, audit logging, retention/deletion controls and a deployment-specific threat model. Do not use this demo to store real patient records.
