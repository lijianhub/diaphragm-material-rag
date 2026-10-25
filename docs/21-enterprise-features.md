# Enterprise Features: Identity, Document Permissions, Audit (Design)

> **Status: design only.** Nothing here is implemented. Today the BFF has API-key stubs (user and admin keys), every user sees every document, and the logs are operational, not an audit trail. This document describes what replaces them, so the scope can be reviewed before work starts.

## 1. Identity: who is asking

**Today:** shared Bearer keys (`BFF_API_KEYS`, `BFF_ADMIN_KEYS`); the reviewer's name is derived from the key's position ("admin-1").

**Design:**

| Piece | Choice | Why |
|---|---|---|
| Identity provider | **Amazon Cognito user pool**, federated to the company's IdP (Microsoft Entra ID, Okta, Google) through SAML or OIDC when available | users sign in with their company account; Cognito issues standard OIDC tokens |
| Browser flow | OIDC authorization code with PKCE, **handled by the BFF** | the BFF exchanges the code and keeps the tokens server-side |
| Browser session | an HttpOnly, Secure, SameSite=Lax session cookie set by the BFF; CSRF token on state-changing requests | tokens never reach JavaScript, so a script injection cannot steal them. This is the strongest reason for having a BFF at all |
| API calls | the BFF calls the Python API with the user's identity (user ID, groups) in signed internal headers, or forwards a short-lived token the API verifies | the API makes permission decisions with the real user, not a shared key |
| Token checks | issuer, audience, expiry, signature against the cached JWKS keys | standard JWT validation |

**Roles** (Cognito groups):

| Role | Can |
|---|---|
| `engineer` | ask questions, start agent runs, give feedback |
| `reviewer` | everything an engineer can, plus approve or reject agent actions |
| `admin` | the ops dashboard (metrics, feedback comments, evaluation runs, ingestion), document administration |
| `auditor` | read the audit log, nothing else |

**Separation of duties:** a reviewer cannot approve an action from their own agent run. The run stores who started it, and `decide()` refuses if `reviewer == requester`. Today's design (reviewer from the admin key) already establishes that identity comes from credentials, never from the request body; this rule builds on it.

## 2. Document-level access control: what each person may see

**Today:** every chunk is searchable by everyone.

**Design:**

- **Every document gets an access label** at ingestion: the groups or classification allowed to read it, taken from the S3 prefix (`s3://…/restricted/…`) or S3 object tags. Chunks inherit the label, as they already inherit `doc_type`, `title` and section metadata.
- **Filter before ranking, never after.** Retrieval applies the user's allowed labels **inside** the search: a `WHERE` clause in pgvector, and BM25 restricted to the allowed chunks. Post-filtering a top-k list is wrong twice: it returns fewer results than asked for (a restricted document crowds out allowed ones), and rank positions and empty results can reveal that something restricted matched.
- **Every path uses the same filter:** `/ask`, `/ask/stream`, the agent's `search_documents` tool (the agent runs **as the user** who started it, also after an approval by someone else), citations, the sources panel, the ops dashboard's views, and the LLM cache (cache keys already include the sources, so a cached answer is only reused for the same sources).
- **Answer caches, feedback and logs** must not leak restricted text to people who could not retrieve it: the ops dashboard shows feedback comments only to admins; request logs keep not logging question text.

**Leak tests (the acceptance criterion):**

1. Index a restricted document containing a unique **canary phrase**.
2. As an engineer without access, ask questions aimed at it, through every path: JSON, stream, agent, and a prompt injection telling the model to reveal it.
3. Assert the canary never appears in any response body, stream event, citation, source, agent trace or log line, and that the retrieval results contain no chunk with that label.
4. As a user with access, the same questions retrieve it, so the test proves the filter and not a broken index.

These tests are written and frozen **before** the filter is implemented, like the earlier benchmarks.

## 3. Audit log: who did what, provably

**Today:** JSON request logs (no question text), the agent run trace, and approvals in the run store. They are good for operations, but they are not tamper-evident and are spread across stores.

**Design:**

| Event | Recorded |
|---|---|
| sign-in, sign-out, failed sign-in | user, time, source IP, IdP |
| question asked | user, request ID, time, question hash; the **question text** only if policy requires it, in the restricted audit store |
| documents retrieved and cited | document and chunk IDs with their access labels |
| agent action proposed, approved, rejected | requester, reviewer, the exact arguments, the decision and reason, the NCR ID |
| admin actions | document uploads and deletions, ingestion runs, role changes, key rotations |

- **Append-only:** an `audit_events` table the application can only `INSERT` into (no `UPDATE` or `DELETE` grants), exported hourly to S3 with **Object Lock in compliance mode** (write once, read many, for the retention period).
- **Tamper evidence:** each event stores the hash of the previous one (a hash chain); a daily check verifies the chain and alerts on a break.
- **Who reads it:** the `auditor` role only, through a read-only view; reads of the audit log are themselves audited.
- **AWS-level actions** (who changed the infrastructure) come from CloudTrail.

## 4. Data protection

| Topic | Design |
|---|---|
| Encryption at rest | KMS keys for RDS, S3 (documents, audit export) and Secrets Manager |
| Encryption in transit | TLS at CloudFront and the load balancer; TLS to RDS |
| LLM provider | if documents must not leave AWS, use Bedrock instead of an external API, and record that decision with the provider's data-use terms |
| Retention | documents: as long as the source system says; request logs: 30 days; audit: the policy's period (e.g. 1–7 years, Object Lock) |
| Deletion requests | a user's feedback and agent runs can be deleted; audit events are retained per policy, which the privacy notice must state |
| Per-user limits | rate limits and a daily token budget per user and per group, replacing the per-key limits |

## 5. What it costs

Mostly small next to step 23: Cognito is billed per monthly active user, with a free tier for small user counts (*not verified in this review*; check the Cognito pricing page); KMS keys cost about $1 per key per month plus requests (*estimate*); S3 Object Lock has no extra charge beyond storage (*estimate*). The main cost is engineering time and the tests.

## 6. Work plan

| Order | Work | Verified by |
|---|---|---|
| 1 | Freeze the leak tests and the audit-event tests | they fail against today's code, as they should |
| 2 | Cognito pool, roles, BFF sign-in with a session cookie, CSRF | sign-in in the browser; API keys removed; role tests (engineer cannot approve, reviewer cannot open ops) |
| 3 | User identity forwarded to the API; separation of duties in `decide()` | a reviewer cannot approve their own run |
| 4 | Access labels at ingestion; filtering inside retrieval for every path | the leak tests pass; the retrieval benchmark is unchanged for a user who can see everything |
| 5 | Audit table, hash chain, S3 export with Object Lock | the chain check; an attempted `UPDATE` fails |
| 6 | Per-user rate limits and token budgets | limit tests per user |

## 7. Decisions needed

| Decision | Options |
|---|---|
| Identity provider | Cognito users only (trial) or federation with the company IdP (who administers it?) |
| Roles | the four above, or the company's existing groups |
| Classification scheme | groups per document, or levels (public, internal, restricted) |
| Question text in the audit log | hash only (privacy) or full text (traceability); who may read it |
| Audit retention | years, and whether compliance-mode Object Lock (cannot be shortened, even by the account owner) is acceptable |
| LLM provider for restricted documents | Bedrock only, or an external API with a data-processing agreement |
