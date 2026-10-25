# AWS Deployment (Design)

> **Status: design only. Nothing in this document has been deployed.** It exists so the resources, the costs and the decisions can be reviewed before any billable resource is created. The step-22 containers ([19-containers-and-ci.md](19-containers-and-ci.md)) are the starting point.

## Goals and non-goals

**Goals:** an enterprise trial for a small group of users, with the same three services as the Compose setup; no long-lived keys anywhere; alerts on the numbers the service already measures; a written runbook; rollback in minutes; a monthly cost that is known in advance and capped by a budget alarm.

**Non-goals for the trial:** multi-region, high availability across zones for the database, autoscaling beyond 1–2 tasks, and the enterprise features (identity, document permissions, audit), which are step 24 ([21-enterprise-features.md](21-enterprise-features.md)).

## Architecture

```
Users ──HTTPS──► CloudFront ──► S3 (React build)                    static files, cached
                     │ /api/*
                     ▼
                Application Load Balancer (HTTPS, ACM certificate)   only public entry for the API path
                     │
                     ▼  private to the VPC (security groups)
               ECS Fargate: bff (Node.js) ──► ECS Fargate: api (Python)
                                                  │        │        │
                                                  │        │        └─► OpenAI / Bedrock (answers, agent)
                                                  │        └─► RDS PostgreSQL + pgvector (chunks, vectors,
                                                  │             feedback, agent runs, reports)
                                                  └─► S3 (source documents) ◄── ingestion task (ECS, on demand)

Secrets Manager (LLM key, BFF keys) · ECR (images) · CloudWatch (logs, metrics, alarms) · AWS Budgets
GitHub Actions ──OIDC role──► ECR push, ECS deploy (no stored AWS keys)
```

The browser sees one origin (CloudFront), as with nginx in Compose: `/api/*` goes to the load balancer, everything else to S3.

## Resources

| Resource | Purpose | Trial size | Why this choice | Main alternative |
|---|---|---|---|---|
| **VPC** with 2 public and 2 private subnets | network isolation | 2 Availability Zones (required by the load balancer and RDS) | standard layout | default VPC (simpler, less isolated) |
| **Application Load Balancer** | HTTPS entry for `/api/*`, health checks, rolling deploys | 1 | ECS integration, deployment circuit breaker, long timeouts for streaming | API Gateway (30 s integration limit breaks streams); App Runner |
| **ECS on Fargate**: `bff`, `api` | run the containers without servers | 0.25 vCPU / 0.5 GB and 0.5 vCPU / 1 GB, 1 task each, ARM | no servers to patch; per-second billing; the same images as Compose | App Runner (simpler, fewer controls), EKS (heavy for two services) |
| **ECS task (on demand)**: ingestion | parse S3 documents, embed, write to the vector store | runs minutes per indexing job | separates indexing from serving; failures do not take the API down | Lambda (15-minute limit, large dependencies) |
| **RDS PostgreSQL** with **pgvector** | vectors and chunks; feedback, agent runs, NCRs, audit log (step 24) | `db.t4g.micro`, 20 GB, single AZ, automated backups | one database for everything; the code has a `pgvector_store.py` placeholder; cheapest managed vector option at this size | OpenSearch Serverless (see costs), Aurora Serverless v2 |
| **S3** | source documents (versioned); the React build | < 1 GB | durable, cheap; versioning gives document history | — |
| **CloudFront** | serves the React app, routes `/api/*` to the load balancer, TLS | 1 distribution | one origin, caching, no nginx container to run | an nginx task (as in Compose) |
| **ECR** | container images, tagged with the git commit | 2 repositories | close to ECS, image scanning on push | GitHub Container Registry |
| **Secrets Manager** | LLM API key, BFF user and admin keys | 3 secrets | injected into tasks at start, rotatable, never in images or env files | SSM Parameter Store (SecureString, cheaper, no rotation) |
| **CloudWatch** | logs (JSON lines already), metrics, alarms, dashboard | 30-day log retention | native; the API's JSON log lines become queryable | Managed Prometheus + Grafana |
| **IAM** | task roles (least privilege), a GitHub OIDC deploy role | — | no long-lived access keys | IAM users with keys (avoid) |
| **ACM** + optional **Route 53** | TLS certificates; a domain name | 1 certificate, 1 hosted zone | free certificates | — |
| **AWS Budgets** | alarm at 50%, 80%, 100% of the monthly cap | 1 budget | cost is a requirement, not an afterthought | — |
| *Optional:* **Bedrock** | embeddings (Titan V2) and reranking | pay per use | already implemented, tested with fakes only | local models in the container |
| *Optional:* **WAF** | rate limiting and common attacks at the edge | 1 web ACL | defence in depth in front of the BFF limits | none for a closed trial |

**No NAT gateway.** A NAT gateway costs about as much as the rest of the compute. For the trial the tasks run in public subnets with public IP addresses, and security groups allow inbound traffic only from the load balancer, so nothing is reachable directly. The alternative for production is private subnets with a NAT gateway or VPC endpoints.

## Cost estimate

List prices for **us-east-1**, a 730-hour month, and the trial sizes above. The first four rows are from AWS pricing pages ([Fargate](https://aws.amazon.com/fargate/pricing/), [Elastic Load Balancing](https://aws.amazon.com/elasticloadbalancing/pricing/), [VPC](https://aws.amazon.com/vpc/pricing/), [RDS for PostgreSQL](https://aws.amazon.com/rds/postgresql/pricing)). The rows marked *estimate* come from general knowledge and were not verified in this review. **Check everything in the [AWS Pricing Calculator](https://calculator.aws/) for the chosen region before deploying.** Other regions cost more (RDS in Tokyo is about 50% more than in Ireland).

| Item | Basis | Per month |
|---|---|---|
| Fargate, 2 tasks (0.75 vCPU, 1.5 GB in total, ARM) | ≈ $0.032 per vCPU-hour (ARM), $0.0044 per GB-hour (x86 rate; ARM is lower) | ≈ $23 |
| Application Load Balancer | $0.0225 per hour + LCUs ($0.008 per LCU-hour; a trial uses well under 1) | ≈ $17–20 |
| Public IPv4 addresses (2 tasks, load balancer) | about $0.005 per address-hour (*estimate*) | ≈ $11 |
| RDS `db.t4g.micro` PostgreSQL | ≈ $0.016 per hour; storage and backups extra | ≈ $12 + storage (*estimate* ≈ $2–3 for 20 GB) |
| Secrets Manager, 3 secrets | *estimate* $0.40 per secret | ≈ $1–2 |
| CloudWatch logs, metrics, ~8 alarms | *estimate* | ≈ $3–8 |
| S3, CloudFront, ECR, Route 53 | *estimate*, small volumes | ≈ $1–3 |
| **Running 24/7** | | **≈ $70–85** |
| Office hours only (ECS tasks scaled to 0 at night and weekends; RDS stopped) | load balancer and IPs still billed | ≈ $45–55 |
| LLM tokens (OpenAI) | step-21 benchmark: ≈ 2,000 tokens per agent task; ≈ 500 per streamed answer | usage-based, external |

**Cost traps avoided by this design:**

| Trap | Cost | Instead |
|---|---|---|
| NAT gateway | $0.045 per hour (≈ $33/month) **plus $0.045 per GB** processed, per Availability Zone | public subnets plus security groups |
| OpenSearch Serverless (classic collections) | billed per OCU-hour at $0.24; a 2-OCU minimum (1 OCU in half-OCU mode) is ≈ $175–350/month even when idle. Newer collection types report no minimum and scale to zero; confirm before choosing | pgvector in the database the system needs anyway |
| A second database for app state | another instance-month | one PostgreSQL for vectors and state |
| Logs kept forever | grows every month | 30-day retention, then S3 if needed |

## What has to change in the code

The containers run unchanged, but several stores are per container today. They must become shared before running in the cloud:

| Today (one machine) | In AWS | Work |
|---|---|---|
| index built into the API image from `data/raw` | ingestion task reads S3, writes chunks and vectors to pgvector | implement `vectorstore/pgvector_store.py`; it must pass the vector-store contract tests (`tests/test_vectorstore.py`) |
| feedback in `logs/feedback.jsonl` | a PostgreSQL table | a store interface with JSONL (local) and PostgreSQL backends |
| agent runs in SQLite (`RunStore`) | PostgreSQL; the compare-and-set becomes `UPDATE … WHERE status = …` (the same SQL) | a PostgreSQL `RunStore` |
| NCRs in `logs/ncr.jsonl` | a PostgreSQL table with a unique idempotency key | as above |
| in-process metrics (`/metrics`), counters per container | CloudWatch metrics (embedded metric format from the JSON logs) | emit EMF fields in the existing log lines |
| `.env` for keys | Secrets Manager, injected as environment variables by ECS | none in the code |
| hashing embedder (offline) | Bedrock Titan V2 or a local model; **the benchmark must be re-run** with the chosen embedder | configuration and a measured run |

## Deployment pipeline

1. **Infrastructure as code: Terraform**, in `infra/`. Plans are reviewed before `apply`, so nothing is created by surprise, and `terraform destroy` removes everything at the end of the trial. (AWS CDK in TypeScript would fit the BFF's language; Terraform is chosen for its plan/review workflow and its wider use outside AWS.)
2. **GitHub Actions with OIDC:** on a merge to `main`, after CI passes, a deploy job assumes an IAM role (no stored keys), builds the images, tags them with the git commit, pushes them to ECR and updates the ECS services.
3. **Rolling deployment with a circuit breaker:** ECS starts the new task, waits for the load balancer health check (`/ready` for the API, `/api/health` for the BFF), and **rolls back automatically** if the new tasks don't become healthy.
4. **Smoke test after deploy:** `scripts/smoke.py --base-url https://<domain>` from the pipeline; a failure triggers a rollback to the previous image tag.

## Alerts

Built on numbers the service already produces ([14-production-rag.md](14-production-rag.md)):

| Alarm | Condition | Why |
|---|---|---|
| API not ready | load balancer healthy hosts < 1 for 2 minutes | nothing can answer |
| Server errors | 5xx rate > 2% over 5 minutes | breakage after a deploy or an outage |
| Latency | `/ask` p95 > 8 s over 10 minutes | LLM slowness or overload (step 16: generation is ~99% of the time) |
| LLM fallbacks | > 10% of answers use the extractive fallback | the LLM provider is failing or the key is wrong |
| Approval queue | an action waiting > 24 hours | a person's decision is stuck |
| Token spend | input + output tokens per hour above an agreed ceiling | runaway cost |
| Budget | 50 / 80 / 100% of the monthly cap | the account bill |
| Database | free storage < 20%, CPU credits low | the micro instance's limits |

## Operations runbook (outline)

| Situation | First checks | Action |
|---|---|---|
| Alarm: API not ready | ECS task events; `/ready` reason (index missing? database unreachable?) | roll back to the previous image tag; re-run the ingestion task if the index is missing |
| Alarm: LLM fallbacks | provider status page; the secret's value and age | rotate the key; answers keep working (extractive) in the meantime |
| Alarm: latency | token use per request; the provider's latency; task CPU | scale the API to 2 tasks; lower `top_k` |
| A user reports a wrong answer | the request ID in the UI → CloudWatch Logs Insights: model, prompt version, sources | add the case to the benchmark (frozen before any fix) |
| A document must be removed | delete from S3 (versioned), re-run ingestion (`sync` deletes missing sources) | verify with a search |
| Ingestion failures | the ingestion report (failed files with reasons) | fix the file; failed files keep their previous version |
| Suspected key leak | CloudTrail; unusual token spend | rotate in Secrets Manager; tasks restart with the new value |

## Rollback

- **Application:** every image is tagged with its git commit; rolling back means pointing the ECS service at the previous tag (automatic on failed health checks, one command otherwise).
- **Database:** schema changes are additive (new columns or tables, never dropping in the same release), so the previous application version keeps working; automated daily snapshots with 7-day retention.
- **Infrastructure:** Terraform state is versioned; a change is reverted by applying the previous commit's configuration.

## Failure drills (to run once deployed)

| Drill | Expected result |
|---|---|
| Stop the API task | the load balancer marks it unhealthy; ECS starts a new one; the alarm fires and clears |
| Deploy an image that fails `/ready` | the circuit breaker rolls back; no user-visible downtime |
| Revoke the LLM key | answers fall back to extractive quotes; the fallback alarm fires |
| Stop the database | `/ready` fails; the runbook entry is followed |
| Restore a snapshot into a new instance | data is back within the documented recovery time |

## Decisions needed before deployment

| Decision | Options | Recommendation |
|---|---|---|
| AWS account | an existing one, or a new one for the trial | a new account, so the bill and permissions are isolated and `terraform destroy` cleans up completely |
| Region | us-east-1 (cheapest, widest Bedrock model choice), ap-northeast-1 (Tokyo), ap-southeast-1 (Singapore), eu-west-1 | the region closest to the trial users that offers the Bedrock models needed. AWS China regions are a separate partition (separate account, ICP licence, different service availability) and are out of scope |
| Monthly budget cap | e.g. $100 | a budget alarm at 50/80/100% |
| LLM provider | OpenAI (measured), Bedrock (Claude or other models, data stays in AWS) | Bedrock if documents must not leave AWS; it needs a measured benchmark run first |
| Embedder | hashing (offline), MiniLM in the container, Bedrock Titan V2 | MiniLM or Titan, chosen by a benchmark run |
| Domain name | a custom domain (Route 53) or the CloudFront default | the default for a closed trial |
| Who can access | API keys (today) or real sign-in (step 24) | step 24 before real users |

## Phased plan

| Phase | Creates | Verified by |
|---|---|---|
| 0. Account setup | budget alarm, Terraform state bucket, GitHub OIDC role | the budget e-mail arrives; `terraform plan` works |
| 1. Network and data | VPC, RDS with pgvector, S3 buckets, Secrets Manager entries | connection from a one-off task; the vector-store contract tests against RDS |
| 2. Services | ECR, ECS cluster and services, load balancer, CloudFront | `smoke.py` against the CloudFront URL |
| 3. Pipeline | the deploy job in GitHub Actions | a commit to `main` deploys; a broken image rolls back |
| 4. Observability | log retention, metrics, alarms, dashboard | each alarm triggered once on purpose |
| 5. Drills | — | the failure drills above |
| End of trial | — | `terraform destroy`; the bill drops to zero |
