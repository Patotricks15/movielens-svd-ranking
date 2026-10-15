# Infrastructure

Terraform for the serving path: an S3 bucket for model artifacts, a DynamoDB
table of precomputed top-N lists, and a Lambda behind IAM that reads them.

The default profile targets **Floci/LocalStack**, so `apply` costs nothing and
needs no AWS account.

## Local (default)

```bash
docker compose up -d                       # start LocalStack on :4566
terraform init
terraform validate
terraform apply -auto-approve
terraform output
terraform destroy -auto-approve
docker compose down
```

## Real AWS

The same configuration deploys to a real account by flipping one variable:

```bash
terraform apply -var use_localstack=false -var aws_region=sa-east-1
```

With `use_localstack = false` the provider stops injecting dummy credentials
and custom endpoints, and the two features LocalStack does not emulate
(DynamoDB point-in-time recovery and server-side encryption) switch on.

## Variables

| Variable | Default | Purpose |
|---|---|---|
| `project_name` | `movielens-svd` | Prefix for every resource name |
| `environment` | `dev` | Environment tag and name suffix |
| `aws_region` | `us-east-1` | Target region |
| `use_localstack` | `true` | Target LocalStack instead of real AWS |
| `localstack_endpoint` | `http://localhost:4566` | LocalStack address |
| `artifact_retention_days` | `90` | Expiry for non-current model versions in S3 |
| `training_schedule` | `cron(0 3 * * ? *)` | Nightly retraining trigger |

## Design notes

**Why precompute top-N instead of scoring in Lambda?** SVD scoring is a dot
product over 32 dimensions and would run comfortably in Lambda. The ranking,
however, must exclude each user's watch history, which would mean loading the
full rating history on every request. Precomputing moves that cost offline
and bounds p99 latency by DynamoDB rather than by model size.

**Why `PAY_PER_REQUEST`?** Portfolio traffic is bursty and near zero on
average. On-demand billing avoids paying for provisioned capacity that sits
idle, and removes capacity planning from the deployment.

**IAM scope.** The inference role is read-only: `GetItem`/`BatchGetItem` on
the top-N table and `GetObject` restricted to the `models/` prefix. The
request path never writes, so it is not granted write permissions.

## Cost

Everything above is free on LocalStack. On real AWS at portfolio traffic:
S3 under $0.05/month for less than 1 GB, DynamoDB on-demand under $1/month,
and Lambda within the free tier. `terraform destroy` plus `docker compose
down` removes all of it.
