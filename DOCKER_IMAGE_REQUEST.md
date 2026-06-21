# Docker image request — Postgres + pgvector

For a restricted environment where you can request a specific image rather
than run arbitrary `docker pull`s yourself.

## What to request

| | |
|---|---|
| **Image** | `pgvector/pgvector` |
| **Registry** | Docker Hub (`docker.io`) |
| **Recommended tag** | `pgvector/pgvector:pg16` -- but see "Which Postgres version" below |
| **What it is** | The official Postgres Docker image, with the open-source `pgvector` extension (vector similarity search) compiled in. Maintained by the pgvector project itself, not a third party. |
| **Source / Dockerfile** | https://github.com/pgvector/pgvector/blob/master/Dockerfile -- fully auditable, nothing hidden |
| **Approx size** | ~150-170 MB compressed |
| **Port** | 5432/tcp (standard Postgres) |
| **Outbound network needed at runtime** | None. The container doesn't call home or fetch anything after it starts. |
| **Volumes** | One: a data directory for persistence (`/var/lib/postgresql/data`) |

This is the same image referenced in `docker-compose.yml` in your project
folder -- that file IS the precise, machine-readable spec of what gets
pulled, what ports open, and what gets mounted. If your request process
accepts an attachment, **that file is the cleanest thing to hand over** --
it's more convincing to a reviewer than a prose description, because they
can read exactly what it does instead of trusting a summary.

## Which Postgres version to actually request

Don't default to `pg16` just because that's what's in the example
`docker-compose.yml` -- ask whoever owns your company's database
standards (DBA / platform team) which Postgres major version they support,
and request that one instead. The `pgvector/pgvector` image publishes a
build for every supported Postgres version (14 through 18 as of mid-2026),
tagged like `pgvector/pgvector:pg17`, `pgvector/pgvector:pg18`, etc. --
swap the tag in `docker-compose.yml`'s `image:` line to match. For a
banking application specifically, matching your org's existing Postgres
standard matters more than picking the newest version.

## For a formal security/image-whitelisting request

If your process requires a pinned, immutable reference rather than a
floating tag (`pg16` tracks the latest pgvector release for that Postgres
version, so it changes over time -- a specific tag like `0.8.0-pg16`
doesn't), do this once you (or whoever has Docker Hub access) can run:

```bash
docker pull pgvector/pgvector:pg16
docker inspect --format='{{index .RepoDigests 0}}' pgvector/pgvector:pg16
```

That gives you a value like `pgvector/pgvector@sha256:<digest>` -- the
fully immutable reference, suitable for a security review that wants to
pin exactly what gets deployed rather than "whatever pg16 currently
points to." Put that digest in the request, not a value I generate here --
digests change every time the image is rebuilt, so one I hand you now
could already be stale.

## If `pgvector/pgvector` itself can't be approved

Two fallbacks, in order of preference:

1. **Ask your platform team to mirror it** into whatever internal registry
   you're actually allowed to pull from (Artifactory / Nexus / Harbor /
   similar). This is extremely common in locked-down corporate
   environments -- the image doesn't change, only where you pull it from.
   Update `docker-compose.yml`'s `image:` line to the internal path once
   they confirm it (e.g. `your-registry.company.com/pgvector/pgvector:pg16`).

2. **Request the official `postgres` image plus a build step**, if your
   org already trusts the base `postgres` image and the friction is
   specifically about third-party images. `pgvector`'s Dockerfile shows
   exactly what it adds on top of `postgres:16` -- it's a small,
   auditable diff (install build deps, `make && make install` the
   extension, remove build deps). This is more work to get approved
   (now it's "build a custom image" rather than "pull an existing one")
   but may be necessary if your security policy specifically disallows
   pulling pre-built images from individual maintainers' Docker Hub
   namespaces.

## What this is NOT requesting

Only the database is containerized. The Python ingestion/query scripts
(`ingest_intents.py`, `query_intent.py`, etc.) still run directly on
whatever machine actually executes them -- there's no request needed for
those beyond normal Python package installation. If you later need those
containerized too (e.g. deploying `query_intent.py` as a service other
systems call), that's a separate image/request, built from a different
Dockerfile -- not a change to this one.
