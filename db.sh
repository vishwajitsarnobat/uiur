#!/usr/bin/env bash
# db.sh - convenience wrapper around docker compose for the pgvector container
#
# usage: ./db.sh {up|down|reset|shell|logs|apply-schema}

set -euo pipefail

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

cmd="${1:-}"

case "$cmd" in
  up)
    docker compose up -d --wait
    echo "pgvector is up and healthy on port ${DB_PORT:-5432}"
    ;;
  down)
    docker compose down
    ;;
  reset)
    echo "This permanently deletes all data in the pgvector volume. Continue? (y/N)"
    read -r confirm
    if [ "$confirm" = "y" ]; then
      docker compose down -v
      docker compose up -d --wait
      echo "Volume reset, schema re-applied from db_schema.sql"
    else
      echo "aborted"
    fi
    ;;
  shell)
    docker compose exec pgvector psql -U "${DB_USER:-postgres}" -d "${DB_NAME:-ragdb}"
    ;;
  logs)
    docker compose logs -f pgvector
    ;;
  apply-schema)
    # Use this after editing db_schema.sql on an EXISTING container --
    # the docker-entrypoint-initdb.d mount only runs on first init.
    docker compose exec -T pgvector psql -U "${DB_USER:-postgres}" -d "${DB_NAME:-ragdb}" < db_schema.sql
    echo "schema re-applied"
    ;;
  *)
    echo "usage: ./db.sh {up|down|reset|shell|logs|apply-schema}"
    exit 1
    ;;
esac
