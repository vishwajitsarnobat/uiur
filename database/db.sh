#!/usr/bin/env bash
# db.sh -- Run from the database/ directory.
# Usage: ./db.sh {up|down|reset|shell|logs|apply-schema|status}

set -euo pipefail

if [ -f .env ]; then
  set -a; source .env; set +a
fi

cmd="${1:-}"
case "$cmd" in
  up)
    docker compose up -d --wait
    echo "pgvector is up on port ${DB_PORT:-5432}"
    ;;
  down)
    docker compose down
    ;;
  status)
    docker compose ps
    ;;
  reset)
    echo "Permanently wipes all data in the volume. Continue? (y/N)"
    read -r confirm
    [ "$confirm" = "y" ] || { echo "aborted"; exit 0; }
    docker compose down -v
    docker compose up -d --wait
    echo "Volume reset; schema re-applied from db_schema.sql"
    ;;
  shell)
    docker compose exec pgvector \
      psql -U "${DB_USER:-postgres}" -d "${DB_NAME:-ragdb}"
    ;;
  logs)
    docker compose logs -f pgvector
    ;;
  apply-schema)
    # Safe to re-run -- all CREATE statements use IF NOT EXISTS.
    docker compose exec -T pgvector \
      psql -U "${DB_USER:-postgres}" -d "${DB_NAME:-ragdb}" < db_schema.sql
    echo "schema applied"
    ;;
  *)
    echo "usage: ./db.sh {up|down|reset|shell|logs|apply-schema|status}"
    exit 1
    ;;
esac
