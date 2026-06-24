"""
app/database/connection.py

Thread-safe psycopg2 connection pool with pgvector type registration.

Four functions used everywhere else:
  init_pool(dsn, minconn, maxconn)  -- call once at startup
  get_conn()                         -- borrow a connection
  release_conn(conn)                 -- return it (always in a finally block)
  close_pool()                       -- call at shutdown

pgvector's register_vector() is called on every checkout so repository.py
never needs to call it directly.
"""

from typing import Optional

import psycopg2
from psycopg2.pool import ThreadedConnectionPool
from pgvector.psycopg2 import register_vector

_pool: Optional[ThreadedConnectionPool] = None


def init_pool(dsn: str, minconn: int = 1, maxconn: int = 10) -> None:
    global _pool
    _pool = ThreadedConnectionPool(minconn, maxconn, dsn)


def get_conn():
    if _pool is None:
        raise RuntimeError(
            "DB pool not initialised. Call database.connection.init_pool() "
            "at application startup before any DB operations."
        )
    conn = _pool.getconn()
    register_vector(conn)   # idempotent; safe on every checkout
    return conn


def release_conn(conn) -> None:
    if _pool is not None:
        _pool.putconn(conn)


def close_pool() -> None:
    global _pool
    if _pool is not None:
        _pool.closeall()
        _pool = None
