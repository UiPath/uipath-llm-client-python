"""Custom VCR.py persister that stores all cassettes in a single SQLite database file.

Instead of creating one YAML file per test cassette (which can produce 1000+ files),
this persister stores all cassettes as rows in a single SQLite database.
"""

import sqlite3
from pathlib import Path

from vcr.cassette import CassetteNotFoundError
from vcr.serialize import deserialize, serialize


class SQLitePersister:
    """Stores VCR cassettes in a single SQLite database file."""

    db_path: str = "tests/cassettes.db"

    @classmethod
    def _get_connection(cls) -> sqlite3.Connection:
        conn = sqlite3.connect(cls.db_path)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cassettes (
                cassette_path TEXT PRIMARY KEY,
                data TEXT NOT NULL
            )
            """
        )
        return conn

    @classmethod
    def load_cassette(cls, cassette_path: str, serializer: str) -> tuple[list, list]:
        # Normalize the path to use as a key
        key = str(Path(cassette_path))
        conn = cls._get_connection()
        try:
            row = conn.execute(
                "SELECT data FROM cassettes WHERE cassette_path = ?", (key,)
            ).fetchone()
            if row is None:
                raise CassetteNotFoundError(f"Cassette not found: {key}")
            cassette_content = row[0]
            return deserialize(cassette_content, serializer)
        finally:
            conn.close()

    @classmethod
    def save_cassette(cls, cassette_path: str, cassette_dict: dict, serializer: str) -> None:
        key = str(Path(cassette_path))
        data = serialize(cassette_dict, serializer)
        conn = cls._get_connection()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO cassettes (cassette_path, data) VALUES (?, ?)",
                (key, data),
            )
            conn.commit()
        finally:
            conn.close()
