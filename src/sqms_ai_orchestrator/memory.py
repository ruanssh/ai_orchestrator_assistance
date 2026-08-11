import sqlite3
import threading
from pathlib import Path

from .models import ChatMessage


class ConversationMemory:
    def __init__(self, database: Path, max_messages: int):
        database.parent.mkdir(parents=True, exist_ok=True)
        self.database = database
        self.max_messages = max_messages
        self.lock = threading.Lock()
        with self._connect() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS messages ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, "
                "role TEXT NOT NULL, content TEXT NOT NULL, created_at TEXT DEFAULT CURRENT_TIMESTAMP)"
            )
            connection.execute("CREATE INDEX IF NOT EXISTS messages_session ON messages(session_id, id)")

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.database)

    def get(self, session_id: str) -> list[ChatMessage]:
        with self.lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                (session_id, self.max_messages),
            ).fetchall()
        return [ChatMessage(role=role, content=content) for role, content in reversed(rows)]

    def append(self, session_id: str, *messages: ChatMessage) -> None:
        with self.lock, self._connect() as connection:
            connection.executemany(
                "INSERT INTO messages(session_id, role, content) VALUES (?, ?, ?)",
                [(session_id, message.role, message.content) for message in messages],
            )
            connection.execute(
                "DELETE FROM messages WHERE session_id = ? AND id NOT IN ("
                "SELECT id FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?)",
                (session_id, session_id, self.max_messages),
            )

    def clear(self, session_id: str) -> None:
        with self.lock, self._connect() as connection:
            connection.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
