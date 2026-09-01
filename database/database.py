import sqlite3
import os
from pathlib import Path
from typing import Optional
from src.utils.logger import setup_logger
from src.utils.config import get_project_root, load_config

logger = setup_logger("Database")

def get_db_path() -> Path:
    """Return database file path."""
    root = get_project_root()
    db_path = root / "database" / "emipredict.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return db_path

def get_db_connection() -> sqlite3.Connection:
    """Create a thread-safe connection to SQLite database."""
    path = get_db_path()
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db(schema_file: Optional[str] = None) -> None:
    """Initialize database tables using schema.sql."""
    root = get_project_root()
    if schema_file is None:
        schema_path = root / "database" / "schema.sql"
    else:
        schema_path = Path(schema_file)

    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found at {schema_path}")

    with open(schema_path, "r", encoding="utf-8") as f:
        schema_sql = f.read()

    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.executescript(schema_sql)
        conn.commit()
        logger.info(f"Database initialized successfully at {get_db_path()}")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise
    finally:
        conn.close()
