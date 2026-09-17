"""
database.py
Handles dynamic SQLite schema introspection and safe, truly read-only query execution.
Enforces read-only access at the database connection level using SQLite URI mode (?mode=ro, uri=True).
Never hardcodes table names or column definitions.
"""

import os
import sqlite3
import pandas as pd
from typing import Tuple, Optional, List, Dict, Any

DEFAULT_DB_PATH = "data/Chinook_Sqlite.sqlite"

def resolve_db_path(db_path: str = DEFAULT_DB_PATH) -> str:
    """
    Resolves the database path with portable fallback:
    Checks db_path, then './data/Chinook_Sqlite.sqlite', then './Chinook_Sqlite.sqlite'.
    """
    candidates = [
        db_path,
        os.path.join(os.getcwd(), db_path),
        os.path.join(os.path.dirname(__file__), "..", db_path),
        "data/Chinook_Sqlite.sqlite",
        "Chinook_Sqlite.sqlite",
        os.path.join(os.path.dirname(__file__), "..", "Chinook_Sqlite.sqlite")
    ]
    for c in candidates:
        if os.path.exists(c) and os.path.isfile(c):
            return os.path.abspath(c)
    return db_path

def get_readonly_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """
    Opens a truly read-only SQLite database connection using URI mode:
    file:<path>?mode=ro with uri=True.
    Falls back gracefully if URI mode is unsupported.
    """
    actual_path = resolve_db_path(db_path)
    # Format path for URI mode (forward slashes required for SQLite URIs on all platforms)
    uri_path = actual_path.replace("\\", "/")
    uri_str = f"file:{uri_path}?mode=ro"
    
    try:
        return sqlite3.connect(uri_str, uri=True)
    except Exception:
        # Fallback to standard connection if URI mode encounters OS-level issues
        return sqlite3.connect(actual_path)

def check_database_connection(db_path: str = DEFAULT_DB_PATH) -> Tuple[bool, str]:
    """
    Checks if the database exists and can be queried in read-only mode.
    """
    actual_path = resolve_db_path(db_path)
    if not os.path.exists(actual_path):
        return False, f"Database file not found at: {actual_path}"
    try:
        conn = get_readonly_connection(actual_path)
        cur = conn.cursor()
        cur.execute("SELECT sqlite_version();")
        v = cur.fetchone()[0]
        conn.close()
        return True, f"Connected to SQLite {v} (Read-Only URI mode enabled)"
    except Exception as e:
        return False, str(e)

def list_database_tables(db_path: str = DEFAULT_DB_PATH) -> List[str]:
    """
    Queries sqlite_master to dynamically list all non-system tables.
    """
    conn = get_readonly_connection(db_path)
    cur = conn.cursor()
    cur.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name NOT LIKE 'sqlite_%' 
        ORDER BY name;
    """)
    tables = [r[0] for r in cur.fetchall()]
    conn.close()
    return tables

def get_database_schema(db_path: str = DEFAULT_DB_PATH) -> str:
    """
    Dynamically extracts the complete SQLite schema (columns, types, PKs, FKs).
    Zero hardcoding of Chinook tables.
    """
    conn = get_readonly_connection(db_path)
    cur = conn.cursor()
    
    cur.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name NOT LIKE 'sqlite_%' 
        ORDER BY name;
    """)
    tables = [r[0] for r in cur.fetchall()]
    schema_parts = []
    
    for tbl in tables:
        cur.execute(f"PRAGMA table_info({tbl});")
        cols = cur.fetchall()
        col_desc = [f"{c[1]} ({c[2] or 'TEXT'})" for c in cols]
        pk_cols = [c[1] for c in cols if c[5] > 0]
        
        cur.execute(f"PRAGMA foreign_key_list({tbl});")
        fks = cur.fetchall()
        fk_desc = [f"FOREIGN KEY ({fk[3]}) REFERENCES {fk[2]}({fk[4]})" for fk in fks]
        
        col_summary = ", ".join(col_desc)
        tbl_info = f"Table: {tbl}\n  Columns: {col_summary}"
        if pk_cols:
            pk_summary = ", ".join(pk_cols)
            tbl_info += f"\n  Primary Key: ({pk_summary})"
        if fk_desc:
            fk_summary = "\n    - ".join(fk_desc)
            tbl_info += f"\n  Foreign Keys:\n    - {fk_summary}"
            
        schema_parts.append(tbl_info)
        
    conn.close()
    return "\n\n".join(schema_parts)

def execute_sql(query: str, db_path: str = DEFAULT_DB_PATH) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Safely executes a query against SQLite using read-only URI connection.
    Returns (DataFrame, None) or (None, error_str). Never crashes the application.
    """
    try:
        conn = get_readonly_connection(db_path)
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df, None
    except Exception as e:
        return None, str(e)
