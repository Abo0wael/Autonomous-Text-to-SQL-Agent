"""
sql_validator.py
Security validation layer for the Autonomous Text-to-SQL Agent.
Guarantees zero-trust execution by enforcing SELECT-only queries and blocking destructive DDL/DML.
"""

import re
from typing import Tuple, List

FORBIDDEN_KEYWORDS: List[str] = [
    r"\bINSERT\b", r"\bUPDATE\b", r"\bDELETE\b", r"\bDROP\b",
    r"\bALTER\b", r"\bCREATE\b", r"\bTRUNCATE\b", r"\bREPLACE\b",
    r"\bATTACH\b", r"\bDETACH\b", r"\bEXEC\b", r"\bEXECUTE\b",
    r"\bPRAGMA\b", r"\bMERGE\b", r"\bUPSERT\b"
]

def clean_sql_output(raw_sql: str) -> str:
    """
    Strips markdown code fences, backticks, and extraneous whitespace from LLM SQL output.
    """
    if not raw_sql:
        return ""
    cleaned = raw_sql.strip()
    match = re.search(r"```(?:sql)?\s*(.*?)\s*```", cleaned, re.DOTALL | re.IGNORECASE)
    if match:
        cleaned = match.group(1).strip()
    cleaned = re.sub(r"^```sql\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()
    if cleaned.endswith(";"):
        cleaned = cleaned[:-1].strip()
    return cleaned

def validate_sql(query: str) -> Tuple[bool, str]:
    """
    Validates that a SQL query is safe to execute.
    
    Returns:
        (True, "Query is valid.") if safe.
        (False, "<Security alert or syntax error>") if dangerous, non-SELECT, or empty.
    """
    if not query or not query.strip():
        return False, "Query is empty."
        
    cleaned = query.strip()
    
    # 1. Check for destructive DDL/DML keywords (case-insensitive word boundaries)
    for kw in FORBIDDEN_KEYWORDS:
        if re.search(kw, cleaned, re.IGNORECASE):
            kw_name = kw.replace(r"\b", "")
            return False, f"Security Alert: Forbidden destructive keyword '{kw_name}' detected."
            
    # 2. Check that query begins with SELECT or WITH (for CTEs)
    first_token_match = re.match(r"^\s*([a-zA-Z]+)", cleaned)
    if not first_token_match:
        return False, "Invalid SQL syntax: Cannot determine statement type."
        
    first_token = first_token_match.group(1).upper()
    if first_token not in ("SELECT", "WITH"):
        return False, f"Security Alert: Only SELECT queries are permitted. Found statement starting with '{first_token}'."
        
    # 3. Check for multiple chained statements (SQL injection prevention)
    statements = [s.strip() for s in cleaned.split(";") if s.strip()]
    if len(statements) > 1:
        return False, "Security Alert: Multiple SQL statements are not permitted."
        
    return True, "Query is valid."
