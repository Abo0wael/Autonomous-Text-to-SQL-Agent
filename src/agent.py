"""
agent.py
LangGraph workflow orchestrating the Autonomous Text-to-SQL Agent.
Coordinates dynamic schema loading, SQL generation, validation, truly read-only execution,
self-correction, and natural-language explanation.
Agent Trace records only observable workflow events (no hidden reasoning).
"""

from typing import TypedDict, Optional, List, Dict, Any, Callable
import pandas as pd
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END

from src.models import get_llm
from src.database import get_database_schema, execute_sql, resolve_db_path, DEFAULT_DB_PATH
from src.sql_validator import validate_sql, clean_sql_output
from src.prompts import (
    build_sql_generation_prompt,
    build_sql_correction_prompt,
    build_answer_prompt
)

MAX_RETRIES = 3

class AgentState(TypedDict):
    question: str
    schema: str
    sql_query: str
    result_df: Optional[pd.DataFrame]
    result_records: Optional[List[Dict[str, Any]]]
    error: Optional[str]
    attempt_count: int
    explanation: Optional[str]
    success: bool
    history: List[str]
    correction_details: Optional[Dict[str, Any]]
    model_id: str
    db_path: str

def is_rate_limit_error(e: Exception) -> bool:
    """Detects if an exception is a Groq 429 rate limit error."""
    msg = str(e).lower()
    return "rate limit" in msg or "429" in msg or "rate_limit_exceeded" in msg or "otpm" in msg

def load_schema_node(state: AgentState) -> Dict[str, Any]:
    """Dynamically introspects database schema if not already present."""
    schema = state.get("schema")
    db_path = state.get("db_path", DEFAULT_DB_PATH)
    if not schema:
        schema = get_database_schema(db_path)
    history = list(state.get("history", []))
    history.append("Schema loaded successfully from database.")
    return {"schema": schema, "history": history}

def generate_sql_node(state: AgentState) -> Dict[str, Any]:
    """Generates initial candidate SQL using selected Groq model."""
    question = state["question"]
    schema = state["schema"]
    model_id = state.get("model_id")
    history = list(state.get("history", []))
    
    try:
        llm = get_llm(model_id)
        prompt = build_sql_generation_prompt(question, schema)
        response = llm.invoke([HumanMessage(content=prompt)])
        sql = clean_sql_output(response.content)
        history.append("SQL generated (Attempt 1).")
        return {"sql_query": sql, "attempt_count": 1, "history": history, "error": None}
    except Exception as e:
        if is_rate_limit_error(e):
            err_msg = "RATE_LIMIT_ERROR"
            history.append("Rate limit reached for this model.")
        else:
            err_msg = f"LLM Generation Error: {str(e)}"
            history.append(f"SQL generation failed: {str(e)}")
        return {"error": err_msg, "success": False, "history": history}

def validate_sql_node(state: AgentState) -> Dict[str, Any]:
    """Security check ensuring single SELECT and blocking destructive DDL/DML."""
    if state.get("error") and "RATE_LIMIT" in state["error"]:
        return {"success": False}
        
    sql = state.get("sql_query", "")
    is_valid, msg = validate_sql(sql)
    history = list(state.get("history", []))
    
    if is_valid:
        history.append("Validation passed: Safe single SELECT query confirmed.")
        return {"error": None, "history": history}
    else:
        history.append(f"Validation failed: {msg}")
        return {"error": msg, "success": False, "history": history}

def execute_sql_node(state: AgentState) -> Dict[str, Any]:
    """Safely executes validated SQL against SQLite without crashing."""
    if state.get("error"):
        return {"success": False}
        
    sql = state.get("sql_query", "")
    db_path = state.get("db_path", DEFAULT_DB_PATH)
    attempt = state.get("attempt_count", 1)
    history = list(state.get("history", []))
    
    df, err = execute_sql(sql, db_path)
    if err:
        history.append(f"Execution failed on Attempt {attempt}: {err}")
        return {"error": err, "result_df": None, "result_records": None, "success": False, "history": history}
    else:
        rows = len(df) if df is not None else 0
        history.append(f"Execution succeeded on Attempt {attempt} (returned {rows} rows).")
        records = df.to_dict(orient="records") if df is not None else []
        return {"error": None, "result_df": df, "result_records": records, "success": True, "history": history}

def correct_sql_node(state: AgentState) -> Dict[str, Any]:
    """Self-Correction Node: Diagnoses error and regenerates corrected SQL query."""
    question = state["question"]
    schema = state["schema"]
    failed_sql = state["sql_query"]
    error_msg = state.get("error", "Unknown error")
    attempt = state.get("attempt_count", 1) + 1
    model_id = state.get("model_id")
    history = list(state.get("history", []))
    
    existing_details = state.get("correction_details") or {
        "original_sql": failed_sql,
        "initial_error": error_msg,
        "corrections": []
    }
    
    try:
        llm = get_llm(model_id)
        prompt = build_sql_correction_prompt(question, schema, failed_sql, error_msg)
        response = llm.invoke([HumanMessage(content=prompt)])
        corrected_sql = clean_sql_output(response.content)
        
        existing_details["corrections"].append({
            "attempt": attempt,
            "failed_sql": failed_sql,
            "error": error_msg,
            "corrected_sql": corrected_sql
        })
        
        history.append(f"Correction Attempt {attempt}: Corrected SQL generated.")
        return {
            "sql_query": corrected_sql,
            "attempt_count": attempt,
            "error": None,
            "history": history,
            "correction_details": existing_details
        }
    except Exception as e:
        if is_rate_limit_error(e):
            err_msg = "RATE_LIMIT_ERROR"
            history.append("Rate limit reached during self-correction.")
        else:
            err_msg = f"Self-Correction LLM Error: {str(e)}"
            history.append(f"Self-correction failed: {str(e)}")
        return {"error": err_msg, "success": False, "history": history}

def generate_answer_node(state: AgentState) -> Dict[str, Any]:
    """Synthesizes query results into an executive natural-language answer."""
    question = state["question"]
    sql = state["sql_query"]
    df = state.get("result_df")
    model_id = state.get("model_id")
    history = list(state.get("history", []))
    
    if df is None or df.empty:
        records_preview = "The query executed successfully but returned 0 rows / empty result set."
    else:
        records_preview = df.head(10).to_string(index=False)
        
    try:
        llm = get_llm(model_id)
        prompt = build_answer_prompt(question, sql, records_preview)
        response = llm.invoke([HumanMessage(content=prompt)])
        explanation = response.content.strip()
        history.append("Natural-language explanation generated.")
        return {"explanation": explanation, "history": history}
    except Exception as e:
        if is_rate_limit_error(e):
            err_msg = "Rate limit reached while generating answer summary."
            history.append("Rate limit reached during answer generation.")
        else:
            err_msg = f"Answer generation error: {str(e)}"
            history.append(f"Answer generation error: {str(e)}")
        return {"explanation": err_msg, "history": history}

# Routing logic
def route_after_validation(state: AgentState) -> str:
    if state.get("error"):
        if "RATE_LIMIT" in state["error"]:
            return "END"
        if state.get("attempt_count", 1) < MAX_RETRIES:
            return "correct_sql"
        return "END"
    return "execute_sql"

def route_after_execution(state: AgentState) -> str:
    if state.get("success", False):
        return "generate_answer"
    if state.get("error") and "RATE_LIMIT" in state["error"]:
        return "END"
    if state.get("attempt_count", 1) < MAX_RETRIES:
        return "correct_sql"
    return "END"

def build_agent_graph():
    """Builds and compiles the LangGraph StateGraph."""
    workflow = StateGraph(AgentState)
    
    workflow.add_node("load_schema", load_schema_node)
    workflow.add_node("generate_sql", generate_sql_node)
    workflow.add_node("validate_sql", validate_sql_node)
    workflow.add_node("execute_sql", execute_sql_node)
    workflow.add_node("correct_sql", correct_sql_node)
    workflow.add_node("generate_answer", generate_answer_node)
    
    workflow.add_edge(START, "load_schema")
    workflow.add_edge("load_schema", "generate_sql")
    workflow.add_edge("generate_sql", "validate_sql")
    
    workflow.add_conditional_edges(
        "validate_sql",
        route_after_validation,
        {
            "execute_sql": "execute_sql",
            "correct_sql": "correct_sql",
            "END": END
        }
    )
    
    workflow.add_conditional_edges(
        "execute_sql",
        route_after_execution,
        {
            "generate_answer": "generate_answer",
            "correct_sql": "correct_sql",
            "END": END
        }
    )
    
    workflow.add_edge("correct_sql", "validate_sql")
    workflow.add_edge("generate_answer", END)
    
    return workflow.compile()

# Compile the graph
agent_graph = build_agent_graph()

def run_agent_workflow(
    question: str,
    model_id: str,
    db_path: str = DEFAULT_DB_PATH,
    progress_callback: Optional[Callable[[str], None]] = None
) -> Dict[str, Any]:
    """
    User-facing runner function with optional progress status updates.
    """
    actual_db = resolve_db_path(db_path)
    
    if progress_callback:
        progress_callback("Analyzing database schema...")
        
    initial_state: AgentState = {
        "question": question,
        "schema": "",
        "sql_query": "",
        "result_df": None,
        "result_records": None,
        "error": None,
        "attempt_count": 0,
        "explanation": None,
        "success": False,
        "history": [],
        "correction_details": None,
        "model_id": model_id,
        "db_path": actual_db
    }
    
    if progress_callback:
        progress_callback(f"Generating and validating SQL query...")
        
    final_state = agent_graph.invoke(initial_state)
    
    if progress_callback:
        if final_state.get("success"):
            progress_callback("Query completed successfully!")
        else:
            progress_callback("Completed with notifications.")
            
    return final_state
