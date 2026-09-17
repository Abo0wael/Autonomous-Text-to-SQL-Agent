"""
prompts.py
System prompts and prompt builders for SQL generation, self-correction, and answer synthesis.
"""

def build_sql_generation_prompt(question: str, schema: str) -> str:
    """
    Builds the system prompt instructing the LLM to generate a compliant SQLite query.
    """
    return f"""You are an expert SQLite database engineer.
Given the SQLite database schema below, generate a single valid SQLite SELECT query to answer the user's question.

Rules:
1. Use ONLY the tables and columns present in the schema.
2. Generate ONLY one valid SELECT query (or WITH ... SELECT).
3. Do NOT hallucinate any table names, column names, or relationships.
4. Use proper SQL joins (JOIN ... ON) when needed based on foreign keys.
5. SQLite syntax rules: use LIMIT for top N, strftime for dates, etc.
6. Return ONLY the executable SQL query. Do not include conversational markdown, greetings, or explanations.

Database Schema:
{schema}

User Question: {question}

SQL Query:"""

def build_sql_correction_prompt(question: str, schema: str, failed_sql: str, error_message: str) -> str:
    """
    Builds the diagnostic self-correction prompt providing the failed query and exact error message.
    """
    return f"""You are an expert SQLite database debugger.
A previously generated SQL query failed with an error.

Task:
Analyze the user question, the database schema, the failed SQL query, and the exact error message.
Generate a corrected, valid SQLite SELECT query that resolves the error.

Database Schema:
{schema}

User Question: {question}
Failed SQL: {failed_sql}
Error Message: {error_message}

Rules:
1. Fix the error while accurately answering the user's question.
2. Use ONLY tables and columns that exist in the schema.
3. Return ONLY the corrected SQL query. No markdown explanation or conversational text.

Corrected SQL:"""

def build_answer_prompt(question: str, sql_query: str, records_preview: str) -> str:
    """
    Builds the prompt instructing the LLM to synthesize tabular data into a business summary.
    """
    return f"""You are a professional business data analyst.
Provide a clear, concise, direct natural-language answer to the user's question based on the executed SQL query and its results.

User Question: {question}
Executed SQL Query:
{sql_query}

Query Results:
{records_preview}

Instructions:
- Give a direct answer in 2 to 4 sentences.
- Explicitly cite relevant numbers, names, or values from the results.
- If results are empty, state that no matching records were found.
- Do NOT output markdown code blocks or conversational chatter."""
