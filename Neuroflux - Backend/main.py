# ======================================================================
# NeuroFlux AGRAG Backend - v21.0 (The Ghostwriter Protocol)
# ======================================================================
# This is the definitive, no-compromise build. It uses the "Trinity"
# architecture with the final "Ghostwriter" prompting strategy to
# generate deep, insightful, and novel "white paper" style reports.
#
# - Mind (Strategist): Creates a narrative plan & synthesizes a rich briefing.
# - Soul (Memory):     Provides internal knowledge via RAG.
# - Voice (Ghostwriter): Expands the briefing into a full, polished HTML report.
# ======================================================================

# --- Core Imports ---
import os
import uvicorn
import httpx
import json
import logging
import asyncio
import re
import html
import time
import uuid
import torch
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional, AsyncIterator
from enum import Enum

# --- FastAPI and Middleware ---
from fastapi import FastAPI, HTTPException, Request, Depends, BackgroundTasks, status
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import structlog

# --- Pydantic and Validation ---
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

# --- Production Features ---
from pybreaker import CircuitBreaker, CircuitBreakerError
from async_lru import alru_cache # Used for caching LLM responses now

# --- Data and ML Libraries ---
from bs4 import BeautifulSoup
import google.generativeai as genai
from sentence_transformers import CrossEncoder # NEW: For re-ranking

# --- LlamaIndex and Vector Store (Using SimpleVectorStore for stability) ---
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings, Document
from llama_index.core.query_engine import RetrieverQueryEngine
# REMOVED QdrantVectorStore import
# import qdrant_client # REMOVED Qdrant client import
# from qdrant_client.http.async_client import AsyncQdrantClient # REMOVED AsyncQdrantClient import
from llama_index.embeddings.fastembed import FastEmbedEmbedding

# Removed RecursiveCharacterTextSplitter import due to persistent issues
# Removed MetadataExtractor and related imports for stability
from llama_index.core.schema import TextNode # Still needed for node type hinting
from llama_index.llms.gemini import Gemini # LlamaIndex's wrapper for Gemini
from llama_index.core.vector_stores import SimpleVectorStore # For in-memory RAG


# PostgreSQL and SQL validation imports
import asyncpg # NEW: For async PostgreSQL client
from sqlglot import parse_one, exp # NEW: For SQL parsing and validation
from sqlglot.errors import ParseError # NEW: For SQL parsing errors


# ======================================================================
# 1. Configuration & Settings
# ======================================================================
class AppSettings(BaseSettings):
    OLLAMA_API_BASE: str = "http://localhost:11434"
    GOOGLE_API_KEY: str = Field(..., description="A Google API Key is required for the 'Mind'.")
    GOOGLE_CSE_ID: Optional[str] = Field(default=None, description="Google Custom Search Engine ID for web search.")

    KNOWLEDGE_BASE_DIR: str = "knowledge_docs"
    # PERSIST_DIR is now irrelevant for SimpleVectorStore, but keeping for future persistent RAG DB re-integration
    PERSIST_DIR: str = "storage"
    # QDRANT_COLLECTION_NAME is now irrelevant
    QDRANT_COLLECTION_NAME: str = "neuroflux_ghostwriter_v21" # Keeping for future Qdrant re-integration

    EMBEDDING_MODEL_NAME: str = "BAAI/bge-small-en-v1.5"
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    RATE_LIMIT: str = "30/minute"
    REQUEST_TIMEOUT: int = 300 # Increased for very long reports

    RERANK_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2" # Recommended for re-ranking
    RERANK_TOP_N: int = 5 # Number of top results to return after re-ranking

    # NEW: PostgreSQL connection details
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_USER: str = "user"
    POSTGRES_PASSWORD: str = "password"
    POSTGRES_DB: str = "your_database"
    
    # NEW: Database schema definition for LLM and validation
    POSTGRES_SCHEMA_DEFINITION: Dict[str, List[str]] = Field(
        default={
            "users": ["id", "name", "email", "signup_date", "age"],
            "products": ["id", "name", "price", "category"],
            "orders": ["id", "user_id", "product_id", "quantity", "order_date"]
        },
        description="Defines accessible tables and columns for SQL generation."
    )

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'

settings = AppSettings()
structlog.configure(processors=[structlog.stdlib.add_log_level, structlog.processors.TimeStamper(fmt="iso"), structlog.processors.JSONRenderer()], logger_factory=structlog.stdlib.LoggerFactory(), cache_logger_on_first_use=True)
logger = structlog.get_logger("neuroflux")
logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[logging.StreamHandler()])

try: genai.configure(api_key=settings.GOOGLE_API_KEY)
except Exception as e: logger.error("FATAL: Could not configure Gemini client.", error=str(e)); exit(1)

class IndexingStatus(str, Enum): IDLE = "idle"; SCANNING = "scanning"; BUILDING = "building"; ERROR = "error"

# Initialize re-ranker model globally for efficiency
try:
    RERANKER = CrossEncoder(settings.RERANK_MODEL, max_length=512)
    logger.info(f"Initialized re-ranker model: {settings.RERANK_MODEL}")
except Exception as e:
    logger.warning(f"Could not load re-ranker model {settings.RERANK_MODEL}: {e}. Re-ranking will be skipped.", error=str(e))
    RERANKER = None

# ======================================================================
# 2. Resource Manager (The Soul)
# ======================================================================
class ResourceManager:
    def __init__(self, config: AppSettings):
        self.config = config
        self.http_client: Optional[httpx.AsyncClient] = None
        # Qdrant client members removed
        self.qdrant_client = None
        self.async_qdrant_client = None
        self.pg_pool: Optional[asyncpg.Pool] = None # NEW: PostgreSQL connection pool
        self.pg_schema: Dict[str, List[str]] = config.POSTGRES_SCHEMA_DEFINITION # NEW: DB schema for validation

        self.rag_query_engine: Optional[RetrieverQueryEngine] = None
        self.indexing_status: Dict[str, Any] = {"status": IndexingStatus.IDLE, "message": "Ready"}
        self.gemini_breaker = CircuitBreaker(fail_max=3, reset_timeout=60, name="gemini")
        self.ollama_breaker = CircuitBreaker(fail_max=3, reset_timeout=60, name="ollama")
        # Store a reference to the in-memory vector store for re-indexing
        self._in_memory_vector_store: Optional[SimpleVectorStore] = None # Used for in-memory RAG

    async def startup(self):
        logger.info("Starting ResourceManager", device=self.config.DEVICE)
        self.http_client = httpx.AsyncClient(timeout=self.config.REQUEST_TIMEOUT)
        for path in [self.config.PERSIST_DIR, self.config.KNOWLEDGE_BASE_DIR]: os.makedirs(path, exist_ok=True)
        
        Settings.embed_model = FastEmbedEmbedding(model_name=self.config.EMBEDDING_MODEL_NAME)
        
        # Qdrant client initialization removed
        
        # NEW: Initialize PostgreSQL connection pool
        try:
            self.pg_pool = await asyncpg.create_pool(
                user=self.config.POSTGRES_USER,
                password=self.config.POSTGRES_PASSWORD,
                host=self.config.POSTGRES_HOST,
                port=self.config.POSTGRES_PORT,
                database=self.config.POSTGRES_DB,
                min_size=1, # Minimum connections in pool
                max_size=10, # Maximum connections in pool
                timeout=self.config.REQUEST_TIMEOUT # Connection timeout
            )
            logger.info("✅ PostgreSQL connection pool initialized.")
        except Exception as e:
            logger.error(f"FATAL: Could not connect to PostgreSQL: {e}", exc_info=True)
            self.pg_pool = None # Ensure it's None if connection fails

        # LlamaIndex's global LLM setting
        llama_index_gemini_llm = Gemini(api_key=self.config.GOOGLE_API_KEY, model_name="models/gemini-1.5-flash-latest")
        Settings.llm = llama_index_gemini_llm 

        await self.load_rag_engine()

    async def shutdown(self):
        if self.http_client: await self.http_client.aclose()
        if self.pg_pool: await self.pg_pool.close() # Close PostgreSQL pool
        logger.info("✅ All resource managers shut down.")

    async def load_rag_engine(self):
        # With SimpleVectorStore, there's no persistent client to check or load from disk.
        # The index is built fresh into memory during the build_index_task.
        # So, we'll indicate it's dormant until indexing is manually triggered.
        self.indexing_status = {"status": IndexingStatus.IDLE, "message": "RAG 'Soul' is dormant (in-memory, requires indexing)."}
        logger.warning("⚠️ RAG 'Soul' is dormant (in-memory, requires indexing).")
        # The rag_query_engine will be set once indexing is complete.

# ======================================================================
# 3. Core Logic: The Scholarly Architect Agent
# ======================================================================
@alru_cache(maxsize=128) # Caching web search results
async def run_web_search_tool(rm: ResourceManager, query: str) -> str:
    if not settings.GOOGLE_CSE_ID: return "Web search tool disabled."
    logger.info("Executing tool: web_search", query=query)
    params = {"key": settings.GOOGLE_API_KEY, "cx": settings.GOOGLE_CSE_ID, "q": query, "num": 5}
    try:
        response = await rm.http_client.get("https://www.googleapis.com/customsearch/v1", params=params)
        response.raise_for_status()
        items = response.json().get("items", [])
        return "\n\n".join([f"Source: {i.get('link')}\nSnippet: {i.get('snippet')}" for i in items]) if items else f"No web results for '{query}'."
    except Exception as e:
        logger.error("Error during web search", error=str(e), exc_info=True) # Added error logging
        return f"Error during web search: {e}"

@alru_cache(maxsize=128) # Caching RAG search results
async def run_rag_search_tool(rm: ResourceManager, query: str, filters: Optional[Dict[str, Any]] = None) -> str:
    # Check if rag_query_engine is actually initialized from in-memory index
    if not rm.rag_query_engine: return "Local document knowledge base (Soul) is not available or not yet indexed."
    logger.info("Executing tool: vector_database_search", query=query, filters=filters)
    try:
        response = await rm.rag_query_engine.aquery(query)
        source_nodes = response.source_nodes

        if RERANKER and source_nodes:
            logger.info(f"Re-ranking {len(source_nodes)} retrieved nodes for query: {query}")
            query_and_nodes_content = [(query, node.get_content(metadata_mode="llm")) for node in source_nodes]
            scores = RERANKER.predict(query_and_nodes_content).tolist()
            scored_nodes = sorted(zip(scores, source_nodes), key=lambda x: x[0], reverse=True)
            
            reranked_nodes_info = []
            for score, node in scored_nodes[:settings.RERANK_TOP_N]:
                file_name = node.metadata.get('file_name', 'N/A')
                title = node.metadata.get('document_title', node.metadata.get('title', node.metadata.get('original_title', 'N/A'))) 
                reranked_nodes_info.append(f"Source: {file_name} (Title: {title}, Score: {score:.4f})\nContent: {node.get_content(metadata_mode='llm')}")
            
            return "\n\n".join(reranked_nodes_info)
        else:
            return str(response)

    except Exception as e:
        logger.error("Error during RAG search", error=str(e), exc_info=True)
        return f"Error during RAG search: {e}"

# NEW: Function to validate LLM-generated SQL
def validate_sql_query(sql_query: str, allowed_schema: Dict[str, List[str]]) -> Optional[str]:
    """
    Validates an LLM-generated SQL query against a simplified schema and security rules.
    Returns the validated SQL string if valid, None otherwise.
    """
    if not sql_query:
        logger.warning("Empty SQL query provided for validation.")
        return None

    # 1. Basic Security Check (Whitelist SELECT, Disallow Harmful Keywords)
    lower_sql = sql_query.lower()
    if not lower_sql.startswith("select"):
        logger.warning(f"SQL query not a SELECT statement (blocked): {sql_query}")
        return None
    
    forbidden_keywords = ["insert", "update", "delete", "drop", "alter", "truncate", "create", "grant", "revoke", "delete", "update", "insert", "union", "sleep", "benchmark"]
    if any(keyword in lower_sql for keyword in forbidden_keywords):
        logger.warning(f"SQL query contains forbidden keywords (blocked): {sql_query}")
        return None

    try:
        # 2. Syntax and Schema Validation using sqlglot
        parsed_sql = parse_one(sql_query, read="postgres") # Read as PostgreSQL dialect

        # Perform basic schema validation by checking table and column names
        for table_exp in parsed_sql.find_all(exp.Table):
            table_name = table_exp.name
            if table_name not in allowed_schema:
                logger.warning(f"SQL query uses unauthorized table: {table_name}")
                return None
            
            # Check columns if explicit columns are specified (more complex for '*')
            for column_exp in parsed_sql.find_all(exp.Column):
                col_name = column_exp.name
                # If column has table prefix, check against specific table
                if column_exp.table: # Column has a table prefix
                    if column_exp.table not in allowed_schema:
                         logger.warning(f"SQL query uses unauthorized table prefix: {column_exp.table}")
                         return None
                    if col_name not in allowed_schema[column_exp.table] and col_name != "*":
                        logger.warning(f"SQL query uses unauthorized column '{col_name}' for table '{column_exp.table}'")
                        return None
                else: # Column no table prefix, check all allowed tables
                    if not any(col_name in cols for cols in allowed_schema.values()) and col_name != "*":
                        logger.warning(f"SQL query uses unauthorized column '{col_name}' without table prefix.")
                        return None
        
        # You could add more sophisticated checks here, e.g.,
        # - Check for excessive JOINs
        # - Enforce LIMIT clauses
        # - Prevent subqueries if not desired

        return parsed_sql.sql(dialect="postgres") # Return canonicalized SQL

    except ParseError as e:
        logger.warning(f"SQL syntax error during validation: {e} for query: {sql_query}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error during SQL validation: {e}", exc_info=True)
        return None

@alru_cache(maxsize=128) # Cache results of successful SQL queries
async def run_postgres_query_tool(rm: ResourceManager, natural_language_query: str) -> str:
    if not rm.pg_pool:
        return "PostgreSQL database connection is not available."
    if not rm.pg_schema:
        return "PostgreSQL schema definition is not loaded for validation."

    logger.info("Executing tool: postgres_query", query=natural_language_query)

    generated_sql = "" # Initialize to ensure it's always defined
    try:
        # Use Gemini to generate SQL
        llm_model = genai.GenerativeModel('gemini-1.5-flash-latest')
        schema_str = "\n".join([
            f"- Table '{table_name}': columns ({', '.join(columns)})"
            for table_name, columns in rm.pg_schema.items()
        ])
        llm_sql_generator_prompt = f"""You are an expert PostgreSQL query generator. Your task is to generate a single, valid PostgreSQL SELECT query based on the user's natural language request.
        
        You have access to the following tables and their columns:
        {schema_str}
        
        Relationships: users.id = orders.user_id, products.id = orders.product_id
        
        Constraints:
        - ONLY generate SELECT queries. DO NOT use INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, TRUNCATE, GRANT, REVOKE, or any other DDL/DML statements.
        - If you cannot generate a valid and safe SELECT query for the request, respond with 'N/A'.
        - ONLY output the SQL query, nothing else.
        
        Natural Language Query: '{natural_language_query}'
        SQL Query:"""

        sql_query_response = await llm_model.generate_content_async(
            llm_sql_generator_prompt,
            generation_config=genai.types.GenerationConfig(
                response_mime_type="text/plain", # Ensure plain text output for SQL
                temperature=0.1 # Keep temperature low for deterministic SQL
            )
        )
        generated_sql = sql_query_response.text.strip()

        if generated_sql.upper().startswith("N/A"):
            logger.warning(f"LLM indicated it cannot generate SQL for: {natural_language_query}")
            return f"The AI could not generate a suitable SQL query for '{natural_language_query}' based on the available schema."

        # --- Phase 2: Validation Against Schema and Allowed Patterns ---
        validated_sql = validate_sql_query(generated_sql, rm.pg_schema)

        if not validated_sql:
            logger.error(f"LLM-generated SQL failed validation: '{generated_sql}' for query: '{natural_language_query}'")
            return f"The generated SQL query failed validation and could not be executed for '{natural_language_query}'. (Possible unsafe query or schema mismatch)."

        # --- Phase 3: Execution and Result Formatting ---
        async with rm.pg_pool.acquire() as connection:
            # fetch() returns a list of asyncpg.Record objects
            records = await connection.fetch(validated_sql)
            
            if records:
                # Convert records to list of dicts for easier parsing by LLM
                formatted_results = [dict(r) for r in records]
                # Summarize results if too long for LLM context, or return raw JSON
                return f"PostgreSQL Results for '{natural_language_query}':\n```json\n{json.dumps(formatted_results, indent=2)}\n```"
            else:
                return f"No results found in PostgreSQL for '{natural_language_query}'."

    except asyncpg.exceptions.PostgresError as pg_err:
        logger.error(f"PostgreSQL query execution error: {pg_err} for generated SQL: '{generated_sql}' and NL query: '{natural_language_query}'", exc_info=True)
        return f"PostgreSQL database error during query execution: {pg_err}. Query: '{natural_language_query}'"
    except Exception as e:
        logger.error(f"An unexpected error occurred during PostgreSQL query generation/execution: {e}", exc_info=True)
        return f"An unexpected error occurred during PostgreSQL query for '{natural_language_query}': {e}"


async def get_best_ollama_model(rm: ResourceManager) -> str:
    try:
        response = await rm.http_client.get(f"{settings.OLLAMA_API_BASE}/api/tags")
        models = response.json().get("models", [])
        if not models: raise ValueError("No Ollama models found.")
        priority_order = ["llama3:8b-instruct", "command-r", "mistral", "llama3", "qwen", "codellama", "phi3"]
        for p_model in priority_order:
            for m in models:
                if p_model in m['name']:
                    logger.info(f"Dynamically selected Scribe model: {m['name']}")
                    return m['name']
        fallback = models[0]['name']; logger.info(f"Falling back to first available model: {fallback}"); return fallback
    except Exception as e:
        logger.warning("Could not determine best Ollama model, defaulting to llama3:8b-instruct.", error=str(e))
        return "llama3:8b-instruct"

# Removed @alru_cache from agent_event_generator as it causes RuntimeErrors with FastAPI's StreamingResponse
async def agent_event_generator(request: BaseModel, rm: ResourceManager) -> AsyncIterator[str]:
    request_id = uuid.uuid4(); logger.info("Ghostwriter Agent execution started", request_id=str(request_id), query=request.query)
    try:
        # === Phase 1: The Genesis Prompt (Mind) ===
        yield f'data: {json.dumps({"type": "log", "content": "Phase 1: Genesis - Creating narrative context...", "class": "phase-title"})}\n\n'
        genesis_prompt = f"""You are a master storyteller and strategist. For the query: '{request.query}', create the narrative and strategic context. Generate a JSON object with keys: 'backstory', 'core_tension', 'stakeholders', and a 'research_plan' (an array of tasks with 'tool_to_use' and 'query_for_tool').
        The available tools are:
        - 'web_search': For general internet knowledge, current events, and industry trends.
        - 'vector_database_search': For detailed, in-depth knowledge from the local document repository (e.g., internal research, academic papers).
        - 'postgres_query': For structured, transactional data from the PostgreSQL database (e.g., user demographics, product information, order details).
            - **Database Schema**:
                - `users` table: `id`, `name`, `email`, `signup_date`, `age`
                - `products` table: `id`, `name`, `price`, `category`
                - `orders` table: `id`, `user_id`, `product_id`, `quantity`, `order_date`
            - Use this tool for queries about specific user characteristics (e.g., 'users over 30'), product attributes (e.g., 'products in electronics category'), or order aggregates (e.g., 'orders placed last month').
            - Only ask for data that can be retrieved with a SELECT statement.

        The entire context and research plan should be focused on answering and elaborating on the provided query. Ensure research tasks explicitly target finding:
        - Quantifiable data, statistics, or percentages relevant to the query. **If internal data might be limited, also seek general industry statistics or publicly available benchmarks that can illustrate potential impacts or trends.**
        - Specific, named open-source tools, frameworks, or projects that address aspects of the query (e.g., for ethical AI, data anonymization, bias detection).
        - Real-world examples or case studies if applicable.
        - **Important**: If a specific data point or tool name is requested that might not be in the internal DB, leverage 'web_search' to find publicly available, analogous information or general principles.
        """

        if rm.gemini_breaker.current_state == "open": raise CircuitBreakerError("Gemini API is unavailable.")
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        response = await model.generate_content_async(genesis_prompt, generation_config=genai.types.GenerationConfig(response_mime_type="application/json"))
        genesis_plan = json.loads(response.text); yield f'data: {json.dumps({"type": "plan_generated", "plan": genesis_plan})}\n\n'

        # === Phase 2: Parallel Research (Soul & Web) ===
        yield f'data: {json.dumps({"type": "log", "content": "Phase 2: Executing hybrid research plan...", "class": "phase-title"})}\n\n'
        research_plan_obj = genesis_plan.get("research_plan", [])
        task_list = research_plan_obj if isinstance(research_plan_obj, list) else research_plan_obj.get("tasks", [])
        tasks_to_run = []
        for t in task_list:
            tool_type = t.get("tool_to_use")
            query_for_tool = t.get("query_for_tool", "")
            if tool_type == "web_search":
                tasks_to_run.append(run_web_search_tool(rm, query_for_tool))
            elif tool_type == "vector_database_search":
                tasks_to_run.append(run_rag_search_tool(rm, query_for_tool, filters=None)) # Filters are a placeholder
            elif tool_type == "postgres_query": # NEW: Handle postgres_query tool
                tasks_to_run.append(run_postgres_query_tool(rm, query_for_tool))
            else: # Handle unknown tools gracefully
                logger.warning(f"Unknown tool requested by Mind: {tool_type}. Skipping task.")
                tasks_to_run.append(asyncio.sleep(0.01)) # Add a tiny delay to prevent tight loop if many unknown

        results = await asyncio.gather(*tasks_to_run, return_exceptions=True)
        research_log = f"Genesis Context:\n{json.dumps(genesis_plan, indent=2)}\n\n"
        research_log += "".join([f"--- Research Result ---\n{res}\n\n" for res in results])

        # === Phase 3: Synthesis & Insight (Mind) ===
        yield f'data: {json.dumps({"type": "log", "content": "Phase 3: Synthesizing novel insights...", "class": "phase-title"})}\n\n'
        synthesis_prompt = f"""You are a senior analyst and theorist. Based on the context and research data below, write a definitive 'Intelligence Briefing' as a JSON object. Your mission is to generate novel, synthesized insights that directly answer and elaborate on the query: '{request.query}'. Your JSON must contain:
        'title' (directly addressing the query),
        'executive_summary' (summarizing the key answer and benefits. **Prioritize substantive findings.** If quantifiable impacts/benefits are EXPLICITLY verifiable from the 'research_log', include them. OTHERWISE, use well-reasoned qualitative descriptions like 'significant improvement', 'enhanced precision', 'potential for bias reduction'. **Avoid directly stating 'data was not available' here; instead, formulate the summary based on what *was* found or can be inferred generally.**),
        'introduction_and_context' (setting the stage for the query's topic),
        'core_analysis' (a deep dive into the topic, providing 'why' explanations. **Prioritize specific examples and deeper reasoning.** If specific open-source tools/frameworks/solutions are EXPLICITLY verifiable from the 'research_log', mention them. OTHERWISE, use general terms like 'various robust frameworks' or 'common mitigation strategies' and describe *types* of solutions.),
        'novel_synthesis' (a JSON object with keys: 'name' (a concise, descriptive name for the core synthesis or main conceptual contribution derived from the query's elaboration), and 'description' (a detailed explanation of this concept/synthesis. This should reflect a deeper analytical leap from the raw data. Do NOT invent a name if no genuine novel concept emerges; instead, use a highly descriptive phrase like 'Balancing Ethical AI with Business Imperatives' or 'The Dual Challenge of Data Utility and Privacy').),
        'perspectives' (an object with 'utopian_view' and 'dystopian_view' related to the query's topic. **If quantifiable impacts are EXPLICITLY verifiable from the 'research_log', include them. OTHERWISE, use qualitative descriptions that align with the respective view.**),
        'verifiable_quotes' (an array of real, direct quotes from the research data, with source if available),
        'verifiable_sources' (an array of real, complete URLs or publication details from web search. **ONLY include sources that are explicitly present in the 'research_log' and are actual links/references. DO NOT invent or assume sources.**),
        and 'mermaid_code' (a simple, syntactically correct flowchart or sequence diagram for a process, architecture, or data flow related to the query's answer. **It MUST start with 'graph TD', 'graph LR', 'sequenceDiagram', 'flowchart TD', or similar valid Mermaid syntax keyword. If there is no clear process or architecture to visualize, provide an empty string.**).
        The overall content of the briefing should be entirely driven by the original query: '{request.query}', and **strictly grounded in the provided 'research_log'. Do NOT hallucinate facts, statistics, or specific tool names not present in the research.**

        --- Context and Research Data ---
        {research_log}
        """
        response = await model.generate_content_async(synthesis_prompt, generation_config=genai.types.GenerationConfig(response_mime_type="application/json"))
        intelligence_briefing = json.loads(response.text)
        yield f'data: {json.dumps({"type": "log", "content": "✅ Intelligence briefing complete."})}\n\n'

        # === Phase 4: The Ghostwriter Protocol (Voice) ===
        yield f'data: {json.dumps({"type": "log", "content": "Phase 4: Commissioning Ghostwriter for final report...", "class": "phase-title"})}\n\n'
        # THIS IS THE CRITICAL PROMPT FOR GENERATING THE DETAILED HTML REPORT
        ghostwriter_prompt = f"""
        You are a world-class author and analyst, renowned for your ability to write deep, insightful, and engaging research reports. You have been commissioned to write a definitive white paper of professional scholar investigative standards. English only, no thinking text.

        Your senior analyst (The Mind) has provided you with a comprehensive 'Intelligence Briefing' in JSON format. This briefing contains foundational research, key insights, a potentially novel concept, and direct quotes.

        **CRITICAL INSTRUCTION: ALL FACTS, STATISTICS, OPEN-SOURCE TOOLS, AND EXAMPLES MUST BE STRICTLY DERIVED FROM THE PROVIDED 'INTELLIGENCE BRIEFING'. DO NOT INVENT OR HALLUCINATE ANY INFORMATION. IF THE BRIEFING DOES NOT CONTAIN A SPECIFIC DETAIL (e.g., A PRECISE PERCENTAGE, THE NAME OF A TOOL), YOU MUST EITHER USE QUALITATIVE LANGUAGE (e.g., "significant improvement") OR STATE THAT SUCH SPECIFIC DATA WAS NOT IDENTIFIED IN THE RESEARCH, BUT ONLY WHERE NECESSARY FOR CLARITY, NOT AS A REPETITIVE DISCLAIMER.**

        Your task is to use this briefing as your **sole source of truth and primary outline**. Expand significantly upon it to write a full, detailed, and beautifully structured HTML report, adhering to the depth and format of a scholarly investigative white paper.

        **Instructions for Report Generation:**
        1.  **Comprehensive HTML Structure:** The report MUST include the following HTML sections in this exact order. **Dynamically use content from the Intelligence Briefing for titles and content where specified.**
            *   `<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>{intelligence_briefing.get('title', 'Generated White Paper')}</title>
            <script src="https://cdn.jsdelivr.net/npm/mermaid@10.8.0/dist/mermaid.min.js"></script>
            <style>
            :root {{ --grid-line-color: rgba(255, 255, 255, 0.07); --bg-color: #0d0d0d; --text-color: #e0e0e0; --primary-color: #00aaff; --accent-green: #20c997; --accent-orange: #ffab70; --border-color: #333; }}
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; line-height: 1.7; background-color: var(--bg-color); color: var(--text-color); max-width: 1400px; margin: 2rem auto; padding: 0 2rem; background-image: linear-gradient(var(--grid-line-color) 1px, transparent 1px), linear-gradient(90deg, var(--grid-line-color) 1px, transparent 1px); background-size: 40px 40px; }}
            header {{ text-align: center; border-bottom: 2px solid var(--primary-color); padding-bottom: 1rem; margin-bottom: 2rem; }}
            .subtitle {{ font-size: 1.3rem; color: #a0a0a0; margin-top: -1rem; margin-bottom: 2rem; }}
            h1, h2, h3, h4 {{ color: var(--primary-color); }} h1 {{ font-size: 2.5rem; }} h2 {{ font-size: 2rem; border-bottom: 1px solid var(--border-color); padding-bottom: 0.5rem; margin-top: 3rem; }} h3 {{ font-size: 1.5rem; color: var(--accent-green); }}
            .report-section {{ margin-bottom: 2.5rem; background: rgba(20, 20, 20, 0.85); border: 1px solid var(--border-color); border-radius: 8px; padding: 1.5rem 2.5rem; backdrop-filter: blur(8px); }}
            blockquote {{ border-left: 4px solid var(--accent-green); padding-left: 1.5rem; margin: 2rem 0; font-style: italic; color: #c0c0c0; }}
            pre.mermaid {{ background-color: rgba(0,0,0,0.2); border: 1px solid var(--border-color); border-radius: 8px; padding: 1rem; margin-top: 2rem; text-align: center; overflow-x: auto; }}
            footer {{ font-size: 0.9rem; color: #888; text-align: center; margin-top: 4rem; border-top: 1px solid var(--border-color); padding-top: 1.5rem; }}
            .references-list {{ list-style-type: none; padding-left: 0; }}
            .references-list li {{ margin-bottom: 0.8rem; font-size: 0.95em; line-height: 1.5; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 1.5rem; border: 1px solid var(--border-color); }}
            th, td {{ padding: 0.75rem; text-align: left; border: 1px solid var(--border-color); }}
            th {{ background-color: rgba(50, 50, 50, 0.8); color: var(--primary-color); }}
            tbody tr:nth-child(odd) {{ background-color: rgba(30, 30, 30, 0.8); }}
            tbody tr:hover {{ background-color: rgba(60, 60, 60, 0.8); }}
        </style>
        <script>
            // Initialize Mermaid.js after the DOM is loaded
            document.addEventListener('DOMContentLoaded', function() {{
                mermaid.initialize({{ startOnLoad: true }});
            }});
        </script>
        </head><body>`
            *   `<header>` (with `<h1>` for main title derived from `intelligence_briefing['title']` and `<p class="subtitle">` for a relevant subtitle, **dynamically generated to reflect the scholarly depth of the report's actual topic, perhaps leveraging the `novel_synthesis` name or core analytical insight**.)
            *   `<section class="report-section">` for `<h2>Abstract</h2>` (Generated based on the briefing's summary, capturing main points, problem, solution, benefits, and impact. Strictly adhere to briefing for quantifiable impacts.)
            *   `<section class="report-section">` for `<h2>Executive Summary</h2>` (Expand on `intelligence_briefing['executive_summary']`. **Focus on the core findings and recommendations first.** Only include plausible, general quantifiable impacts/benefits if explicitly found in the briefing. Otherwise, use well-reasoned qualitative terms. **Avoid repetitive statements about data unavailability; integrate it gracefully if crucial for context, or pivot to general industry insights.**)
            *   `<section class="report-section">` for `<h2>Introduction and Context</h2>` (Expand on `intelligence_briefing['introduction_and_context']`. Strengthen narrative, provide specific examples of issues/context from briefing. )
            *   `<section class="report-section">` for `<h2>Related Work</h2>` (This section is essential. Discuss existing literature, concepts, and relevant open-source tools related to the *report's actual topic*. Explain how the report's content builds upon/differentiates itself. **ONLY mention specific names of tools/frameworks/research if they are explicitly present in the `intelligence_briefing`'s research findings.** If not, discuss general types of approaches or theoretical concepts found. Use placeholder citations like `[1]`, `[2]`, `[3]` etc. as you discuss.)
            *   `<section class="report-section">` for `<h2>Core Analysis</h2>` (Expand on `intelligence_briefing['core_analysis']`. Provide detailed 'why' explanations. **ONLY mention specific open-source tools/solutions or methodologies if explicitly present in the `intelligence_briefing`'s research findings.** If specific internal data for quantification was not found, discuss the implications qualitatively or use general industry data from the briefing. Use placeholder citations `[N]`.)
            *   `<section class="report-section">` for `<h2>Novel Synthesis: {intelligence_briefing.get('novel_synthesis', {}).get('name', 'Core Insights')}</h2>` (Expand on `intelligence_briefing['novel_synthesis'].get('description', '')`. If it describes a concept, detail its architecture/functionality based strictly on the briefing. If it's a general synthesis of ideas, elaborate on those insights. **If `intelligence_briefing.get('mermaid_code')` exists and starts with a valid Mermaid keyword (e.g., 'graph', 'sequence', 'flowchart'), include it here in `<pre class="mermaid">`; otherwise, display a "Diagram not available for this topic." message.**)
            *   `<section class="report-section">` for `<h2>Evaluation and Results</h2>` (If the briefing contains quantifiable results or implies evaluation, expand on this. **If the briefing explicitly suggests or provides metrics/percentages, create hypothetical tables or detailed descriptions. If only qualitative benefits or challenges of evaluation are present in the briefing, discuss those in detail here.** Use placeholder citations `[N]`.)
            *   `<section class="report-section">` for `<h2>Perspectives</h2>` (with `<h3>` for Utopian/Dystopian. Expand on `intelligence_briefing['perspectives']`. Include plausible, general quantifiable impacts **only if found in the briefing**.)
            *   `<section class="report-section">` for `<h2>Actionable Recommendations</h2>` (Expand on `intelligence_briefing`'s recommendations. Provide highly specific, actionable, and technically detailed recommendations relevant to the *report's actual topic*. Quantify targets **only if found in the briefing**. Mention relevant open-source tools/solutions or best practices **only if found in the briefing**. Use placeholder citations `[N]`.)
            *   `<section class="report-section">` for `<h2>Conclusion & Future Work</h2>` (Summarize the main contribution and significance. Outline specific, forward-looking areas for "Future Work" relevant to the *report's actual topic*. Use placeholder citations `[N]`.)
            *   `<section class="report-section">` for `<h2>References</h2>` (This is CRITICAL. Create a formal, numbered list of references. **For each `[N]` citation used in the text, provide a plausible, formatted academic reference (e.g., using a consistent style like IEEE or APA) ONLY IF it is present in the `intelligence_briefing['verifiable_sources']` array.** **DO NOT invent references or make up links/titles.** If a concept is discussed and no direct source for it is in `verifiable_sources`, do NOT include a reference. **If `verifiable_sources` is empty, this section should simply state "No direct verifiable sources were identified in the research."** Use `<ul class="references-list">` for this section. Ensure links are clickable if provided in sources.)
            *   `<footer>` (with copyright and contact info, dynamically referencing the report's topic or authoring entity if possible)
            *   `</body></html>`

        3.  **Overall Quality:**
            *   Maintain a formal, objective, and authoritative tone suitable for a scholarly white paper.
            *   Ensure smooth transitions between paragraphs and sections.
            *   Avoid colloquialisms or overly simplistic language.
            *   The entire output MUST be a single, complete HTML file.

        --- INTELLIGENCE BRIEFING (Your Outline and Facts) ---
        ```json
        {json.dumps(intelligence_briefing, indent=2)}
        ```

        Begin writing the full HTML report now. Ensure all content is within the specified HTML section tags and follows ALL instructions rigorously, adapting to the actual topic from the briefing and **strictly adhering to the "no hallucination" rule.**
        """

        selected_model = await get_best_ollama_model(rm)
        payload = {"model": selected_model, "messages": [{"role": "user", "content": ghostwriter_prompt}], "stream": True}

        if rm.ollama_breaker.current_state == "open": raise CircuitBreakerError("Ollama is unavailable.")
        async with rm.http_client.stream("POST", f"{settings.OLLAMA_API_BASE}/api/chat", json=payload) as response:
            if response.status_code != 200:
                error_body = await response.aread()
                logger.error("Ollama API call failed", status_code=response.status_code, details=error_body.decode())
                raise HTTPException(status_code=response.status_code, detail=f"Ollama error: {error_body.decode()}")

            async for line in response.aiter_lines():
                if not line: continue
                chunk = json.loads(line)
                if content := chunk.get("message", {}).get("content"):
                    yield f'data: {json.dumps({"type": "synthesis_token", "content": content})}\n\n'
                if chunk.get("done"): break

    except Exception as e:
        logger.error("Agent execution failed", error=str(e), exc_info=True); yield f'data: {json.dumps({"type": "error", "content": f"An error occurred: {e}"})}\n\n'
    finally:
        yield f'data: {json.dumps({"type": "complete", "content": "Stream ended."})}\n\n'; logger.info("Ghostwriter Agent execution finished", request_id=str(request_id))

# ======================================================================
# 4. FastAPI App Setup & Endpoints
# ======================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.resource_manager = ResourceManager(settings); await app.state.resource_manager.startup(); yield; await app.state.resource_manager.shutdown()

app = FastAPI(title="NeuroFlux Backend (Ghostwriter)", version="21.0", lifespan=lifespan)
app.state.limiter = Limiter(key_func=get_remote_address)
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

def get_resource_manager(request: Request) -> ResourceManager: return request.app.state.resource_manager
class AgentRequest(BaseModel): query: str

@app.post("/api/agent/execute")
@app.state.limiter.limit(settings.RATE_LIMIT)
async def agent_execute(request: Request, agent_request: AgentRequest, rm: ResourceManager = Depends(get_resource_manager)):
    return StreamingResponse(agent_event_generator(agent_request, rm), media_type="text/event-stream")

# ... (RAG indexing and other endpoints remain the same and are included for completeness)
@app.post("/api/rag/start-indexing", status_code=status.HTTP_202_ACCEPTED)
async def start_indexing_endpoint(background_tasks: BackgroundTasks, rm: ResourceManager = Depends(get_resource_manager)):
    if rm.indexing_status.get("status") == IndexingStatus.BUILDING: raise HTTPException(status_code=409, detail="Indexing is already in progress.")
    async def build_index_task(rm_instance: ResourceManager):
        rm_instance.indexing_status = {"status": IndexingStatus.BUILDING, "message": "Scanning document directory..."}
        try:
            knowledge_dir = settings.KNOWLEDGE_BASE_DIR
            if not os.path.exists(knowledge_dir) or not os.listdir(knowledge_dir):
                rm_instance.indexing_status["message"] = "Knowledge base directory is empty."; return
            
            # Load documents normally
            documents = SimpleDirectoryReader(knowledge_dir, recursive=True).load_data()
            if not documents: rm_instance.indexing_status["message"] = "No supported documents found."; return

            nodes = documents # 'nodes' are just the raw Document objects loaded by SimpleDirectoryReader
            
            if not nodes: rm_instance.indexing_status["message"] = "No nodes generated from documents."; return

            rm_instance.indexing_status["message"] = f"Found {len(documents)} documents. Building vector index..."
            
            # Use SimpleVectorStore for in-memory indexing as a fallback for Qdrant issues
            vector_store = SimpleVectorStore() # In-memory store
            store_type_message = "in-memory (Qdrant disabled due to persistent errors)"

            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            VectorStoreIndex.from_documents(nodes, storage_context=storage_context, show_progress=True)
            
            # After successful indexing, set the query engine for the resource manager
            rm_instance.rag_query_engine = VectorStoreIndex.from_vector_store(vector_store).as_query_engine(similarity_top_k=settings.RERANK_TOP_N * 3)
            
            rm_instance.indexing_status["message"] = f"Index built successfully ({store_type_message})."
            logger.info(f"✅ RAG 'Soul' initialized ({store_type_message}).")
        except Exception as e:
            rm_instance.indexing_status = {"status": IndexingStatus.ERROR, "message": f"Indexing failed: {e}"}
            logger.error(f"FATAL: Indexing failed in build_index_task: {e}", exc_info=True) # Ensure this error is logged
    background_tasks.add_task(build_index_task, rm); return {"message": "Knowledge base indexing has been initiated."}

@app.get("/api/rag/indexing-status")
async def get_indexing_status(rm: ResourceManager = Depends(get_resource_manager)): return rm.indexing_status

@app.get("/api/ollama/models", tags=["Utilities"])
async def get_ollama_models(rm: ResourceManager = Depends(get_resource_manager)):
    try: response = await rm.http_client.get(f"{settings.OLLAMA_API_BASE}/api/tags"); response.raise_for_status(); return response.json()
    except Exception as e: raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Could not connect to Ollama: {e}")

frontend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")
    @app.get("/", include_in_schema=False)
    async def serve_index(): return FileResponse(os.path.join(frontend_dir, 'index.html'))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)