# ======================================================================
# NeuroFlux AGRAG Backend - v21.0 (The Ghostwriter Protocol)
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
# NEW: Import SimpleVectorStore for in-memory RAG
from llama_index.core.vector_stores import SimpleVectorStore

# ======================================================================
# 1. Configuration & Settings
# ======================================================================
class AppSettings(BaseSettings):
    OLLAMA_API_BASE: str = "http://localhost:11434"
    GOOGLE_API_KEY: str = Field(..., description="A Google API Key is required for the 'Mind'.")
    GOOGLE_CSE_ID: Optional[str] = Field(default=None, description="Google Custom Search Engine ID for web search.")

    KNOWLEDGE_BASE_DIR: str = "knowledge_docs"
    # PERSIST_DIR is now irrelevant for SimpleVectorStore, but keeping for future Qdrant re-integration
    PERSIST_DIR: str = "storage"
    # QDRANT_COLLECTION_NAME is now irrelevant
    QDRANT_COLLECTION_NAME: str = "neuroflux_ghostwriter_v21" # Keeping for future Qdrant re-integration

    EMBEDDING_MODEL_NAME: str = "BAAI/bge-small-en-v1.5"
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    RATE_LIMIT: str = "30/minute"
    REQUEST_TIMEOUT: int = 300 # Increased for very long reports

    RERANK_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2" # Recommended for re-ranking
    RERANK_TOP_N: int = 5 # Number of top results to return after re-ranking

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
        # self.qdrant_client: Optional[qdrant_client.QdrantClient] = None # REMOVED
        # self.async_qdrant_client: Optional[AsyncQdrantClient] = None # REMOVED
        self.rag_query_engine: Optional[RetrieverQueryEngine] = None
        self.indexing_status: Dict[str, Any] = {"status": IndexingStatus.IDLE, "message": "Ready"}
        self.gemini_breaker = CircuitBreaker(fail_max=3, reset_timeout=60, name="gemini")
        self.ollama_breaker = CircuitBreaker(fail_max=3, reset_timeout=60, name="ollama")
        # Store a reference to the in-memory vector store for re-indexing
        self._in_memory_vector_store: Optional[SimpleVectorStore] = None

    async def startup(self):
        logger.info("Starting ResourceManager", device=self.config.DEVICE)
        self.http_client = httpx.AsyncClient(timeout=self.config.REQUEST_TIMEOUT)
        for path in [self.config.PERSIST_DIR, self.config.KNOWLEDGE_BASE_DIR]: os.makedirs(path, exist_ok=True)
        
        Settings.embed_model = FastEmbedEmbedding(model_name=self.config.EMBEDDING_MODEL_NAME)
        
        # REMOVED Qdrant client initialization
        # self.qdrant_client = qdrant_client.QdrantClient(...)
        # self.async_qdrant_client = AsyncQdrantClient(...)

        llama_index_gemini_llm = Gemini(api_key=self.config.GOOGLE_API_KEY, model_name="models/gemini-1.5-flash-latest")
        Settings.llm = llama_index_gemini_llm 

        await self.load_rag_engine()

    async def shutdown(self):
        if self.http_client: await self.http_client.aclose()
        # REMOVED Qdrant async client close
        # if self.async_qdrant_client: await self.async_qdrant_client.close()

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
    # ... (No changes)
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
    # NEW: Check if rag_query_engine is actually initialized from in-memory index
    if not rm.rag_query_engine: return "Local knowledge base (Soul) is not available or not yet indexed."
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
        The entire context and research plan should be focused on answering and elaborating on the provided query. Ensure research tasks explicitly target finding:
        - Quantifiable data, statistics, or percentages relevant to the query.
        - Specific, named open-source tools, frameworks, or projects that address aspects of the query.
        - Real-world examples or case studies if applicable."""

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
            else: # Defaults to rag_search_tool
                tasks_to_run.append(run_rag_search_tool(rm, query_for_tool, filters=None)) # Filters are a placeholder for future implementation

        results = await asyncio.gather(*tasks_to_run, return_exceptions=True)
        research_log = f"Genesis Context:\n{json.dumps(genesis_plan, indent=2)}\n\n"
        research_log += "".join([f"--- Research Result ---\n{res}\n\n" for res in results])

        # === Phase 3: Synthesis & Insight (Mind) ===
        yield f'data: {json.dumps({"type": "log", "content": "Phase 3: Synthesizing novel insights...", "class": "phase-title"})}\n\n'
        synthesis_prompt = f"""You are a senior analyst and theorist. Based on the context and research data below, write a definitive 'Intelligence Briefing' as a JSON object. Your mission is to generate novel, synthesized insights that directly answer and elaborate on the query: '{request.query}'. Your JSON must contain:
        'title' (directly addressing the query),
        'executive_summary' (summarizing the key answer and benefits. **ONLY include quantifiable impacts/benefits if explicitly verifiable from the 'research_log'. Otherwise, use qualitative descriptions like 'significant improvement'.**),
        'introduction_and_context' (setting the stage for the query's topic),
        'core_analysis' (a deep dive into the topic, providing 'why' explanations. **ONLY mention specific open-source tools/frameworks/solutions if explicitly verifiable from the 'research_log'. Otherwise, use general terms like 'various frameworks'.**),
        'novel_synthesis' (a JSON object with keys: 'name' (a concise name for the novel insight or core synthesis derived from the query's elaboration), and 'description' (a detailed explanation of the concept/synthesis). If a novel concept emerges, name it specifically; otherwise, use a generic name like 'Key Insights' or 'Core Synthesis'. Do NOT invent a new concept or filter name like 'Chronos Filter' unless the research explicitly supports it),
        'perspectives' (an object with 'utopian_view' and 'dystopian_view' related to the query's topic. **ONLY include quantifiable impacts if explicitly verifiable from the 'research_log'.**),
        'verifiable_quotes' (an array of real quotes from the data),
        'verifiable_sources' (an array of real links from web search, if any. Ensure these are actual, complete URLs or publication details.),
        and 'mermaid_code' (a simple, syntactically correct flowchart for a process or architecture related to the query's answer. **It MUST start with 'graph TD', 'graph LR', 'sequenceDiagram', 'flowchart TD', or similar valid Mermaid syntax keyword. If no relevant diagram, provide an empty string.**).
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

        **CRITICAL INSTRUCTION: ALL FACTS, STATISTICS, OPEN-SOURCE TOOLS, AND EXAMPLES MUST BE STRICTLY DERIVED FROM THE PROVIDED 'INTELLIGENCE BRIEFING'. DO NOT INVENT OR HALLUCINATE ANY INFORMATION. IF THE BRIEFING DOES NOT CONTAIN A SPECIFIC DETAIL (e.g., A PRECISE PERCENTAGE, THE NAME OF A TOOL), YOU MUST EITHER USE QUALITATIVE LANGUAGE (e.g., "significant improvement") OR STATE THAT SUCH SPECIFIC DATA WAS NOT IDENTIFIED IN THE RESEARCH.**

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
            *   `<header>` (with `<h1>` for main title derived from `intelligence_briefing['title']` and `<p class="subtitle">` for a relevant subtitle, dynamically generated to reflect the scholarly depth of the report's actual topic.)
            *   `<section class="report-section">` for `<h2>Abstract</h2>` (Generated based on the briefing's summary, capturing main points, problem, solution, benefits, and impact. Strictly adhere to briefing for quantifiable impacts.)
            *   `<section class="report-section">` for `<h2>Executive Summary</h2>` (Expand on `intelligence_briefing['executive_summary']`. Include plausible, general quantifiable impacts/benefits **only if found in the briefing**. Otherwise, use qualitative terms.)
            *   `<section class="report-section">` for `<h2>Introduction and Context</h2>` (Expand on `intelligence_briefing['introduction_and_context']`. Strengthen narrative, provide specific examples of issues/context from briefing. )
            *   `<section class="report-section">` for `<h2>Related Work</h2>` (This section is essential. Discuss existing literature, concepts, and relevant open-source tools related to the *report's actual topic*. Explain how the report's content builds upon/differentiates itself. **ONLY mention specific names of tools/frameworks/research if they are explicitly present in the `intelligence_briefing`'s research findings.** Use placeholder citations like `[1]`, `[2]`, `[3]` etc. as you discuss.)
            *   `<section class="report-section">` for `<h2>Core Analysis</h2>` (Expand on `intelligence_briefing['core_analysis']`. Provide detailed 'why' explanations. **ONLY mention specific open-source tools/solutions or methodologies if explicitly present in the `intelligence_briefing`'s research findings.** Use placeholder citations `[N]`.)
            *   `<section class="report-section">` for `<h2>Novel Synthesis: {intelligence_briefing.get('novel_synthesis', {}).get('name', 'Core Insights')}</h2>` (Expand on `intelligence_briefing['novel_synthesis'].get('description', '')`. If it describes a concept, detail its architecture/functionality based strictly on the briefing. If it's a general synthesis of ideas, elaborate on those insights. **If `intelligence_briefing.get('mermaid_code')` and it starts with a valid Mermaid keyword (e.g., 'graph', 'sequence', 'flowchart'), include it here in `<pre class="mermaid">` tags; otherwise, display a "Diagram not available." message.**)
            *   `<section class="report-section">` for `<h2>Evaluation and Results</h2>` (If the briefing contains quantifiable results or implies evaluation, expand on this. **Create hypothetical tables with metrics/percentages *ONLY if the briefing explicitly suggests or provides such quantification*. Otherwise, discuss qualitative benefits or challenges in this section.** Use placeholder citations `[N]`.)
            *   `<section class="report-section">` for `<h2>Perspectives</h2>` (with `<h3>` for Utopian/Dystopian. Expand on `intelligence_briefing['perspectives']`. Include plausible, general quantifiable impacts **only if found in the briefing**.)
            *   `<section class="report-section">` for `<h2>Actionable Recommendations</h2>` (Expand on `intelligence_briefing`'s recommendations. Provide highly specific, actionable, and technically detailed recommendations relevant to the *report's actual topic*. Quantify targets **only if found in the briefing**. Mention relevant open-source tools/solutions or best practices **only if found in the briefing**. Use placeholder citations `[N]`.)
            *   `<section class="report-section">` for `<h2>Conclusion & Future Work</h2>` (Summarize the main contribution and significance. Outline specific, forward-looking areas for "Future Work" relevant to the *report's actual topic*. Use placeholder citations `[N]`.)
            *   `<section class="report-section">` for `<h2>References</h2>` (This is CRITICAL. Create a formal, numbered list of references. **For each `[N]` citation used in the text, provide a plausible, formatted academic reference (e.g., using a consistent style like IEEE or APA) related to the *actual topic of the report*. IMPORTANT: ONLY include references that are present in the `intelligence_briefing['verifiable_sources']` array.** If a concept is discussed and no direct source for it is in `verifiable_sources`, do NOT invent a reference. **If `verifiable_sources` is empty, this section should simply state "No direct verifiable sources were identified in the research."** Use `<ul class="references-list">` for this section. Ensure links are clickable if provided in sources.)
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

            # When text_splitter is removed, SimpleDirectoryReader directly returns Document objects.
            # VectorStoreIndex.from_documents will handle the default splitting internally
            nodes = documents 
            
            if not nodes: rm_instance.indexing_status["message"] = "No nodes generated from documents."; return

            rm_instance.indexing_status["message"] = f"Found {len(documents)} documents. Building vector index..."
            # Use SimpleVectorStore for in-memory indexing as a fallback for Qdrant issues
            vector_store = SimpleVectorStore() # In-memory store
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            # Use from_documents with the raw Document objects; LlamaIndex will apply default chunking
            VectorStoreIndex.from_documents(nodes, storage_context=storage_context, show_progress=True)
            
            # After successful indexing, set the query engine for the resource manager
            rm_instance.rag_query_engine = VectorStoreIndex.from_vector_store(vector_store).as_query_engine(similarity_top_k=settings.RERANK_TOP_N * 3)
            
            rm_instance.indexing_status["message"] = "Index built successfully (in-memory)."
            logger.info("✅ RAG 'Soul' initialized (in-memory).")
        except Exception as e:
            rm_instance.indexing_status = {"status": IndexingStatus.ERROR, "message": f"Indexing failed: {e}"}
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
