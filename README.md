NeuroFlux AGRAG (Ghostwriter Protocol)
🌌 Advanced Generative Research Agent - v21.0


![image](https://github.com/user-attachments/assets/26600f57-5b07-40ad-9576-cc653b773b43)

NeuroFlux AGRAG, powered by the "Ghostwriter Protocol," is an innovative AI agent designed to generate deep, insightful, and polished "white paper" style reports. It leverages a sophisticated "Trinity" architecture to combine strategic planning, robust knowledge retrieval, and professional content synthesis, enabling it to tackle complex queries with multi-faceted information needs.
✨ Features
"Trinity" Architecture: A powerful collaboration between three core AI components:
Mind (Strategist): Powered by Google Gemini, it creates a comprehensive narrative plan, synthesizes research findings, and generates novel insights.
Soul (Memory): Facilitates Retrieval-Augmented Generation (RAG) from a local document knowledge base (in-memory SimpleVectorStore) and integrates structured data from an asynchronous PostgreSQL database.
Voice (Ghostwriter): Utilizes Ollama-compatible local LLMs (e.g., Mistral, Llama3) to expand the synthesized briefing into a full, polished, HTML-formatted research report.
Hybrid Data Integration: Seamlessly combines information from:
Web Search: For general internet knowledge and current trends.
Local Knowledge Base (RAG): For in-depth insights from your private documents.
PostgreSQL Database: For querying structured, transactional data with AI-generated and validated SQL queries.
AI-Powered SQL Generation & Validation: The agent can dynamically generate and validate SQL SELECT queries against a defined database schema, executing them and incorporating results into its research.
Real-time Insights: Provides a live log of the agent's thought process, showing each phase of research and synthesis.
Exportable Reports: Generated white papers can be easily exported as complete HTML files.
Production-Ready Backend: Built with FastAPI, featuring rate limiting, circuit breakers, and asynchronous operations for high performance and resilience.
Sleek, Professional UI: A refined, dark greyscale user interface inspired by the "Orbe" aesthetic, designed for intuitive control and clear presentation of results.
🧠 Architecture Overview
The NeuroFlux AGRAG operates on a "Trinity" principle, where distinct AI roles collaborate:
Mind (Google Gemini):
Receives the initial strategic query.
Generates a research_plan that outlines necessary information gathering tasks.
Chooses appropriate tools (web_search, vector_database_search, postgres_query) for each task.
Synthesizes all gathered research into a structured "Intelligence Briefing" (JSON format), identifying core analysis, novel insights, and relevant perspectives.
Soul (LlamaIndex RAG & AsyncPG):
Executes vector_database_search queries against an in-memory LlamaIndex vector store (populated from your knowledge_docs). Includes re-ranking for improved relevance.
Manages postgres_query requests: leverages Gemini to generate SQL, validates it against a predefined schema, executes it via asyncpg, and returns formatted results.
Voice (Ollama LLM):
Receives the complete "Intelligence Briefing" from the Mind.
Acts as the "Ghostwriter," expanding the briefing's content into a full, detailed, and stylistically consistent HTML report, strictly adhering to the provided facts and insights.
⚙️ Prerequisites
Before you begin, ensure you have the following installed and configured:
Python: Version 3.9 or higher.
Ollama: Download and install Ollama from ollama.com.
After installation, pull the required models. The system attempts to select optimal models (llama3:8b-instruct, command-r, mistral, llama3), but mistral:latest is preferred for the "Ghostwriter."
To pull mistral: ollama pull mistral
To pull llama3: ollama pull llama3
PostgreSQL: A running PostgreSQL instance.
You'll need a database and a user with access.
The main.py defines a sample schema (users, products, orders). You should create these tables in your database if you want to test the postgres_query tool effectively.
Google Cloud Account:
Gemini API Key: Required for the "Mind" component. Enable the Gemini API in your Google Cloud Project.
Google Custom Search Engine (CSE) ID (Optional): Required if you want the web_search tool to function. Configure a CSE and link it to your Google API Key.
Git: For cloning the repository.
🚀 Installation & Setup
Follow these steps to get NeuroFlux AGRAG up and running:
