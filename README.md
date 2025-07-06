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
Clone the Repository:
Generated bash
git clone https://github.com/your-username/neuroflux-agrag.git
cd neuroflux-agrag
Use code with caution.
Bash
(Replace your-username/neuroflux-agrag.git with the actual repository URL once you upload it.)
Set Up Python Virtual Environment:
It's highly recommended to use a virtual environment to manage dependencies.
Generated bash
python -m venv venv
Use code with caution.
Bash
On Linux/macOS:
Generated bash
source venv/bin/activate
Use code with caution.
Bash
On Windows:
Generated bash
.\venv\Scripts\activate
Use code with caution.
Bash
Install Python Dependencies:
Generated bash
pip install -r requirements.txt
Use code with caution.
Bash
Configure Environment Variables:
Create a .env file in the root directory of the project (where main.py is located) and populate it with your credentials and settings.
Copy the .env.example to .env:
Generated bash
cp .env.example .env
Use code with caution.
Bash
Then, edit the .env file:
Generated code
# --- Required API Keys ---
GOOGLE_API_KEY="YOUR_GOOGLE_GEMINI_API_KEY"

# --- Ollama Configuration (defaults to localhost:11434, change if different) ---
OLLAMA_API_BASE="http://localhost:11434"

# --- Google Custom Search Engine (Optional, for web_search tool) ---
# GOOGLE_CSE_ID="YOUR_GOOGLE_CSE_ID"

# --- PostgreSQL Database Configuration ---
POSTGRES_HOST="localhost"
POSTGRES_PORT=5432
POSTGRES_USER="your_pg_user"
POSTGRES_PASSWORD="your_pg_password"
POSTGRES_DB="your_pg_database_name"

# --- Optional: Define your PostgreSQL schema for validation ---
# This is critical for the LLM to understand what tables/columns it can query.
# The default in main.py is:
# POSTGRES_SCHEMA_DEFINITION={"users": ["id", "name", "email", "signup_date", "age"], "products": ["id", "name", "price", "category"], "orders": ["id", "user_id", "product_id", "quantity", "order_date"]}
# If you change your schema, update this variable in .env or main.py.
Use code with caution.
IMPORTANT: Never commit your .env file to version control. It contains sensitive API keys. .gitignore should already exclude it.
Prepare Knowledge Base (Optional but Recommended):
Create a knowledge_docs directory in the project root. Place any .pdf, .txt, .md, etc., files you want the "Soul" to draw knowledge from into this directory.
Generated bash
mkdir knowledge_docs
# cp your_documents/* knowledge_docs/
Use code with caution.
Bash
Run the FastAPI Application:
Generated bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
Use code with caution.
Bash
The --reload flag is useful for development as it automatically restarts the server when code changes are detected.
💻 Usage
Access the UI:
Open your web browser and navigate to http://localhost:8000.
Initialize Models:
The "Ghostwriter" (Target Model) dropdown should auto-populate with available Ollama models. If it shows "Connection failed," ensure Ollama is running and accessible. You can click the 🔄 (refresh) button to retry fetching models.
The "Mind" (Drafter Model) dropdown is pre-configured with Gemini models.
Index Knowledge Base (If Applicable):
If you've placed documents in knowledge_docs/, click the "Index Knowledge Base" button. This will build an in-memory RAG index. The status will update in the UI.
Enter Your Strategic Query:
In the large text area at the bottom, enter a detailed strategic query. Examples:
"Discuss the ethical challenges of using LLMs in critical decision-making processes, particularly when combining data from diverse sources (like structured databases and unstructured documents). Name at least two open-source frameworks or methodologies designed to address these ethical challenges in AI applications, and explain their role."
"Analyze the privacy implications of integrating user data from an internal PostgreSQL database (specifically focusing on user age, email, and signup date) into LLM-driven decision-making processes. Provide quantifiable insights derived from this data to illustrate potential biases or risks. Furthermore, identify and describe at least two open-source frameworks or methodologies designed for ethical data governance and privacy protection in AI applications."
Execute the Agent:
Click the ▶️ (Send/Execute) button. The "Agent's Thought Process" log at the top will show real-time updates as the agent executes its phases: Genesis (Mind planning), Research (Soul & Web tool calls), Synthesis (Mind insight generation), and Ghostwriting (Voice report generation).
Export the Report:
Once the agent completes its process, the ⬇️ (Download/Export) button will become active. Click it to download the generated HTML report to your local machine.
📁 Project Structure
Generated code
neuroflux-agrag/
├── main.py                   # FastAPI backend, LLM orchestration, RAG, PostgreSQL integration
├── requirements.txt          # Python dependencies
├── .env.example              # Template for environment variables (copy to .env and fill)
├── knowledge_docs/           # Directory for your local knowledge base documents (PDFs, TXTs, etc.)
│   └── <your_documents_here>
└── frontend/
    ├── index.html            # Main HTML UI structure
    ├── style.css             # Frontend styling (CSS)
    └── app.js                # Frontend interactivity (JavaScript)
Use code with caution.
🤝 Contributing
Contributions are welcome! If you have suggestions for improvements, bug fixes, or new features, please open an issue or submit a pull request.
📄 License
This project is open-source and available under the MIT License.
🙏 Acknowledgements
Google Gemini: For powerful large language models driving the "Mind" component.
Ollama: For enabling local, open-source LLM inference for the "Voice" component.
FastAPI: For the robust and efficient backend framework.
LlamaIndex: For the RAG framework.
Asyncpg & SQLGlot: For asynchronous PostgreSQL interaction and SQL validation.
Feather Icons: For the clean and modern UI icons.
