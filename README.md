# Persona Twin

> An AI-powered digital persona system that transforms a person's documents, web content, and other source material into a searchable knowledge base and conversational AI persona.

**Persona Twin** is a Python-based system designed to create an AI representation of a person from their existing information. It extracts and processes personal content, builds a **persona knowledge graph**, creates **vector embeddings for semantic search**, and uses an LLM to generate responses grounded in the persona's information.

---

## Features

* Multi-source data ingestion

  * PDF documents
  * DOCX documents
  * Images
  * Web pages and URLs
  * Directories containing supported files

* Automatic text extraction and chunking

  * Converts raw content into manageable chunks
  * Stores processed data in JSONL format

* Persona Knowledge Graph

  * Extracts relationships and information about the persona
  * Uses Neo4j for graph-based storage

* Semantic Search

  * Generates embeddings from extracted content
  * Stores and searches vectors using Qdrant

* LLM-powered Persona

  * Uses OpenRouter-compatible LLM APIs
  * Retrieves relevant persona information before generating responses

* Persona-aware conversations

  * Responses can be grounded in the persona's stored knowledge

* Command-line interface

  * Run ingestion, graph construction, embedding, search, and debugging operations from the terminal

* FastAPI backend

  * Provides the foundation for serving the Persona Twin as an API

---

## Architecture

Persona Twin follows a multi-stage pipeline:

```text
                    +----------------------+
                    |      Raw Sources     |
                    |                      |
                    | PDF / DOCX / Images  |
                    | URLs / Web Content   |
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    |   Data Ingestion     |
                    |                      |
                    | Extraction + Chunking|
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    |      JSONL Data      |
                    |   Processed Chunks   |
                    +-------+-------+------+
                            |       |
                 +----------+       +----------+
                 |                             |
                 v                             v
       +-------------------+         +-------------------+
       |  Persona Graph    |         |   Vector Store    |
       |                   |         |                   |
       |      Neo4j        |         |      Qdrant       |
       +---------+---------+         +---------+---------+
                 |                             |
                 +--------------+--------------+
                                |
                                v
                     +----------------------+
                     |    Persona Engine    |
                     |                      |
                     | Context + Retrieval  |
                     +----------+-----------+
                                |
                                v
                     +----------------------+
                     |         LLM          |
                     |                      |
                     |  Persona Response    |
                     +----------------------+
```

---

## Project Structure

```text
Persona-Twin/
│
├── app/
│   ├── graph/          # Persona knowledge graph functionality
│   ├── ingestion/      # Document, image and URL ingestion
│   ├── llm/            # LLM/OpenRouter integration
│   ├── persona/        # Persona logic and response generation
│   ├── static/         # Static application resources
│   ├── storage/        # Data/storage utilities
│   ├── vector/         # Embedding and Qdrant functionality
│   │
│   ├── cli.py          # Command-line interface
│   ├── main.py         # FastAPI application
│   └── __init__.py
│
├── data/
│   ├── output/         # Processed/generated data
│   ├── personas/       # Persona-related data
│   └── raw/            # Raw input material
│
├── requirements.txt    # Python dependencies
└── README.md
```

---

## Tech Stack

| Technology    | Purpose                   |
| ------------- | ------------------------- |
| Python        | Core programming language |
| FastAPI       | Backend/API server        |
| Uvicorn       | ASGI server               |
| OpenRouter    | LLM API access            |
| Neo4j         | Persona knowledge graph   |
| Qdrant        | Vector database           |
| FastEmbed     | Text embedding generation |
| PyPDF         | PDF extraction            |
| python-docx   | DOCX processing           |
| BeautifulSoup | HTML parsing              |
| Trafilatura   | Web content extraction    |
| Requests      | HTTP requests             |
| python-dotenv | Environment configuration |

---

# Getting Started

## 1. Clone the Repository

```bash
git clone https://github.com/Abhay-dev-hub/Persona-Twin.git
cd Persona-Twin
```

---

## 2. Create a Virtual Environment

### Windows

```powershell
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Configure Environment Variables

Create a `.env` file in the project root.

Example:

```env
OPENROUTER_API_KEY=your_openrouter_api_key

NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_neo4j_password

QDRANT_URL=http://localhost:6333
```

> Never commit your `.env` file or API keys to GitHub.

---

# Data Ingestion

Persona Twin can ingest different types of source material and convert them into structured chunks.

The CLI supports:

```text
File
URL
Directory
```

### Process a File

```bash
python -m app.cli file path/to/document.pdf
```

### Process an Image

```bash
python -m app.cli file path/to/image.jpg
```

### Process a URL

```bash
python -m app.cli url https://example.com/article
```

### Process a Directory

```bash
python -m app.cli dir data/raw
```

The processed content is written to JSONL data, which can then be used by the graph and vector stages.

---

# Build the Persona Knowledge Graph

After generating the processed chunks:

```bash
python -m app.cli graph --persona "Your Name" data/output/chunks.jsonl
```

This stage extracts persona-related information and stores it in **Neo4j**.

The graph can represent information such as:

```text
Persona
   |
   +-- Interests
   +-- Skills
   +-- Education
   +-- Projects
   +-- Experiences
   +-- Preferences
   +-- Relationships
```

---

# Generate Embeddings

Create vector embeddings for the processed chunks:

```bash
python -m app.cli embed --collection your_persona data/output/chunks.jsonl
```

The embeddings are stored in **Qdrant**, allowing the system to perform semantic similarity searches.

---

# Semantic Search

Search the persona's knowledge base:

```bash
python -m app.cli search \
  --collection your_persona \
  "What does this person like?"
```

This can be used to test whether relevant information is being retrieved from the vector database.

---

# Debug Persona Prompts

Before sending a request to the LLM, you can inspect the generated persona prompt:

```bash
python -m app.cli debug-prompt \
  --persona "Your Name" \
  --collection your_persona \
  "What do you think about your hobbies?"
```

This is useful for debugging retrieval and persona-context construction.

---

# Running the API

Persona Twin includes a FastAPI backend.

Start the development server with:

```bash
uvicorn app.main:app --reload
```

The API will normally be available at:

```text
http://127.0.0.1:8000
```

FastAPI also provides interactive API documentation at:

```text
http://127.0.0.1:8000/docs
```

---

# Complete Pipeline

A typical workflow looks like this:

```bash
# 1. Extract source data
python -m app.cli dir data/raw

# 2. Build the persona graph
python -m app.cli graph \
  --persona "Your Name" \
  data/output/chunks.jsonl

# 3. Generate embeddings
python -m app.cli embed \
  --collection your_persona \
  data/output/chunks.jsonl

# 4. Test retrieval
python -m app.cli search \
  --collection your_persona \
  "Tell me about this person's interests"

# 5. Debug the generated persona prompt
python -m app.cli debug-prompt \
  --persona "Your Name" \
  --collection your_persona \
  "Tell me about yourself"

# 6. Start the API
uvicorn app.main:app --reload
```

---

# How It Works

## 1. Ingestion

Raw information is collected from documents, images, URLs, and other supported sources.

## 2. Chunking

Large documents are divided into smaller pieces so that individual pieces of information can be processed and retrieved efficiently.

## 3. Knowledge Graph

Important relationships and persona information are extracted and stored in Neo4j.

## 4. Embeddings

Source chunks are converted into vector representations using FastEmbed.

## 5. Retrieval

When a user asks a question, relevant information can be retrieved from Qdrant.

## 6. Persona Context

Retrieved information and persona graph information are combined to construct contextual information for the LLM.

## 7. Response Generation

The LLM uses the retrieved context to generate a response representing the configured persona.

---

# Goals

The project aims to explore how an AI system can represent a person using their existing digital information while maintaining a structured connection between generated responses and source knowledge.

Potential applications include:

* Personal AI assistants
* Digital persona systems
* Personal knowledge management
* Interactive portfolios
* AI-powered personal profiles
* Research into AI memory and persona modeling
* Knowledge-grounded conversational agents

---

# Security and Privacy

Persona Twin can process highly personal information, so security should be considered carefully.

### Never commit:

```text
.env
API keys
Passwords
Private documents
Personal credentials
Private database credentials
```

Use environment variables for secrets and keep sensitive source material outside public repositories.

---

# Development

The project is organized into independent modules so individual parts of the pipeline can be developed and tested separately.

```text
app/
├── ingestion/
├── graph/
├── llm/
├── persona/
├── storage/
└── vector/
```

The CLI provides a convenient way to test individual pipeline stages without running the entire application.

---

# Project Status

**Active Development**

Persona Twin is currently a work-in-progress project. APIs, data structures, retrieval behavior, and persona-generation functionality may change as development continues.

---

# Future Improvements

* [ ] Improved persona consistency
* [ ] Better source attribution
* [ ] Conversation memory
* [ ] More ingestion formats
* [ ] Improved graph extraction
* [ ] Hybrid graph + vector retrieval
* [ ] Streaming LLM responses
* [ ] Authentication and user management
* [ ] Web-based persona management interface
* [ ] Docker deployment
* [ ] Automated testing
* [ ] Evaluation framework for persona accuracy
* [ ] Better handling of conflicting information
* [ ] Multi-persona support

---

# Contributing

Contributions, suggestions, and improvements are welcome.

1. Fork the repository.
2. Create a feature branch.

```bash
git checkout -b feature/your-feature
```

3. Make your changes.
4. Commit your changes.

```bash
git commit -m "Add your feature"
```

5. Push the branch.

```bash
git push origin feature/your-feature
```

6. Open a Pull Request.

---

# License

Add your preferred license to the repository before publishing a final release.

---

# Author

**Abhay Waghamode**

GitHub: https://github.com/Abhay-dev-hub

---

# Support

If you find Persona Twin interesting, consider giving the repository a star on GitHub.

Repository:

https://github.com/Abhay-dev-hub/Persona-Twin

---

> Persona Twin — turning personal information into an interactive AI persona.
