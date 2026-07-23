# Semantic-Search-with-Embeddings-and-Chroma
A small semantic search system built with sentence embeddings and a persistent vector database (Chroma). Demonstrates meaning-based document search — including metadata filtering — over a sample set of company policy documents.

## What this project demonstrates
 
- **Embeddings** — converting text into numeric vectors using `sentence-transformers`.
- **Semantic search** — finding the most relevant document for a natural-language question, even with little to no word overlap (e.g. "How many vacation days do I get?" correctly matches "Employees are entitled to 20 days of paid vacation per year.").
- **A real, persisted vector database** — using Chroma, so embeddings survive between separate script runs instead of living only in memory.
- **Metadata filtering** — narrowing search results by fields like department or document type, both in plain Python and natively inside Chroma.
## Setup
 
1. Install dependencies:
```bash
   pip install sentence-transformers chromadb
```
2. The embedding model (`all-MiniLM-L6-v2`) downloads automatically on first run and is cached locally afterward — no API key required, everything runs on your own machine.
## Project structure
 
```
.
├── embed_demo.py       # Step-by-step exploration: embeddings, similarity,
│                        #   in-memory search, and in-memory metadata filtering
├── chroma_setup.py      # Builds the persistent vector database (run once)
├── chroma_search.py      # Connects to the existing database and runs
│                        #   filtered semantic searches
├── chroma_db/          # The persisted vector database (created by
│                        #   chroma_setup.py — do not edit by hand)
└── README.md
```
 
## How to run
 
Run these in order:
 
1. **Build the database** (only needs to be run once, or again if you change the document set):
```bash
   python chroma_setup.py
```
   This embeds the sample documents and saves them, along with their metadata, into the `chroma_db/` folder.
 
2. **Search it**:
```bash
   python chroma_search.py
```
   This reconnects to the existing `chroma_db/` folder (no re-embedding needed) and runs a few example searches, including filtered ones like "only Security documents" or "only documents of type roadmap".
 
`embed_demo.py` is the earlier, in-memory version used to build and test the core ideas before moving to Chroma — kept here as a reference for how the same search and filtering logic looks without a real database behind it.
 
## Example result
 
```
Question: How many vacation days do I get?
 
Distance: 0.6722
Text: Employees are entitled to 20 days of paid vacation per year.
Metadata: {'source': 'hr_policy_handbook.pdf', 'department': 'HR', 'date': '2024-01-15', 'doc_type': 'policy'}
```
 
Note: Chroma returns **distance**, not similarity — lower distance means a closer match (the opposite direction from the cosine similarity scores used in `embed_demo.py`).
 
## Metadata fields
 
Each document is tagged with:
- `source` — the originating file name
- `department` — HR, IT, Finance, Security, or Product
- `date` — when the document was written
- `doc_type` — policy, guide, checklist, or roadmap
These fields can be filtered on independently or combined (e.g. department **and** doc_type at once).
 
## Lessons learned
 
- Embedding models judge overall semantic theme, not just shared keywords — two sentences with an obvious shared word can score lower than two sentences with no words in common but a stronger shared topic.
- Cosine similarity (higher = more similar) and Chroma's distance metric (lower = more similar) run in opposite directions — easy to misread if switching between the two.
- Metadata never affects the embedding vector itself; it's stored and filtered separately, entirely independent of semantic meaning.
## Known limitations
 
- Only 8 sample documents — not tested at a scale where indexing performance would matter.
- No chunking logic — each document is a single short sentence rather than a split-up longer document.
- Filtering only covers exact-match fields (e.g. `department == "HR"`); no range filtering (e.g. date ranges) implemented yet.
 
