# Financial Analytics — Application README

## Project Overview

**DualLens Analytics** is a Retrieval-Augmented Generation (RAG) powered investment analysis system that combines **quantitative financial data** (stock prices, market cap, P/E ratio) with **qualitative AI initiative insights** (extracted from company PDF reports) to score and rank companies for investment potential.

The system analyzes five major tech companies: **GOOGL, MSFT, IBM, NVDA, AMZN**.

---

## Architecture & Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    DualLens Analytics                        │
├─────────────────┬───────────────────────────────────────────┤
│  Quantitative   │           Qualitative (RAG)               │
│  (YFinance)     │   PDF → Chunks → Embeddings → ChromaDB   │
├─────────────────┴───────────────────────────────────────────┤
│         LLM Synthesis (gpt-4o-mini via LangChain)           │
├─────────────────────────────────────────────────────────────┤
│              Scoring, Ranking & Recommendation              │
└─────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| LLM | OpenAI `gpt-4o-mini` | Generation, scoring, evaluation |
| LLM Framework | LangChain `0.3.25` | Prompt management, document loading, text splitting |
| Embeddings | OpenAI `text-embedding-ada-002` | Vectorizing PDF document chunks |
| Vector Store | ChromaDB `1.3.4` | Storing and retrieving document embeddings |
| PDF Processing | PyPDF `5.4.0` + LangChain `PyPDFDirectoryLoader` | Loading and parsing company AI initiative PDFs |
| Text Splitting | LangChain `RecursiveCharacterTextSplitter` | Chunking PDFs using tiktoken `cl100k_base` encoding |
| Financial Data | YFinance | Fetching stock prices and financial metrics |
| Visualization | Matplotlib + Pandas | Plotting stock trends and financial metric comparisons |
| Environment | Google Colab | Notebook execution platform |

---

## LLM Features & Configuration

### Model Configuration
```python
ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,          # Deterministic output for consistency
    max_tokens=5000,        # Extended output for detailed analysis
    top_p=0.95,             # Near-full nucleus sampling
    frequency_penalty=1.2,  # Reduces repetitive phrasing
    stop_sequences=['INST'] # Custom stop token for prompt formatting
)
```

### LLM Usage Across the Pipeline

1. **RAG Question Answering** — The LLM answers questions about company AI initiatives using retrieved context from ChromaDB. Uses `[INST]...[/INST]` prompt formatting with system + user message templates.

2. **LLM-as-Judge (Groundedness)** — A separate LLM instance evaluates whether the RAG answer is fully supported by the retrieved context. Rates on a 1–5 scale with JSON-structured output including score and reasoning.

3. **LLM-as-Judge (Relevance)** — Another LLM instance evaluates whether the RAG answer directly addresses the question asked. Also rates 1–5 with structured JSON output.

4. **Investment Scoring & Ranking** — The LLM acts as a senior financial analyst, receiving both the financial metrics DataFrame and the full AI initiative document corpus, then produces weighted scores (60% quantitative, 40% qualitative) and a final investment ranking.

---

## Pipeline Walkthrough

### 1. Organization Selection
Five companies selected for analysis: `GOOGL`, `MSFT`, `IBM`, `NVDA`, `AMZN`.

### 2. LLM Setup
- API credentials loaded from `config.json` containing `API_KEY` and `OPENAI_API_BASE`
- Credentials set as environment variables for LangChain's `ChatOpenAI`

### 3. Visualization & Insight Extraction
- **Stock Price Trends**: 3-year historical closing prices fetched via `yf.Ticker(symbol).history(period="3y")` and plotted with Matplotlib
- **Financial Metrics**: Market Cap, P/E Ratio, Dividend Yield, Beta, and Total Revenue fetched from `ticker.info` and compiled into a Pandas DataFrame
- Market Cap and Total Revenue converted to billions for readability
- Each metric visualized as a separate bar chart for cross-company comparison

### 4. RAG-Driven Analysis

#### A. Document Loading
- AI initiative PDFs extracted from `Companies-AI-Initiatives.zip`
- Loaded using `PyPDFDirectoryLoader` from LangChain

#### B. Text Splitting & Vectorization
- Documents split using `RecursiveCharacterTextSplitter` with tiktoken `cl100k_base` encoding
  - `chunk_size=500` tokens
  - `chunk_overlap=100` tokens
- Produced **118 chunks** total across all company documents
- Chunks embedded using `text-embedding-ada-002` and stored in ChromaDB with collection name `AI_Initiatives`

#### C. Retrieval & Generation
- Retriever configured for `similarity` search returning top `k=10` chunks per query
- RAG function built with:
  - System message: instructs LLM to use ONLY provided context
  - User message template: injects context and question
  - `[INST]...[/INST]` prompt wrapping
- Test queries executed covering IBM projects, GOOGL vs MSFT comparison, NVDA timelines, AMZN AI investments, and GOOGL risks

#### D. Evaluation (LLM-as-Judge)
- **Groundedness evaluation**: Scored **5/5** — every claim in the answer was directly supported by the retrieved context
- **Relevance evaluation**: Scored **5/5** — the answer directly and completely addressed the question asked

### 5. Scoring & Ranking
- Combined financial DataFrame (`df.to_string()`) and full AI initiative documents (`vectorstore.get()['documents']`) into a single prompt
- LLM scored each company on quantitative (1–10) and qualitative (1–10) factors
- Combined weighted score: `(0.6 × Quantitative) + (0.4 × Qualitative)`
- **Final ranking**: GOOGL → MSFT → NVDA → AMZN → IBM

### 6. Summary & Future Scope
- Observations on combining quantitative + qualitative analysis
- RAG pipeline reliability validated through evaluation scores
- Data preprocessing challenges documented (dividend yield conversion)
- Future improvements: real-time sentiment analysis, expanded evaluation metrics, agentic workflows

---

## Key LLM Patterns Used

| Pattern | Description |
|---|---|
| **RAG (Retrieval-Augmented Generation)** | Grounds LLM responses in actual document content rather than relying on training data |
| **LLM-as-Judge** | Uses a separate LLM to evaluate the quality of another LLM's output (groundedness + relevance) |
| **System/User Prompt Separation** | Distinct system instructions and user queries for controlled behavior |
| **Structured Output** | JSON-formatted evaluation responses with score + reasoning |
| **Multi-Source Synthesis** | LLM integrates quantitative data (DataFrame) with qualitative data (retrieved documents) in a single analysis |
| **Temperature=0 Determinism** | Ensures reproducible, consistent outputs across runs |
| **Frequency Penalty** | Reduces repetitive language in generated analysis |

---

## File Structure

```
├── config.json                      # API credentials (API_KEY + OPENAI_API_BASE)
├── Companies-AI-Initiatives.zip     # Zipped PDF reports for 5 companies
├── Companies-AI-Initiatives/        # Extracted PDF directory
│   ├── GOOGL_AI_Initiatives.pdf
│   ├── MSFT_AI_Initiatives.pdf
│   ├── IBM_AI_Initiatives.pdf
│   ├── NVDA_AI_Initiatives.pdf
│   └── AMZN_AI_Initiatives.pdf
├── Stock_Price_Trends_3Y.png        # Generated stock chart
└── DualLens_Analytics.ipynb         # Main notebook
```

---

## How to Run

1. Upload `config.json` with valid API credentials to Google Colab
2. Upload `Companies-AI-Initiatives.zip` to Google Colab
3. Run the installation cell (Cell 1) — then **restart the runtime**
4. Execute all remaining cells sequentially from top to bottom
5. Export to HTML: `!jupyter nbconvert --to html "notebook_name.ipynb"`

---

## Dependencies

```
langchain==0.3.25
langchain-core==0.3.65
langchain-openai==0.3.24
chromadb==1.3.4
langchain-community==0.3.20
pypdf==5.4.0
yfinance (pre-installed in Colab)
matplotlib (pre-installed in Colab)
pandas (pre-installed in Colab)
```
