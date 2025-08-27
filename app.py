import os
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# LangChain bits
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, UnstructuredMarkdownLoader, CSVLoader
)
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate

load_dotenv()

# ----------------------------
# Config
# ----------------------------
DATA_DIR = os.getenv("DATA_DIR", "data")
DB_DIR = os.getenv("DB_DIR", "chroma_db")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-large")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")  # cheap+fast default

# ----------------------------
# Helpers
# ----------------------------
def make_embedder():
    return OpenAIEmbeddings(model=EMBED_MODEL)

def make_llm(temperature: float = 0.0):
    # If you want Azure, set environment variables and pass model/deployment accordingly.
    return ChatOpenAI(model=LLM_MODEL, temperature=temperature)

def split_docs(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1100,
        chunk_overlap=150,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(docs)

def file_to_docs(path: str) -> List[Document]:
    path_lower = path.lower()
    if path_lower.endswith(".pdf"):
        loader = PyPDFLoader(path)
    elif path_lower.endswith(".md") or path_lower.endswith(".markdown"):
        loader = UnstructuredMarkdownLoader(path, mode="elements")
    elif path_lower.endswith(".txt"):
        loader = TextLoader(path, encoding="utf-8")
    elif path_lower.endswith(".csv"):
        loader = CSVLoader(path)
    else:
        raise ValueError(f"Unsupported file type: {path}")
    return loader.load()

def get_or_create_vs(persist_dir: str = DB_DIR) -> Chroma:
    return Chroma(
        collection_name="rag-docs",
        embedding_function=make_embedder(),
        persist_directory=persist_dir,
    )

# ----------------------------
# Prompt (strictly grounded)
# ----------------------------
SYSTEM_PROMPT = """You are a careful assistant that answers ONLY using the provided context.
- If the answer is not fully supported by the context, say: "I don’t have enough information in the documents to answer that."
- Always cite sources as [source_name] with a short filename or title.
- Keep answers concise and factual.
"""

USER_PROMPT = """Question: {question}

Context:
{context}

Return:
- A short, precise answer grounded in the context.
- A bullet list of sources like [filename_or_title].
"""

prompt = ChatPromptTemplate.from_messages(
    [("system", SYSTEM_PROMPT), ("user", USER_PROMPT)]
)

# ----------------------------
# FastAPI app
# ----------------------------
app = FastAPI(title="RAG Grounded QA", version="1.0.0", docs_url="/")

class QueryIn(BaseModel):
    question: str
    k: int = 5
    score_threshold: Optional[float] = 0.0  # set >0 to filter weak matches

class QueryOut(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    used_chunks: int

@app.post("/ingest")
async def ingest(files: List[UploadFile] = File(...)):
    saved_paths = []
    for f in files:
        dest = os.path.join(DATA_DIR, f.filename)
        with open(dest, "wb") as out:
            out.write(await f.read())
        saved_paths.append(dest)

    # Load+split docs
    all_docs: List[Document] = []
    for p in saved_paths:
        try:
            docs = file_to_docs(p)
            all_docs.extend(docs)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to load {p}: {e}")

    chunks = split_docs(all_docs)

    # Index
    vs = get_or_create_vs()
    vs.add_documents(chunks)
    vs.persist()

    return {"status": "ok", "files_ingested": [os.path.basename(p) for p in saved_paths], "chunks": len(chunks)}

@app.post("/query", response_model=QueryOut)
async def query(q: QueryIn):
    vs = get_or_create_vs()
    retriever = vs.as_retriever(search_kwargs={"k": q.k})
    docs: List[Document] = retriever.get_relevant_documents(q.question)

    # Optional score filtering if metadata contains distances (Chroma returns scores separately via similarity_search_with_score)
    filtered_docs = []
    try:
        docs_scores = vs.similarity_search_with_score(q.question, k=q.k)
        for d, s in docs_scores:
            if q.score_threshold is None or s >= q.score_threshold:
                filtered_docs.append(d)
        if filtered_docs:
            docs = filtered_docs
    except Exception:
        pass  # fallback to docs

    if not docs:
        return QueryOut(
            answer="I don’t have enough information in the documents to answer that.",
            sources=[], used_chunks=0
        )

    # Build context text
    def short_name(meta: dict) -> str:
        # Try filename, title, or source
        for key in ("source", "file_path", "filename", "title"):
            if key in meta and meta[key]:
                return os.path.basename(str(meta[key]))
        return "document"

    context_blocks = []
    unique_sources = []
    for d in docs:
        src = short_name(d.metadata or {})
        if src not in unique_sources:
            unique_sources.append(src)
        # Include minimal metadata for grounding
        context_blocks.append(f"[{src}]\n{d.page_content}")

    context = "\n\n---\n\n".join(context_blocks[:q.k])

    llm = make_llm()
    chain = prompt | llm
    response = chain.invoke({"question": q.question, "context": context})

    # Collect sources (unique order-preserving)
    srcs = [{"name": s} for s in unique_sources]

    return QueryOut(
        answer=response.content.strip(),
        sources=srcs,
        used_chunks=min(len(docs), q.k),
    )

@app.post("/clear")
async def clear():
    # Danger: wipes the local vector store; useful during development
    import shutil
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)
    os.makedirs(DB_DIR, exist_ok=True)
    return {"status": "cleared"}
