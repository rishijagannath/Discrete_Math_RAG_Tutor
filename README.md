# RAG CS173 Tutor

A Retrieval-Augmented Generation system built to answer CS173 (Discrete Mathematics) questions at UIUC with course-specific accuracy, notation, and proof conventions.

## Try my chatbot out here!

https://discretemathragtutor.streamlit.app/

## Motivation

I used to use LLMs like ChatGPT and Claude for my discrete math questions - which both consistently hallucinated and struggled with CS173 questions. They lacked the course-specific context needed to give notation-consistent, assumption-accurate answers. This project grounds model responses in the official CS173 textbook to fix that.

## Tech Stack

- **LLM:** OpenAI O3-mini
- **Embeddings:** OpenAI text-embedding-3-large
- **Vector Store:** FAISS
- **Framework:** LangChain
- **Evaluation:** Manual Grading, secondary RAGAS testing
- **Deployment:** Streamlit on Hugging Face Spaces

## Rag Outline

PDFs
  ↓
Chunking
  ↓
Embeddings
  ↓
FAISS
  ↓
Retriever
  ↓
GPT
  ↓
Answer

## Results

| Model | Tuning Set (30Q) | Evaluation Set (50Q) |
|---|---|---|
| Baseline O3-mini | 82% | 94% |
| RAG (basic prompt) | 90% | — |
| RAG (optimized prompt) | 96.7% | 99% |

## Textbook

*Building Blocks for Theoretical Computer Science* — Margaret M. Fleck, 2017. Official CS173 textbook, 273 pages. Used to build the FAISS vector store (chunk size 500, overlap 100).

## Project Notes & Raw Data

Full methodology, evaluation results, graded responses, and datasets available here:(https://drive.google.com/drive/folders/1jBtt9ocdpcxgXxC5fBJrJ1kBpDXiPNq4?usp=drive_link)
