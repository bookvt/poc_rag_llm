# CV RAG Lab

A local Streamlit proof of concept for asking questions about a CV using retrieval-augmented generation (RAG). Upload a text-based PDF or DOCX file, inspect the text and chunks the app extracts, and review the source passages used to answer each question.

The Streamlit interface is currently in Thai. Answers are requested in the same language as the question.

## Requirements

- Python 3.9 or later
- An OpenAI API key with access to `text-embedding-3-small` and `gpt-6-luna`
- An internet connection for OpenAI API calls

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set `OPENAI_API_KEY` in `.env` (or in your environment), then start the app:

```bash
streamlit run app.py
```

Open the URL shown by Streamlit, upload a CV, and enter a question. The app accepts one PDF or DOCX file at a time, up to 10 MiB. By default, Streamlit listens on `localhost` only.

## How it works

1. **Extract text.** The app reads text from PDF pages or DOCX paragraphs and table cells, retaining page or document locations as source labels.
2. **Build an index.** It splits the text into overlapping chunks and creates embeddings with `text-embedding-3-small`. The index stays in the current Streamlit session.
3. **Retrieve evidence.** Each question gets its own embedding. The app ranks chunks by cosine similarity, with local keyword and contact-detail matching to help surface exact terms, phone numbers, and email addresses.
4. **Generate an answer.** Up to three retrieved chunks and the question are sent to `gpt-6-luna`. The app asks the model to answer from those chunks and cite them with references such as `[1]`. You can expand each answer to inspect the retrieved text and its source location.

The keyword and contact-detail checks run locally and do not make additional model calls. Each question is handled independently; previous chat messages are not sent as context. Similarity scores describe embedding proximity, not answer confidence.

## Data handling and limitations

- Uploaded files, extracted text, embeddings, and chat history are held in Streamlit session memory. Switching files or ending the session clears the working index; the app has no vector database.
- CV text is sent to OpenAI to create embeddings. Questions and retrieved CV excerpts are sent to OpenAI to generate answers. The generation request sets `store=False`. Use documents you are authorized to send to the API.
- Scanned or image-only PDFs need OCR before upload. If text is missing from the app's extracted-text panel, it cannot be retrieved. Text inside DOCX text boxes may also be missed.
- Answers depend on the extracted text and retrieved chunks. Check the displayed source passages before relying on an answer.
- The app displays API-reported token usage and an estimated cost for indexing and answers. The estimate uses Standard API rates hard-coded in `rag.py` as of September 29, 2026; actual charges and current pricing may differ.

`.gitignore` excludes `.env`, PDF files, and DOCX files. Keep API keys and personal CV data out of commits.

## Run tests

```bash
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m unittest discover -s tests -v
```

The test suite uses sample data and mocked OpenAI responses; it does not require an API key. Running the app against the OpenAI API incurs usage charges.
