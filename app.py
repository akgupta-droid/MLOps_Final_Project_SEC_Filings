import math
import os
from typing import List, Optional, Sequence, Tuple

import gradio as gr
import numpy as np
import psycopg2
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()

# -----------------------------
# Configuration
# -----------------------------
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gpt-5-mini")
TOP_K = int(os.getenv("TOP_K", "5"))

DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "port": int(os.getenv("DB_PORT", "5432")),
    "database": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "sslmode": os.getenv("DB_SSLMODE", "require"),
}

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

REQUIRED_ENV_VARS = {
    "DB_HOST": DB_CONFIG["host"],
    "DB_NAME": DB_CONFIG["database"],
    "DB_USER": DB_CONFIG["user"],
    "DB_PASSWORD": DB_CONFIG["password"],
    "OPENAI_API_KEY": OPENAI_API_KEY,
}


def validate_env() -> None:
    missing = [key for key, value in REQUIRED_ENV_VARS.items() if not value]
    if missing:
        raise RuntimeError(
            "Missing required environment variables: " + ", ".join(missing)
        )


validate_env()
client = OpenAI(api_key=OPENAI_API_KEY)


# -----------------------------
# Helpers
# -----------------------------
def clean_text(x: Optional[object]) -> Optional[str]:
    if x is None:
        return None
    x = str(x).strip()
    return x if x else None


def clean_number(x: Optional[object]) -> Optional[float]:
    if x is None:
        return None
    try:
        if isinstance(x, float) and math.isnan(x):
            return None
    except Exception:
        pass
    return x


def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)


def health_check() -> Tuple[bool, str]:
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
                cur.fetchone()
        return True, "Database connection OK."
    except Exception as e:
        return False, f"Database connection failed: {e}"


def get_query_embedding(text: str) -> np.ndarray:
    response = client.embeddings.create(
        input=text,
        model=EMBEDDING_MODEL,
    )
    return np.array(response.data[0].embedding, dtype=float)


def build_metadata_filters(
    symbol=None,
    industry=None,
    company_name=None,
    sector=None,
    quarter=None,
    accession_number=None,
    start_datetime=None,
    end_datetime=None,
    min_word_count=None,
):
    conditions = []
    params = []

    symbol = clean_text(symbol)
    industry = clean_text(industry)
    company_name = clean_text(company_name)
    sector = clean_text(sector)
    accession_number = clean_text(accession_number)
    start_datetime = clean_text(start_datetime)
    end_datetime = clean_text(end_datetime)
    min_word_count = clean_number(min_word_count)

    if symbol:
        conditions.append("UPPER(TRIM(symbol)) = UPPER(TRIM(%s))")
        params.append(symbol)

    if company_name:
        conditions.append("company_name ILIKE %s")
        params.append(f"%{company_name}%")

    if industry:
        conditions.append("industry ILIKE %s")
        params.append(f"%{industry}%")

    if sector:
        conditions.append("sector ILIKE %s")
        params.append(f"%{sector}%")

    if quarter not in (None, "", "All"):
        conditions.append('"quarter" = %s')
        params.append(int(quarter))

    if accession_number:
        conditions.append("TRIM(sec_accession_number) = TRIM(%s)")
        params.append(accession_number)

    if start_datetime:
        conditions.append("release_datetime >= %s::timestamptz")
        params.append(start_datetime)

    if end_datetime:
        conditions.append("release_datetime <= %s::timestamptz")
        params.append(end_datetime)

    if min_word_count is not None:
        conditions.append("word_count >= %s")
        params.append(int(min_word_count))

    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    return where_clause, params


def retrieve_chunks(
    query: str,
    top_k: int = TOP_K,
    symbol=None,
    industry=None,
    company_name=None,
    sector=None,
    quarter=None,
    accession_number=None,
    start_datetime=None,
    end_datetime=None,
    min_word_count=None,
):
    query_embedding = get_query_embedding(query)
    vector_str = "[" + ",".join(map(str, query_embedding.tolist())) + "]"

    where_clause, metadata_params = build_metadata_filters(
        symbol=symbol,
        industry=industry,
        company_name=company_name,
        sector=sector,
        quarter=quarter,
        accession_number=accession_number,
        start_datetime=start_datetime,
        end_datetime=end_datetime,
        min_word_count=min_word_count,
    )

    sql = f"""
    SELECT
        id,
        sec_accession_number,
        symbol,
        industry,
        company_name,
        release_datetime,
        quarter,
        sector,
        chunk_index,
        word_count,
        chunk_text,
        embedding <=> %s::vector AS distance
    FROM chunks
    {where_clause}
    ORDER BY embedding <=> %s::vector
    LIMIT %s;
    """

    params = [vector_str] + metadata_params + [vector_str, top_k]

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

    return rows


def build_context(results: Sequence[tuple]) -> str:
    if not results:
        return "No relevant context found."

    parts = []
    for i, row in enumerate(results, start=1):
        (
            row_id,
            sec_accession_number,
            symbol,
            industry,
            company_name,
            release_datetime,
            quarter,
            sector,
            chunk_index,
            word_count,
            chunk_text,
            distance,
        ) = row

        parts.append(
            f"""[Source {i}]
Row ID: {row_id}
SEC Accession Number: {sec_accession_number}
Symbol: {symbol}
Industry: {industry}
Company Name: {company_name}
Release Datetime: {release_datetime}
Quarter: {quarter}
Sector: {sector}
Chunk Index: {chunk_index}
Word Count: {word_count}
Distance: {distance:.4f}

Text:
{chunk_text}"""
        )

    return "\n\n------------------------\n\n".join(parts)


def generate_answer(query: str, context: str) -> str:
    prompt = f"""
You are a retrieval-based financial filings assistant.

Use ONLY the retrieved context below.
Do not invent facts.
If the answer is not supported by the context, say:
"I could not find the answer in the retrieved documents."

When possible, cite sources like [Source 1], [Source 2].

Retrieved Context:
{context}

User Question:
{query}
"""

    response = client.responses.create(
        model=GENERATION_MODEL,
        input=prompt,
    )
    return response.output_text.strip()


def rag_chat(
    message,
    history,
    symbol,
    industry,
    company_name,
    sector,
    quarter,
    accession_number,
    start_datetime,
    end_datetime,
    min_word_count,
):
    try:
        results = retrieve_chunks(
            query=message,
            top_k=TOP_K,
            symbol=symbol or None,
            industry=industry or None,
            company_name=company_name or None,
            sector=sector or None,
            quarter=quarter if quarter not in ("", "All", None) else None,
            accession_number=accession_number or None,
            start_datetime=start_datetime or None,
            end_datetime=end_datetime or None,
            min_word_count=min_word_count if min_word_count not in ("", None) else None,
        )

        if not results:
            return "I could not find any matching chunks for the selected metadata filters."

        context = build_context(results)
        answer = generate_answer(message, context)

        source_preview = "\n\nRetrieved sources:\n"
        for i, row in enumerate(results, start=1):
            (
                row_id,
                sec_accession_number,
                symbol_r,
                industry_r,
                company_name_r,
                release_datetime,
                quarter_r,
                sector_r,
                chunk_index,
                word_count,
                chunk_text,
                distance,
            ) = row

            preview = chunk_text[:180].replace("\n", " ")
            source_preview += (
                f"- [Source {i}] row_id={row_id}, accession={sec_accession_number}, "
                f"symbol={symbol_r}, industry={industry_r}, company={company_name_r}, "
                f"release_datetime={release_datetime}, quarter={quarter_r}, "
                f"sector={sector_r}, chunk_index={chunk_index}, "
                f"word_count={word_count}, distance={distance:.4f}\n"
                f"  preview={preview}...\n"
            )

        return answer + source_preview

    except Exception as e:
        return f"Error: {e}"


# -----------------------------
# Gradio UI
# -----------------------------
with gr.Blocks() as demo:
    gr.Markdown("## SEC Filings Metadata-Aware RAG Chatbot")

    chatbot = gr.Chatbot(type="messages")
    msg = gr.Textbox(label="Ask a question about the filings")

    with gr.Row():
        symbol = gr.Textbox(label="Symbol")
        industry = gr.Textbox(label="Industry")
        company_name = gr.Textbox(label="Company Name")
        sector = gr.Textbox(label="Sector")
        quarter = gr.Dropdown(
            choices=["All", 1, 2, 3, 4],
            value="All",
            label="Quarter"
        )

    with gr.Row():
        accession_number = gr.Textbox(label="SEC Accession Number")
        start_datetime = gr.Textbox(label="Start Datetime (e.g. 2024-01-01)")
        end_datetime = gr.Textbox(label="End Datetime (e.g. 2024-12-31)")
        min_word_count = gr.Number(label="Minimum Word Count", value=None)

    def respond(
        message,
        chat_history,
        symbol,
        industry,
        company_name,
        sector,
        quarter,
        accession_number,
        start_datetime,
        end_datetime,
        min_word_count
    ):
        chat_history = chat_history or []

        answer = rag_chat(
            message=message,
            history=chat_history,
            symbol=symbol,
            industry=industry,
            company_name=company_name,
            sector=sector,
            quarter=quarter,
            accession_number=accession_number,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            min_word_count=min_word_count
        )

        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": answer})
        return "", chat_history

    msg.submit(
        respond,
        inputs=[
            msg,
            chatbot,
            symbol,
            industry,
            company_name,
            sector,
            quarter,
            accession_number,
            start_datetime,
            end_datetime,
            min_word_count
        ],
        outputs=[msg, chatbot]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)