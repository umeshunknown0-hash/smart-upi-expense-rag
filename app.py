import streamlit as st
import pandas as pd
import numpy as np
import io
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from flashrank import Ranker, RerankRequest
from google import genai

st.set_page_config(page_title="AI Expense & Receipt RAG", layout="wide")
st.title("💳 AI Personal Finance & Receipt Analyzer (Hybrid RAG)")
st.caption("Upload any Indian bank statement (CSV) and PDF bills/warranties for full expense breakdown and document search.")

# Smart CSV Parser to skip bank header metadata rows automatically
def load_indian_bank_csv(uploaded_file):
    raw_bytes = uploaded_file.getvalue()
    text = raw_bytes.decode("utf-8", errors="ignore")
    lines = text.splitlines()
    
    # Find the row where the actual transaction table starts (usually has most commas/columns)
    header_idx = 0
    max_cols = 0
    for i, line in enumerate(lines[:30]):
        cols = len(line.split(","))
        if cols > max_cols:
            max_cols = cols
            header_idx = i
            
    df = pd.read_csv(io.StringIO(text), skiprows=header_idx, on_bad_lines="skip")
    df = df.dropna(how="all").dropna(axis=1, how="all")
    return df

# Sidebar for API Key and File Uploads
with st.sidebar:
    st.header("1. Setup & Upload")
    api_key = st.text_input("Enter Gemini API Key", type="password")
    uploaded_csv = st.file_uploader("Upload Bank Statement (.csv)", type=["csv"])
    uploaded_pdfs = st.file_uploader("Upload Receipts / Warranties (.pdf)", type=["pdf"], accept_multiple_files=True)
    st.info("Privacy Tip: Your data is processed in memory and never saved.")

@st.cache_resource
def load_models():
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")
    return embed_model, ranker

embed_model, ranker = load_models()

# Process Bank Statement CSV
df_statement = None
if uploaded_csv:
    try:
        df_statement = load_indian_bank_csv(uploaded_csv)
        with st.expander("Preview Detected Bank Transactions"):
            st.dataframe(df_statement.head(15))
    except Exception as e:
        st.error(f"Could not read CSV format: {e}")

# Process PDFs into chunks
pdf_chunks = []
if uploaded_pdfs:
    for pdf in uploaded_pdfs:
        reader = PdfReader(pdf)
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                pdf_chunks.append({
                    "id": len(pdf_chunks),
                    "source": f"{pdf.name} (Page {i+1})",
                    "text": text.strip()
                })
    st.sidebar.success(f"Indexed {len(pdf_chunks)} pages from {len(uploaded_pdfs)} PDF(s).")

# Chat Interface
query = st.text_input(
    "Ask anything (e.g., 'Diversify all my expenses into categories with totals' or 'Find my laptop warranty')"
)

if st.button("Analyze") and query:
    if not api_key:
        st.error("Please enter your Gemini API Key in the sidebar first.")
        st.stop()
    if df_statement is None and not pdf_chunks:
        st.warning("Please upload at least a CSV bank statement or a PDF receipt.")
        st.stop()

    client = genai.Client(api_key=api_key)
    
    with st.spinner("Running Hybrid Search & Full Financial Diversification..."):
        # 1. Prepare CSV Data Context
        statement_context = ""
        if df_statement is not None:
            statement_context = f"BANK STATEMENT TRANSACTIONS:\n{df_statement.to_string(index=False)}\n\n"

        # 2. Run Hybrid RAG (BM25 + Dense Vector + FlashRank Re-ranker) on PDFs
        pdf_context = ""
        reranked_sources = []
        if pdf_chunks:
            bm25 = BM25Okapi([c["text"].lower().split() for c in pdf_chunks])
            bm25_scores = bm25.get_scores(query.lower().split())
            top_bm25 = np.argsort(bm25_scores)[::-1][:4]

            doc_embeddings = embed_model.encode([c["text"] for c in pdf_chunks], normalize_embeddings=True)
            q_emb = embed_model.encode([query], normalize_embeddings=True)
            vec_scores = np.dot(doc_embeddings, q_emb)
            top_vec = np.argsort(vec_scores)[::-1][:4]

            candidates = list(set(top_bm25).union(set(top_vec)))
            passages = [{"id": pdf_chunks[i]["id"], "text": pdf_chunks[i]["text"], "meta": pdf_chunks[i]["source"]} for i in candidates]
            
            reranked_sources = ranker.rerank(RerankRequest(query=query, passages=passages))[:3]
            pdf_context = "VERIFIED RECEIPT EXCERPTS:\n" + "\n".join([f"[{r['meta']}]: {r['text']}" for r in reranked_sources])

        # 3. Comprehensive Financial & Document Analysis Prompt
        prompt = f"""You are an expert Personal Finance & Document Analyst.
        Use ONLY the provided data below to answer the user's question accurately.
        
        Rules:
        1. When analyzing bank statement expenses, examine EVERY debit/withdrawal transaction and diversify ALL of them into logical categories (e.g., Food & Dining, Groceries, Online Shopping, Travel & Fuel, Bills & Utilities, Rent/Housing, Health/Medical, Entertainment/Subscriptions, Peer-to-Peer UPI Transfers, and Other).
        2. Ignore credits/deposits unless specifically asked about income, and clearly separate Total Credits vs. Total Debits.
        3. Present a clean Markdown summary table showing: Category | Number of Transactions | Total Amount (INR ₹) | % of Total Spending, followed by key insights (like top merchants or largest single expense).
        4. If answering from PDF receipts/warranties, cite the exact PDF filename and page number.

        {statement_context}
        {pdf_context}

        User Question: {query}
        """

        response = client.models.generate_content(model="gemini-3.8-flash", contents=prompt)
        
        st.subheader("Analysis Result")
        st.markdown(response.text)

        if reranked_sources:
            with st.expander("View Retrieved PDF Sources & Re-ranker Scores"):
                for s in reranked_sources:
                    st.write(f"**{s['meta']}** — Relevance Score: `{s['score']:.3f}`")
                    st.caption(s['text'][:300] + "...")
