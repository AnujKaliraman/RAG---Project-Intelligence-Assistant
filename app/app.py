"""
app.py

Minimal Streamlit interface for the source-linked RAG system. Lets a user
pick a role (department access level), ask a question, and see the answer
alongside the exact source documents and pages it was grounded in.

Run with: streamlit run app.py
"""

import sys
sys.path.append("../retrieval")

import streamlit as st
from retriever import retrieve
from generator import generate_answer


st.set_page_config(page_title="Project Intelligence Assistant", layout="centered")

st.title("Project intelligence assistant")
st.caption(
    "Source-linked Q&A over engineering, legal, finance, and procurement documents. "
    "Every answer cites the exact document and page it came from."
)

DEPARTMENT_OPTIONS = {
    "Engineering": ["engineering"],
    "Legal": ["legal"],
    "Finance": ["finance"],
    "Procurement": ["procurement"],
    "Admin (all departments)": None,
}

role = st.selectbox("Your role / access level", list(DEPARTMENT_OPTIONS.keys()))
allowed_departments = DEPARTMENT_OPTIONS[role]

question = st.text_input(
    "Ask a question about the project documents",
    placeholder="e.g. What are the interconnection timeline requirements?",
)

if st.button("Ask") and question:
    with st.spinner("Retrieving relevant chunks and generating a grounded answer..."):
        chunks = retrieve(question, allowed_departments=allowed_departments)

        if not chunks:
            st.warning(
                "No relevant chunks found for your access level. "
                "Try a different question or role."
            )
        else:
            result = generate_answer(question, chunks)

            st.subheader("Answer")
            st.write(result.answer)

            st.subheader("Sources used")
            for source in result.sources_used:
                st.markdown(f"- `{source}`")

            with st.expander("Show retrieved chunks (raw)"):
                for chunk in chunks:
                    st.markdown(
                        f"**{chunk.source_file}, p.{chunk.page_number}** "
                        f"(similarity: {chunk.similarity}, department: {chunk.department})"
                    )
                    st.text(chunk.content[:400] + ("..." if len(chunk.content) > 400 else ""))
                    st.divider()

st.markdown("---")
st.caption(
    "Built to demonstrate a source-linked, permission-aware retrieval pattern: "
    "documents are chunked, embedded, and stored in PostgreSQL + pgvector; "
    "retrieval is filtered by access level before generation; every answer "
    "is grounded in cited source material rather than free-form LLM output."
)
