import streamlit as st
import requests

st.set_page_config(page_title="Knowledge Assistant", page_icon="📚")
st.title("RAG Knowledge Assistant")
st.caption("Ask questions over documents indexed by the FastAPI service.")
api = st.sidebar.text_input("API URL", "http://127.0.0.1:8000")
question = st.text_area("Your question")
top_k = st.slider("Retrieved passages", 1, 10, 3)
if st.button("Ask", type="primary"):
    if not question.strip():
        st.warning("Enter a question first.")
    else:
        try:
            response = requests.post(f"{api}/ask", json={"question": question, "top_k": top_k}, timeout=30)
            response.raise_for_status()
            result = response.json()
            st.markdown(result["answer"])
            st.subheader("Sources")
            for cite in result["citations"]:
                st.write(f'{cite["source"]} · chunk {cite["chunk"]} · score {cite["score"]:.3f}')
        except requests.RequestException as exc:
            st.error(f"Could not reach API: {exc}")
