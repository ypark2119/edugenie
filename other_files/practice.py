import streamlit as st
import os
import json
from datetime import date
import requests
from google.cloud import storage
from ics import Calendar
import fitz  # from PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
import numpy as np
import re
from typing import List, Dict, Any

# === CONFIGURATION ===
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "radiant-wall-456420-s3-7a6e386e29a2.json"
BUCKET_NAME = st.secrets["BUCKET_NAME"]
STRUCTURE_FILE = "course_structure.json"

# Use the API key from Streamlit secrets
google_api_key = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=google_api_key)
gemini = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# === MATH FORMATTING ===
def format_math_expressions(text: str) -> str:
    text = re.sub(r'\\begin\{([^}]+)\}(.*?)\\end\{\1\}',
                  lambda m: re.sub(r'(?<![\\])\s*\\\s+', r'\\\\', m.group(0)),
                  text, flags=re.DOTALL)
    text = re.sub(r'\$\$(.*?)\$\$', r'$$\1$$', text, flags=re.DOTALL)
    text = re.sub(r'(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)', r'$\1$', text)
    text = re.sub(r'\${3,}', '$$', text)
    text = re.sub(r'([^\s])\$\$', r'\1 $$', text)
    text = re.sub(r'\$\$([^\s])', r'$$ \1', text)
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if re.search(r'\b(-?\d+\s+){1,}-?\d+\b', line):
            entries = [list(map(str.strip, re.split(r'\s+', l.strip()))) for l in lines[i:i+5] if re.search(r'\b(-?\d+\s+){1,}-?\d+\b', l)]
            if entries:
                matrix = '\\begin{bmatrix}\n' + " \\\\n".join([" & ".join(row) for row in entries]) + '\\n\\end{bmatrix}'
                lines[i:i+len(entries)] = [f"$$\n{matrix}\n$$"]
                break
    return "\n".join(lines)

# === CLASS DEFINITIONS ===
class PDFRagSystem:
    def __init__(self, google_api_key=None):
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vector_store = None
        self.qa_chain = None
        if google_api_key:
            os.environ["GOOGLE_API_KEY"] = google_api_key

    def process_pdf_bytes(self, pdf_bytes: bytes):
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = "".join(page.get_text() for page in doc)
        chunks = self.text_splitter.split_text(text)
        
        # ✅ Automatically set vector store here
        self.vector_store = FAISS.from_texts(chunks, self.embeddings)
        
        return text, chunks

    def initialize_qa_chain(self):
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Process a PDF first.")

        question_prompt_template = """
        You are an assistant that answers questions based on the provided context from a PDF document.

        Context: {context}
        Question: {question}

        Answer the question based only on the provided context. If you cannot find the answer in the context, 
        say "I don't have enough information to answer this question based on the document content." 
        Do not make up information.

        When you're explaining mathematical concepts or equations:
        1. Use proper LaTeX notation for formulas and equations
        2. Format subscripts properly (e.g., x_i instead of xi)
        3. Use proper notation for derivatives, integrals, sums, etc.
        4. For inline math, use single dollar signs like $x^2$
        5. For display equations, use double dollar signs like $$E=mc^2$$
        """
        QUESTION_PROMPT = PromptTemplate(template=question_prompt_template, input_variables=["context", "question"])

        combine_prompt_template = """
        The following is a set of summaries from different sections of a document:

        {summaries}
        Based on these summaries, please provide a comprehensive answer to the question: {question}

        When you're explaining mathematical concepts or equations:
        1. Use proper LaTeX notation for formulas and equations
        2. Format subscripts properly (e.g., x_i instead of xi)
        3. Use proper notation for derivatives, integrals, sums, etc.
        4. For inline math, use single dollar signs like $x^2$
        5. For display equations, use double dollar signs like $$E=mc^2$$

        If the information isn't available in the summaries, say "I don't have enough information to answer this question based on the document content."
        """
        COMBINE_PROMPT = PromptTemplate(template=combine_prompt_template, input_variables=["summaries", "question"])

        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.1)

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="map_reduce",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 5}),
            chain_type_kwargs={
                "question_prompt": QUESTION_PROMPT,
                "combine_prompt": COMBINE_PROMPT
            }
        )

    def query(self, question: str) -> Dict[str, Any]:
        if not self.qa_chain:
            if not os.environ.get("GOOGLE_API_KEY"):
                return {"result": "Google API key not set. Please provide a key to enable querying."}
            if not self.vector_store:
                return {"result": "No PDF has been processed yet. Please process a PDF first."}
            self.initialize_qa_chain()

        try:
            result = self.qa_chain({"query": question})
            return result
        except Exception as e:
            return {"result": f"Error processing query: {str(e)}"}

    def similarity_search(self, query: str, k: int = 3) -> List[tuple]:
        if not self.vector_store:
            return [("No PDF has been processed yet. Please process a PDF first.", 0)]
        try:
            results = self.vector_store.similarity_search_with_score(query, k=k)
            return [(doc.page_content, score) for doc, score in results]
        except Exception as e:
            return [(f"Error during similarity search: {str(e)}", 0)]

    def answer(self, question: str) -> str:
        result = self.query(question)
        return format_math_expressions(result.get("result", "No answer found."))

# === UTILS ===
def gcs_client():
    return storage.Client()

def upload_to_gcs(bucket_name, destination_blob_name, file):
    client = gcs_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_file(file, rewind=True)
    return f"gs://{bucket_name}/{destination_blob_name}"

def delete_from_gcs(path):
    try:
        client = gcs_client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(path)
        if blob.exists():
            blob.delete()
            return True
        return False
    except Exception as e:
        print(f"Error deleting {path}: {e}")
        return False

def list_gcs_files(prefix):
    client = gcs_client()
    bucket = client.bucket(BUCKET_NAME)
    return [b.name.split("/")[-1] for b in bucket.list_blobs(prefix=prefix) if not b.name.endswith(".keep")]

def download_structure():
    client = gcs_client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(STRUCTURE_FILE)
    if not blob.exists():
        return {}
    return json.loads(blob.download_as_text())

def upload_structure(data):
    client = gcs_client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(STRUCTURE_FILE)
    blob.upload_from_string(json.dumps(data, indent=2))

def ensure_structure():
    if "structure" not in st.session_state:
        st.session_state.structure = download_structure()

# === STREAMLIT SETUP ===
st.set_page_config(page_title="AI Tutor Dashboard", layout="wide")
st.sidebar.title("AI Tutor")
if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

page = st.sidebar.radio("Go to", ["Dashboard", "Chatbot", "Calendar", "Courses", "Settings"], index=["Dashboard", "Chatbot", "Calendar", "Courses", "Settings"].index(st.session_state.page))
st.session_state.page = page
ical_url = st.sidebar.text_input("\U0001F4C5 Paste public iCal URL", placeholder="https://...")

# === DASHBOARD PAGE ===
if page == "Dashboard":
    st.title("Dashboard")
    st.subheader("Upcoming Assignments")
    if ical_url:
        try:
            r = requests.get(ical_url)
            cal = Calendar(r.text)
            events = sorted([{"name": e.name, "date": e.begin.datetime.date()} for e in cal.events if e.begin.datetime.date() >= date.today()], key=lambda x: x["date"])
            for e in events[:5]:
                st.write(f"• {e['name']} due {e['date'].strftime('%b %d')}")
        except:
            st.error("Could not parse calendar.")
    else:
        st.info("Please enter a valid iCal URL.")

# === CHATBOT PAGE ===
elif page == "Chatbot":
    st.title("AI Tutor Chatbot")

    if "rag_system" not in st.session_state:
        st.session_state.rag_system = PDFRagSystem(google_api_key=google_api_key)
    if "messages" not in st.session_state:
        st.session_state.messages = []

    selected_paths = st.session_state.get("selected_files", [])
    if selected_paths:
        for path in selected_paths:
            client = storage.Client()
            bucket = client.bucket(BUCKET_NAME)
            blob = bucket.blob(path)
            with blob.open("rb") as f:
                pdf_bytes = f.read()
                _, _ = st.session_state.rag_system.process_pdf_bytes(pdf_bytes)

        try:
            st.session_state.rag_system.initialize_qa_chain()
            st.success(f"✅ Loaded {len(selected_paths)} file(s) into QA chain.")
        except Exception as e:
            st.error(f"Error initializing QA chain: {e}")

        st.session_state["selected_files"] = []


    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"], unsafe_allow_html=True)

    # if prompt := st.chat_input("Ask your question..."):
    #     with st.chat_message("user"):
    #         st.markdown(prompt)
    #     response = st.session_state.rag_system.answer(prompt)
    #     with st.chat_message("assistant"):
    #         st.markdown(response, unsafe_allow_html=True)
    #     st.session_state.messages.append({"role": "user", "content": prompt})
    #     st.session_state.messages.append({"role": "assistant", "content": response})

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"], unsafe_allow_html=True)

    # Chat input
    if prompt := st.chat_input("Ask your question..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        response = st.session_state.rag_system.answer(prompt)
        with st.chat_message("assistant"):
            st.markdown(response, unsafe_allow_html=True)
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant", "content": response})

# === COURSES PAGE ===
elif page == "Courses":
    st.title("Courses")
    ensure_structure()
    structure = st.session_state.structure
    show_advanced = st.checkbox("Show advanced options")

    tab1, tab2, tab3 = st.tabs(["📘 Course Setup", "📁 Categories", "📂 Files"])

    # === TAB 1: COURSE SETUP ===
    with tab1:
        st.subheader("Manage Courses")

        new_course = st.text_input("➕ Add new course")
        if st.button("Add Course") and new_course:
            if new_course not in structure:
                structure[new_course] = []
                upload_structure(structure)
                st.rerun()

        course_names = list(structure.keys())
        if course_names:
            selected_course = st.selectbox("🎓 Select a course", course_names)

            if show_advanced:
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🗑 Delete Course"):
                        del structure[selected_course]
                        upload_structure(structure)
                        st.rerun()
                with col2:
                    rename = st.text_input("✏️ Rename Course", key="rename_course")
                    if st.button("Rename Course") and rename:
                        structure[rename] = structure.pop(selected_course)
                        upload_structure(structure)
                        st.rerun()

    # === TAB 2: CATEGORY SETUP ===
    with tab2:
        if course_names:
            selected_course = st.selectbox("🎓 Select a course (for categories)", course_names, key="cat_course")
            st.subheader(f"Categories for {selected_course}")

            new_cat = st.text_input("➕ Add new category", key="new_cat")
            if st.button("Add Category") and new_cat not in structure[selected_course]:
                structure[selected_course].append(new_cat)
                upload_structure(structure)
                st.rerun()

            if structure[selected_course]:
                selected_cat = st.selectbox("📂 Select a category", structure[selected_course], key="selected_cat")

                if show_advanced:
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("🗑 Delete Category"):
                            structure[selected_course].remove(selected_cat)
                            upload_structure(structure)
                            st.rerun()
                    with col2:
                        rename_cat = st.text_input("✏️ Rename Category", key="rename_cat")
                        if st.button("Rename Category") and rename_cat:
                            files = list_gcs_files(f"courses/{selected_course}/{selected_cat}")
                            for f in files:
                                old_path = f"courses/{selected_course}/{selected_cat}/{f}"
                                new_path = f"courses/{selected_course}/{rename_cat}/{f}"
                                client = gcs_client()
                                bucket = client.bucket(BUCKET_NAME)
                                bucket.rename_blob(bucket.blob(old_path), new_path)
                            structure[selected_course].remove(selected_cat)
                            structure[selected_course].append(rename_cat)
                            upload_structure(structure)
                            st.rerun()

    # === TAB 3: FILE MANAGEMENT ===
    with tab3:
        if "selected_files" not in st.session_state:
            st.session_state["selected_files"] = []

        if course_names:
            selected_course = st.selectbox("🎓 Select a course (for file management)", course_names, key="file_course_all")
            if structure[selected_course]:
                selected_cat = st.selectbox("📂 Select a category", structure[selected_course], key="file_cat_all")

                st.markdown("#### 📤 Upload a file")
                file = st.file_uploader("Choose file to upload", key=f"{selected_course}_{selected_cat}_upload")
                if file:
                    path = f"courses/{selected_course}/{selected_cat}/{file.name}"
                    upload_to_gcs(BUCKET_NAME, path, file)
                    st.success("✅ File uploaded.")
                    st.rerun()

                st.markdown("#### 📁 Uploaded Files")
                files = list_gcs_files(f"courses/{selected_course}/{selected_cat}")
                for fname in files:
                    file_path = f"courses/{selected_course}/{selected_cat}/{fname}"
                    with st.container():
                        col1, col2, col3 = st.columns([6, 1, 1])
                        col1.write(fname)

                        # Checkbox for chatbot selection
                        selected = col2.checkbox("Chat", key=f"select_{file_path}")
                        if selected and file_path not in st.session_state["selected_files"]:
                            st.session_state["selected_files"].append(file_path)
                        elif not selected and file_path in st.session_state["selected_files"]:
                            st.session_state["selected_files"].remove(file_path)

                        # Optional deletion
                        if show_advanced and col3.button("🗑", key=f"delete_{fname}"):
                            if delete_from_gcs(file_path):
                                st.success(f"Deleted {fname}")
                                st.rerun()
                            else:
                                st.error("Failed to delete.")

                # Button to trigger chatbot loading
                if st.session_state["selected_files"]:
                    st.markdown("### 🤖 Ready to Chat with Selected Files?")
                    if st.button("💬 Chat"):
                        st.session_state.page = "Chatbot"
                        st.rerun()

