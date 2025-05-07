import streamlit as st
import os
import json
from datetime import date
import requests
from google.cloud import storage
from ics import Calendar
import fitz  # from PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import google.generativeai as genai
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
import re
import pandas as pd
from streamlit_calendar import calendar
import json
from datetime import datetime, date, time, timedelta
import uuid


# === CONFIGURATION ===
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "radiant-wall-456420-s3-7a6e386e29a2.json"
BUCKET_NAME = st.secrets["BUCKET_NAME"]
STRUCTURE_FILE = "course_structure.json"
google_api_key = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=google_api_key)
gemini = genai.GenerativeModel("models/gemini-2.0-flash")


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
                matrix = '\\begin{bmatrix}\n' + " \\\\\n".join([" & ".join(row) for row in entries]) + '\\n\\end{bmatrix}'
                lines[i:i+len(entries)] = [f"$$\n{matrix}\n$$"]
                break
    return "\n".join(lines)


# === CLASS DEFINITIONS ===
class PDFRagSystem:
    def __init__(self, google_api_key):
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vector_store = None
        self.qa_chain = None

    def process_pdf_bytes(self, pdf_bytes):
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = "".join(page.get_text() for page in doc)
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        documents = splitter.create_documents([text])
        chunks = [doc.page_content for doc in documents]
        return text, chunks

    def initialize_qa_chain(self):
        if not self.vector_store:
            raise ValueError("Vector store is not initialized.")
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=os.getenv("GOOGLE_API_KEY"))
        self.qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=False)

    def answer(self, query: str) -> str:
        if not self.qa_chain:
            raise ValueError("QA chain is not initialized.")
        result = self.qa_chain.run(query)
        return format_math_expressions(result)


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

def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    return "".join(page.get_text() for page in doc)

def chunk_text(text, chunk_size=1000, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.create_documents([text])

def create_vector_store(documents):
    texts = [doc.page_content for doc in documents]
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return {"index": index, "documents": documents, "model": model}

def build_rag_qa_chain(vector_store):
    def qa(query):
        query_embedding = vector_store["model"].encode([query])[0]
        D, I = vector_store["index"].search(np.array([query_embedding]), k=3)
        relevant_docs = [vector_store["documents"][i].page_content for i in I[0]]
        prompt = "Use the following context to answer the question:\n\n" + "\n\n".join(relevant_docs) + f"\n\nQuestion: {query}"
        return gemini.generate_content(prompt).text
    return qa

CALENDAR_EVENTS_FILE = "user_calendar_events.json"

def save_calendar_events(events: list):
    client = gcs_client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(CALENDAR_EVENTS_FILE)
    blob.upload_from_string(json.dumps(events, indent=2))

def load_calendar_events():
    client = gcs_client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(CALENDAR_EVENTS_FILE)
    if blob.exists():
        return json.loads(blob.download_as_text())
    return []

def deduplicate_events(events):
    """Remove duplicate events based on title and start time."""
    seen = set()
    unique = []
    for e in events:
        key = (e["title"], e["start"])
        if key not in seen:
            seen.add(key)
            unique.append(e)
    return sorted(unique, key=lambda x: x["start"])

def get_event_start(e):
    return getattr(e.begin, 'datetime', None) or e.begin.date()

def get_event_end(e):
    return getattr(e.end, 'datetime', None) or e.end.date() if e.end else get_event_start(e)

def load_all_calendar_events():
    all_events = []
    for file in ["calendar_assignments.json", "calendar_classes.json"]:
        try:
            blob = gcs_client().bucket(BUCKET_NAME).blob(file)
            if blob.exists():
                all_events += json.loads(blob.download_as_text())
        except Exception as e:
            st.warning(f"⚠️ Could not load {file}: {e}")
    return all_events


# === STREAMLIT SETUP ===
st.set_page_config(page_title="EduGenie Dashboard", layout="wide")
# === SIDEBAR: Logo ===
st.sidebar.image("logo.png", width=400)
# === PAGE DEFINITIONS ===
pages = {
    "Dashboard": "🏠 Dashboard",
    "Smart Scheduler": "🧠 Smart Scheduler",
    "Courses": "📚 Courses",
    "Chatbot": "🧞‍♂️ EduGenie"
}
# === PAGE STATE SETUP ===
if "page" not in st.session_state or st.session_state.page not in pages:
    st.session_state.page = "Dashboard"  # Default page
# === SIDEBAR BUTTON STYLING ===
button_style = """
    <style>
    div.stButton > button {
        font-size: 18px !important;
        padding: 0.75rem 1rem;
        width: 100%;
        margin-bottom: 12px;
        border-radius: 8px;
        background-color: #3c3c3c;
        color: white;
    }
    </style>
"""
st.sidebar.markdown(button_style, unsafe_allow_html=True)
# === SIDEBAR NAVIGATION ===
for key, label in pages.items():
    if st.sidebar.button(label, key=f"nav_{key}"):
        st.session_state.page = key
# === SELECTED PAGE ===
page = st.session_state.page



# === DASHBOARD PAGE ===
if page == "Dashboard":
    st.title("🏠 Dashboard")

    # === Sidebar Controls ===
    with st.sidebar:
        st.markdown("### 📥 Calendar Import")
        ical_url = st.text_input("iCal URL", key="ical_url")

        st.markdown("### ⚙️ Calendar Settings")
        calendar_type = st.selectbox("Select calendar type", ["Assignments", "Classes"], key="calendar_type")
        reload_requested = st.button("🔁 Load Calendar Events")

    calendar_filename = f"calendar_{calendar_type.lower()}.json"

    if "prev_ical_url" not in st.session_state:
        st.session_state["prev_ical_url"] = None
    if "edit_event_index" not in st.session_state:
        st.session_state["edit_event_index"] = None

    # === Load or Reparse iCal ===
    reparse_calendar = ical_url and reload_requested
    if reparse_calendar:
        st.session_state["prev_ical_url"] = ical_url
        try:
            r = requests.get(ical_url)
            if r.status_code != 200 or "text/calendar" not in r.headers.get("Content-Type", ""):
                st.error("❌ Invalid iCal URL.")
            else:
                cal = Calendar(r.text)
                parsed_events = []
                for e in cal.events:
                    start = get_event_start(e)
                    end = get_event_end(e)
                    is_all_day = calendar_type == "Assignments"
                    start_str = start.date().isoformat() if is_all_day else start.isoformat()
                    end_str = end.date().isoformat() if is_all_day else end.isoformat()

                    parsed_events.append({
                        "id": str(uuid.uuid4()),
                        "title": e.name or "Untitled Event",
                        "start": start_str,
                        "end": end_str,
                        "allDay": is_all_day,
                        "color": "blue" if is_all_day else "green"
                    })

                parsed_events = deduplicate_events(parsed_events)
                # Load existing events from GCS
                try:
                    blob = gcs_client().bucket(BUCKET_NAME).blob(calendar_filename)
                    existing_events = json.loads(blob.download_as_text()) if blob.exists() else []
                except:
                    existing_events = []

                # Combine and deduplicate
                combined_events = deduplicate_events(existing_events + parsed_events)

                # Save merged list back to GCS
                gcs_client().bucket(BUCKET_NAME).blob(calendar_filename).upload_from_string(
                    json.dumps(combined_events, indent=2)
                )
                st.success(f"✅ Calendar saved to {calendar_filename}")
        except Exception as ex:
            st.error("❌ Could not parse calendar.")
            st.exception(ex)

    # === Load Events ===
    events = load_all_calendar_events()
    st.session_state["calendar_events"] = events

    # === Filtered View ===
    selected_view = st.radio("View", ["Upcoming Assignments", "Upcoming Classes"], horizontal=True)
    is_all_day_view = selected_view == "Upcoming Assignments"
    st.subheader(f"📌 {selected_view}")

    filtered = [e for e in events if e.get("allDay", False) == is_all_day_view and e.get("start") >= date.today().isoformat()]
    filtered = sorted(filtered, key=lambda x: x["start"])

    for i, e in enumerate(filtered):
        if "id" not in e:
            e["id"] = str(uuid.uuid4())
        try:
            if e.get("allDay"):
                formatted = datetime.fromisoformat(e["start"]).strftime("%b %d")
                short_label = f"{e['title']} due {formatted}"
            else:
                s = datetime.fromisoformat(e["start"])
                t = datetime.fromisoformat(e["end"])
                short_label = f"{e['title']} - {s.strftime('%b %d - %I:%M %p')} ~ {t.strftime('%I:%M %p')}"
        except:
            short_label = e["title"]

        if st.session_state["edit_event_index"] == i:
            with st.form(f"edit_form_{i}"):
                new_title = st.text_input("Event Title", e["title"])
                if e.get("allDay"):
                    new_date = st.date_input("Due Date", value=datetime.fromisoformat(e["start"]).date())
                    new_start = new_date.isoformat()
                    new_end = new_start
                else:
                    start = datetime.fromisoformat(e["start"])
                    end = datetime.fromisoformat(e["end"])
                    d = st.date_input("Class Date", value=start.date())
                    t1 = st.time_input("Start Time", value=start.time())
                    t2 = st.time_input("End Time", value=end.time())
                    new_start = datetime.combine(d, t1).replace(tzinfo=None).isoformat()
                    new_end = datetime.combine(d, t2).replace(tzinfo=None).isoformat()
                col1, col2 = st.columns(2)
                with col1:
                    if st.form_submit_button("💾 Save"):
                        e.update({
                            "title": new_title,
                            "start": new_start,
                            "end": new_end,
                            "color": "blue" if e.get("allDay") else "green"
                        })
                        file = "calendar_assignments.json" if e.get("allDay") else "calendar_classes.json"
                        gcs_client().bucket(BUCKET_NAME).blob(file).upload_from_string(
                            json.dumps([x for x in events if x.get("allDay") == e.get("allDay")], indent=2))
                        st.session_state["calendar_events"] = load_all_calendar_events()
                        st.session_state["edit_event_index"] = None
                        st.rerun()
                with col2:
                    if st.form_submit_button("Cancel"):
                        st.session_state["edit_event_index"] = None
        else:
            col1, col2, col3 = st.columns([8, 1, 1])
            with col1:
                st.markdown(f"• **{short_label}**")
            with col2:
                if st.button("✏️", key=f"edit_{i}"):
                    st.session_state["edit_event_index"] = i
            with col3:
                if st.button("❌", key=f"del_{i}"):
                    events.remove(e)
                    file = "calendar_assignments.json" if e.get("allDay") else "calendar_classes.json"
                    gcs_client().bucket(BUCKET_NAME).blob(file).upload_from_string(
                        json.dumps([x for x in events if x.get("allDay") == e.get("allDay")], indent=2))
                    st.session_state["calendar_events"] = load_all_calendar_events()
                    st.rerun()

    # === Add New Assignment ===
    with st.expander("➕ Add New Assignment"):
        with st.form("add_assignment_form"):
            a_title = st.text_input("Assignment Title")
            a_date = st.date_input("Due Date")
            dt_str = datetime.combine(a_date, datetime.min.time()).replace(tzinfo=None).isoformat()
            if st.form_submit_button("Add Assignment"):
                new_event = {
                    "id": str(uuid.uuid4()),
                    "title": a_title,
                    "start": dt_str,
                    "end": dt_str,
                    "allDay": True,
                    "color": "blue"
                }
                assignments = [e for e in events if e.get("allDay")]
                assignments.append(new_event)
                gcs_client().bucket(BUCKET_NAME).blob("calendar_assignments.json").upload_from_string(
                    json.dumps(assignments, indent=2))
                st.session_state["calendar_events"] = load_all_calendar_events()
                st.rerun()

    # === Add New Class ===
    with st.expander("➕ Add New Class"):
        with st.form("add_class_form"):
            c_title = st.text_input("Class Title")
            c_date = st.date_input("Class Date")
            now = datetime.now().replace(second=0, microsecond=0)
            c_start = st.time_input("Start Time", value=now.time())
            c_end = st.time_input("End Time", value=(now + timedelta(hours=1)).time())
            start_str = datetime.combine(c_date, c_start).isoformat()
            end_str = datetime.combine(c_date, c_end).isoformat()
            if st.form_submit_button("Add Class"):
                new_event = {
                    "id": str(uuid.uuid4()),
                    "title": c_title,
                    "start": start_str,
                    "end": end_str,
                    "allDay": False,
                    "color": "green"
                }
                classes = [e for e in events if not e.get("allDay")]
                classes.append(new_event)
                gcs_client().bucket(BUCKET_NAME).blob("calendar_classes.json").upload_from_string(
                    json.dumps(classes, indent=2))
                st.session_state["calendar_events"] = load_all_calendar_events()
                st.rerun()

    # === Monthly Calendar ===
    if "edit_event_index" not in st.session_state or st.session_state["edit_event_index"] is None:
        st.subheader("📆 Monthly Calendar")
        for e in st.session_state["calendar_events"]:
            if "color" not in e:
                e["color"] = "blue" if e.get("allDay") else "green"
        calendar(
            events=st.session_state["calendar_events"],
            options={
                "initialView": "dayGridMonth",
                "height": 650,
                "editable": False,
            },
            key=f"calendar_{len(st.session_state['calendar_events'])}"
        )




# === SMART SCHEDULER PAGE ===
elif page == "Smart Scheduler":
    st.title("🧠 Smart Scheduler")
    st.info("This feature is coming soon!")



# === COURSES PAGE ===
elif page == "Courses":
    st.title("📚 Courses")
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
                if file and not st.session_state.get("file_uploaded"):
                    path = f"courses/{selected_course}/{selected_cat}/{file.name}"
                    upload_to_gcs(BUCKET_NAME, path, file)
                    st.success("✅ File uploaded.")
                    st.session_state["file_uploaded"] = True
                    st.rerun()

                # Reset flag after rerun
                if "file_uploaded" in st.session_state:
                    del st.session_state["file_uploaded"]

                st.markdown("#### 📁 Uploaded Files")
                files = list_gcs_files(f"courses/{selected_course}/{selected_cat}")
                for fname in files:
                    file_path = f"courses/{selected_course}/{selected_cat}/{fname}"
                    with st.container():
                        col1, col2, col3 = st.columns([6, 1, 1])
                        col1.write(fname)

                        # Checkbox to select for chatbot
                        selected = col2.checkbox("Chat", key=f"select_{file_path}")
                        if selected and file_path not in st.session_state["selected_files"]:
                            st.session_state["selected_files"].append(file_path)
                        elif not selected and file_path in st.session_state["selected_files"]:
                            st.session_state["selected_files"].remove(file_path)

                        # Replace/Delete (optional)
                        if show_advanced:
                            if col3.button("🗑", key=f"delete_{fname}"):
                                if delete_from_gcs(file_path):
                                    st.success(f"Deleted {fname}")
                                    st.rerun()
                                else:
                                    st.error("Failed to delete.")

                # Chat button
                if st.session_state["selected_files"]:
                    st.markdown("### 🤖 Ready to Chat with Selected Files?")
                    if st.button("💬 Chat"):
                        st.session_state.page = "Chatbot"
                        st.rerun()



# === CHATBOT PAGE ===
elif page == "Chatbot":
    if "selected_files" not in st.session_state:
        st.session_state["selected_files"] = []

    left, main, right = st.columns([0.2, 2.5, 1.3])
    with main:
        st.title("🧞‍♂️ How can I help you? ✨")

        if "rag_system" not in st.session_state:
            st.session_state.rag_system = PDFRagSystem(google_api_key=os.getenv("GOOGLE_API_KEY"))
        if "messages" not in st.session_state:
            st.session_state.messages = []

        selected_paths = st.session_state["selected_files"]
        if selected_paths:
            all_chunks = []
            for path in selected_paths:
                blob = gcs_client().bucket(BUCKET_NAME).blob(path)
                with blob.open("rb") as f:
                    _, chunks = st.session_state.rag_system.process_pdf_bytes(f.read())
                    all_chunks.extend(chunks)

            if all_chunks:
                st.session_state.rag_system.vector_store = FAISS.from_texts(all_chunks, st.session_state.rag_system.embeddings)
                try:
                    st.session_state.rag_system.initialize_qa_chain()
                    st.success(f"✅ Loaded {len(selected_paths)} file(s).")
                except Exception as e:
                    st.error(f"Failed to initialize QA chain: {e}")
            else:
                st.error("No extractable content found.")
            st.session_state["selected_files"] = []

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"], unsafe_allow_html=True)

        if prompt := st.chat_input("Ask your question..."):
            st.chat_message("user").markdown(prompt)
            response = st.session_state.rag_system.answer(prompt)
            st.chat_message("assistant").markdown(response, unsafe_allow_html=True)
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.messages.append({"role": "assistant", "content": response})

    with right:
        st.markdown("### 📂 Uploaded Files")
        ensure_structure()
        structure = st.session_state.structure

        if structure:
            selected_course = st.selectbox("📘 Select Course", list(structure.keys()))
            selected_cat = st.selectbox("📂 Select Category", structure[selected_course])
            folder = f"courses/{selected_course}/{selected_cat}"
            files = list_gcs_files(folder)
            if files:
                for fname in files:
                    path = f"{folder}/{fname}"
                    selected = st.checkbox(fname, key=f"select_{path}")
                    if selected and path not in st.session_state["selected_files"]:
                        st.session_state["selected_files"].append(path)
                    elif not selected and path in st.session_state["selected_files"]:
                        st.session_state["selected_files"].remove(path)
            else:
                st.info("No files uploaded in this category.")
        else:
            st.info("No courses found. Add one in the Courses tab.")

        if st.button("💬 Load Selected Files"):
            st.rerun()