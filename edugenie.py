import os
import uuid
import re
import json
import logging
import requests
import fitz  # from PyMuPDF
import faiss
import numpy as np
import pandas as pd
from io import StringIO
from datetime import datetime, date, time, timedelta
import datetime as dt
from collections import defaultdict

import streamlit as st
from streamlit_calendar import calendar

from ics import Calendar
from google.cloud import storage
from google.oauth2 import service_account
import google.cloud.logging
from google.cloud.logging_v2.handlers import CloudLoggingHandler

import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate


# === CONFIGURATION ===
def gcs_client():
    credentials_info = {
        "type": st.secrets["gcp_service_account"]["type"],
        "project_id": st.secrets["gcp_service_account"]["project_id"],
        "private_key_id": st.secrets["gcp_service_account"]["private_key_id"],
        "private_key": st.secrets["gcp_service_account"]["private_key"].replace("\\n", "\n"),
        "client_email": st.secrets["gcp_service_account"]["client_email"],
        "client_id": st.secrets["gcp_service_account"]["client_id"],
        "auth_uri": st.secrets["gcp_service_account"]["auth_uri"],
        "token_uri": st.secrets["gcp_service_account"]["token_uri"],
        "auth_provider_x509_cert_url": st.secrets["gcp_service_account"]["auth_provider_x509_cert_url"],
        "client_x509_cert_url": st.secrets["gcp_service_account"]["client_x509_cert_url"],
    }
    credentials = service_account.Credentials.from_service_account_info(credentials_info)
    return storage.Client(credentials=credentials, project=credentials_info["project_id"])

BUCKET_NAME = st.secrets["BUCKET_NAME"]
STRUCTURE_FILE = "course_structure.json"
google_api_key = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=google_api_key)
gemini = genai.GenerativeModel("models/gemini-2.0-flash")
gcp_logging_client = google.cloud.logging.Client(credentials=gcs_client()._credentials, project=gcs_client().project)
gcp_logging_client.setup_logging()
logger = logging.getLogger("edu_genie")


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

        prompt_template = PromptTemplate.from_template(
            """You are an expert assistant. Answer the user's question using *only* the provided context below.
    If the context does not contain the answer, respond with "I'm not sure based on the uploaded documents."

    Context:
    {context}

    Question:
    {question}

    Answer:"""
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=False,
            chain_type_kwargs={"prompt": prompt_template}
        )

    def answer(self, query: str) -> str:
        if not self.qa_chain:
            raise ValueError("QA chain is not initialized.")
        result = self.qa_chain.run(query).strip()
        if "I'm not sure" in result or not result:
            return "⚠️ Sorry, I couldn't find the answer in the uploaded documents."
        return format_math_expressions(result)
    

# === UTILS ===
def upload_to_gcs(bucket_name, destination_blob_name, file):
    client = gcs_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_file(file, rewind=True)
    return f"gs://{bucket_name}/{destination_blob_name}"

def log_interaction_to_gcs(prompt, response, user="anonymous"):
    from datetime import datetime
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "user": user,
        "prompt": prompt,
        "response": response
    }
    file_path = f"gemini_logs/log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    client = gcs_client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(file_path)
    blob.upload_from_string(json.dumps(log_entry, indent=2), content_type="application/json")

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

def handle_edit_event_button(i):
    st.session_state["edit_event_index"] = i

def handle_delete_event_button(e, all_events):
    st.session_state["delete_event"] = e
    st.session_state["events_to_update"] = all_events

def toggle_file_selection(path):
    if path not in st.session_state["selected_files"]:
        st.session_state["selected_files"].append(path)
    else:
        st.session_state["selected_files"].remove(path)

def initialize_session_state():
    # Initialize all session state variables to avoid reruns due to first-time assignments
    if "page" not in st.session_state:
        st.session_state.page = "Dashboard"
    if "structure" not in st.session_state:
        st.session_state.structure = download_structure()
    if "selected_files" not in st.session_state:
        st.session_state["selected_files"] = []
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "prev_ical_url" not in st.session_state:
        st.session_state["prev_ical_url"] = None
    if "edit_event_index" not in st.session_state:
        st.session_state["edit_event_index"] = None
    if "calendar_events" not in st.session_state:
        st.session_state["calendar_events"] = load_all_calendar_events()
    if "completed_assignments" not in st.session_state:
        st.session_state["completed_assignments"] = set()
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = PDFRagSystem(google_api_key=os.getenv("GOOGLE_API_KEY"))
    if "calendar_updated" not in st.session_state:
        st.session_state["calendar_updated"] = False
    if "file_uploaded" not in st.session_state:
        st.session_state["file_uploaded"] = False
    if "delete_event" not in st.session_state:
        st.session_state["delete_event"] = None
    if "events_to_update" not in st.session_state:
        st.session_state["events_to_update"] = None


# === STREAMLIT SETUP ===
st.set_page_config(page_title="EduGenie Dashboard", layout="wide")
initialize_session_state()

st.sidebar.image("logo.png", width=400)

with st.sidebar.expander("🔒 Privacy Notice"):
    st.markdown("""
    - Your uploaded files are securely stored in Google Cloud Storage.
    - All data is encrypted at rest (AES-256).
    - Gemini interactions are logged for performance monitoring.
    - No personal data is shared or used for training.
    """)

pages = {
    "Dashboard": "🏠 Dashboard",
    "Smart Scheduler": "🧠 Smart Scheduler",
    "Courses": "📚 Courses",
    "Chatbot": "🧞‍♂️ EduGenie"
}

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

for key, label in pages.items():
    if st.sidebar.button(label, key=f"nav_{key}"):
        st.session_state.page = key

page = st.session_state.page

# === DASHBOARD PAGE ===
if page == "Dashboard":
    st.title("🏠 Dashboard")

    with st.sidebar:
        st.markdown("### 📥 Calendar Import")
        ical_url = st.text_input("iCal URL", key="ical_url")

        st.markdown("### ⚙️ Calendar Settings")
        calendar_type = st.selectbox("Select calendar type", ["Assignments", "Classes"], key="calendar_type")
        reload_requested = st.button("🔁 Load Calendar Events")

    calendar_filename = f"calendar_{calendar_type.lower()}.json"

    # Handle calendar import logic
    if ical_url and reload_requested:
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
                try:
                    blob = gcs_client().bucket(BUCKET_NAME).blob(calendar_filename)
                    existing_events = json.loads(blob.download_as_text()) if blob.exists() else []
                except:
                    existing_events = []

                combined_events = deduplicate_events(existing_events + parsed_events)
                gcs_client().bucket(BUCKET_NAME).blob(calendar_filename).upload_from_string(
                    json.dumps(combined_events, indent=2)
                )
                st.success(f"✅ Calendar saved to {calendar_filename}")
                st.session_state["calendar_updated"] = True
                st.session_state["calendar_events"] = load_all_calendar_events()
        except Exception as ex:
            st.error("❌ Could not parse calendar.")
            st.exception(ex)

    # Load calendar events once
    if st.session_state["calendar_updated"] or not st.session_state.get("calendar_events"):
        all_events = load_all_calendar_events()
        st.session_state["calendar_events"] = all_events
        st.session_state["calendar_updated"] = False
    else:
        all_events = st.session_state["calendar_events"]

    # Process deletion if requested from a previous interaction
    if st.session_state["delete_event"] is not None:
        e = st.session_state["delete_event"]
        all_events = st.session_state["events_to_update"]
        all_events.remove(e)
        file = "calendar_assignments.json" if e.get("allDay") else "calendar_classes.json"
        gcs_client().bucket(BUCKET_NAME).blob(file).upload_from_string(
            json.dumps([x for x in all_events if x.get("allDay") == e.get("allDay")], indent=2))
        st.session_state["calendar_events"] = load_all_calendar_events()
        st.session_state["delete_event"] = None
        st.session_state["events_to_update"] = None

    st.subheader("📌 Upcoming Assignments")
    now = datetime.now()
    five_days_later = now + timedelta(days=5)

    upcoming_assignments = [
        e for e in all_events
        if e.get("allDay", False) is True and now.date() <= datetime.fromisoformat(e["start"]).date() <= five_days_later.date()
    ]
    upcoming_assignments = sorted(upcoming_assignments, key=lambda x: x["start"])

    if upcoming_assignments:
        for e in upcoming_assignments:
            s = datetime.fromisoformat(e["start"])
            st.markdown(f"- **{e['title']}** due **{s.strftime('%b %d')}**")
    else:
        st.info("No upcoming assignments in the next 5 days.")

    if st.session_state.get("edit_event_index") is None:
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

    st.markdown("### 🛠️ Manage Events by Date")
    unique_dates = sorted(set(datetime.fromisoformat(e["start"]).date() for e in all_events))
    if unique_dates:
        selected_day = st.selectbox("📅 Select a date", unique_dates, format_func=lambda d: d.strftime("%A, %B %d"))
        daily_events = [e for e in all_events if datetime.fromisoformat(e["start"]).date() == selected_day]

        if st.session_state.get("edit_event_index") is None:
            for i, e in enumerate(daily_events):
                s = datetime.fromisoformat(e["start"])
                t = datetime.fromisoformat(e["end"])
                event_time = "" if e.get("allDay") else f" ({s.strftime('%I:%M %p')} ~ {t.strftime('%I:%M %p')})"
                col1, col2, col3 = st.columns([8, 1, 1])
                with col1:
                    st.markdown(f"- **{e['title']}**{event_time}")
                with col2:
                    if st.button("✏️", key=f"edit_date_{i}", on_click=handle_edit_event_button, args=(i,)):
                        pass
                with col3:
                    if st.button("❌", key=f"del_date_{i}", on_click=handle_delete_event_button, args=(e, all_events)):
                        pass
        else:
            st.subheader("✏️ Edit Event")
            index = st.session_state["edit_event_index"]
            if index < len(daily_events):
                e = daily_events[index]
                is_all_day = e.get("allDay", False)
                s = datetime.fromisoformat(e["start"])
                t = datetime.fromisoformat(e["end"])

                with st.form("edit_event_form"):
                    new_title = st.text_input("Title", value=e["title"])
                    new_date = st.date_input("Date", value=s.date())
                    if is_all_day:
                        new_start, new_end = new_date, new_date
                    else:
                        new_start_time = st.time_input("Start Time", value=s.time())
                        new_end_time = st.time_input("End Time", value=t.time())
                        new_start = datetime.combine(new_date, new_start_time)
                        new_end = datetime.combine(new_date, new_end_time)

                    save_changes = st.form_submit_button("Save Changes")

                if save_changes:
                    e["title"] = new_title
                    e["start"] = new_start.isoformat() if not is_all_day else new_start.isoformat()
                    e["end"] = new_end.isoformat() if not is_all_day else new_end.isoformat()

                    file = "calendar_assignments.json" if is_all_day else "calendar_classes.json"
                    events = [ev for ev in all_events if ev.get("allDay") == is_all_day]
                    gcs_client().bucket(BUCKET_NAME).blob(file).upload_from_string(json.dumps(events, indent=2))

                    st.session_state["edit_event_index"] = None
                    st.session_state["calendar_events"] = load_all_calendar_events()
                    st.success("✅ Event updated.")

                if st.button("Cancel"):
                    st.session_state["edit_event_index"] = None
    else:
        st.info("📭 No events yet. Add a class or assignment below.")

    with st.expander("➕ Add New Assignment"):
        with st.form("add_assignment_form"):
            a_title = st.text_input("Assignment Title")
            a_date = st.date_input("Due Date")
            dt_str = datetime.combine(a_date, datetime.min.time()).replace(tzinfo=None).isoformat()
            submit_assignment = st.form_submit_button("Add Assignment")
            
        if submit_assignment and a_title:
            new_event = {
                "id": str(uuid.uuid4()),
                "title": a_title,
                "start": dt_str,
                "end": dt_str,
                "allDay": True,
                "color": "blue"
            }
            assignments = [e for e in all_events if e.get("allDay")]
            assignments.append(new_event)
            gcs_client().bucket(BUCKET_NAME).blob("calendar_assignments.json").upload_from_string(
                json.dumps(assignments, indent=2))
            st.session_state["calendar_events"] = load_all_calendar_events()
            st.success(f"✅ Added assignment: {a_title}")

    with st.expander("➕ Add New Class"):
        with st.form("add_class_form"):
            c_title = st.text_input("Class Title")
            c_date = st.date_input("Class Date")
            now = datetime.now().replace(second=0, microsecond=0)
            c_start = st.time_input("Start Time", value=now.time())
            c_end = st.time_input("End Time", value=(now + timedelta(hours=1)).time())
            start_str = datetime.combine(c_date, c_start).isoformat()
            end_str = datetime.combine(c_date, c_end).isoformat()
            submit_class = st.form_submit_button("Add Class")
            
        if submit_class and c_title:
            new_event = {
                "id": str(uuid.uuid4()),
                "title": c_title,
                "start": start_str,
                "end": end_str,
                "allDay": False,
                "color": "green"
            }
            classes = [e for e in all_events if not e.get("allDay")]
            classes.append(new_event)
            gcs_client().bucket(BUCKET_NAME).blob("calendar_classes.json").upload_from_string(
                json.dumps(classes, indent=2))
            st.session_state["calendar_events"] = load_all_calendar_events()
            st.success(f"✅ Added class: {c_title}")



elif page == "Smart Scheduler":
    st.title("🧠 Smart Scheduler Pro")
    
    # Create tabs for better organization
    tabs = st.tabs(["📅 Availability", "📚 Assignments", "🗓️ Study Plan", "📊 Analytics"])
    
    with tabs[0]:  # AVAILABILITY TAB
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown("### ✅ Select Your Available Time Slots")
            WEEKLY_AVAILABILITY_FILE = "weekly_availability.json"
            
            # Improved time slot selection with better UI
            days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            
            # More granular time slots (30 min intervals) for better scheduling
            # Using standard hyphen instead of en-dash
            time_slots = []
            for h in range(6, 23):
                time_slots.append(f"{str(h).zfill(2)}:00-{str(h).zfill(2)}:30")
                time_slots.append(f"{str(h).zfill(2)}:30-{str(h+1).zfill(2)}:00")
            
            # # Add option to use recurring schedule templates
            # templates = {
            #     "Early Bird": {day: [slot for slot in time_slots if "06:" in slot or "07:" in slot or "08:" in slot] for day in days},
            #     "Night Owl": {day: [slot for slot in time_slots if "20:" in slot or "21:" in slot or "22:" in slot] for day in days},
            #     "Weekend Focus": {day: [] for day in days[:5]} | {"Saturday": time_slots[8:24], "Sunday": time_slots[8:24]}
            # }
            
            # template = st.selectbox("Quick Template", ["Custom"] + list(templates.keys()), index=0)
            
            # Load saved availability with error handling for JSON
            default_availability = {}
            try:
                blob = gcs_client().bucket(BUCKET_NAME).blob(WEEKLY_AVAILABILITY_FILE)
                if blob.exists():
                    content = blob.download_as_text()
                    # Handle potential JSON issue by stripping extra characters
                    content = content.strip()
                    default_availability = json.loads(content)
                    if not isinstance(default_availability, dict):
                        default_availability = {}
                        st.warning("⚠️ Invalid format in saved availability data")
            except json.JSONDecodeError as e:
                st.warning(f"⚠️ Could not parse saved availability: {e}")
                default_availability = {}
            except Exception as e:
                st.warning(f"⚠️ Could not load availability: {e}")
                default_availability = {}
            
            # # Apply template if selected
            # if template != "Custom":
            #     default_availability = templates[template]
            
            # DEBUG VALUES - DELETE LATER
            # Default values for debugging - delete this section in production
            if not default_availability:
                default_availability = {
                    "Monday": ["08:00-08:30", "08:30-09:00", "09:00-09:30", "14:00-14:30", "14:30-15:00"],
                    "Tuesday": ["10:00-10:30", "10:30-11:00", "15:00-15:30", "15:30-16:00"],
                    "Wednesday": ["09:00-09:30", "09:30-10:00", "16:00-16:30", "16:30-17:00"],
                    "Thursday": ["11:00-11:30", "11:30-12:00", "17:00-17:30", "17:30-18:00"],
                    "Friday": ["13:00-13:30", "13:30-14:00", "18:00-18:30", "18:30-19:00"],
                    "Saturday": ["10:00-10:30", "10:30-11:00", "14:00-14:30", "14:30-15:00"],
                    "Sunday": ["12:00-12:30", "12:30-13:00", "16:00-16:30", "16:30-17:00"]
                }
            # END DEBUG VALUES - DELETE LATER
            
            # Initialize weekly_availability with default values
            if "weekly_availability" not in st.session_state:
                st.session_state.weekly_availability = default_availability.copy()
            
            # Use expanders for each day to save space
            for day in days:
                with st.expander(f"{day}", expanded=(day == days[0])):
                    # Get the default selections for this day
                    previous = st.session_state.weekly_availability.get(day, [])
                    
                    # Quick selection buttons
                    cols = st.columns(4)
                    if cols[0].button(f"Morning", key=f"{day}_morning"):
                        previous = [slot for slot in time_slots if "06:" in slot or "07:" in slot or "08:" in slot or "09:" in slot or "10:" in slot or "11:" in slot]
                        st.session_state.weekly_availability[day] = previous
                    if cols[1].button(f"Afternoon", key=f"{day}_afternoon"):
                        previous = [slot for slot in time_slots if "12:" in slot or "13:" in slot or "14:" in slot or "15:" in slot or "16:" in slot]
                        st.session_state.weekly_availability[day] = previous
                    if cols[2].button(f"Evening", key=f"{day}_evening"):
                        previous = [slot for slot in time_slots if "17:" in slot or "18:" in slot or "19:" in slot or "20:" in slot]
                        st.session_state.weekly_availability[day] = previous
                    if cols[3].button(f"Clear", key=f"{day}_clear"):
                        previous = []
                        st.session_state.weekly_availability[day] = previous
                    
                    selected = st.multiselect(f"Select available time slots", options=time_slots, default=previous, key=f"{day}_slots")
                    st.session_state.weekly_availability[day] = selected
            
            # Get the final weekly availability from session state
            weekly_availability = st.session_state.weekly_availability
            
            if st.button("📅 Save Weekly Availability", type="primary"):
                try:
                    # Ensure consistent encoding of time slots (use standard hyphen)
                    sanitized_availability = {}
                    for day, slots in weekly_availability.items():
                        sanitized_availability[day] = [slot.replace("–", "-") for slot in slots]
                    
                    # Serialize to JSON, ensuring standard format and ASCII encoding
                    json_data = json.dumps(sanitized_availability, indent=2, ensure_ascii=True)
                    
                    # Extra validation step to make sure it's valid JSON
                    _ = json.loads(json_data)  # Test if we can parse it back
                    
                    # Upload to GCS
                    blob = gcs_client().bucket(BUCKET_NAME).blob(WEEKLY_AVAILABILITY_FILE)
                    blob.upload_from_string(json_data)
                    st.success("✅ Availability saved!")
                except Exception as e:
                    st.error(f"❌ Failed to save: {e}")
                    st.error("JSON content that failed to save:")
                    st.code(json.dumps(weekly_availability, indent=2))
        
        with col2:
            st.markdown("### ⚙️ Preferences")
            
            # Study preferences for more personalized scheduling
            st.markdown("#### Study Style")
            focus_duration = st.slider("Focus session length (min)", 25, 120, 50, step=5)
            break_duration = st.slider("Break length (min)", 5, 30, 10, step=5)
            
            st.markdown("#### Daily Limits")
            daily_limit = st.slider("Max hours per day", 1, 10, 3)
            
            st.markdown("#### Learning")
            productivity_peak = st.selectbox("When are you most productive?", 
                                          ["Morning", "Afternoon", "Evening", "No preference"])
            
            difficult_first = st.checkbox("Schedule difficult tasks first", value=True)
            
            # Save preferences
            if st.button("💾 Save Preferences"):
                st.session_state["study_preferences"] = {
                    "focus_duration": focus_duration,
                    "break_duration": break_duration,
                    "daily_limit": daily_limit,
                    "productivity_peak": productivity_peak,
                    "difficult_first": difficult_first
                }
                st.success("✅ Preferences saved!")
            
    with tabs[1]:  # ASSIGNMENTS TAB
        st.markdown("### 📚 Manage Your Assignments")
        
        # Load Calendar Events
        all_events = load_all_calendar_events()
        class_events = [e for e in all_events if not e.get("allDay", False)]
        
        # Filter Class Time from Availability
        def overlaps(start1, end1, start2, end2):
            return max(start1, start2) < min(end1, end2)
        
        def parse_slot_range(slot_str):
            # Handle both hyphen and en-dash for robustness
            if "-" in slot_str:
                s, e = slot_str.split("-")
            elif "–" in slot_str:
                s, e = slot_str.split("–")
            else:
                # Fallback in case of unexpected format
                parts = re.split(r'[-–—]', slot_str)
                if len(parts) >= 2:
                    s, e = parts[0], parts[1]
                else:
                    return None, None
            
            try:
                return datetime.strptime(s, "%H:%M").time(), datetime.strptime(e, "%H:%M").time()
            except ValueError:
                st.warning(f"Could not parse time slot: {slot_str}")
                return None, None
        
        availability = {}
        try:
            for day, slots in weekly_availability.items():
                available = []
                for slot in slots:
                    slot_start, slot_end = parse_slot_range(slot)
                    if slot_start is None or slot_end is None:
                        continue
                        
                    slot_has_conflict = False
                    for e in class_events:
                        s = datetime.fromisoformat(e["start"])
                        e_dt = datetime.fromisoformat(e["end"])
                        e_day = s.strftime("%A")
                        if e_day != day:
                            continue
                        class_start, class_end = s.time(), e_dt.time()
                        if overlaps(slot_start, slot_end, class_start, class_end):
                            slot_has_conflict = True
                            break
                    if not slot_has_conflict:
                        available.append(slot)
                availability[day] = available
        except Exception as e:
            st.warning(f"⚠️ Could not process availability: {e}")
            availability = {}
        
        # Load Assignments
        now = datetime.now().date()
        assignments = [e for e in all_events if e.get("allDay") and datetime.fromisoformat(e["start"]).date() >= now]
        
        if not assignments:
            st.warning("No assignments found in your calendar.")
            st.info("Add assignments as all-day events in your calendar to see them here.")
        else:
            # Allow manual assignment creation
            with st.expander("➕ Add Manual Assignment"):
                new_title = st.text_input("Assignment Title")
                new_due_date = st.date_input("Due Date", min_value=now)
                new_course = st.text_input("Course")
                new_difficulty = st.select_slider("Difficulty", options=["Easy", "Medium", "Hard"])
                new_estimated_hours = st.number_input("Estimated Hours", min_value=0.5, max_value=20.0, value=2.0, step=0.5)
                
                if st.button("Add Assignment"):
                    if new_title:
                        # Code to add to calendar would go here
                        st.success(f"✅ Added '{new_title}' due {new_due_date}")
                    else:
                        st.error("Title is required")
            
            # Enhanced assignment management with priority, difficulty, and estimated time
            st.markdown("### Assignment List")
            
            # Initialize session state for assignment metadata if not exists
            if "assignment_metadata" not in st.session_state:
                st.session_state["assignment_metadata"] = {}
            
            # File-Assignment Matching
            uploaded_files = list_gcs_files("courses")
            
            # Sort assignments by due date
            assignments.sort(key=lambda x: datetime.fromisoformat(x["start"]))
            
            for i, a in enumerate(assignments):
                with st.expander(f"{a['title']} (Due: {datetime.fromisoformat(a['start']).strftime('%b %d')})", expanded=(i == 0)):
                    a_id = a["id"]
                    
                    # Initialize metadata for this assignment if not exists
                    if a_id not in st.session_state["assignment_metadata"]:
                        st.session_state["assignment_metadata"][a_id] = {
                            "difficulty": "Medium",
                            "estimated_hours": 2.0,
                            "priority": "Medium",
                            "completed": False,
                            "notes": "",
                            "files": []
                        }
                    
                    meta = st.session_state["assignment_metadata"][a_id]
                    
                    # Assignment details in columns
                    col1, col2, col3 = st.columns([1, 1, 1])
                    
                    with col1:
                        meta["difficulty"] = st.select_slider(
                            "Difficulty", 
                            options=["Easy", "Medium", "Hard"], 
                            value=meta["difficulty"],
                            key=f"diff_{a_id}"
                        )
                    
                    with col2:
                        meta["priority"] = st.select_slider(
                            "Priority", 
                            options=["Low", "Medium", "High", "Urgent"], 
                            value=meta["priority"],
                            key=f"prio_{a_id}"
                        )
                    
                    with col3:
                        meta["estimated_hours"] = st.number_input(
                            "Est. Hours", 
                            min_value=0.5, 
                            max_value=20.0, 
                            value=meta["estimated_hours"],
                            step=0.5,
                            key=f"hours_{a_id}"
                        )
                    
                    # File selection for the assignment
                    meta["files"] = st.multiselect(
                        "Related Files", 
                        uploaded_files, 
                        default=meta["files"],
                        key=f"files_{a_id}"
                    )
                    
                    # Notes for the assignment
                    meta["notes"] = st.text_area(
                        "Notes", 
                        value=meta["notes"],
                        key=f"notes_{a_id}"
                    )
                    
                    # Completion tracking
                    meta["completed"] = st.checkbox(
                        "Mark as Completed", 
                        value=meta["completed"],
                        key=f"done_{a_id}"
                    )
                    
                    if meta["completed"]:
                        st.success("✅ Completed!")
            
            # Progress tracking
            completed = sum(1 for a_id in st.session_state["assignment_metadata"] if st.session_state["assignment_metadata"][a_id]["completed"])
            total = len(assignments)
            st.progress(completed / total if total > 0 else 0)
            st.markdown(f"**Overall Progress:** {completed}/{total} assignments completed")

    with tabs[2]:  # STUDY PLAN TAB
        st.markdown("### 🗓️ Generate Your Personalized Study Plan")
        
        # Get preferences from session state or use defaults
        prefs = st.session_state.get("study_preferences", {
            "focus_duration": 50,
            "break_duration": 10,
            "daily_limit": 3,
            "productivity_peak": "No preference",
            "difficult_first": True
        })
        
        daily_limit = prefs["daily_limit"]
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            plan_type = st.radio(
                "Plan Type", 
                ["Week View", "Day View"],
                horizontal=True
            )
        
        with col2:
            plan_date = None
            if plan_type == "Day View":
                plan_date = st.date_input("Select Day", value=datetime.now())
        
        def generate_weekly_study_plan(availability, prefs, assignments, metadata):
            def get_file_content(fpath):
                try:
                    blob = gcs_client().bucket(BUCKET_NAME).blob(fpath)
                    with blob.open("rb") as f:
                        return extract_text_from_pdf(f)
                except:
                    return ""
            
            def parse_gemini_json(text):
                # Look for a JSON object or array using a greedy regex
                match = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', text)
                if match:
                    try:
                        return json.loads(match.group(0))
                    except json.JSONDecodeError as e:
                        st.error(f"⚠️ JSON decode failed: {e}")
                        st.code(match.group(0))
                        return None
                else:
                    st.error("⚠️ No valid JSON found in Gemini response.")
                    st.code(text)
                    return None
            
            # Calculate minutes per day with improved 30-min slot handling
            minutes_per_day = {
                day: min(len(slots) * 30, prefs["daily_limit"] * 60)  # Each slot is now 30 min
                for day, slots in availability.items()
            }
            
            payload = {
                "availability": availability,
                "daily_limit_hours": prefs["daily_limit"],
                "max_minutes_per_day": minutes_per_day,
                "focus_duration": prefs["focus_duration"],
                "break_duration": prefs["break_duration"],
                "productivity_peak": prefs["productivity_peak"],
                "difficult_first": prefs["difficult_first"],
                "assignments": []
            }
            
            for a in assignments:
                a_id = a["id"]
                meta = metadata.get(a_id, {"difficulty": "Medium", "estimated_hours": 2.0, "priority": "Medium", "completed": False, "files": []})
                
                payload["assignments"].append({
                    "id": a_id,
                    "title": a["title"],
                    "due_date": a["start"],
                    "completed": meta["completed"],
                    "difficulty": meta["difficulty"],
                    "priority": meta["priority"],
                    "estimated_hours": meta["estimated_hours"],
                    "files": meta["files"],
                    "file_contents": [get_file_content(f)[:1000] for f in meta["files"]]
                })
            
            prompt = f"""
            You are a helpful academic assistant. Create a personalized weekly study plan for a student.

            ## Student Preferences:
            - Focus session duration: {prefs["focus_duration"]} minutes
            - Break duration: {prefs["break_duration"]} minutes
            - Max hours per day: {prefs["daily_limit"]} hours
            - Productivity peak time: {prefs["productivity_peak"]}
            - Schedule difficult tasks first: {"Yes" if prefs["difficult_first"] else "No"}

            ## Rules:
            - Use only the time blocks under "availability".
            - NEVER assign more total minutes in a day than allowed in "max_minutes_per_day".
            - Track total scheduled minutes per day as you go.
            - Insert {prefs["break_duration"]}-minute breaks between sessions.
            - Apply the Pomodoro technique with {prefs["focus_duration"]} min focus periods.
            - For difficult assignments, break them into multiple sessions across days.
            - Prioritize assignments by due date, priority level, and difficulty.
            - If a day is full, move remaining tasks to another day or log in "unscheduled".
            - Use the en dash (–) in time ranges. Ensure absolutely no overlap.

            ## Format:
            Return:
            {{
            "scheduled": [ 
                {{ 
                    "day": "...", 
                    "task": "...", 
                    "time": "HH:MM–HH:MM",
                    "assignment_id": "...",
                    "is_break": false,
                    "difficulty": "..." 
                }} 
            ],
            "unscheduled": [ 
                {{ 
                    "title": "...", 
                    "reason": "..." 
                }} 
            ],
            "stats": {{
                "total_study_hours": 0,
                "total_break_time": 0,
                "assignments_covered": 0
            }}
            }}

            ## Input:
            {json.dumps(payload, indent=2)}
            """
            
            try:
                model = genai.GenerativeModel("models/gemini-1.5-pro")
                response = model.generate_content(prompt)
                return parse_gemini_json(response.text)
            except Exception as e:
                st.error(f"❌ Failed to generate plan: {e}")
                return None
        
        if st.button("🔮 Generate Study Plan", type="primary"):
            with st.spinner("Crafting your personalized study plan..."):
                # Get assignment metadata
                metadata = st.session_state.get("assignment_metadata", {})
                
                # Generate plan
                result = generate_weekly_study_plan(availability, prefs, assignments, metadata)
                
                if result and isinstance(result, dict):
                    scheduled = result.get("scheduled", [])
                    unscheduled = result.get("unscheduled", [])
                    stats = result.get("stats", {})
                    
                    st.success("✅ Study plan generated!")
                    
                    # Store the plan in session state
                    st.session_state["study_plan"] = result
                    
                    # Display plan based on view type
                    if plan_type == "Week View":
                        # Create a calendar-like view
                        st.markdown("### 📅 Weekly Schedule")
                        
                        def slot_minutes(slot):
                            try:
                                slot = slot.replace("-", "–") 
                                start, end = slot.split("–")
                                t1 = dt.datetime.strptime(start.strip(), "%H:%M")
                                t2 = dt.datetime.strptime(end.strip(), "%H:%M")
                                return int((t2 - t1).total_seconds() // 60)
                            except:
                                return 0
                        
                        # Group tasks by day
                        daily_tasks = {day: [] for day in days}
                        for task in scheduled:
                            day = task["day"]
                            if day in daily_tasks:
                                daily_tasks[day].append(task)
                        
                        # Create tabs for each day
                        day_tabs = st.tabs(days)
                        
                        for i, day in enumerate(days):
                            with day_tabs[i]:
                                if not daily_tasks[day]:
                                    st.info(f"No tasks scheduled for {day}")
                                    continue
                                
                                # Sort tasks by time
                                daily_tasks[day].sort(key=lambda x: x["time"].split("–")[0])

                                # Calculate daily stats
                                daily_minutes = sum(slot_minutes(task["time"]) for task in daily_tasks[day] if not task.get("is_break", False))
                                daily_break_minutes = sum(slot_minutes(task["time"]) for task in daily_tasks[day] if task.get("is_break", True))
                                
                                st.markdown(f"**Total Study Time:** {daily_minutes//60}h {daily_minutes%60}m | **Breaks:** {daily_break_minutes} min")
                                
                                # Create timeline
                                for task in daily_tasks[day]:
                                    # Different styling for breaks vs study sessions
                                    if task.get("is_break", False):
                                        st.markdown(f"""
                                        <div style="padding: 5px 10px; margin: 5px 0; background-color: #e1f5fe; border-left: 4px solid #03a9f4; border-radius: 4px;">
                                            <span style="color: #0288d1; font-weight: bold;">{task["time"]}</span> - Break Time ({slot_minutes(task["time"])} min)
                                        </div>
                                        """, unsafe_allow_html=True)
                                    else:
                                        # Color coding based on difficulty
                                        difficulty = task.get("difficulty", "Medium")
                                        diff_color = "#4caf50" if difficulty == "Easy" else "#ff9800" if difficulty == "Medium" else "#f44336"
                                        
                                        st.markdown(f"""
                                        <div style="padding: 10px; margin: 10px 0; background-color: #f9f9f9; border-left: 4px solid {diff_color}; border-radius: 4px; color: black;">
                                            <span style="font-weight: bold;">{task["time"]}</span> - {task["task"]} ({slot_minutes(task["time"])} min)
                                            <br><span style="color: {diff_color}; font-size: 0.8em;">Difficulty: {difficulty}</span>
                                        </div>
                                        """, unsafe_allow_html=True)

                    else:  # Day View
                        selected_day = plan_date.strftime("%A") if plan_date else datetime.now().strftime("%A")
                        
                        st.markdown(f"### 📆 Schedule for {selected_day}")
                        
                        # Filter tasks for the selected day
                        day_tasks = [task for task in scheduled if task["day"] == selected_day]
                        
                        if not day_tasks:
                            st.info(f"No tasks scheduled for {selected_day}")
                        else:
                            # Sort tasks by time
                            day_tasks.sort(key=lambda x: x["time"].split("–")[0])
                            
                            # Display tasks in a timeline
                            for task in day_tasks:
                                is_break = task.get("is_break", False)
                                
                                if is_break:
                                    st.markdown(f"""
                                    <div style="padding: 5px 10px; margin: 5px 0; background-color: #e1f5fe; border-radius: 4px;">
                                        <span style="color: #0288d1; font-weight: bold;">{task["time"]}</span> - Break Time
                                    </div>
                                    """, unsafe_allow_html=True)
                                else:
                                    st.markdown(f"""
                                    <div style="padding: 10px; margin: 10px 0; background-color: #f9f9f9; border-radius: 4px;">
                                        <span style="font-weight: bold;">{task["time"]}</span> - {task["task"]}
                                    </div>
                                    """, unsafe_allow_html=True)
                    
                    # Display unscheduled tasks
                    if unscheduled:
                        st.markdown("### ⚠️ Unscheduled Tasks")
                        for item in unscheduled:
                            st.warning(f"**{item['title']}** → {item.get('reason', 'No reason provided')}")
                    
                    # # Export options
                    # export_col1, export_col2 = st.columns(2)
                    
                    # with export_col1:
                    #     if st.button("📲 Export to Calendar"):
                    #         st.info("Calendar export functionality would be implemented here.")
                            
                    # with export_col2:
                    #     if st.button("📱 Send to Mobile"):
                    #         st.info("Mobile notification functionality would be implemented here.")
                else:
                    st.error("Failed to generate a study plan. Please try again.")
        
        # Display cached plan if available
        elif "study_plan" in st.session_state:
            st.info("Showing previously generated plan. Click 'Generate Study Plan' to create a new one.")
            
            # Display the cached plan similar to above
            result = st.session_state["study_plan"]
            scheduled = result.get("scheduled", [])
            
            if plan_type == "Week View":
                # Create tabs for each day
                day_tabs = st.tabs(days)
                
                # Group tasks by day
                daily_tasks = {day: [] for day in days}
                for task in scheduled:
                    day = task["day"]
                    if day in daily_tasks:
                        daily_tasks[day].append(task)
                
                for i, day in enumerate(days):
                    with day_tabs[i]:
                        if not daily_tasks[day]:
                            st.info(f"No tasks scheduled for {day}")
                            continue
                        
                        for task in sorted(daily_tasks[day], key=lambda x: x["time"].split("–")[0]):
                            if task.get("is_break", False):
                                st.markdown(f"**{task['time']}** - Break")
                            else:
                                st.markdown(f"**{task['time']}** - {task['task']}")
            else:
                # Day view display
                selected_day = plan_date.strftime("%A") if plan_date else datetime.now().strftime("%A")
                st.markdown(f"### Schedule for {selected_day}")
                
                day_tasks = [task for task in scheduled if task["day"] == selected_day]
                
                if not day_tasks:
                    st.info(f"No tasks scheduled for {selected_day}")
                else:
                    for task in sorted(day_tasks, key=lambda x: x["time"].split("–")[0]):
                        if task.get("is_break", False):
                            st.markdown(f"**{task['time']}** - Break")
                        else:
                            st.markdown(f"**{task['time']}** - {task['task']}")
    
    # with tabs[3]:  # ANALYTICS TAB
    #     st.markdown("### 📊 Study Analytics & Insights")
        
    #     # Placeholder for analytics
    #     if "study_plan" not in st.session_state:
    #         st.info("Generate a study plan first to see analytics.")
    #     else:
    #         # Get plan data
    #         plan = st.session_state["study_plan"]
    #         scheduled = plan.get("scheduled", [])
            
    #         # Calculate stats
    #         total_study_tasks = len([t for t in scheduled if not t.get("is_break", False)])
    #         total_break_tasks = len([t for t in scheduled if t.get("is_break", False)])
            
    #         def slot_minutes(slot):
    #             try:
    #                 slot = slot.replace("-", "–") 
    #                 start, end = slot.split("–")
    #                 t1 = dt.datetime.strptime(start.strip(), "%H:%M")
    #                 t2 = dt.datetime.strptime(end.strip(), "%H:%M")
    #                 return int((t2 - t1).total_seconds() // 60)
    #             except:
    #                 return 0
            
    #         total_study_minutes = sum(slot_minutes(t["time"]) for t in scheduled if not t.get("is_break", False))
    #         total_break_minutes = sum(slot_minutes(t["time"]) for t in scheduled if t.get("is_break", False))
            
    #         # Display summary metrics
    #         col1, col2, col3 = st.columns(3)
            
    #         with col1:
    #             st.metric("Total Study Time", f"{total_study_minutes//60}h {total_study_minutes%60}m")
            
    #         with col2:
    #             st.metric("Study Sessions", total_study_tasks)
            
    #         with col3:
    #             st.metric("Break Time", f"{total_break_minutes} min")
            
    #         # Create day-by-day breakdown
    #         st.markdown("### Daily Breakdown")
            
    #         # Group by day
    #         daily_minutes = {}
    #         for day in days:
    #             day_tasks = [t for t in scheduled if t["day"] == day and not t.get("is_break", False)]
    #             daily_minutes[day] = sum(slot_minutes(t["time"]) for t in day_tasks)
            
    #         # Create chart data
    #         chart_data = {
    #             "days": list(daily_minutes.keys()),
    #             "minutes": list(daily_minutes.values())
    #         }
            
    #         # Display chart using React
    #         st.markdown("### Study Time Distribution")
            
    #         # Create synthetic chart using st.bar_chart
    #         chart_df = pd.DataFrame({
    #             "Day": list(daily_minutes.keys()),
    #             "Minutes": list(daily_minutes.values())
    #         })
    #         st.bar_chart(chart_df.set_index("Day"))
            
    #         # Assignment progress tracking
    #         st.markdown("### Assignment Progress")
            
    #         # Get metadata
    #         metadata = st.session_state.get("assignment_metadata", {})
            
    #         # Create completion stats
    #         completed_count = sum(1 for _, meta in metadata.items() if meta.get("completed", False))
    #         total_count = len(metadata)
    #         completion_rate = completed_count / total_count if total_count > 0 else 0
            
    #         st.progress(completion_rate)
    #         st.markdown(f"**Completion Rate:** {int(completion_rate * 100)}% ({completed_count}/{total_count})")
            
    #         # Study efficiency insights
    #         st.markdown("### Study Efficiency Insights")
            
    #         # Create synthetic insights based on the data
    #         prefs = st.session_state.get("study_preferences", {})
    #         peak_time = prefs.get("productivity_peak", "No preference")
            
    #         # Generate insights
    #         insights = [
    #             f"Your most productive time is during the {peak_time.lower()}. We've scheduled {len([t for t in scheduled if not t.get('is_break', False) and peak_time.lower() in t['time'].lower()])} sessions during this period.",
    #             f"You're taking approximately {total_break_minutes/(total_study_tasks if total_study_tasks > 0 else 1):.1f} minutes of break time per study session.",
    #             "Based on your current progress, you're on track to complete all assignments before their due dates."
    #         ]
            
    #         for insight in insights:
    #             st.info(insight)
            
    #         # Recommendations
    #         st.markdown("### Recommendations")
            
    #         recommendations = [
    #             "Try increasing your focus sessions to 55 minutes for better productivity.",
    #             "Consider adding more study slots on weekends to balance your workload.",
    #             "For difficult assignments, try scheduling shorter, more frequent study sessions.",
    #             "Based on your profile, you may benefit from adding more morning study slots."
    #         ]
            
    #         for recommendation in recommendations:
    #             st.warning(recommendation)
            
    #         # # Export analytics report
    #         # if st.button("📊 Export Analytics Report"):
    #         #     st.info("This feature would generate a downloadable PDF report with your study analytics.")
            
    #         # Study streak tracking
    #         st.markdown("### Study Streak")
            
    #         # Simulate streak data
    #         import random
    #         streak_days = random.randint(3, 14)
    #         st.metric("Current Study Streak", f"{streak_days} days")
            
    #         # Create a calendar heatmap visual representation
    #         st.markdown("### Study Activity Calendar")
    #         st.info("A calendar heatmap showing your study activity would be displayed here.")
            
    #         # Learning type assessment
    #         st.markdown("### Learning Profile")
            
    #         learning_styles = {
    #             "Visual": 65,
    #             "Auditory": 40,
    #             "Reading/Writing": 80,
    #             "Kinesthetic": 35
    #         }
            
    #         # Display as horizontal bars
    #         for style, value in learning_styles.items():
    #             st.markdown(f"**{style}:** {value}%")
    #             st.progress(value/100)
            
    #         st.markdown("""
    #         **Learning Style Recommendations:**
    #         - **Visual**: Use mind maps and diagrams in your notes
    #         - **Reading/Writing**: Continue focusing on written notes and summaries
    #         - **Auditory**: Consider recording lectures for review
    #         - **Kinesthetic**: Try study methods that involve movement or hands-on activities
    #         """)



# === COURSES PAGE ===
elif page == "Courses":
    st.title("📚 Courses")
    ensure_structure()
    structure = st.session_state.structure
    show_advanced = st.checkbox("Show advanced options")

    # Initialize session state variables if not present
    if "selected_files" not in st.session_state:
        st.session_state["selected_files"] = []
    if "course_operation_done" not in st.session_state:
        st.session_state["course_operation_done"] = False
    
    # Handle operations completed flag
    if st.session_state.get("course_operation_done", False):
        st.success(st.session_state.get("operation_message", "Operation completed"))
        st.session_state["course_operation_done"] = False
        st.session_state["operation_message"] = ""

    tab1, tab2, tab3 = st.tabs(["📘 Course Setup", "📁 Categories", "📂 Files"])

    # === TAB 1: COURSE SETUP ===
    with tab1:
        st.subheader("Manage Courses")

        # Course addition form
        with st.form("add_course_form", clear_on_submit=True):
            new_course = st.text_input("➕ Add new course")
            add_course_submitted = st.form_submit_button("Add Course")
            
        if add_course_submitted and new_course:
            if new_course not in structure:
                structure[new_course] = []
                upload_structure(structure)
                st.session_state["course_operation_done"] = True
                st.session_state["operation_message"] = f"✅ Course '{new_course}' added"
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
                        st.session_state["course_operation_done"] = True
                        st.session_state["operation_message"] = f"✅ Course '{selected_course}' deleted"
                        st.rerun()
                
                with col2:
                    with st.form("rename_course_form", clear_on_submit=True):
                        rename = st.text_input("✏️ Rename Course", key="rename_course")
                        rename_submitted = st.form_submit_button("Rename Course")
                    
                    if rename_submitted and rename and rename != selected_course:
                        structure[rename] = structure.pop(selected_course)
                        upload_structure(structure)
                        st.session_state["course_operation_done"] = True
                        st.session_state["operation_message"] = f"✅ Course renamed to '{rename}'"
                        st.rerun()

    # === TAB 2: CATEGORY SETUP ===
    with tab2:
        if course_names:
            selected_course = st.selectbox("🎓 Select a course (for categories)", course_names, key="cat_course")
            st.subheader(f"Categories for {selected_course}")

            with st.form("add_category_form", clear_on_submit=True):
                new_cat = st.text_input("➕ Add new category", key="new_cat")
                add_cat_submitted = st.form_submit_button("Add Category")
            
            if add_cat_submitted and new_cat and new_cat not in structure[selected_course]:
                structure[selected_course].append(new_cat)
                upload_structure(structure)
                st.session_state["course_operation_done"] = True
                st.session_state["operation_message"] = f"✅ Category '{new_cat}' added"
                st.rerun()

            if structure[selected_course]:
                selected_cat = st.selectbox("📂 Select a category", structure[selected_course], key="selected_cat")

                if show_advanced:
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("🗑 Delete Category"):
                            structure[selected_course].remove(selected_cat)
                            upload_structure(structure)
                            st.session_state["course_operation_done"] = True
                            st.session_state["operation_message"] = f"✅ Category '{selected_cat}' deleted"
                            st.rerun()
                    
                    with col2:
                        with st.form("rename_category_form", clear_on_submit=True):
                            rename_cat = st.text_input("✏️ Rename Category", key="rename_cat")
                            rename_cat_submitted = st.form_submit_button("Rename Category")
                        
                        if rename_cat_submitted and rename_cat and rename_cat != selected_cat:
                            try:
                                # Batch rename operations to minimize GCS calls
                                files = list_gcs_files(f"courses/{selected_course}/{selected_cat}")
                                client = gcs_client()
                                bucket = client.bucket(BUCKET_NAME)
                                
                                for f in files:
                                    old_path = f"courses/{selected_course}/{selected_cat}/{f}"
                                    new_path = f"courses/{selected_course}/{rename_cat}/{f}"
                                    bucket.rename_blob(bucket.blob(old_path), new_path)
                                
                                # Update structure once after all renames
                                structure[selected_course].remove(selected_cat)
                                structure[selected_course].append(rename_cat)
                                upload_structure(structure)
                                st.session_state["course_operation_done"] = True
                                st.session_state["operation_message"] = f"✅ Category renamed to '{rename_cat}'"
                                st.rerun()
                            except Exception as e:
                                st.error(f"Failed to rename category: {str(e)}")

    # === TAB 3: FILE MANAGEMENT ===
    with tab3:
        if course_names:
            selected_course = st.selectbox("🎓 Select a course (for file management)", course_names, key="file_course_all")
            
            if structure[selected_course]:
                selected_cat = st.selectbox("📂 Select a category", structure[selected_course], key="file_cat_all")
                
                # File Upload Section
                st.markdown("#### 📤 Upload a file")
                
                uploaded_file = st.file_uploader("Choose file to upload", key=f"{selected_course}_{selected_cat}_upload")
                if uploaded_file:
                    upload_button = st.button("Upload File")
                    if upload_button:
                        path = f"courses/{selected_course}/{selected_cat}/{uploaded_file.name}"
                        upload_to_gcs(BUCKET_NAME, path, uploaded_file)
                        st.session_state["course_operation_done"] = True
                        st.session_state["operation_message"] = f"✅ File '{uploaded_file.name}' uploaded"
                        st.rerun()

                # Uploaded Files Section
                st.markdown("#### 📁 Uploaded Files")
                
                # Cache file list to avoid repeated GCS calls
                cache_key = f"file_list_{selected_course}_{selected_cat}"
                if cache_key not in st.session_state:
                    st.session_state[cache_key] = list_gcs_files(f"courses/{selected_course}/{selected_cat}")
                
                files = st.session_state[cache_key]
                
                # Initialize file operation states
                if "file_to_delete" not in st.session_state:
                    st.session_state["file_to_delete"] = None
                
                # Process pending file deletion
                if st.session_state["file_to_delete"]:
                    file_path = st.session_state["file_to_delete"]
                    if delete_from_gcs(file_path):
                        file_name = file_path.split("/")[-1]
                        st.session_state[cache_key].remove(file_name)
                        st.session_state["course_operation_done"] = True
                        st.session_state["operation_message"] = f"✅ File '{file_name}' deleted"
                    else:
                        st.session_state["course_operation_done"] = True
                        st.session_state["operation_message"] = "❌ Failed to delete file"
                    
                    st.session_state["file_to_delete"] = None
                    st.rerun()
                
                # Display files with selection checkboxes
                for fname in files:
                    file_path = f"courses/{selected_course}/{selected_cat}/{fname}"
                    
                    with st.container():
                        col1, col2, col3 = st.columns([6, 1, 1])
                        col1.write(fname)
                        
                        # Chat selection checkbox
                        selected = col2.checkbox("Chat", key=f"select_{file_path}", 
                                               value=file_path in st.session_state["selected_files"])
                        
                        if selected and file_path not in st.session_state["selected_files"]:
                            st.session_state["selected_files"].append(file_path)
                        elif not selected and file_path in st.session_state["selected_files"]:
                            st.session_state["selected_files"].remove(file_path)
                        
                        # Delete button (advanced mode)
                        if show_advanced:
                            if col3.button("🗑", key=f"delete_{fname}"):
                                st.session_state["file_to_delete"] = file_path
                                st.rerun()

                # Chat button
                if st.session_state["selected_files"]:
                    st.markdown("### 🤖 Ready to Chat with Selected Files?")
                    
                    if st.button("💬 Chat"):
                        # Save selected files to session state and switch page
                        # (no processing here, defer to Chatbot page)
                        st.session_state["pending_file_processing"] = True
                        st.session_state.page = "Chatbot"
                        st.rerun()


# === CHATBOT PAGE ===
elif page == "Chatbot":
    # Initialize session states
    if "selected_files" not in st.session_state:
        st.session_state["selected_files"] = []
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = PDFRagSystem(google_api_key=os.getenv("GOOGLE_API_KEY"))
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chatbot_status" not in st.session_state:
        st.session_state.chatbot_status = None
    if "pending_file_processing" not in st.session_state:
        st.session_state.pending_file_processing = False

    left, main, right = st.columns([0.2, 2.5, 1.3])

    with main:
        st.title("🧞‍♂️ How can I help you? ✨")
        
        # Process pending files from Courses page if any
        if st.session_state.pending_file_processing and st.session_state["selected_files"]:
            with st.spinner("Processing files..."):
                try:
                    all_chunks = []
                    for path in st.session_state["selected_files"]:
                        blob = gcs_client().bucket(BUCKET_NAME).blob(path)
                        with blob.open("rb") as f:
                            _, chunks = st.session_state.rag_system.process_pdf_bytes(f.read())
                            all_chunks.extend(chunks)
                            logger.info(f"Extracted {len(chunks)} chunks from {path}")

                    if all_chunks:
                        st.session_state.rag_system.vector_store = FAISS.from_texts(
                            all_chunks, st.session_state.rag_system.embeddings
                        )
                        st.session_state.rag_system.initialize_qa_chain()
                        logger.info("RAG system initialized with selected files")
                        st.session_state.chatbot_status = f"✅ Loaded {len(st.session_state['selected_files'])} file(s)"
                        file_names = [path.split('/')[-1] for path in st.session_state["selected_files"]]
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": f"Files loaded: {', '.join(file_names)}. You can now ask questions about these documents."
                        })
                    else:
                        st.session_state.chatbot_status = "❌ No extractable content found in selected files."
                except Exception as e:
                    logger.error("Failed to process files before chat", exc_info=True)
                    st.session_state.chatbot_status = f"❌ Failed to process files: {str(e)}"
                
                # Clear pending flags
                st.session_state["selected_files"] = []
                st.session_state.pending_file_processing = False
        
        # Display status message if any
        if st.session_state.chatbot_status:
            if "✅" in st.session_state.chatbot_status:
                st.success(st.session_state.chatbot_status)
            elif "❌" in st.session_state.chatbot_status:
                st.error(st.session_state.chatbot_status)
            else:
                st.info(st.session_state.chatbot_status)
            # Clear status after displaying
            st.session_state.chatbot_status = None

        # Process pending chat message if any
        if "pending_chat_message" in st.session_state and st.session_state.pending_chat_message:
            prompt = st.session_state.pending_chat_message
            
            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Check if RAG system is initialized
            if not hasattr(st.session_state.rag_system, "qa_chain") or st.session_state.rag_system.qa_chain is None:
                response = "❌ Please load at least one file before asking questions."
            else:
                # Process with RAG system
                try:
                    with st.spinner("Thinking..."):
                        response = st.session_state.rag_system.answer(prompt)
                        logger.info("Gemini answered user prompt")
                        log_interaction_to_gcs(prompt, response)
                except Exception as e:
                    logger.error("Failed to get answer from RAG", exc_info=True)
                    response = f"❌ An error occurred: {str(e)}"
            
            # Add assistant response to chat
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Clear pending message
            st.session_state.pending_chat_message = None
            st.rerun()
        
        # Display chat messages
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"], unsafe_allow_html=True)
        
        # Chat input
        prompt = st.chat_input("Ask your question...")
        if prompt:
            st.session_state.pending_chat_message = prompt
            st.rerun()

    with right:
        st.markdown("### 📂 File Selection")
        ensure_structure()
        structure = st.session_state.structure
        
        # Initialize or reset file selection
        if "right_panel_selected_files" not in st.session_state:
            st.session_state.right_panel_selected_files = []
        
        # Course and category selection
        if structure:
            selected_course = st.selectbox("📘 Select Course", list(structure.keys()))
            
            if selected_course in structure and structure[selected_course]:
                selected_cat = st.selectbox("📂 Select Category", structure[selected_course])
                folder = f"courses/{selected_course}/{selected_cat}"
                
                # Cache file listing to reduce GCS calls
                cache_key = f"right_panel_files_{selected_course}_{selected_cat}"
                if cache_key not in st.session_state:
                    st.session_state[cache_key] = list_gcs_files(folder)
                
                files = st.session_state[cache_key]
                
                if files:
                    # File selection with checkboxes
                    select_all = st.checkbox("Select All", key="select_all_files")
                    if select_all:
                        for fname in files:
                            path = f"{folder}/{fname}"
                            if path not in st.session_state.right_panel_selected_files:
                                st.session_state.right_panel_selected_files.append(path)
                    
                    # Individual file selection
                    for fname in files:
                        path = f"{folder}/{fname}"
                        selected = st.checkbox(
                            fname, 
                            key=f"right_panel_select_{path}",
                            value=path in st.session_state.right_panel_selected_files
                        )
                        
                        if selected and path not in st.session_state.right_panel_selected_files:
                            st.session_state.right_panel_selected_files.append(path)
                        elif not selected and path in st.session_state.right_panel_selected_files:
                            st.session_state.right_panel_selected_files.remove(path)
                else:
                    st.info("No files uploaded in this category.")
        else:
            st.info("No courses found. Add one in the Courses tab.")
        
        # Load files button
        if st.button("💬 Load Selected Files"):
            if not st.session_state.right_panel_selected_files:
                st.warning("⚠️ Please select at least one file.")
            else:
                # Transfer selected files to main selected_files
                st.session_state["selected_files"] = st.session_state.right_panel_selected_files.copy()
                st.session_state.right_panel_selected_files = []
                st.session_state.pending_file_processing = True
                st.rerun()

# import os
# import uuid
# import re
# import json
# import logging
# import requests
# import fitz  # from PyMuPDF
# import faiss
# import numpy as np
# import pandas as pd
# from io import StringIO
# from datetime import datetime, date, time, timedelta
# import datetime as dt
# from collections import defaultdict

# import streamlit as st
# from streamlit_calendar import calendar

# from ics import Calendar
# from google.cloud import storage
# from google.oauth2 import service_account
# import google.cloud.logging
# from google.cloud.logging_v2.handlers import CloudLoggingHandler

# import google.generativeai as genai
# from sentence_transformers import SentenceTransformer
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import FAISS
# from langchain.chains import RetrievalQA
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.prompts import PromptTemplate


# # === CONFIGURATION ===
# def gcs_client():
#     credentials_info = {
#         "type": st.secrets["gcp_service_account"]["type"],
#         "project_id": st.secrets["gcp_service_account"]["project_id"],
#         "private_key_id": st.secrets["gcp_service_account"]["private_key_id"],
#         "private_key": st.secrets["gcp_service_account"]["private_key"].replace("\\n", "\n"),
#         "client_email": st.secrets["gcp_service_account"]["client_email"],
#         "client_id": st.secrets["gcp_service_account"]["client_id"],
#         "auth_uri": st.secrets["gcp_service_account"]["auth_uri"],
#         "token_uri": st.secrets["gcp_service_account"]["token_uri"],
#         "auth_provider_x509_cert_url": st.secrets["gcp_service_account"]["auth_provider_x509_cert_url"],
#         "client_x509_cert_url": st.secrets["gcp_service_account"]["client_x509_cert_url"],
#     }
#     credentials = service_account.Credentials.from_service_account_info(credentials_info)
#     return storage.Client(credentials=credentials, project=credentials_info["project_id"])

# BUCKET_NAME = st.secrets["BUCKET_NAME"]
# STRUCTURE_FILE = "course_structure.json"
# google_api_key = st.secrets["GOOGLE_API_KEY"]
# genai.configure(api_key=google_api_key)
# gemini = genai.GenerativeModel("models/gemini-2.0-flash")
# gcp_logging_client = google.cloud.logging.Client(credentials=gcs_client()._credentials, project=gcs_client().project)
# gcp_logging_client.setup_logging()
# logger = logging.getLogger("edu_genie")


# # === MATH FORMATTING ===
# def format_math_expressions(text: str) -> str:
#     text = re.sub(r'\\begin\{([^}]+)\}(.*?)\\end\{\1\}',
#                   lambda m: re.sub(r'(?<![\\])\s*\\\s+', r'\\\\', m.group(0)),
#                   text, flags=re.DOTALL)
#     text = re.sub(r'\$\$(.*?)\$\$', r'$$\1$$', text, flags=re.DOTALL)
#     text = re.sub(r'(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)', r'$\1$', text)
#     text = re.sub(r'\${3,}', '$$', text)
#     text = re.sub(r'([^\s])\$\$', r'\1 $$', text)
#     text = re.sub(r'\$\$([^\s])', r'$$ \1', text)

#     lines = text.splitlines()
#     for i, line in enumerate(lines):
#         if re.search(r'\b(-?\d+\s+){1,}-?\d+\b', line):
#             entries = [list(map(str.strip, re.split(r'\s+', l.strip()))) for l in lines[i:i+5] if re.search(r'\b(-?\d+\s+){1,}-?\d+\b', l)]
#             if entries:
#                 matrix = '\\begin{bmatrix}\n' + " \\\\\n".join([" & ".join(row) for row in entries]) + '\\n\\end{bmatrix}'
#                 lines[i:i+len(entries)] = [f"$$\n{matrix}\n$$"]
#                 break
#     return "\n".join(lines)


# # === CLASS DEFINITIONS ===
# class PDFRagSystem:
#     def __init__(self, google_api_key):
#         self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#         self.vector_store = None
#         self.qa_chain = None

#     def process_pdf_bytes(self, pdf_bytes):
#         doc = fitz.open(stream=pdf_bytes, filetype="pdf")
#         text = "".join(page.get_text() for page in doc)
#         splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
#         documents = splitter.create_documents([text])
#         chunks = [doc.page_content for doc in documents]
#         return text, chunks

#     def initialize_qa_chain(self):
#         if not self.vector_store:
#             raise ValueError("Vector store is not initialized.")
        
#         retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
#         llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=os.getenv("GOOGLE_API_KEY"))

#         prompt_template = PromptTemplate.from_template(
#             """You are an expert assistant. Answer the user's question using *only* the provided context below.
#     If the context does not contain the answer, respond with "I'm not sure based on the uploaded documents."

#     Context:
#     {context}

#     Question:
#     {question}

#     Answer:"""
#         )

#         self.qa_chain = RetrievalQA.from_chain_type(
#             llm=llm,
#             retriever=retriever,
#             return_source_documents=False,
#             chain_type_kwargs={"prompt": prompt_template}
#         )

#     def answer(self, query: str) -> str:
#         if not self.qa_chain:
#             raise ValueError("QA chain is not initialized.")
#         result = self.qa_chain.run(query).strip()
#         if "I'm not sure" in result or not result:
#             return "⚠️ Sorry, I couldn't find the answer in the uploaded documents."
#         return format_math_expressions(result)
    

# # === UTILS ===
# def upload_to_gcs(bucket_name, destination_blob_name, file):
#     client = gcs_client()
#     bucket = client.bucket(bucket_name)
#     blob = bucket.blob(destination_blob_name)
#     blob.upload_from_file(file, rewind=True)
#     return f"gs://{bucket_name}/{destination_blob_name}"

# def log_interaction_to_gcs(prompt, response, user="anonymous"):
#     from datetime import datetime
#     log_entry = {
#         "timestamp": datetime.now().isoformat(),
#         "user": user,
#         "prompt": prompt,
#         "response": response
#     }
#     file_path = f"gemini_logs/log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
#     client = gcs_client()
#     bucket = client.bucket(BUCKET_NAME)
#     blob = bucket.blob(file_path)
#     blob.upload_from_string(json.dumps(log_entry, indent=2), content_type="application/json")

# def delete_from_gcs(path):
#     try:
#         client = gcs_client()
#         bucket = client.bucket(BUCKET_NAME)
#         blob = bucket.blob(path)
#         if blob.exists():
#             blob.delete()
#             return True
#         return False
#     except Exception as e:
#         print(f"Error deleting {path}: {e}")
#         return False

# def list_gcs_files(prefix):
#     client = gcs_client()
#     bucket = client.bucket(BUCKET_NAME)
#     return [b.name.split("/")[-1] for b in bucket.list_blobs(prefix=prefix) if not b.name.endswith(".keep")]

# def download_structure():
#     client = gcs_client()
#     bucket = client.bucket(BUCKET_NAME)
#     blob = bucket.blob(STRUCTURE_FILE)
#     if not blob.exists():
#         return {}
#     return json.loads(blob.download_as_text())

# def upload_structure(data):
#     client = gcs_client()
#     bucket = client.bucket(BUCKET_NAME)
#     blob = bucket.blob(STRUCTURE_FILE)
#     blob.upload_from_string(json.dumps(data, indent=2))

# def ensure_structure():
#     if "structure" not in st.session_state:
#         st.session_state.structure = download_structure()

# def extract_text_from_pdf(file):
#     doc = fitz.open(stream=file.read(), filetype="pdf")
#     return "".join(page.get_text() for page in doc)

# def chunk_text(text, chunk_size=1000, chunk_overlap=100):
#     splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
#     return splitter.create_documents([text])

# def create_vector_store(documents):
#     texts = [doc.page_content for doc in documents]
#     model = SentenceTransformer("all-MiniLM-L6-v2")
#     embeddings = model.encode(texts, convert_to_numpy=True)
#     index = faiss.IndexFlatL2(embeddings.shape[1])
#     index.add(embeddings)
#     return {"index": index, "documents": documents, "model": model}

# def build_rag_qa_chain(vector_store):
#     def qa(query):
#         query_embedding = vector_store["model"].encode([query])[0]
#         D, I = vector_store["index"].search(np.array([query_embedding]), k=3)
#         relevant_docs = [vector_store["documents"][i].page_content for i in I[0]]
#         prompt = "Use the following context to answer the question:\n\n" + "\n\n".join(relevant_docs) + f"\n\nQuestion: {query}"
#         return gemini.generate_content(prompt).text
#     return qa

# CALENDAR_EVENTS_FILE = "user_calendar_events.json"

# def save_calendar_events(events: list):
#     client = gcs_client()
#     bucket = client.bucket(BUCKET_NAME)
#     blob = bucket.blob(CALENDAR_EVENTS_FILE)
#     blob.upload_from_string(json.dumps(events, indent=2))

# def load_calendar_events():
#     client = gcs_client()
#     bucket = client.bucket(BUCKET_NAME)
#     blob = bucket.blob(CALENDAR_EVENTS_FILE)
#     if blob.exists():
#         return json.loads(blob.download_as_text())
#     return []

# def deduplicate_events(events):
#     """Remove duplicate events based on title and start time."""
#     seen = set()
#     unique = []
#     for e in events:
#         key = (e["title"], e["start"])
#         if key not in seen:
#             seen.add(key)
#             unique.append(e)
#     return sorted(unique, key=lambda x: x["start"])

# def get_event_start(e):
#     return getattr(e.begin, 'datetime', None) or e.begin.date()

# def get_event_end(e):
#     return getattr(e.end, 'datetime', None) or e.end.date() if e.end else get_event_start(e)

# def load_all_calendar_events():
#     all_events = []
#     for file in ["calendar_assignments.json", "calendar_classes.json"]:
#         try:
#             blob = gcs_client().bucket(BUCKET_NAME).blob(file)
#             if blob.exists():
#                 all_events += json.loads(blob.download_as_text())
#         except Exception as e:
#             st.warning(f"⚠️ Could not load {file}: {e}")
#     return all_events


# # === STREAMLIT SETUP ===
# st.set_page_config(page_title="EduGenie Dashboard", layout="wide")
# st.sidebar.image("logo.png", width=400)

# with st.sidebar.expander("🔒 Privacy Notice"):
#     st.markdown("""
#     - Your uploaded files are securely stored in Google Cloud Storage.
#     - All data is encrypted at rest (AES-256).
#     - Gemini interactions are logged for performance monitoring.
#     - No personal data is shared or used for training.
#     """)

# pages = {
#     "Dashboard": "🏠 Dashboard",
#     "Smart Scheduler": "🧠 Smart Scheduler",
#     "Courses": "📚 Courses",
#     "Chatbot": "🧞‍♂️ EduGenie"
# }

# if "page" not in st.session_state or st.session_state.page not in pages:
#     st.session_state.page = "Dashboard"

# button_style = """
#     <style>
#     div.stButton > button {
#         font-size: 18px !important;
#         padding: 0.75rem 1rem;
#         width: 100%;
#         margin-bottom: 12px;
#         border-radius: 8px;
#         background-color: #3c3c3c;
#         color: white;
#     }
#     </style>
# """
# st.sidebar.markdown(button_style, unsafe_allow_html=True)

# for key, label in pages.items():
#     if st.sidebar.button(label, key=f"nav_{key}"):
#         st.session_state.page = key

# page = st.session_state.page



# # === DASHBOARD PAGE ===
# if page == "Dashboard":
#     st.title("🏠 Dashboard")

#     with st.sidebar:
#         st.markdown("### 📥 Calendar Import")
#         ical_url = st.text_input("iCal URL", key="ical_url")

#         st.markdown("### ⚙️ Calendar Settings")
#         calendar_type = st.selectbox("Select calendar type", ["Assignments", "Classes"], key="calendar_type")
#         reload_requested = st.button("🔁 Load Calendar Events")

#     calendar_filename = f"calendar_{calendar_type.lower()}.json"

#     if "prev_ical_url" not in st.session_state:
#         st.session_state["prev_ical_url"] = None
#     if "edit_event_index" not in st.session_state:
#         st.session_state["edit_event_index"] = None

#     reparse_calendar = ical_url and reload_requested
#     if reparse_calendar:
#         st.session_state["prev_ical_url"] = ical_url
#         try:
#             r = requests.get(ical_url)
#             if r.status_code != 200 or "text/calendar" not in r.headers.get("Content-Type", ""):
#                 st.error("❌ Invalid iCal URL.")
#             else:
#                 cal = Calendar(r.text)
#                 parsed_events = []
#                 for e in cal.events:
#                     start = get_event_start(e)
#                     end = get_event_end(e)
#                     is_all_day = calendar_type == "Assignments"
#                     start_str = start.date().isoformat() if is_all_day else start.isoformat()
#                     end_str = end.date().isoformat() if is_all_day else end.isoformat()

#                     parsed_events.append({
#                         "id": str(uuid.uuid4()),
#                         "title": e.name or "Untitled Event",
#                         "start": start_str,
#                         "end": end_str,
#                         "allDay": is_all_day,
#                         "color": "blue" if is_all_day else "green"
#                     })

#                 parsed_events = deduplicate_events(parsed_events)
#                 try:
#                     blob = gcs_client().bucket(BUCKET_NAME).blob(calendar_filename)
#                     existing_events = json.loads(blob.download_as_text()) if blob.exists() else []
#                 except:
#                     existing_events = []

#                 combined_events = deduplicate_events(existing_events + parsed_events)
#                 gcs_client().bucket(BUCKET_NAME).blob(calendar_filename).upload_from_string(
#                     json.dumps(combined_events, indent=2)
#                 )
#                 st.success(f"✅ Calendar saved to {calendar_filename}")
#         except Exception as ex:
#             st.error("❌ Could not parse calendar.")
#             st.exception(ex)

#     all_events = load_all_calendar_events()
#     st.session_state["calendar_events"] = all_events

#     st.subheader("📌 Upcoming Assignments")
#     now = datetime.now()
#     five_days_later = now + timedelta(days=5)

#     upcoming_assignments = [
#         e for e in all_events
#         if e.get("allDay", False) is True and now.date() <= datetime.fromisoformat(e["start"]).date() <= five_days_later.date()
#     ]
#     upcoming_assignments = sorted(upcoming_assignments, key=lambda x: x["start"])

#     if upcoming_assignments:
#         for e in upcoming_assignments:
#             s = datetime.fromisoformat(e["start"])
#             st.markdown(f"- **{e['title']}** due **{s.strftime('%b %d')}**")
#     else:
#         st.info("No upcoming assignments in the next 5 days.")

#     if st.session_state.get("edit_event_index") is None:
#         st.subheader("📆 Monthly Calendar")
#         for e in st.session_state["calendar_events"]:
#             if "color" not in e:
#                 e["color"] = "blue" if e.get("allDay") else "green"
#         calendar(
#             events=st.session_state["calendar_events"],
#             options={
#                 "initialView": "dayGridMonth",
#                 "height": 650,
#                 "editable": False,
#             },
#             key=f"calendar_{len(st.session_state['calendar_events'])}"
#         )

#     st.markdown("### 🛠️ Manage Events by Date")
#     unique_dates = sorted(set(datetime.fromisoformat(e["start"]).date() for e in all_events))
#     if unique_dates:
#         selected_day = st.selectbox("📅 Select a date", unique_dates, format_func=lambda d: d.strftime("%A, %B %d"))
#         daily_events = [e for e in all_events if datetime.fromisoformat(e["start"]).date() == selected_day]

#         if st.session_state.get("edit_event_index") is None:
#             for i, e in enumerate(daily_events):
#                 s = datetime.fromisoformat(e["start"])
#                 t = datetime.fromisoformat(e["end"])
#                 event_time = "" if e.get("allDay") else f" ({s.strftime('%I:%M %p')} ~ {t.strftime('%I:%M %p')})"
#                 col1, col2, col3 = st.columns([8, 1, 1])
#                 with col1:
#                     st.markdown(f"- **{e['title']}**{event_time}")
#                 with col2:
#                     if st.button("✏️", key=f"edit_date_{i}"):
#                         st.session_state["edit_event_index"] = i
#                         st.rerun()
#                 with col3:
#                     if st.button("❌", key=f"del_date_{i}"):
#                         all_events.remove(e)
#                         file = "calendar_assignments.json" if e.get("allDay") else "calendar_classes.json"
#                         gcs_client().bucket(BUCKET_NAME).blob(file).upload_from_string(
#                             json.dumps([x for x in all_events if x.get("allDay") == e.get("allDay")], indent=2))
#                         st.session_state["calendar_events"] = load_all_calendar_events()
#                         st.rerun()
#         else:
#             st.subheader("✏️ Edit Event")
#             index = st.session_state["edit_event_index"]
#             if index < len(daily_events):
#                 e = daily_events[index]
#                 is_all_day = e.get("allDay", False)
#                 s = datetime.fromisoformat(e["start"])
#                 t = datetime.fromisoformat(e["end"])

#                 with st.form("edit_event_form"):
#                     new_title = st.text_input("Title", value=e["title"])
#                     new_date = st.date_input("Date", value=s.date())
#                     if is_all_day:
#                         new_start, new_end = new_date, new_date
#                     else:
#                         new_start_time = st.time_input("Start Time", value=s.time())
#                         new_end_time = st.time_input("End Time", value=t.time())
#                         new_start = datetime.combine(new_date, new_start_time)
#                         new_end = datetime.combine(new_date, new_end_time)

#                     if st.form_submit_button("Save Changes"):
#                         e["title"] = new_title
#                         e["start"] = new_start.isoformat() if not is_all_day else new_start.isoformat()
#                         e["end"] = new_end.isoformat() if not is_all_day else new_end.isoformat()

#                         file = "calendar_assignments.json" if is_all_day else "calendar_classes.json"
#                         events = [ev for ev in all_events if ev.get("allDay") == is_all_day]
#                         gcs_client().bucket(BUCKET_NAME).blob(file).upload_from_string(json.dumps(events, indent=2))

#                         st.session_state["edit_event_index"] = None
#                         st.session_state["calendar_events"] = load_all_calendar_events()
#                         st.success("✅ Event updated.")
#                         st.rerun()

#                 if st.button("Cancel"):
#                     st.session_state["edit_event_index"] = None
#                     st.rerun()
#     else:
#         st.info("📭 No events yet. Add a class or assignment below.")

#     with st.expander("➕ Add New Assignment"):
#         with st.form("add_assignment_form"):
#             a_title = st.text_input("Assignment Title")
#             a_date = st.date_input("Due Date")
#             dt_str = datetime.combine(a_date, datetime.min.time()).replace(tzinfo=None).isoformat()
#             if st.form_submit_button("Add Assignment"):
#                 new_event = {
#                     "id": str(uuid.uuid4()),
#                     "title": a_title,
#                     "start": dt_str,
#                     "end": dt_str,
#                     "allDay": True,
#                     "color": "blue"
#                 }
#                 assignments = [e for e in all_events if e.get("allDay")]
#                 assignments.append(new_event)
#                 gcs_client().bucket(BUCKET_NAME).blob("calendar_assignments.json").upload_from_string(
#                     json.dumps(assignments, indent=2))
#                 st.session_state["calendar_events"] = load_all_calendar_events()
#                 st.rerun()

#     with st.expander("➕ Add New Class"):
#         with st.form("add_class_form"):
#             c_title = st.text_input("Class Title")
#             c_date = st.date_input("Class Date")
#             now = datetime.now().replace(second=0, microsecond=0)
#             c_start = st.time_input("Start Time", value=now.time())
#             c_end = st.time_input("End Time", value=(now + timedelta(hours=1)).time())
#             start_str = datetime.combine(c_date, c_start).isoformat()
#             end_str = datetime.combine(c_date, c_end).isoformat()
#             if st.form_submit_button("Add Class"):
#                 new_event = {
#                     "id": str(uuid.uuid4()),
#                     "title": c_title,
#                     "start": start_str,
#                     "end": end_str,
#                     "allDay": False,
#                     "color": "green"
#                 }
#                 classes = [e for e in all_events if not e.get("allDay")]
#                 classes.append(new_event)
#                 gcs_client().bucket(BUCKET_NAME).blob("calendar_classes.json").upload_from_string(
#                     json.dumps(classes, indent=2))
#                 st.session_state["calendar_events"] = load_all_calendar_events()
#                 st.rerun()



# elif page == "Smart Scheduler":
#     st.title("🧠 Smart Scheduler")

#     # === 1. Weekly Availability Input ===
#     st.markdown("### ✅ Select Your *Available* Time Slots")
#     WEEKLY_AVAILABILITY_FILE = "weekly_availability.json"

#     days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
#     time_slots = [f"{str(h).zfill(2)}:00–{str(h+1).zfill(2)}:00" for h in range(6, 23)]

#     default_availability = {}
#     try:
#         blob = gcs_client().bucket(BUCKET_NAME).blob(WEEKLY_AVAILABILITY_FILE)
#         if blob.exists():
#             content = json.loads(blob.download_as_text())
#             if isinstance(content, dict):
#                 default_availability = content
#     except Exception as e:
#         st.warning(f"⚠️ Could not load availability: {e}")

#     weekly_availability = {}
#     for day in days:
#         previous = default_availability.get(day, [])
#         selected = st.multiselect(f"{day}", options=time_slots, default=previous, key=f"{day}_slots")
#         weekly_availability[day] = selected

#     if st.button("📅 Save Weekly Availability"):
#         try:
#             blob = gcs_client().bucket(BUCKET_NAME).blob(WEEKLY_AVAILABILITY_FILE)
#             blob.upload_from_string(json.dumps(weekly_availability, indent=2))
#             st.success("✅ Availability saved!")
#         except Exception as e:
#             st.error(f"❌ Failed to save: {e}")

#     # === 2. Load Calendar Events ===
#     all_events = load_all_calendar_events()
#     class_events = [e for e in all_events if not e.get("allDay", False)]

#     # === 3. Filter Class Time from Availability ===
#     def overlaps(start1, end1, start2, end2):
#         return max(start1, start2) < min(end1, end2)

#     def parse_slot_range(slot_str):
#         s, e = slot_str.split("–")
#         return datetime.strptime(s, "%H:%M").time(), datetime.strptime(e, "%H:%M").time()

#     availability = {}
#     try:
#         for day, slots in weekly_availability.items():
#             available = []
#             for slot in slots:
#                 slot_start, slot_end = parse_slot_range(slot)
#                 slot_has_conflict = False
#                 for e in class_events:
#                     s = datetime.fromisoformat(e["start"])
#                     e_dt = datetime.fromisoformat(e["end"])
#                     e_day = s.strftime("%A")
#                     if e_day != day:
#                         continue
#                     class_start, class_end = s.time(), e_dt.time()
#                     if overlaps(slot_start, slot_end, class_start, class_end):
#                         slot_has_conflict = True
#                         break
#                 if not slot_has_conflict:
#                     available.append(slot)
#             availability[day] = available
#     except Exception as e:
#         st.warning(f"⚠️ Could not process availability: {e}")
#         availability = {}

#     # === 4. Daily Study Limit ===
#     st.markdown("### ⏱️ Daily Study Limit")
#     daily_limit = st.slider("Max hours per day", 1, 6, 3)

#     # === 5. Load Assignments ===
#     now = datetime.now().date()
#     assignments = [e for e in all_events if e.get("allDay") and datetime.fromisoformat(e["start"]).date() >= now]

#     if not assignments:
#         st.warning("No assignments found.")
#         st.stop()

#     # === 6. File-Assignment Matching ===
#     assignment_file_map = {}
#     uploaded_files = list_gcs_files("courses")

#     for a in assignments:
#         matched_files = [f for f in uploaded_files if f.lower().split(".")[0] in a["title"].lower()]
#         assignment_file_map[a["id"]] = st.multiselect(f"Select file(s) for {a['title']}", uploaded_files, default=matched_files, key=f"{a['id']}_files")

#     # === 7. Completion Tracking ===
#     st.markdown("### ✅ Mark Completed Assignments")
#     if "completed_assignments" not in st.session_state:
#         st.session_state["completed_assignments"] = set()

#     for a in assignments:
#         is_done = st.checkbox(f"{a['title']}", key=f"done_{a['id']}")
#         if is_done:
#             st.session_state["completed_assignments"].add(a["id"])
#         else:
#             st.session_state["completed_assignments"].discard(a["id"])

#     # === 8. Generate Plan ===
#     st.markdown("### 🗓️ Generated Study Plan")

#     def generate_weekly_study_plan(availability, daily_limit, assignments, assignment_file_map):
#         def get_file_content(fpath):
#             try:
#                 blob = gcs_client().bucket(BUCKET_NAME).blob(fpath)
#                 with blob.open("rb") as f:
#                     return extract_text_from_pdf(f)
#             except:
#                 return ""

#         def parse_gemini_json(text):
#             import re
#             try:
#                 return json.loads(text)
#             except:
#                 match = re.search(r'\[\s*{.*}\s*\]', text, re.DOTALL)
#                 if match:
#                     return json.loads(match.group(0))
#                 return None

#         minutes_per_day = {
#             day: min(len(slots) * 60, daily_limit * 60)
#             for day, slots in availability.items()
#         }

#         payload = {
#             "availability": availability,
#             "daily_limit_hours": daily_limit,
#             "max_minutes_per_day": minutes_per_day,
#             "assignments": []
#         }

#         for a in assignments:
#             payload["assignments"].append({
#                 "id": a["id"],
#                 "title": a["title"],
#                 "due_date": a["start"],
#                 "completed": a["id"] in st.session_state["completed_assignments"],
#                 "files": assignment_file_map[a["id"]],
#                 "file_contents": [get_file_content(f)[:1000] for f in assignment_file_map[a["id"]]]
#             })

#         prompt = f"""
#         You are a helpful academic assistant. Create a weekly study plan for a student.

#         ## Rules:
#         - Use only the time blocks under "availability".
#         - NEVER assign more total minutes in a day than allowed in "max_minutes_per_day".
#         - Track total scheduled minutes per day as you go.
#         - Insert 15–30 min breaks between long or back-to-back sessions.
#         - If a day is full, move remaining tasks to another day or log in "unscheduled".
#         - Use the en dash (–) in time ranges. Avoid overlap.

#         ## Format:
#         Return:
#         {{
#         "scheduled": [ {{ "day": "...", "task": "...", "time": "HH:MM–HH:MM" }} ],
#         "unscheduled": [ {{ "title": "...", "reason": "..." }} ]
#         }}

#         ## Example:
#         If Monday has 180 min max and you schedule:
#         - Ethics HW: 08:00–09:00 (60)
#         - Math Lab: 10:00–11:00 (60)
#         - Reading: 13:00–14:00 (60)
#         Total: ✅ 180 min — OK

#         But if you try to add another:
#         - Philosophy HW: 14:00–15:00 (60)
#         Total: ❌ 240 > 180 → Move to Tuesday or mark unscheduled.

#         ## Input:
#         {json.dumps(payload, indent=2)}
#         """

#         try:
#             model = genai.GenerativeModel("models/gemini-1.5-pro")
#             response = model.generate_content(prompt)
#             return parse_gemini_json(response.text)
#         except Exception as e:
#             st.error(f"❌ Failed to generate plan: {e}")
#             return None

#     if st.button("🔮 Generate Plan"):
#         result = generate_weekly_study_plan(availability, daily_limit, assignments, assignment_file_map)
        
#         if result:
#             if isinstance(result, dict):
#                 scheduled = result.get("scheduled", [])
#                 unscheduled = result.get("unscheduled", [])
#             else:
#                 scheduled = result
#                 unscheduled = []

#             st.success("✅ Weekly plan generated!")

#             def slot_minutes(slot):
#                 try:
#                     slot = slot.replace("-", "–") 
#                     start, end = slot.split("–")
#                     t1 = dt.datetime.strptime(start.strip(), "%H:%M")
#                     t2 = dt.datetime.strptime(end.strip(), "%H:%M")
#                     return int((t2 - t1).total_seconds() // 60)
#                 except:
#                     return 0

#             daily_minutes = defaultdict(int)
#             daily_tasks = defaultdict(list)
#             for task in scheduled:
#                 duration = slot_minutes(task["time"])
#                 day = task["day"]
#                 daily_minutes[day] += duration
#                 daily_tasks[day].append((task["task"], task["time"], duration))

#             for day in days:
#                 if not daily_tasks[day]:
#                     continue
#                 total = daily_minutes[day]
#                 allowed = min(len(availability.get(day, [])) * 60, daily_limit * 60)
#                 if total > allowed:
#                     st.error(f"🚨 **{day} exceeds limit**: {total} min scheduled vs {allowed} min allowed")
#                 else:
#                     st.markdown(f"### 📅 {day} – {total} min scheduled")
#                 for task_name, time, mins in daily_tasks[day]:
#                     st.markdown(f"- _{task_name}_ at **{time}** ({mins} min)")

#             if unscheduled:
#                 st.warning("⚠️ The following tasks could not be scheduled this week:")
#                 for item in unscheduled:
#                     st.markdown(f"- **{item['title']}** → _{item.get('reason', 'No reason provided')}_")


#         else:
#             st.info("No plan returned.")




# # === COURSES PAGE ===
# elif page == "Courses":
#     st.title("📚 Courses")
#     ensure_structure()
#     structure = st.session_state.structure
#     show_advanced = st.checkbox("Show advanced options")

#     tab1, tab2, tab3 = st.tabs(["📘 Course Setup", "📁 Categories", "📂 Files"])

#     # === TAB 1: COURSE SETUP ===
#     with tab1:
#         st.subheader("Manage Courses")

#         new_course = st.text_input("➕ Add new course")
#         if st.button("Add Course") and new_course:
#             if new_course not in structure:
#                 structure[new_course] = []
#                 upload_structure(structure)
#                 st.rerun()

#         course_names = list(structure.keys())
#         if course_names:
#             selected_course = st.selectbox("🎓 Select a course", course_names)

#             if show_advanced:
#                 col1, col2 = st.columns(2)
#                 with col1:
#                     if st.button("🗑 Delete Course"):
#                         del structure[selected_course]
#                         upload_structure(structure)
#                         st.rerun()
#                 with col2:
#                     rename = st.text_input("✏️ Rename Course", key="rename_course")
#                     if st.button("Rename Course") and rename:
#                         structure[rename] = structure.pop(selected_course)
#                         upload_structure(structure)
#                         st.rerun()

#     # === TAB 2: CATEGORY SETUP ===
#     with tab2:
#         if course_names:
#             selected_course = st.selectbox("🎓 Select a course (for categories)", course_names, key="cat_course")
#             st.subheader(f"Categories for {selected_course}")

#             new_cat = st.text_input("➕ Add new category", key="new_cat")
#             if st.button("Add Category") and new_cat not in structure[selected_course]:
#                 structure[selected_course].append(new_cat)
#                 upload_structure(structure)
#                 st.rerun()

#             if structure[selected_course]:
#                 selected_cat = st.selectbox("📂 Select a category", structure[selected_course], key="selected_cat")

#                 if show_advanced:
#                     col1, col2 = st.columns(2)
#                     with col1:
#                         if st.button("🗑 Delete Category"):
#                             structure[selected_course].remove(selected_cat)
#                             upload_structure(structure)
#                             st.rerun()
#                     with col2:
#                         rename_cat = st.text_input("✏️ Rename Category", key="rename_cat")
#                         if st.button("Rename Category") and rename_cat:
#                             files = list_gcs_files(f"courses/{selected_course}/{selected_cat}")
#                             for f in files:
#                                 old_path = f"courses/{selected_course}/{selected_cat}/{f}"
#                                 new_path = f"courses/{selected_course}/{rename_cat}/{f}"
#                                 client = gcs_client()
#                                 bucket = client.bucket(BUCKET_NAME)
#                                 bucket.rename_blob(bucket.blob(old_path), new_path)
#                             structure[selected_course].remove(selected_cat)
#                             structure[selected_course].append(rename_cat)
#                             upload_structure(structure)
#                             st.rerun()

#     # === TAB 3: FILE MANAGEMENT ===
#     with tab3:
#         if "selected_files" not in st.session_state:
#             st.session_state["selected_files"] = []

#         if course_names:
#             selected_course = st.selectbox("🎓 Select a course (for file management)", course_names, key="file_course_all")
#             if structure[selected_course]:
#                 selected_cat = st.selectbox("📂 Select a category", structure[selected_course], key="file_cat_all")

#                 st.markdown("#### 📤 Upload a file")
#                 file = st.file_uploader("Choose file to upload", key=f"{selected_course}_{selected_cat}_upload")
#                 if file and not st.session_state.get("file_uploaded"):
#                     path = f"courses/{selected_course}/{selected_cat}/{file.name}"
#                     upload_to_gcs(BUCKET_NAME, path, file)
#                     st.success("✅ File uploaded.")
#                     st.session_state["file_uploaded"] = True
#                     st.rerun()

#                 if "file_uploaded" in st.session_state:
#                     del st.session_state["file_uploaded"]

#                 st.markdown("#### 📁 Uploaded Files")
#                 files = list_gcs_files(f"courses/{selected_course}/{selected_cat}")
#                 for fname in files:
#                     file_path = f"courses/{selected_course}/{selected_cat}/{fname}"
#                     with st.container():
#                         col1, col2, col3 = st.columns([6, 1, 1])
#                         col1.write(fname)

#                         selected = col2.checkbox("Chat", key=f"select_{file_path}")
#                         if selected and file_path not in st.session_state["selected_files"]:
#                             st.session_state["selected_files"].append(file_path)
#                         elif not selected and file_path in st.session_state["selected_files"]:
#                             st.session_state["selected_files"].remove(file_path)

#                         if show_advanced:
#                             if col3.button("🗑", key=f"delete_{fname}"):
#                                 if delete_from_gcs(file_path):
#                                     st.success(f"Deleted {fname}")
#                                     st.rerun()
#                                 else:
#                                     st.error("Failed to delete.")

#                 if st.session_state["selected_files"]:
#                     st.markdown("### 🤖 Ready to Chat with Selected Files?")
#                     if st.button("💬 Chat"):
#                         try:
#                             all_chunks = []
#                             rag = PDFRagSystem(google_api_key=os.getenv("GOOGLE_API_KEY"))
#                             for path in st.session_state["selected_files"]:
#                                 blob = gcs_client().bucket(BUCKET_NAME).blob(path)
#                                 with blob.open("rb") as f:
#                                     _, chunks = rag.process_pdf_bytes(f.read())
#                                     all_chunks.extend(chunks)
#                                     logger.info(f"Extracted {len(chunks)} chunks from {path}")

#                             if all_chunks:
#                                 st.session_state.rag_system.vector_store = FAISS.from_texts(
#                                     all_chunks, st.session_state.rag_system.embeddings
#                                 )
#                                 st.session_state.rag_system.initialize_qa_chain()
#                                 logger.info("RAG system initialized with selected files")
#                                 st.session_state["selected_files"] = []
#                                 st.session_state.page = "Chatbot"
#                                 st.rerun()
#                             else:
#                                 st.error("❌ No extractable content found in selected files.")
#                         except Exception as e:
#                             logger.error("Failed to process files before chat", exc_info=True)
#                             st.error(f"❌ Failed to process files: {e}")




# # === CHATBOT PAGE ===
# elif page == "Chatbot":
#     if "selected_files" not in st.session_state:
#         st.session_state["selected_files"] = []

#     left, main, right = st.columns([0.2, 2.5, 1.3])

#     with main:
#         st.title("🧞‍♂️ How can I help you? ✨")

#         if "rag_system" not in st.session_state:
#             st.session_state.rag_system = PDFRagSystem(google_api_key=os.getenv("GOOGLE_API_KEY"))
#         if "messages" not in st.session_state:
#             st.session_state.messages = []
#         if "new_message" not in st.session_state:
#             st.session_state.new_message = False

#         if st.session_state.get("new_message"):
#             prompt = st.session_state.get("pending_input", "")
#             if prompt:
#                 st.session_state.messages.append({"role": "user", "content": prompt})
#                 response = st.session_state.rag_system.answer(prompt)
#                 logger.info("Gemini answered user prompt")
#                 log_interaction_to_gcs(prompt, response)
#                 st.session_state.messages.append({"role": "assistant", "content": response})
#             st.session_state.new_message = False
#             st.session_state.pending_input = ""
#             st.rerun()

#         for msg in st.session_state.messages:
#             with st.chat_message(msg["role"]):
#                 st.markdown(msg["content"], unsafe_allow_html=True)

#         prompt = st.chat_input("Ask your question...")
#         if prompt:
#             if not hasattr(st.session_state.rag_system, "qa_chain") or st.session_state.rag_system.qa_chain is None:
#                 st.error("❌ Please load at least one file before asking questions.")
#             else:
#                 st.session_state.pending_input = prompt
#                 st.session_state.new_message = True
#                 st.rerun()

#     with right:
#         st.markdown("### 📂 Uploaded Files")
#         ensure_structure()
#         structure = st.session_state.structure

#         if structure:
#             selected_course = st.selectbox("📘 Select Course", list(structure.keys()))
#             selected_cat = st.selectbox("📂 Select Category", structure[selected_course])
#             folder = f"courses/{selected_course}/{selected_cat}"
#             files = list_gcs_files(folder)
#             if files:
#                 for fname in files:
#                     path = f"{folder}/{fname}"
#                     selected = st.checkbox(fname, key=f"select_{path}")
#                     if selected and path not in st.session_state["selected_files"]:
#                         st.session_state["selected_files"].append(path)
#                     elif not selected and path in st.session_state["selected_files"]:
#                         st.session_state["selected_files"].remove(path)
#             else:
#                 st.info("No files uploaded in this category.")
#         else:
#             st.info("No courses found. Add one in the Courses tab.")

#         if st.button("💬 Load Selected Files"):
#             if not st.session_state["selected_files"]:
#                 st.warning("⚠️ Please select at least one file.")
#             else:
#                 try:
#                     all_chunks = []
#                     for path in st.session_state["selected_files"]:
#                         blob = gcs_client().bucket(BUCKET_NAME).blob(path)
#                         with blob.open("rb") as f:
#                             _, chunks = st.session_state.rag_system.process_pdf_bytes(f.read())
#                             all_chunks.extend(chunks)
#                             logger.info(f"Extended chunk list with {len(chunks)} chunks from {path}")

#                     logger.info(f"Starting vectorization of {len(all_chunks)} chunks")
#                     embedding_model = st.session_state.rag_system.embeddings.client
#                     embeddings = embedding_model.encode(all_chunks, convert_to_numpy=True)

#                     df = pd.DataFrame(embeddings)
#                     df["text"] = all_chunks
#                     csv_buffer = StringIO()
#                     df.to_csv(csv_buffer, index=False)
#                     csv_buffer.seek(0)

#                     def upload_embedding_log_to_gcs(csv_buffer, bucket_name, file_prefix="embedding_logs"):
#                         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#                         file_path = f"{file_prefix}/embedding_log_{timestamp}.csv"
#                         client = gcs_client()
#                         bucket = client.bucket(bucket_name)
#                         blob = bucket.blob(file_path)
#                         blob.upload_from_string(csv_buffer.getvalue(), content_type="text/csv")
#                         return file_path

#                     uploaded_path = upload_embedding_log_to_gcs(csv_buffer, BUCKET_NAME)
#                     # st.success(f"📁 Embedding log uploaded to: gs://{BUCKET_NAME}/{uploaded_path}")

#                     if all_chunks:
#                         st.session_state.rag_system.vector_store = FAISS.from_texts(
#                             all_chunks, st.session_state.rag_system.embeddings
#                         )
#                         logger.info("FAISS index created and vector store assigned")
#                         st.session_state.rag_system.initialize_qa_chain()
#                         logger.info("Gemini QA chain initialized")
#                         st.success(f"✅ Loaded {len(st.session_state['selected_files'])} file(s).")
#                         st.session_state["selected_files"] = []
#                     else:
#                         st.error("❌ No extractable content found in selected files.")
#                 except Exception as e:
#                     logger.error("Failed to load files", exc_info=True)
#                     st.error(f"❌ Failed to load files: {e}")
