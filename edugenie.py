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




# === MODEL SETUP ===
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
        


# === STREAMLIT SETUP ===
st.set_page_config(page_title="EduGenie Dashboard", layout="wide")
st.markdown("""
    <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS_HTML"></script>
    <script>
    MathJax.Hub.Config({
        tex2jax: {
            inlineMath: [ ['$','$'], ['\\(','\\)'] ],
            displayMath: [ ['$$','$$'], ['\\[','\\]'] ],
            processEscapes: true,
            processEnvironments: true
        },
        displayAlign: 'center',
        "HTML-CSS": { linebreaks: { automatic: true } },
        SVG: { linebreaks: { automatic: true } }
    });
    </script>
""", unsafe_allow_html=True)

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

    if st.session_state["calendar_updated"] or not st.session_state.get("calendar_events"):
        all_events = load_all_calendar_events()
        st.session_state["calendar_events"] = all_events
        st.session_state["calendar_updated"] = False
    else:
        all_events = st.session_state["calendar_events"]

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
                    st.rerun()

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
            st.rerun()

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
            st.rerun()



elif page == "Smart Scheduler":
    st.title("🧠 Smart Scheduler")

    tabs = st.tabs(["📅 Availability", "📚 Assignments", "🗓️ Study Plan"])
    
    # AVAILABILITY TAB
    with tabs[0]: 
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown("### ✅ Select Your Available Time Slots")
            WEEKLY_AVAILABILITY_FILE = "weekly_availability.json"

            days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

            time_slots = []
            for h in range(6, 23):
                time_slots.append(f"{str(h).zfill(2)}:00-{str(h).zfill(2)}:30")
                time_slots.append(f"{str(h).zfill(2)}:30-{str(h+1).zfill(2)}:00")
            
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
            
            # # DEBUG VALUES - DELETE LATER
            # if not default_availability:
            #     default_availability = {
            #         "Monday": [],
            #         "Tuesday": ["10:00-10:30", "10:30-11:00", "15:00-15:30", "15:30-16:00"],
            #         "Wednesday": [],
            #         "Thursday": ["11:00-11:30", "11:30-12:00", "17:00-17:30", "17:30-18:00"],
            #         "Friday": ["13:00-13:30", "13:30-14:00", "18:00-18:30", "18:30-19:00"],
            #         "Saturday": ["10:00-10:30", "10:30-11:00", "14:00-14:30", "14:30-15:00"],
            #         "Sunday": []
            #     }
            # # END DEBUG VALUES - DELETE LATER

            if "weekly_availability" not in st.session_state:
                st.session_state.weekly_availability = default_availability.copy()
            
            for day in days:
                with st.expander(f"{day}", expanded=(day == days[0])):
                    previous = st.session_state.weekly_availability.get(day, [])

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

            weekly_availability = st.session_state.weekly_availability
            
            if st.button("📅 Save Weekly Availability", type="primary"):
                try:
                    sanitized_availability = {}
                    for day, slots in weekly_availability.items():
                        sanitized_availability[day] = [slot.replace("–", "-") for slot in slots]

                    json_data = json.dumps(sanitized_availability, indent=2, ensure_ascii=True)

                    _ = json.loads(json_data)

                    blob = gcs_client().bucket(BUCKET_NAME).blob(WEEKLY_AVAILABILITY_FILE)
                    blob.upload_from_string(json_data)
                    st.success("✅ Availability saved!")
                except Exception as e:
                    st.error(f"❌ Failed to save: {e}")
                    st.error("JSON content that failed to save:")
                    st.code(json.dumps(weekly_availability, indent=2))
        
        with col2:
            st.markdown("### ⚙️ Preferences")

            st.markdown("#### Study Style")
            focus_duration = st.slider("Focus session length (min)", 25, 120, 50, step=5)
            break_duration = st.slider("Break length (min)", 5, 30, 10, step=5)
            
            st.markdown("#### Daily Limits")
            daily_limit = st.slider("Max hours per day", 1, 10, 3)
            
            st.markdown("#### Learning")
            productivity_peak = st.selectbox("When are you most productive?", 
                                          ["Morning", "Afternoon", "Evening", "No preference"])
            
            difficult_first = st.checkbox("Schedule difficult tasks first", value=True)

            if st.button("💾 Save Preferences"):
                st.session_state["study_preferences"] = {
                    "focus_duration": focus_duration,
                    "break_duration": break_duration,
                    "daily_limit": daily_limit,
                    "productivity_peak": productivity_peak,
                    "difficult_first": difficult_first
                }
                st.success("✅ Preferences saved!")
    
    # ASSIGNMENTS TAB
    with tabs[1]:
        st.markdown("### 📚 Manage Your Assignments")

        all_events = load_all_calendar_events()
        class_events = [e for e in all_events if not e.get("allDay", False)]

        def overlaps(start1, end1, start2, end2):
            return max(start1, start2) < min(end1, end2)
        
        def parse_slot_range(slot_str):
            if "-" in slot_str:
                s, e = slot_str.split("-")
            elif "–" in slot_str:
                s, e = slot_str.split("–")
            else:
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

        now = datetime.now().date()
        assignments = [e for e in all_events if e.get("allDay") and datetime.fromisoformat(e["start"]).date() >= now]
        
        if not assignments:
            st.warning("No assignments found in your calendar.")
            st.info("Add assignments as all-day events in your calendar to see them here.")
        else:
            with st.expander("➕ Add Manual Assignment"):
                new_title = st.text_input("Assignment Title")
                new_due_date = st.date_input("Due Date", min_value=now)
                new_course = st.text_input("Course")
                new_difficulty = st.select_slider("Difficulty", options=["Easy", "Medium", "Hard"])
                new_estimated_hours = st.number_input("Estimated Hours", min_value=0.5, max_value=20.0, value=2.0, step=0.5)
                
                if st.button("Add Assignment"):
                    if new_title:
                        st.success(f"✅ Added '{new_title}' due {new_due_date}")
                    else:
                        st.error("Title is required")

            st.markdown("### Assignment List")

            if "assignment_metadata" not in st.session_state:
                st.session_state["assignment_metadata"] = {}

            uploaded_files = list_gcs_files("courses")

            assignments.sort(key=lambda x: datetime.fromisoformat(x["start"]))
            
            for i, a in enumerate(assignments):
                with st.expander(f"{a['title']} (Due: {datetime.fromisoformat(a['start']).strftime('%b %d')})", expanded=(i == 0)):
                    a_id = a["id"]
 
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

                    meta["files"] = st.multiselect(
                        "Related Files", 
                        uploaded_files, 
                        default=meta["files"],
                        key=f"files_{a_id}"
                    )

                    meta["notes"] = st.text_area(
                        "Notes", 
                        value=meta["notes"],
                        key=f"notes_{a_id}"
                    )

                    meta["completed"] = st.checkbox(
                        "Mark as Completed", 
                        value=meta["completed"],
                        key=f"done_{a_id}"
                    )
                    
                    if meta["completed"]:
                        st.success("✅ Completed!")

            completed = sum(1 for a_id in st.session_state["assignment_metadata"] if st.session_state["assignment_metadata"][a_id]["completed"])
            total = len(assignments)
            st.progress(completed / total if total > 0 else 0)
            st.markdown(f"**Overall Progress:** {completed}/{total} assignments completed")

    # STUDY PLAN TAB
    with tabs[2]:  
        st.markdown("### 🗓️ Generate Your Personalized Study Plan")

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
            plan_type = "Week View"

        def generate_weekly_study_plan(availability, prefs, assignments, metadata):
            def get_file_content(fpath):
                try:
                    blob = gcs_client().bucket(BUCKET_NAME).blob(fpath)
                    with blob.open("rb") as f:
                        return extract_text_from_pdf(f)
                except:
                    return ""
            
            def parse_gemini_json(text):
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
            
            minutes_per_day = {
                day: min(len(slots) * 30, prefs["daily_limit"] * 60)
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
                metadata = st.session_state.get("assignment_metadata", {})

                result = generate_weekly_study_plan(availability, prefs, assignments, metadata)
                
                if result and isinstance(result, dict):
                    scheduled = result.get("scheduled", [])
                    unscheduled = result.get("unscheduled", [])
                    stats = result.get("stats", {})
                    
                    st.success("✅ Study plan generated!")

                    st.session_state["study_plan"] = result

                    if plan_type == "Week View":
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

                        daily_tasks = {day: [] for day in days}
                        for task in scheduled:
                            day = task["day"]
                            if day in daily_tasks:
                                daily_tasks[day].append(task)

                        day_tabs = st.tabs(days)
                        
                        for i, day in enumerate(days):
                            with day_tabs[i]:
                                if not daily_tasks[day]:
                                    st.info(f"No tasks scheduled for {day}")
                                    continue
     
                                daily_tasks[day].sort(key=lambda x: x["time"].split("–")[0])

                                daily_minutes = sum(slot_minutes(task["time"]) for task in daily_tasks[day] if not task.get("is_break", False))
                                daily_break_minutes = sum(slot_minutes(task["time"]) for task in daily_tasks[day] if task.get("is_break", True))
                                
                                st.markdown(f"**Total Study Time:** {daily_minutes//60}h {daily_minutes%60}m | **Breaks:** {daily_break_minutes} min")

                                for task in daily_tasks[day]:
                                    if task.get("is_break", False):
                                        st.markdown(f"""
                                        <div style="padding: 5px 10px; margin: 5px 0; background-color: #e1f5fe; border-left: 4px solid #03a9f4; border-radius: 4px;">
                                            <span style="color: black; font-weight: bold;">{task["time"]}</span> - <span style="color: black;">Break Time ({slot_minutes(task["time"])} min)</span>
                                        </div>
                                        """, unsafe_allow_html=True)
                                    else:
                                        difficulty = task.get("difficulty", "Medium")
                                        diff_color = "#4caf50" if difficulty == "Easy" else "#ff9800" if difficulty == "Medium" else "#f44336"
                                        
                                        st.markdown(f"""
                                        <div style="padding: 10px; margin: 10px 0; background-color: #f9f9f9; border-left: 4px solid {diff_color}; border-radius: 4px; color: black;">
                                            <span style="font-weight: bold;">{task["time"]}</span> - {task["task"]} ({slot_minutes(task["time"])} min)
                                            <br><span style="color: {diff_color}; font-size: 0.8em;">Difficulty: {difficulty}</span>
                                        </div>
                                        """, unsafe_allow_html=True)
                else:
                    st.error("Failed to generate a study plan. Please try again.")

        elif "study_plan" in st.session_state:
            st.info("Showing previously generated plan. Click 'Generate Study Plan' to create a new one.")

            result = st.session_state["study_plan"]
            scheduled = result.get("scheduled", [])
            
            if plan_type == "Week View":
                day_tabs = st.tabs(days)

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

                if "file_uploaded" in st.session_state:
                    del st.session_state["file_uploaded"]

                st.markdown("#### 📁 Uploaded Files")
                files = list_gcs_files(f"courses/{selected_course}/{selected_cat}")
                for fname in files:
                    file_path = f"courses/{selected_course}/{selected_cat}/{fname}"
                    with st.container():
                        col1, col2, col3 = st.columns([6, 1, 1])
                        col1.write(fname)

                        if show_advanced:
                            if col3.button("🗑", key=f"delete_{fname}"):
                                if delete_from_gcs(file_path):
                                    st.success(f"Deleted {fname}")
                                    st.rerun()
                                else:
                                    st.error("Failed to delete.")



# === CHATBOT PAGE ===
elif page == "Chatbot":
    if "selected_files" not in st.session_state:
        st.session_state["selected_files"] = []

    if "loaded_files" not in st.session_state:
        st.session_state["loaded_files"] = []

    if "checked_files" not in st.session_state:
        st.session_state["checked_files"] = {}

    if "load_message" not in st.session_state:
        st.session_state["load_message"] = None

    left, main, right = st.columns([0.2, 2.5, 1.3])

    with main:
        st.title("🧞‍♂️ How can I help you? ✨")

        if "rag_system" not in st.session_state:
            st.session_state.rag_system = PDFRagSystem(google_api_key=os.getenv("GOOGLE_API_KEY"))
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "new_message" not in st.session_state:
            st.session_state.new_message = False

        if st.session_state.get("new_message"):
            prompt = st.session_state.get("pending_input", "")
            if prompt:
                st.session_state.messages.append({"role": "user", "content": prompt})
                raw_response = st.session_state.rag_system.answer(prompt)
                logger.info("Gemini answered user prompt")
                log_interaction_to_gcs(prompt, raw_response)

                # Optional: ensure math renders correctly
                st.session_state.messages.append({"role": "assistant", "content": raw_response})
            st.session_state.new_message = False
            st.session_state.pending_input = ""
            st.rerun()

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"], unsafe_allow_html=True)

        prompt = st.chat_input("Ask your question...")
        if prompt:
            if not hasattr(st.session_state.rag_system, "qa_chain") or st.session_state.rag_system.qa_chain is None:
                st.error("❌ Please load at least one file before asking questions.")
            else:
                st.session_state.pending_input = prompt
                st.session_state.new_message = True
                st.rerun()

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
                    is_checked = st.session_state["checked_files"].get(path, False)
                    selected = st.checkbox(fname, value=is_checked, key=f"select_{path}")
                    st.session_state["checked_files"][path] = selected
                    if selected and path not in st.session_state["selected_files"]:
                        st.session_state["selected_files"].append(path)
                    elif not selected and path in st.session_state["selected_files"]:
                        st.session_state["selected_files"].remove(path)
            else:
                st.info("No files uploaded in this category.")
        else:
            st.info("No courses found. Add one in the Courses tab.")

        if st.button("💬 Load Selected Files"):
            if not st.session_state["selected_files"]:
                st.warning("⚠️ Please select at least one file.")
                st.session_state["load_message"] = None
            else:
                try:
                    all_chunks = []
                    newly_loaded_files = []

                    for path in st.session_state["selected_files"]:
                        filename = path.split('/')[-1]
                        newly_loaded_files.append(filename)
                        blob = gcs_client().bucket(BUCKET_NAME).blob(path)
                        with blob.open("rb") as f:
                            _, chunks = st.session_state.rag_system.process_pdf_bytes(f.read())
                            all_chunks.extend(chunks)
                            logger.info(f"Extended chunk list with {len(chunks)} chunks from {path}")

                    for file in newly_loaded_files:
                        if file not in st.session_state["loaded_files"]:
                            st.session_state["loaded_files"].append(file)

                    logger.info(f"Starting vectorization of {len(all_chunks)} chunks")
                    embedding_model = st.session_state.rag_system.embeddings.client
                    embeddings = embedding_model.encode(all_chunks, convert_to_numpy=True)

                    df = pd.DataFrame(embeddings)
                    df["text"] = all_chunks
                    csv_buffer = StringIO()
                    df.to_csv(csv_buffer, index=False)
                    csv_buffer.seek(0)

                    def upload_embedding_log_to_gcs(csv_buffer, bucket_name, file_prefix="embedding_logs"):
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        file_path = f"{file_prefix}/embedding_log_{timestamp}.csv"
                        client = gcs_client()
                        bucket = client.bucket(bucket_name)
                        blob = bucket.blob(file_path)
                        blob.upload_from_string(csv_buffer.getvalue(), content_type="text/csv")
                        return file_path

                    upload_embedding_log_to_gcs(csv_buffer, BUCKET_NAME)

                    if all_chunks:
                        st.session_state.rag_system.vector_store = FAISS.from_texts(
                            all_chunks, st.session_state.rag_system.embeddings
                        )
                        logger.info("FAISS index created and vector store assigned")
                        st.session_state.rag_system.initialize_qa_chain()
                        logger.info("Gemini QA chain initialized")

                        st.session_state["selected_files"] = []
                        files_text = ", ".join(st.session_state["loaded_files"])
                        st.session_state["load_message"] = f"✅ Files loaded: {files_text}. You can now ask questions about these documents."
                    else:
                        st.session_state["load_message"] = None
                        st.error("❌ No extractable content found in selected files.")
                except Exception as e:
                    logger.error("Failed to load files", exc_info=True)
                    st.session_state["load_message"] = None
                    st.error(f"❌ Failed to load files: {e}")
        else:
            st.session_state["load_message"] = None

        if st.session_state.get("load_message"):
            st.success(st.session_state["load_message"])