import streamlit as st
import PyPDF2
import spacy
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from wordcloud import WordCloud
from collections import Counter
import re
import base64

# ------------ CONFIG + STYLING ------------

def set_background(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
    }}
    .block-container {{
        background-color: rgba(10, 25, 47, 0.80);
        border-radius: 12px;
        padding: 1.5rem 2rem 2rem 2rem;
        color: #f9fafb;
    }}
    h1, h2, h3, h4, h5, h6 {{
        color: #38bdf8;
    }}
    .sidebar .sidebar-content {{
        background-color: #020617;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Load SpaCy model
nlp = spacy.load('en_core_web_sm')

# ------------ HELPER FUNCTIONS ------------

def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + " "
    return text

def extract_skills(text):
    skill_keywords = [
        'python', 'java', 'c++', 'c#', 'javascript',
        'machine learning', 'deep learning', 'data analysis',
        'nlp', 'natural language processing',
        'sql', 'mysql', 'postgresql',
        'html', 'css', 'django', 'flask',
        'aws', 'azure', 'gcp',
        'tensorflow', 'pytorch', 'keras',
        'pandas', 'numpy', 'scikit-learn'
    ]
    text_low = text.lower()
    skills_found = [skill for skill in skill_keywords if skill in text_low]
    return list(set(skills_found))

def extract_contact_info(text):
    emails = re.findall(r'\S+@\S+', text)
    phones = re.findall(r'\b\d{10}\b', text)
    email = emails[0] if emails else ''
    phone = phones[0] if phones else ''
    return email, phone

def extract_experience_years(text):
    text_low = text.lower()
    match = re.search(r'(\d+)\s+years? of experience', text_low)
    if match:
        return int(match.group(1))
    match2 = re.search(r'(\d+)\s*\+?\s*years', text_low)
    if match2:
        return int(match2.group(1))
    return 0

def parse_resume(file):
    raw_text = extract_text_from_pdf(file)
    skills = extract_skills(raw_text)
    email, phone = extract_contact_info(raw_text)
    experience = extract_experience_years(raw_text)
    return {
        "text": raw_text,
        "skills": skills,
        "email": email,
        "phone": phone,
        "experience": experience
    }

def calculate_match_score(resume_skills, job_desc):
    job_low = job_desc.lower()
    if not resume_skills:
        return 0.0
    matched = sum(1 for skill in resume_skills if skill in job_low)
    score = (matched / len(resume_skills)) * 100
    return round(score, 2)

# ------------ STREAMLIT APP ------------

def main():
    st.set_page_config(page_title="AI Resume Screener", page_icon="📄", layout="wide")
    set_background("bg.jpg")  # make sure bg.jpg is in same folder

    st.sidebar.markdown("## 🧠 AI Resume Screener")
    st.sidebar.markdown("Built as part of AI/ML with Python training.")
    st.sidebar.markdown("---")
    option = st.sidebar.radio(
        "Navigate",
        ["Home", "Upload Resumes", "Job Description", "Ranking", "Analytics"]
    )

    if "data" not in st.session_state:
        st.session_state["data"] = []
    if "job_desc" not in st.session_state:
        st.session_state["job_desc"] = ""

    # ---------- HOME ----------
    if option == "Home":
        st.title("AI-Based Resume Screening and Analysis")

        col_left, col_right = st.columns([2, 1])

        with col_left:
            st.markdown("### Project Overview")
            st.markdown("""
This project is an AI/ML-based **Resume Screening and Analysis System** developed after a 45‑day
training in *Python, Machine Learning and NLP*.  
The application helps HR and recruiters automatically filter multiple candidate resumes instead of
reading each resume manually.

The system:
- Reads and parses resumes uploaded in **PDF** format.  
- Extracts key details such as **skills, years of experience, email and phone number**.  
- Compares every resume with a given **job description** using keyword-based NLP logic.  
- Generates a **match score (0–100%)** for each candidate and ranks them accordingly.
            """)

        with col_right:
            st.markdown("### Learning Outcomes")
            st.markdown("""
From this project, the student learned and implemented:
- Text extraction from PDF using Python.  
- Basic **NLP** techniques for keyword and skill extraction.  
- Designing a **Streamlit web dashboard** with multiple interactive pages.  
- Creating simple AI-style scoring logic to support decision making in recruitment.  
            """)

        st.markdown("---")
        st.markdown("""
### How to Use the Application
1. Go to **Upload Resumes** and upload multiple candidate resumes in PDF format.  
2. Open **Job Description**, paste the required job role description and click **Calculate Match Scores**.  
3. Check the **Ranking** page to see candidates sorted by their job match score and download the result as CSV.  
4. Use the **Analytics** page to view the skill word cloud, experience distribution and match score charts,  
   which make this project look like a complete AI/ML dashboard suitable for major project submission.
        """)

    # ---------- UPLOAD ----------
    elif option == "Upload Resumes":
        st.title("Upload Resume PDF Files")
        st.write("Upload one or more candidate resumes in PDF format. The system will parse skills, experience and contact details.")

        uploaded_files = st.file_uploader(
            "Choose one or more resume PDFs",
            type=["pdf"],
            accept_multiple_files=True
        )

        if uploaded_files:
            for file in uploaded_files:
                with st.spinner(f"Parsing {file.name} ..."):
                    parsed = parse_resume(file)
                    record = {
                        "filename": file.name,
                        "email": parsed["email"],
                        "phone": parsed["phone"],
                        "skills": parsed["skills"],
                        "experience": parsed["experience"],
                        "text": parsed["text"]
                    }
                    st.session_state["data"].append(record)
            st.success("Parsing completed! Now open the 'Job Description' page.")

        if st.session_state["data"]:
            st.subheader("Parsed Resumes")
            df_tmp = pd.DataFrame(st.session_state["data"])
            st.dataframe(df_tmp[["filename", "email", "phone", "skills", "experience"]])

    # ---------- JOB DESCRIPTION ----------
    elif option == "Job Description":
        st.title("Job Description and Matching")
        jd = st.text_area(
            "Paste job description here (example: Python / ML Developer with 1–2 years of experience)",
            value=st.session_state["job_desc"],
            height=200
        )
        st.session_state["job_desc"] = jd

        if st.button("Calculate Match Scores"):
            if not st.session_state["data"]:
                st.warning("First upload at least one resume in the 'Upload Resumes' page.")
            elif not jd.strip():
                st.warning("Please paste a job description before calculating scores.")
            else:
                for record in st.session_state["data"]:
                    record["score"] = calculate_match_score(record["skills"], jd)
                st.success("Match scores calculated! Open the 'Ranking' page to see ranked candidates.")

    # ---------- RANKING ----------
    elif option == "Ranking":
        st.title("Candidate Ranking Based on Job Match")
        if not st.session_state["data"]:
            st.info("No resumes found. Please upload resumes first.")
        else:
            df = pd.DataFrame(st.session_state["data"])
            if "score" not in df.columns:
                st.warning("Scores not calculated. Go to 'Job Description' page and click 'Calculate Match Scores'.")
            else:
                df_sorted = df.sort_values(by="score", ascending=False)

                top_score = df_sorted["score"].max()
                avg_score = round(df_sorted["score"].mean(), 2)
                count_res = len(df_sorted)

                c1, c2, c3 = st.columns(3)
                c1.metric("Total Resumes", count_res)
                c2.metric("Highest Match Score", f"{top_score}%")
                c3.metric("Average Match Score", f"{avg_score}%")

                st.subheader("Detailed Candidate Table")
                st.dataframe(df_sorted[["filename", "email", "phone", "skills", "experience", "score"]])

                csv = df_sorted.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download Ranking as CSV",
                    data=csv,
                    file_name="resume_ranking.csv",
                    mime="text/csv"
                )

    # ---------- ANALYTICS ----------
    elif option == "Analytics":
        st.title("Visual Analytics")
        if not st.session_state["data"]:
            st.info("Upload resumes first to see analytics.")
        else:
            df = pd.DataFrame(st.session_state["data"])
            all_skills = []
            for rec in st.session_state["data"]:
                all_skills.extend(rec["skills"])

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Skill Word Cloud")
                if all_skills:
                    skill_counts = Counter(all_skills)
                    wc = WordCloud(
                        width=800,
                        height=400,
                        background_color="white"
                    ).generate_from_frequencies(skill_counts)
                    plt.figure(figsize=(8, 4))
                    plt.imshow(wc, interpolation="bilinear")
                    plt.axis("off")
                    st.pyplot(plt)
                else:
                    st.info("No skills extracted yet from uploaded resumes.")

            with col2:
                st.subheader("Experience (Years) per Candidate")
                if "experience" in df.columns and not df["experience"].isna().all():
                    fig_exp = px.bar(
                        df,
                        x="filename",
                        y="experience",
                        title="Experience in Years",
                        labels={"filename": "Candidate", "experience": "Years"}
                    )
                    st.plotly_chart(fig_exp, use_container_width=True)
                else:
                    st.info("No experience data found in resumes.")

            st.subheader("Match Score Chart")
            if "score" in df.columns:
                fig_score = px.bar(
                    df,
                    x="filename",
                    y="score",
                    color="score",
                    title="Candidate Match Scores",
                    labels={"filename": "Candidate", "score": "Match Score"}
                )
                st.plotly_chart(fig_score, use_container_width=True)
            else:
                st.info("Scores not calculated yet – go to 'Job Description' page and calculate them first.")

if __name__ == "__main__":
    main()
