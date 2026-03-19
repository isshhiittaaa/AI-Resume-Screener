import pdfplumber
import streamlit as st
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("📄 AI Resume Screening System")

# Upload JD
jd_file = st.file_uploader("Upload Job Description", type=["txt", "pdf"])

# Upload resumes
resume_files = st.file_uploader("Upload Resumes", type=["txt", "pdf"], accept_multiple_files=True)

if st.button("Analyze"):

    if jd_file is not None and resume_files:

        # Read JD
        if jd_file.type == "application/pdf":
            with pdfplumber.open(jd_file) as pdf:
                jd = ""
                for page in pdf.pages:
                    jd += page.extract_text() or ""
        else:
            jd = jd_file.read().decode("utf-8")

        # ✅ CORRECT INDENTATION HERE
        resumes = []
        names = []

        for file in resume_files:
            names.append(file.name)

            if file.type == "application/pdf":
                with pdfplumber.open(file) as pdf:
                    text = ""
                    for page in pdf.pages:
                        text += page.extract_text() or ""
                resumes.append(text)

            else:
                content = file.read().decode("utf-8")
                resumes.append(content)
                 # Create DataFrame
        df = pd.DataFrame({"Candidate": names, "Resume": resumes})

        # TF-IDF
        vectorizer = TfidfVectorizer()
        documents = [jd] + df["Resume"].tolist()
        tfidf_matrix = vectorizer.fit_transform(documents)

        # Similarity
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
        df["Score"] = (similarity * 100).round(2)

        # Ranking
        df = df.sort_values(by="Score", ascending=False).reset_index(drop=True)

        # Skills
        skills = ["python", "sql", "excel", "machine learning"]

        def analyze(resume):
            resume_lower = resume.lower()
            found = [s for s in skills if s in resume_lower]
            missing = [s for s in skills if s not in resume_lower]
            return found[:3], missing[:3]

        df["Strengths"], df["Gaps"] = zip(*df["Resume"].apply(analyze))

        # Recommendation
        def recommend(score):
            if score > 30:
                return "Strong Fit"
            elif score > 15:
                return "Moderate Fit"
            else:
                return "Not Fit"

        df["Recommendation"] = df["Score"].apply(recommend)

        # Format display
        df["Strengths"] = df["Strengths"].apply(lambda x: ", ".join(x))
        df["Gaps"] = df["Gaps"].apply(lambda x: ", ".join(x))

        # Show results
        st.subheader("📊 Results")
        st.dataframe(df)

        # ⭐ TOP CANDIDATE
        top_candidate = df.iloc[0]
        st.success(f"🏆 Top Candidate: {top_candidate['Candidate']} (Score: {top_candidate['Score']})")

        # ⭐ DOWNLOAD BUTTON
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Results",
            data=csv,
            file_name='results.csv',
            mime='text/csv',
        )

        # GRAPH
        st.subheader("📈 Candidate Scores")
        st.bar_chart(df.set_index("Candidate")["Score"])

    else:
        st.warning("Please upload JD and resumes")