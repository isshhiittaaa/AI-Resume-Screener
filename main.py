# Step 1: Import libraries
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Step 2: Read resumes
def read_files(folder):
    texts = []
    names = []
    
    for file in os.listdir(folder):
        with open(os.path.join(folder, file), 'r', encoding='utf-8') as f:
            texts.append(f.read())
            names.append(file)
    
    return pd.DataFrame({"Candidate": names, "Resume": texts})

df = read_files("resumes")

# Step 3: Read Job Description
with open("jd.txt", "r") as f:
    jd = f.read()

# Step 4: TF-IDF Vectorization
vectorizer = TfidfVectorizer()
documents = [jd] + df["Resume"].tolist()
tfidf_matrix = vectorizer.fit_transform(documents)

# Step 5: Similarity Score
similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
df["Score"] = (similarity * 100).round(2)

# Step 6: Ranking
df = df.sort_values(by="Score", ascending=False).reset_index(drop=True)

# Step 7: Strengths & Gaps
skills = ["python", "sql", "excel", "machine learning"]

def analyze(resume):
    resume_lower = resume.lower()
    found = [s for s in skills if s in resume_lower]
    missing = [s for s in skills if s not in resume_lower]
    return found[:3], missing[:3]

df["Strengths"], df["Gaps"] = zip(*df["Resume"].apply(analyze))

# Step 8: Recommendation
def recommend(score):
    if score > 30:
        return "Strong Fit"
    elif score > 15:
        return "Moderate Fit"
    else:
        return "Not Fit"

df["Recommendation"] = df["Score"].apply(recommend)

# Step 9: Output

print("\nFinal Candidate Ranking:\n")
print(df.to_string(index=False))
# Step 10: Save results
df.to_csv("results.csv", index=False)
import matplotlib.pyplot as plt

plt.bar(df["Candidate"], df["Score"])
plt.title("Candidate Scores")
plt.xlabel("Candidates")
plt.ylabel("Score")
plt.show()