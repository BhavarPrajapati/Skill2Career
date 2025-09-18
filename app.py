
# from flask import Flask, render_template, request, jsonify
# import pickle
# import pandas as pd
# import pdfplumber
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import logging
# logging.getLogger("pdfminer").setLevel(logging.ERROR)
# import pymysql




# app = Flask(__name__)

# def get_db_connection():
#     conn = pymysql.connect(
#         host="localhost",
#         user="root",
#         password="18Ecocs052@",
#         database="internship_applications"
#     )
#     return conn

# # ---------------------------
# # Load saved internship recommendation model
# # ---------------------------
# with open("internship_model.pkl", "rb") as f:
#     saved_data = pickle.load(f)

# vectorizer = saved_data["vectorizer"]
# internship_vectors = saved_data["internship_vectors"]
# internships = saved_data["internships"]

# # ---------------------------
# # Manual input recommendation
# # ---------------------------
# def recommend_custom(name, role, skills, location, interests="", top_k=5):
#     custom_features = skills + " " + location + " " + interests + " " + role
#     custom_vector = vectorizer.transform([custom_features])
#     scores = cosine_similarity(custom_vector, internship_vectors)[0]
#     top_indices = scores.argsort()[::-1][:top_k]

#     results = []
#     for idx in top_indices:
#         results.append({
#             "internship_id": internships.iloc[idx]['internship_id'],
#             "Company_Name": internships.iloc[idx]['company_name'],
#             "Role": internships.iloc[idx]['title'],
#             "Location": internships.iloc[idx]['location'],
#             "Required_Skills": internships.iloc[idx]['required_skills'],
#             "Match_Score(%)": round(scores[idx]*100, 2)
#         })
#     return results

# # ---------------------------
# # Resume-based recommendation
# # ---------------------------
# # Load internships dataset for resume-based recommendation
# internships_resume = pd.read_csv("internships.csv")

# def extract_text_from_pdf(pdf_path):
#     text = ""
#     with pdfplumber.open(pdf_path) as pdf:
#         for page in pdf.pages:
#             text += page.extract_text() + " "
#     return text

# def extract_skills(text):
#     skills_list = ["python", "machine learning", "deep learning", "tensorflow",
#                    "sql", "java", "c++", "communication", "data analysis", 
#                    "nlp", "excel", "flask", "django", "cloud", "aws"]
#     found_skills = []
#     text = text.lower()
#     for skill in skills_list:
#         if skill.lower() in text:
#             found_skills.append(skill)
#     return ", ".join(found_skills)

# def recommend_from_resume(pdf_path, top_k=5):
#     resume_text = extract_text_from_pdf(pdf_path)
#     resume_skills = extract_skills(resume_text)

#     if not resume_skills:
#         return []

#     vectorizer = TfidfVectorizer()
#     internship_texts = internships_resume['required_skills'].fillna("") + " " + internships_resume['title']
#     vectors = vectorizer.fit_transform([resume_skills] + internship_texts.tolist())
#     scores = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
#     top_indices = scores.argsort()[-top_k:][::-1]

#     results = []
#     for idx in top_indices:
#         results.append({
#             "internship_id": internships_resume.iloc[idx]['internship_id'],
#             "Company_Name": internships_resume.iloc[idx]['company_name'],
#             "Role": internships_resume.iloc[idx]['title'],
#             "Location": internships_resume.iloc[idx]['location'],
#             "Required_Skills": internships_resume.iloc[idx]['required_skills'],
#             "Match_Score(%)": round(scores[idx]*100, 2)
#         })
#     return results

# # ---------------------------
# # Routes
# # ---------------------------
# @app.route("/")
# def home():
#     return render_template("index.html")

# @app.route("/second")
# def second():
#     return render_template("second.html")

# @app.route("/recommend", methods=["POST"])
# def recommend():
#     data = request.get_json()
#     name = data.get("name", "")
#     role = data.get("role", "")
#     location = data.get("location", "")
#     skills = data.get("skills", "")
#     interests = data.get("interests", "")

#     recommendations = recommend_custom(name, role, skills, location, interests, top_k=5)
#     return jsonify({"results": recommendations})

# @app.route("/upload_resume", methods=["POST"])
# def upload_resume():
#     if "resume" not in request.files:
#         return jsonify([])

#     resume_file = request.files["resume"]
#     if resume_file.filename == "":
#         return jsonify([])

#     # Save temporarily and recommend
#     temp_path = "temp_resume.pdf"
#     resume_file.save(temp_path)
#     recommendations = recommend_from_resume(temp_path, top_k=5)
#     return jsonify(recommendations)

# @app.route("/apply", methods=["POST"])
# def apply():
#     data = request.get_json()
#     conn = get_db_connection()
#     cursor = conn.cursor()
#     cursor.execute("""
#         INSERT INTO applications (internship_id, company_name, role, location, required_skills, match_score)
#         VALUES (%s, %s, %s, %s, %s, %s)
#     """, (
#         data.get("internship_id"),
#         data.get("Company_Name"),
#         data.get("Role"),
#         data.get("Location"),
#         data.get("Required_Skills"),
#         data.get("Match_Score(%)")
#     ))
#     conn.commit()
#     cursor.close()
#     conn.close()
#     return jsonify({"message": f"Applied successfully for {data.get('Role')} at {data.get('Company_Name')}!"})


# if __name__ == "__main__":
#     app.run(debug=True)

from flask import Flask, render_template, request, jsonify, redirect
import pickle
import pandas as pd
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
import pymysql
import csv
import os

logging.getLogger("pdfminer").setLevel(logging.ERROR)

app = Flask(__name__)

# ---------------------------
# Database connection (for applications)
# ---------------------------
def get_db_connection():
    conn = pymysql.connect(
        host="localhost",
        user="root",
        password="18Ecocs052@",
        database="internship_applications"
    )
    return conn

# ---------------------------
# Load saved internship recommendation model
# ---------------------------
with open("internship_model.pkl", "rb") as f:
    saved_data = pickle.load(f)

vectorizer = saved_data["vectorizer"]
internship_vectors = saved_data["internship_vectors"]
internships = saved_data["internships"]

# ---------------------------
# Manual input recommendation
# ---------------------------
def recommend_custom(name, role, skills, location, interests="", top_k=5):
    custom_features = skills + " " + location + " " + interests + " " + role
    custom_vector = vectorizer.transform([custom_features])
    scores = cosine_similarity(custom_vector, internship_vectors)[0]
    top_indices = scores.argsort()[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            "internship_id": internships.iloc[idx]['internship_id'],
            "Company_Name": internships.iloc[idx]['company_name'],
            "Role": internships.iloc[idx]['title'],
            "Location": internships.iloc[idx]['location'],
            "Required_Skills": internships.iloc[idx]['required_skills'],
            "Match_Score(%)": round(scores[idx]*100, 2)
        })
    return results

# ---------------------------
# Resume-based recommendation
# ---------------------------
internships_resume = pd.read_csv("internships_demo.csv")

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + " "
    return text

def extract_skills(text):
    skills_list = ["python", "machine learning", "deep learning", "tensorflow",
                   "sql", "java", "c++", "communication", "data analysis", 
                   "nlp", "excel", "flask", "django", "cloud", "aws"]
    found_skills = []
    text = text.lower()
    for skill in skills_list:
        if skill.lower() in text:
            found_skills.append(skill)
    return ", ".join(found_skills)

def recommend_from_resume(pdf_path, top_k=5):
    resume_text = extract_text_from_pdf(pdf_path)
    resume_skills = extract_skills(resume_text)

    if not resume_skills:
        return []

    vectorizer = TfidfVectorizer()
    internship_texts = internships_resume['required_skills'].fillna("") + " " + internships_resume['title']
    vectors = vectorizer.fit_transform([resume_skills] + internship_texts.tolist())
    scores = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
    top_indices = scores.argsort()[-top_k:][::-1]

    results = []
    for idx in top_indices:
        results.append({
            "internship_id": internships_resume.iloc[idx]['internship_id'],
            "Company_Name": internships_resume.iloc[idx]['company_name'],
            "Role": internships_resume.iloc[idx]['title'],
            "Location": internships_resume.iloc[idx]['location'],
            "Required_Skills": internships_resume.iloc[idx]['required_skills'],
            "Match_Score(%)": round(scores[idx]*100, 2)
        })
    return results

# ---------------------------
# Routes
# ---------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/second")
def second():
    return render_template("second.html")

@app.route("/practice")
def practice():
    return render_template("practice.html")

@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json()
    name = data.get("name", "")
    role = data.get("role", "")
    location = data.get("location", "")
    skills = data.get("skills", "")
    interests = data.get("interests", "")

    recommendations = recommend_custom(name, role, skills, location, interests, top_k=5)
    return jsonify({"results": recommendations})

@app.route("/upload_resume", methods=["POST"])
def upload_resume():
    if "resume" not in request.files:
        return jsonify([])

    resume_file = request.files["resume"]
    if resume_file.filename == "":
        return jsonify([])

    temp_path = "temp_resume.pdf"
    resume_file.save(temp_path)
    recommendations = recommend_from_resume(temp_path, top_k=5)
    return jsonify(recommendations)

@app.route("/apply", methods=["POST"])
def apply():
    data = request.get_json()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO applications (internship_id, company_name, role, location, required_skills, match_score)
        VALUES (%s, %s, %s, %s, %s, %s)
    """, (
        data.get("internship_id"),
        data.get("Company_Name"),
        data.get("Role"),
        data.get("Location"),
        data.get("Required_Skills"),
        data.get("Match_Score(%)")
    ))
    conn.commit()
    cursor.close()
    conn.close()
    return jsonify({"message": f"Applied successfully for {data.get('Role')} at {data.get('Company_Name')}!"})

# ---------------------------
# Upload Internship (CSV saving)
# ---------------------------
CSV_FILE = "internships_demo.csv"

if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["internship_id","company_name","title","required_skills",
                         "location","duration_months","seats","stipend_inr","sector","is_demo"])

@app.route("/upload", methods=["POST"])
def upload_internship():
    data = [
        request.form["internship_id"],
        request.form["company_name"],
        request.form["title"],
        request.form["required_skills"],
        request.form["location"],
        request.form["duration_months"],
        request.form["seats"],
        request.form["stipend_inr"],
        request.form["sector"],
        request.form["is_demo"]
    ]

    with open(CSV_FILE, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(data)

    return jsonify({"success": True})


if __name__ == "__main__":
    app.run(debug=True)
