import streamlit as st
import os
import pypdf  # Changed from PyPDF2
import docx
import spacy
import re
import numpy as np
from io import BytesIO
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

# Import sklearn with error handling
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError as e:
    st.error(f"scikit-learn import error: {e}")
    st.info("Try installing: pip install scikit-learn==1.3.2")
    SKLEARN_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="AI Resume Screening Tool",
    page_icon="📄",
    layout="wide"
)

# Load spaCy model with error handling
@st.cache_resource
def load_spacy_model():
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            nlp = spacy.load('en_core_web_sm')
        return nlp
    except OSError:
        st.error("spaCy model not found. Please install it using: `python -m spacy download en_core_web_sm`")
        return None

nlp = load_spacy_model()

def extract_text_from_pdf(file):
    """Extract text from PDF file using pypdf"""
    try:
        pdf_reader = pypdf.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

def extract_text_from_docx(file):
    """Extract text from DOCX file"""
    try:
        doc = docx.Document(file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading DOCX: {e}")
        return ""

def preprocess_text(text):
    """Clean and preprocess text"""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s.,!?;:]', '', text)
    return text.strip()

def extract_skills(text):
    """Extract skills from text using spaCy"""
    if nlp is None:
        return []
    
    doc = nlp(text)
    skills = []
    
    for chunk in doc.noun_chunks:
        if len(chunk.text.split()) <= 3:
            skills.append(chunk.text.lower())
    
    return list(set(skills))

def calculate_match_score(job_description, resume_text):
    """Calculate match score between job description and resume"""
    if not SKLEARN_AVAILABLE:
        return 0, [], []
    
    job_desc_clean = preprocess_text(job_description)
    resume_clean = preprocess_text(resume_text)
    
    vectorizer = TfidfVectorizer(stop_words='english')
    try:
        tfidf_matrix = vectorizer.fit_transform([job_desc_clean, resume_clean])
        cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        match_score = round(cosine_sim * 100, 2)
    except:
        match_score = 0
    
    job_skills = extract_skills(job_desc_clean)
    resume_skills = extract_skills(resume_clean)
    
    if job_skills:
        skill_match = len(set(job_skills) & set(resume_skills)) / len(job_skills) * 100
    else:
        skill_match = 0
    
    final_score = (match_score * 0.7) + (skill_match * 0.3)
    
    return min(100, final_score), job_skills, resume_skills

def generate_suggestions(job_description, resume_text, job_skills, resume_skills, match_score):
    """Generate improvement suggestions for the resume"""
    suggestions = []
    
    missing_skills = set(job_skills) - set(resume_skills)
    if missing_skills:
        suggestions.append(f"Add these missing skills: {', '.join(list(missing_skills)[:5])}")
    
    if len(resume_text) < 500:
        suggestions.append("Consider adding more detail about your experiences and achievements")
    elif len(resume_text) > 2000:
        suggestions.append("Consider making your resume more concise and focused")
    
    if match_score < 50:
        suggestions.append("Focus on aligning your experience more closely with the job requirements")
    elif match_score < 70:
        suggestions.append("Highlight more relevant projects and experiences that match the job description")
    else:
        suggestions.append("Good match! Consider adding quantifiable achievements to stand out")
    
    if not re.search(r'\d+', resume_text):
        suggestions.append("Add quantifiable achievements (e.g., 'increased sales by 20%')")
    
    return suggestions

def get_match_description(score):
    """Get descriptive text for match score"""
    if score >= 90:
        return "Excellent match with strong alignment to job requirements"
    elif score >= 80:
        return "Strong match with good technical and experience alignment"
    elif score >= 70:
        return "Good match with relevant skills and experience"
    elif score >= 60:
        return "Moderate match with some relevant qualifications"
    elif score >= 50:
        return "Basic match, needs improvement in key areas"
    else:
        return "Limited match, significant improvements needed"

# Streamlit UI
def main():
    st.title("📄 AI Resume Screening Tool")
    st.markdown("Upload job description and resumes to analyze candidate matches")
    
    if not SKLEARN_AVAILABLE:
        st.error("⚠️ scikit-learn is not available. Please install compatible versions.")
        st.code("pip install scikit-learn==1.3.2 numpy==1.24.3 scipy==1.10.1")
        return
    
    # Job description input
    st.header("Job Description")
    job_description = st.text_area(
        "Paste the job description here:",
        height=200,
        placeholder="Enter the job description, requirements, and qualifications..."
    )
    
    # Resume upload
    st.header("Upload Resumes")
    uploaded_files = st.file_uploader(
        "Choose resume files (PDF or DOCX)",
        type=['pdf', 'docx', 'doc'],
        accept_multiple_files=True,
        help="Upload multiple resumes to compare"
    )
    
    # Analysis button
    if st.button("Analyze Resumes", type="primary"):
        if not job_description.strip():
            st.error("Please enter a job description")
            return
        
        if not uploaded_files:
            st.error("Please upload at least one resume")
            return
        
        if nlp is None:
            st.error("spaCy model not loaded. Please check the installation.")
            return
        
        results = []
        
        with st.spinner("Analyzing resumes..."):
            for uploaded_file in uploaded_files:
                # Extract text based on file type
                if uploaded_file.type == "application/pdf":
                    resume_text = extract_text_from_pdf(uploaded_file)
                else:
                    resume_text = extract_text_from_docx(uploaded_file)
                
                if not resume_text.strip():
                    st.warning(f"Could not extract text from {uploaded_file.name}")
                    continue
                
                # Calculate match score
                match_score, job_skills, resume_skills = calculate_match_score(job_description, resume_text)
                
                # Generate suggestions
                suggestions = generate_suggestions(job_description, resume_text, job_skills, resume_skills, match_score)
                
                result = {
                    'name': uploaded_file.name.rsplit('.', 1)[0],
                    'score': round(match_score),
                    'matchDetails': get_match_description(match_score),
                    'strengths': list(resume_skills)[:5],
                    'suggestions': suggestions[:4],
                    'filename': uploaded_file.name
                }
                
                results.append(result)
        
        # Sort and display results
        results.sort(key=lambda x: x['score'], reverse=True)
        
        st.header("Analysis Results")
        
        for i, result in enumerate(results, 1):
            with st.container():
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.subheader(f"{i}. {result['name']}")
                    st.write(f"**Match Score:** {result['score']}%")
                    st.write(f"**Assessment:** {result['matchDetails']}")
                    
                    if result['strengths']:
                        st.write("**Key Strengths:**")
                        for strength in result['strengths']:
                            st.write(f"- {strength}")
                
                with col2:
                    # Color code based on score
                    if result['score'] >= 80:
                        st.success(f"🏆 {result['score']}%")
                    elif result['score'] >= 60:
                        st.info(f"👍 {result['score']}%")
                    else:
                        st.error(f"📊 {result['score']}%")
                
                # Suggestions
                with st.expander("Improvement Suggestions"):
                    for suggestion in result['suggestions']:
                        st.write(f"• {suggestion}")
                
                st.divider()

if __name__ == "__main__":
    main()