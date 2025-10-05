from flask import Flask, request, jsonify, render_template
import os
import PyPDF2
import docx
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'doc'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load spaCy model with error handling
try:
    nlp = spacy.load('en_core_web_sm')
    print("spaCy model loaded successfully!")
except OSError:
    print("Downloading spaCy model...")
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load('en_core_web_sm')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(file_path):
    """Extract text from PDF file"""
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""

def extract_text_from_docx(file_path):
    """Extract text from DOCX file"""
    try:
        doc = docx.Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        print(f"Error reading DOCX: {e}")
        return ""

def preprocess_text(text):
    """Clean and preprocess text"""
    # Remove extra whitespace and newlines
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?;:]', '', text)
    return text.strip()

def extract_skills(text):
    """Extract skills from text using spaCy"""
    doc = nlp(text)
    skills = []
    
    # Simple skill extraction based on POS patterns
    for chunk in doc.noun_chunks:
        if len(chunk.text.split()) <= 3:  # Limit to reasonable skill lengths
            skills.append(chunk.text.lower())
    
    # Remove duplicates and return
    return list(set(skills))

def calculate_match_score(job_description, resume_text):
    """Calculate match score between job description and resume"""
    # Preprocess texts
    job_desc_clean = preprocess_text(job_description)
    resume_clean = preprocess_text(resume_text)
    
    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(stop_words='english')
    try:
        tfidf_matrix = vectorizer.fit_transform([job_desc_clean, resume_clean])
        cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        # Convert to percentage (0-100)
        match_score = round(cosine_sim * 100, 2)
    except:
        match_score = 0
    
    # Extract skills for analysis
    job_skills = extract_skills(job_desc_clean)
    resume_skills = extract_skills(resume_clean)
    
    # Calculate skill match
    if job_skills:
        skill_match = len(set(job_skills) & set(resume_skills)) / len(job_skills) * 100
    else:
        skill_match = 0
    
    # Combine scores (weighted average)
    final_score = (match_score * 0.7) + (skill_match * 0.3)
    
    return min(100, final_score), job_skills, resume_skills

def generate_suggestions(job_description, resume_text, job_skills, resume_skills, match_score):
    """Generate improvement suggestions for the resume"""
    suggestions = []
    
    # Skill-based suggestions
    missing_skills = set(job_skills) - set(resume_skills)
    if missing_skills:
        suggestions.append(f"Add these missing skills: {', '.join(list(missing_skills)[:5])}")
    
    # Length-based suggestions
    if len(resume_text) < 500:
        suggestions.append("Consider adding more detail about your experiences and achievements")
    elif len(resume_text) > 2000:
        suggestions.append("Consider making your resume more concise and focused")
    
    # Content suggestions based on match score
    if match_score < 50:
        suggestions.append("Focus on aligning your experience more closely with the job requirements")
    elif match_score < 70:
        suggestions.append("Highlight more relevant projects and experiences that match the job description")
    else:
        suggestions.append("Good match! Consider adding quantifiable achievements to stand out")
    
    # Format suggestions
    if not re.search(r'\d+', resume_text):  # Check for numbers/quantifiable results
        suggestions.append("Add quantifiable achievements (e.g., 'increased sales by 20%')")
    
    return suggestions

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_resumes():
    try:
        # Get form data
        job_description = request.form.get('job_description', '')
        
        # Check if files were uploaded
        if 'resumes' not in request.files:
            return jsonify({'error': 'No files uploaded'}), 400
        
        files = request.files.getlist('resumes')
        if not files or files[0].filename == '':
            return jsonify({'error': 'No files selected'}), 400
        
        results = []
        
        for file in files:
            if file and allowed_file(file.filename):
                # Save file temporarily
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                
                # Extract text based on file type
                if filename.lower().endswith('.pdf'):
                    resume_text = extract_text_from_pdf(file_path)
                else:
                    resume_text = extract_text_from_docx(file_path)
                
                # Clean up temporary file
                os.remove(file_path)
                
                if not resume_text.strip():
                    continue
                
                # Calculate match score and get skills
                match_score, job_skills, resume_skills = calculate_match_score(job_description, resume_text)
                
                # Generate suggestions
                suggestions = generate_suggestions(job_description, resume_text, job_skills, resume_skills, match_score)
                
                # Prepare result
                result = {
                    'name': filename.rsplit('.', 1)[0],  # Use filename without extension as name
                    'score': round(match_score),
                    'matchDetails': get_match_description(match_score),
                    'strengths': list(resume_skills)[:5],  # Top 5 skills as strengths
                    'suggestions': suggestions[:4]  # Limit to 4 suggestions
                }
                
                results.append(result)
        
        # Sort results by score (highest first)
        results.sort(key=lambda x: x['score'], reverse=True)
        
        return jsonify({'results': results})
    
    except Exception as e:
        print(f"Error in analysis: {e}")
        return jsonify({'error': 'An error occurred during analysis'}), 500

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

if __name__ == '__main__':
    # Fix spaCy version warning by disabling it
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module='spacy')
    
    print("Starting AI Resume Screening Tool...")
    print("Access the application at: http://127.0.0.1:5000")
    app.run(debug=True, host='127.0.0.1', port=5000)