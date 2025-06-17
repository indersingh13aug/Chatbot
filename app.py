from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer, util
import os

app = Flask(__name__)
CORS(app)

# Load NLP model
model = SentenceTransformer('paraphrase-MiniLM-L3-v2')

# Directory containing all QA txt files
QA_DIR = "qa_files"

# Cache: site_name -> (qa_data, embeddings)
qa_cache = {}

def load_qa_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        lines = file.read().split('\n')
    qas = []
    question, answer = None, None
    for line in lines:
        if line.startswith("Q:"):
            question = line[2:].strip()
        elif line.startswith("A:"):
            answer = line[2:].strip()
        elif line.strip() == "" and question and answer:
            qas.append((question, answer))
            question, answer = None, None
    if question and answer:
        qas.append((question, answer))
    return qas

def get_qa_data(site):
    if site in qa_cache:
        return qa_cache[site]
    
    path = os.path.join(QA_DIR, f"{site}.txt")
    if not os.path.exists(path):
        return None
    
    qa_data = load_qa_file(path)
    questions = [q for q, _ in qa_data]
    embeddings = model.encode(questions, convert_to_tensor=True)
    qa_cache[site] = (qa_data, embeddings)
    return qa_cache[site]

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    site = data.get("site")
    user_question = data.get("message")

    if not site or not user_question:
        return jsonify({"reply": "Missing site or question."}), 400

    result = get_qa_data(site)
    if not result:
        return jsonify({"reply": f"Site '{site}' not found."}), 404

    qa_data, question_embeddings = result
    user_embedding = model.encode(user_question, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(user_embedding, question_embeddings)[0]
    best_idx = scores.argmax().item()
    best_score = scores[best_idx].item()

    if best_score < 0.4:
        return jsonify({"reply": f"Sorry, I couldn't find a relevant answer. I can give answer related to {site} only."})

    return jsonify({"reply": qa_data[best_idx][1]})

if __name__ == "__main__":
    app.run(debug=True)
