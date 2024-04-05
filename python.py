from flask import Flask, request, jsonify
from flask_asgi import FlaskASGI
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow_hub as hub
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Load Universal Sentence Encoder
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# Initialize Flask app
app = Flask(__name__)
asgi_app = FlaskASGI(app)

# Load NLTK resources
import nltk
nltk.download('punkt')
nltk.download('stopwords')

# Get English stop words
stop_words = set(stopwords.words('english'))

# Function for text preprocessing
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove special characters, numbers, and extra spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove stop words
    tokens = [word for word in tokens if word not in stop_words]
    # Join tokens back into a single string
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

# Function to calculate similarity between two preprocessed sentences
def semantic_similarity(sentence1, sentence2):
    embeddings = embed([sentence1, sentence2])
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return similarity.item()

# Define API endpoint
@app.route('/api/similarity', methods=['POST'])
def calculate_similarity():
    # Get text inputs from request body
    request_data = request.json
    text1 = request_data.get('text1', '')
    text2 = request_data.get('text2', '')

    # Preprocess text inputs
    text1 = preprocess_text(text1)
    text2 = preprocess_text(text2)

    # Calculate similarity
    similarity_score = semantic_similarity(text1, text2)

    # Prepare response body
    response_body = {'similarity_score': similarity_score}

    # Return JSON response
    return jsonify(response_body)

# Define root endpoint
@app.route('/', methods=['GET'])
def index():
    return "Welcome to the similarity calculation API!"

if __name__ == '__main__':
    # Run Flask app
    asgi_app.run(debug=False)
