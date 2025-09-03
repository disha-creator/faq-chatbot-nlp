import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load English language model
nlp = spacy.load("en_core_web_sm")

# Sample FAQs
faqs = [
    {
        "question": "What is your return policy?",
        "answer": "You can return items within 30 days of purchase."
    },
    {
        "question": "How long does shipping take?",
        "answer": "Standard shipping takes 3-5 business days."
    },
    {
        "question": "Do you offer international shipping?",
        "answer": "Yes, we ship to most countries worldwide."
    }
]

# Text preprocessing
def preprocess(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

# Find best answer
def get_answer(user_query):
    # Preprocess all questions
    processed_questions = [preprocess(faq["question"]) for faq in faqs]
    processed_query = preprocess(user_query)
    
    # Combine questions + user query
    all_texts = processed_questions + [processed_query]
    
    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    
    # Calculate similarities
    similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    best_match_idx = similarities.argmax()
    
    return faqs[best_match_idx]["answer"]

# Simple chat interface
print("FAQ Bot: Hello! Ask me anything (type 'quit' to exit)")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'quit':
        break
    response = get_answer(user_input)
    print("Bot:", response)