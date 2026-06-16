from flask import Flask, render_template, request, jsonify
import json
import random
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

with open('intents.json') as file:
    data = json.load(file)

sentences = []
labels = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        sentences.append(pattern)
        labels.append(intent['tag'])

def clean_text(text):
    tokens = nltk.word_tokenize(text.lower())
    return " ".join([lemmatizer.lemmatize(w) for w in tokens])

sentences = [clean_text(s) for s in sentences]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(sentences)
y = labels

model = MultinomialNB()
model.fit(X, y)


def chatbot_response(user_input):
    user_input = clean_text(user_input)
    X_test = vectorizer.transform([user_input])
    predicted_tag = model.predict(X_test)[0]

    for intent in data['intents']:
        if intent['tag'] == predicted_tag:
            return random.choice(intent['responses'])
    return "I'm sorry, I didn't quite understand that. Please rephrase your question."

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get_response", methods=["POST"])
def get_response():
    user_input = request.form["msg"]
    response = chatbot_response(user_input)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)