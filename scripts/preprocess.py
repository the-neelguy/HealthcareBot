import json
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
import pickle
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
import spacy
from textblob import TextBlob

nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
nlp = spacy.load("en_core_web_sm")

words = []
classes = []
documents = []
ignore_words = ['?', '!', '.', ',']
data_file = open('./data/intents.json').read()
intents = json.loads(data_file)

# Store original text patterns for tokenizer
text_patterns = []

for intent in intents['intents']:
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, intent['tag']))
        text_patterns.append(pattern)  # Store the original text pattern
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

classes = sorted(list(set(classes)))

if not os.path.exists('./models/'):
    os.makedirs('./models/')

pickle.dump(words, open('./models/words.pkl', 'wb'))
pickle.dump(classes, open('./models/classes.pkl', 'wb'))

training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

training = np.array(training, dtype=object)
train_x = list(training[:, 0])
train_y = list(training[:, 1])

# Save training data
np.save('./models/train_x.npy', train_x)
np.save('./models/train_y.npy', train_y)

# Tokenizer setup
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(text_patterns)  # Use original text patterns

# Save tokenizer
with open('./models/tokenizer.pkl', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# LabelEncoder setup
label_encoder = LabelEncoder()
train_y_enc = label_encoder.fit_transform([str(y) for y in train_y])

# Save label encoder
with open('./models/label_encoder.pkl', 'wb') as handle:
    pickle.dump(label_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Sentiment analysis and NER features
def extract_entities(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

def analyze_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity, blob.sentiment.subjectivity

ner_features = []
sentiment_features = []

for text in text_patterns:
    entities = extract_entities(text)
    sentiment = analyze_sentiment(text)

    ner_vector = [0] * len(classes)
    for entity in entities:
        if entity[1] in classes:
            ner_vector[classes.index(entity[1])] = 1

    sentiment_vector = [sentiment[0], sentiment[1]]

    ner_features.append(ner_vector)
    sentiment_features.append(sentiment_vector)

ner_features = np.array(ner_features)
sentiment_features = np.array(sentiment_features)

# Save additional features
np.save('./models/ner_features.npy', ner_features)
np.save('./models/sentiment_features.npy', sentiment_features)

print("Preprocessing complete")
