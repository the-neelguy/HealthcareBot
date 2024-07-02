import json
import numpy as np
import pickle
import spacy
from textblob import TextBlob
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Conv1D, GlobalMaxPooling1D, Embedding, Concatenate, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load Spacy model for NER
nlp = spacy.load("en_core_web_sm")

# Load preprocessed data
train_x = np.load('./models/train_x.npy', allow_pickle=True)
train_y = np.load('./models/train_y.npy', allow_pickle=True)

# Load words and classes
with open('./models/words.pkl', 'rb') as handle:
    words = pickle.load(handle)

with open('./models/classes.pkl', 'rb') as handle:
    classes = pickle.load(handle)

# Load tokenizer and label encoder
with open('./models/tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

with open('./models/label_encoder.pkl', 'rb') as handle:
    label_encoder = pickle.load(handle)

# Define the function to extract entities
def extract_entities(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

# Define the function to perform sentiment analysis
def analyze_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity, blob.sentiment.subjectivity

# Add NER and sentiment analysis features to the training data
def add_features(texts):
    ner_features = []
    sentiment_features = []

    for text in texts:
        entities = extract_entities(text)
        sentiment = analyze_sentiment(text)

        ner_vector = [0] * len(classes)
        for entity in entities:
            if entity[1] in classes:
                ner_vector[classes.index(entity[1])] = 1

        sentiment_vector = [sentiment[0], sentiment[1]]

        ner_features.append(ner_vector)
        sentiment_features.append(sentiment_vector)

    return np.array(ner_features), np.array(sentiment_features)

# Tokenize and pad the sequences
sequences = tokenizer.texts_to_sequences(train_x)
padded_sequences = pad_sequences(sequences, maxlen=20, padding='post')

# Add NER and sentiment analysis features
ner_features, sentiment_features = add_features(train_x)

# Combine all features
combined_features = np.concatenate((padded_sequences, ner_features, sentiment_features), axis=1)

# Define the model
input_layer = Input(shape=(combined_features.shape[1],))
embedding_layer = Embedding(input_dim=5000, output_dim=64, input_length=20)(input_layer)
conv_layer = Conv1D(64, 5, activation='relu')(embedding_layer)
pooling_layer = GlobalMaxPooling1D()(conv_layer)
dense_layer1 = Dense(64, activation='relu')(pooling_layer)
dropout_layer = Dropout(0.5)(dense_layer1)
dense_layer2 = Dense(len(classes), activation='softmax')(dropout_layer)

model = Model(inputs=input_layer, outputs=dense_layer2)

model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# Train the model
model.fit(combined_features, np.array(train_y), epochs=200, batch_size=5, verbose=1)

# Save the trained model
model.save('./models/chatbot_model_with_features.h5')

print("Model trained and saved")
