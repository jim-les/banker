import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

# Function to log detected attacks for feedback
def log_attack_details(sentence, predicted_label, actual_label):
    with open('attack_log.txt', 'a') as log_file:
        log_file.write(f"Sentence: {sentence}\n")
        log_file.write(f"Predicted Label: {predicted_label}\n")
        log_file.write(f"Actual Label: {actual_label}\n\n")

# Load your dataset (replace 'd
# ataset.csv' with your actual dataset)
data = pd.read_csv('sqliv2.csv', encoding='ISO-8859-1')  

# Extract sentences and labels
sentences = data['Sentence']
labels = data['Label']

# Preprocess the data
max_words = 1000  # Maximum number of words in your vocabulary
max_len = 50  # Maximum length of a sentence

# Handle non-string values by converting them to empty strings
sentences = sentences.fillna('').astype(str)

# Tokenize and pad sequences
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)
X = pad_sequences(sequences, maxlen=max_len)

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build an advanced LSTM model
model = Sequential()
model.add(Embedding(input_dim=max_words, output_dim=64, input_length=max_len))
model.add(Bidirectional(LSTM(128, return_sequences=True)))
model.add(Bidirectional(LSTM(128)))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Implement early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=64, validation_split=0.2, callbacks=[early_stopping])

# Evaluate the model on the test data
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss:.4f}, Test Accuracy: {accuracy*100:.2f}%')

# Assuming you have a list of sentences to classify for feedback
sentences_to_classify = ["Suspicious SQL injection attempt", "Normal user query", "Another attack example"]

# Classify the sentences and log details for attacks
for sentence in sentences_to_classify:
    # Preprocess the sentence and tokenize it
    sequence = tokenizer.texts_to_sequences([sentence])
    padded_sequence = pad_sequences(sequence, maxlen=max_len)

    # Predict the label for the sentence
    predicted_label = model.predict(padded_sequence)

    # Assuming you have a threshold for classifying attacks
    threshold = 0.5
    if predicted_label > threshold:
        # Log the attack details for feedback
        log_attack_details(sentence, predicted_label, "Attack")
    else:
        log_attack_details(sentence, predicted_label, "Normal")

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load your best classification model (replace 'best_model.h5' with your model's file)
best_model = load_model('model.h5')

# Predict using the loaded model
y_pred = best_model.predict(X_test)  # X_test contains your test data

# Convert predictions to binary values (0 or 1)
y_pred_binary = (y_pred > 0.5).astype(int)

# Calculate evaluation metrics
accuracy = accuracy_score(y_true, y_pred_binary)  # Replace y_true with your true labels
precision = precision_score(y_true, y_pred_binary)
recall = recall_score(y_true, y_pred_binary)
f1 = f1_score(y_true, y_pred_binary)

print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1-score: {f1:.2f}')
