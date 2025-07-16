# Spam-detection-in-email
#This repository provides code for spam detection in email using AI which normal people are not aware about
pip install pandas scikit-learn nltk tensorflow
import pandas as pd
import re
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
def preprocess_text(text):
    text = text.lower()  # lowercase
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    tokens = nltk.word_tokenize(text)  # tokenize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]  # remove stopwords + lemmatize
    return ' '.join(tokens)

data = {
    'text': [
        'Congratulations! You have won a $1000 Walmart gift card. Click to claim now.',
        'Hi, are we still meeting tomorrow?',
        'Earn money from home, no investment needed!',
        'Reminder: Your appointment is scheduled for 3 PM.',
        'Get cheap loans instantly with no credit check!',
        'Lunch at 1 PM?',
        'You have been selected for a chance to win an iPhone.',
        'Project deadline is next week, please prepare slides.'
    ],
    'label': ['spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham']
}

df = pd.DataFrame(data)
df['clean_text'] = df['text'].apply(preprocess_text)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.utils import to_categorical

X = df['clean_text']
y = df['label']

tfidf = TfidfVectorizer(max_features=1000)
X_tfidf = tfidf.fit_transform(X).toarray()

X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

# Tokenization
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(df['clean_text'])
sequences = tokenizer.texts_to_sequences(df['clean_text'])

# Padding sequences
max_len = 20
X_seq = pad_sequences(sequences, maxlen=max_len, padding='post')
y = np.array(df['label'])

X_train, X_test, y_train, y_test = train_test_split(X_seq, y, test_size=0.2, random_state=42)

# LSTM Model
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=64, input_length=max_len))
model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=2, validation_data=(X_test, y_test))



def predict_email_lstm(email_text):
    clean = preprocess_text(email_text)
    seq = tokenizer.texts_to_sequences([clean])
    padded = pad_sequences(seq, maxlen=max_len, padding='post')
    pred = model.predict(padded)[0][0]
    return "spam" if pred > 0.5 else "ham"

print("\n📧 Predictions:")
emails = [
    "Congratulations! You've won a free iPhone!",
    "Team meeting at 2 PM in the conference room.",
    "Urgent loan approval. Claim in 5 minutes!"
]
for e in emails:
    print(f"'{e}' --> {predict_email_lstm(e)}")



