import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import LabelEncoder
import random

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

def load_training_data():
    """Load and prepare training data from intents.json"""
    with open('intents.json', 'r', encoding='utf-8') as file:
        intents = json.load(file)
    
    words = []
    classes = []
    documents = []
    ignore_letters = ['?', '!', '.', ',']
    
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            word_list = nltk.word_tokenize(pattern)
            words.extend(word_list)
            documents.append((word_list, intent['tag']))
            if intent['tag'] not in classes:
                classes.append(intent['tag'])
    
    words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
    words = sorted(set(words))
    classes = sorted(set(classes))
    
    return words, classes, documents

def create_training_data(words, classes, documents):
    """Create training data for the neural network"""
    training = []
    output_empty = [0] * len(classes)
    
    for doc in documents:
        bag = []
        word_patterns = doc[0]
        word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
        
        for word in words:
            bag.append(1) if word in word_patterns else bag.append(0)
        
        output_row = list(output_empty)
        output_row[classes.index(doc[1])] = 1
        training.append([bag, output_row])
    
    random.shuffle(training)
    training = np.array(training, dtype=object)
    
    train_x = list(training[:, 0])
    train_y = list(training[:, 1])
    
    return np.array(train_x), np.array(train_y)

def build_model(input_shape, output_shape):
    """Build the neural network model"""
    model = Sequential()
    model.add(Dense(128, input_shape=(input_shape,), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_shape, activation='softmax'))
    
    sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    
    return model

def train_chatbot():
    """Train the chatbot model"""
    print("Loading training data...")
    words, classes, documents = load_training_data()
    
    print("Creating training data...")
    train_x, train_y = create_training_data(words, classes, documents)
    
    print("Building model...")
    model = build_model(len(train_x[0]), len(train_y[0]))
    
    print("Training model...")
    hist = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)
    
    print("Saving model and data...")
    model.save('chatbot_model.h5')
    
    with open('words.pkl', 'wb') as f:
        pickle.dump(words, f)
    
    with open('classes.pkl', 'wb') as f:
        pickle.dump(classes, f)
    
    print("Training completed successfully!")
    print(f"Model saved as 'chatbot_model.h5'")
    print(f"Words saved as 'words.pkl'")
    print(f"Classes saved as 'classes.pkl'")
    
    return model, words, classes

if __name__ == '__main__':
    train_chatbot()
