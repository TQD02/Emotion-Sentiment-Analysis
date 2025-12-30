
import numpy as np
import pandas as pd
import json
import tensorflow as tf

print(f"TensorFlow version: {tf.__version__}")
import keras
from keras import layers, Model
from tensorflow.keras.layers import TextVectorization  
from keras.preprocessing.sequence import pad_sequences
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score
from sklearn.metrics import accuracy_score
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score
from keras import regularizers
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import zipfile
import os



#Extract and list files
with zipfile.ZipFile('archive.zip', 'r') as zip_ref:
    zip_ref.extractall('extracted_files')
    
print("Files extracted to 'extracted_files' folder")

for root, dirs, files_list in os.walk('extracted_files'):
    for file in files_list:
        print(os.path.join(root, file))



#Inspect Data
train_data = pd.read_csv('extracted_files/data/train.tsv', sep = '\t',header = None, names = ['Text', 'emotion_id', 'id'])
test_data = pd.read_csv('extracted_files/data/test.tsv', sep = '\t',header = None, names = ['Text', 'emotion_id', 'id'])
print(train_data.head())
print(test_data)
print(train_data.count())
multi_label_mask = train_data['emotion_id'].astype(str).str.contains(',')
multi_label_rows = train_data[multi_label_mask]

print(f"Total rows with multiple emotions: {len(multi_label_rows)}")

#Load emotion labels
with open('extracted_files/data/emotions.txt', 'r') as f:
    emotions = [line.strip() for line in f.readlines()]
with open('extracted_files/data/sentiment_dict.json', 'r') as f:
    sentiment_map = json.load(f)
print(emotions)
print(sentiment_map)
sentiment_map.update({"neutral": "neutral"})
sentiment_classes = list(sentiment_map.keys())
print(sentiment_classes)
NUM_CLASSES = len(sentiment_classes)


emotion_idx_to_sentiment_idx = {}
for emotion_idx, emotion_name in enumerate(emotions):
    found = False
    for sent_idx, sent_name in enumerate(sentiment_classes):
        if emotion_name in sentiment_map[sent_name]:
            emotion_idx_to_sentiment_idx[emotion_idx] = sent_idx
            found = True
            break
    if not found:
        print(f"Warning: {emotion_name} not found in any sentiment group!")
print(f"Mapped {len(emotion_idx_to_sentiment_idx)} emotions to {NUM_CLASSES} sentiment classes.\n")

#Inspect emotion mapping index
for emotion_idx, emotion_name in enumerate(emotions):
    if emotion_idx in emotion_idx_to_sentiment_idx:
        sent_idx = emotion_idx_to_sentiment_idx[emotion_idx]
        sent_name = sentiment_classes[sent_idx]
        
        print(f"{emotion_idx:<12} {emotion_name:<20} {sent_idx:<15} {sent_name}")
    else:
        print(f"{emotion_idx:<12} {emotion_name:<20} {'MISSING':<15} {'MISSING'}")


#Check for missing values
print(f"\nMissing values in training data:\n{train_data.isnull().sum()}")

#Emotion distribution
emotion_counts = train_data['emotion_id'].apply(
    lambda x: int(str(x).split(',')[0])
).value_counts().sort_index()
print(f"\nEmotion distribution:\n{emotion_counts}")

#Plot emotion distribution
plt.figure(figsize=(14, 6))
emotion_names = [emotions[i] if i < len(emotions) else f"Emotion_{i}" 
                 for i in emotion_counts.index]
plt.bar(emotion_names, emotion_counts.values, color='steelblue')
plt.xlabel('Emotion', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Distribution of Emotions in Training Data', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('emotion_distribution.png', dpi=300, bbox_inches='tight')
plt.show()


#Data cleaning
def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower() #Convert lower case
    text = re.sub(r'http\S+', '', text) #Remove HTML tags
    text = re.sub(r'[^a-zA-Z\s]', '', text) #Remove special chars w/o basic punctuation
    text = re.sub(r'\s+', ' ', text).strip() #Remove white noise
    return text

train_data['cleaned_text'] = train_data['Text'].apply(clean_text)
test_data['cleaned_text'] = test_data['Text'].apply(clean_text)


#Tokenization
MAX_WORDS = 10000
MAX_LEN = 100
EMBEDDING_DIM = 128

vectorizer = TextVectorization(
    max_tokens=MAX_WORDS,
    output_mode='int',
    output_sequence_length=MAX_LEN,
    standardize=None  # We already cleaned the text
)

# Adapt the vectorizer to the training data
vectorizer.adapt(train_data['cleaned_text'].values)

# Get vocabulary size
vocab_size = len(vectorizer.get_vocabulary())
print(f"Vocabulary size: {vocab_size}")
#Pad sequences to same length
X_train_pad = vectorizer(train_data['cleaned_text'].values).numpy()
X_test_pad = vectorizer(test_data['cleaned_text'].values).numpy()

print(f"Shape of X train: {X_train_pad.shape}")
print(f"Shape of X test: {X_test_pad.shape}")

#Convert to multi-label binary format
def emotion_to_sentiment_vector(emotion_id_str, num_classes=4):
    label_vector = np.zeros(num_classes, dtype=np.float32)
    
    try:
        
        if isinstance(emotion_id_str, str) and ',' in emotion_id_str:
            emotion_ids = [int(x.strip()) for x in emotion_id_str.split(',')]
        else:
            emotion_ids = [int(emotion_id_str)]
        
        for eid in emotion_ids:
            if eid in emotion_idx_to_sentiment_idx:
                sent_idx = emotion_idx_to_sentiment_idx[eid]
                label_vector[sent_idx] = 1.0
                
    except Exception as e:
        print(f"Error parsing emotion_id '{emotion_id_str}': {e}")
    
    return label_vector
y_train = np.array([emotion_to_sentiment_vector(eid, NUM_CLASSES) 
                    for eid in train_data['emotion_id']])
y_test = np.array([emotion_to_sentiment_vector(eid, NUM_CLASSES) 
                   for eid in test_data['emotion_id']])    

X_train, X_val, y_train, y_val = train_test_split(X_train_pad, y_train, test_size=0.15, random_state=42, shuffle=True)
print(f"Shape of Y train: {y_train.shape}")
print(f"Shape of Y test: {y_test.shape}")



#Calculate Class Weights
class_counts = np.sum(y_train, axis=0) 
total_samples = len(y_train)

plt.figure(figsize=(8, 5))
bars = plt.bar(sentiment_classes, class_counts, color=['green', 'red', 'orange', 'gray'])

plt.title('Distribution of Training Samples per Sentiment')
plt.xlabel('Sentiment')
plt.ylabel('Number of Samples')
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 20, int(yval), ha='center', va='bottom')

plt.show()
print(f"Counts per class: {class_counts}")


class_weights = {}
for i in range(NUM_CLASSES):
    
    if class_counts[i] > 0:
        class_weights[i] = total_samples / (NUM_CLASSES * class_counts[i])
    else:
        class_weights[i] = 1.0 

print("\nComputed Class Weights:")
for i, name in enumerate(sentiment_classes):
    print(f"{name}: {class_weights[i]:.4f}")


#Model 1: Convolutional Neural Network (CNN) for text classification
def create_cnn_model(max_words=10000, max_len=100, embedding_dim=128, num_classes=2):
    model = keras.Sequential([
        layers.Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_len),
        layers.SpatialDropout1D(0.4), 

        layers.Conv1D(filters=128, kernel_size=2),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling1D(pool_size=2),

        layers.Conv1D(filters=128, kernel_size=3),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling1D(pool_size=2),

        layers.Conv1D(filters=128, kernel_size=4),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.GlobalMaxPooling1D(),

        layers.Dense(64, activation='relu',kernel_regularizer=regularizers.l2(0.01)),  
        layers.Dropout(0.5),

        layers.Dense(num_classes, activation='sigmoid')
    ])

    return model
 
#Model 2.1: Bi-directional Long Shot Term Memoty (BILSTM) with 2 layers 
def create_bilstm_model_1(max_words=10000, max_len=100, embedding_dim=128, num_classes=2):

    model = keras.Sequential([
        layers.Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_len),
        layers.SpatialDropout1D(0.3),

        layers.Bidirectional(layers.LSTM(64, dropout=0.3, recurrent_dropout=0.2)),
        
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        layers.Dropout(0.5),
        
        layers.Dense(num_classes, activation='sigmoid')
    ])

    return model


#Model 2.2: Bi-directional Long Shot Term Memoty (BILSTM) with 2 layers 
def create_bilstm_model_2(max_words=10000, max_len=100, embedding_dim=128, num_classes=2):
    model = keras.Sequential([
        layers.Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_len),
        layers.SpatialDropout1D(0.3),

        layers.Bidirectional(layers.LSTM(64, dropout=0.3, recurrent_dropout=0.2, return_sequences=True)),
        layers.Dropout(0.2),
        layers.Bidirectional(layers.LSTM(64)),

        layers.Dropout(0.2),

        
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='sigmoid')
    ])

    return model

def compile_and_train(model, X_train, y_train, X_val, y_val, num_classes=4, epochs=10, batch_size=256,class_weight=None):
    #Compile model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics= ['accuracy'])
    #Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.00001),
        ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True)]
    #Train model
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        class_weight=class_weight,   
        verbose=1)
    return history


#Create and train CNN model
model_cnn = create_cnn_model(MAX_WORDS, MAX_LEN, EMBEDDING_DIM, NUM_CLASSES)
history_cnn = compile_and_train(
    model_cnn, 
    X_train, y_train, 
    X_val, y_val,
    num_classes=NUM_CLASSES, 
    epochs=20,
    batch_size=32,
    class_weight=class_weights
)

#Evaluate CNN model on test set
y_pred_probs_cnn = model_cnn.predict(X_test_pad)
y_pred_binary = (y_pred_probs_cnn > 0.2).astype(int)

test_loss_cnn, test_acc_cnn  = model_cnn.evaluate(X_test_pad, y_test, verbose=0)
print(f"\nTest Loss: {test_loss_cnn:.4f}")
print(f"Test Accuracy: {test_acc_cnn:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred_binary, target_names=sentiment_classes, zero_division=0))


#Create and train BiLSTM.1 model
model_bilstm_1 = create_bilstm_model_1(MAX_WORDS, MAX_LEN, EMBEDDING_DIM, NUM_CLASSES)
history_bilstm_1 = compile_and_train(
    model_bilstm_1, 
    X_train, y_train, 
    X_val, y_val,
    num_classes=NUM_CLASSES, 
    epochs=20,
    batch_size=32,
    #class_weight=class_weights

)


#Evaluate BiLSTM.1 model on test set
y_pred_probs_bilstm_1 = model_bilstm_1.predict(X_test_pad)
y_pred_bilstm_1 = (y_pred_probs_bilstm_1 > 0.3).astype(int)
test_loss_bilstm_1, test_acc_bilstm_1  = model_bilstm_1.evaluate(X_test_pad, y_test, verbose=0)

print(f"\nTest Loss: {test_loss_bilstm_1:.4f}")
print(f"Test Accuracy: {test_acc_bilstm_1:4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_bilstm_1, target_names=sentiment_classes, zero_division=0))


#Create and train BiLSTM.2 model
model_bilstm_2 = create_bilstm_model_2(MAX_WORDS, MAX_LEN, EMBEDDING_DIM, NUM_CLASSES)
history_bilstm_2 = compile_and_train(
    model_bilstm_2, 
    X_train, y_train, 
    X_val, y_val,
    num_classes=NUM_CLASSES, 
    epochs=20,
    batch_size=32,
    class_weight=class_weights

)

#Evaluate BiLSTM.2 model on test set
y_pred_probs_bilstm_2 = model_bilstm_2.predict(X_test_pad)
y_pred_bilstm_2 = (y_pred_probs_bilstm_2 > 0.3).astype(int)
test_loss_bilstm_2, test_acc_bilstm_2  = model_bilstm_2.evaluate(X_test_pad, y_test, verbose=0)


# Classification report
print(f"\nTest Loss: {test_loss_bilstm_2:.4f}")
print(f"Test Accuracy: {test_acc_bilstm_2:4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_bilstm_2, target_names=sentiment_classes, zero_division=0))






