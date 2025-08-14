# Decoding-RNN 🎯

A beginner-friendly walkthrough of **Recurrent Neural Networks (RNNs)** for text-based tasks — demonstrated with sentiment analysis on the IMDb movie reviews dataset.

---

## 📌 What Problem Do RNNs Solve?
Traditional neural networks (like feed-forward networks) struggle with **sequential data** because they treat every input independently.  
RNNs solve this by introducing a **memory mechanism**, allowing the network to **retain context** from earlier parts of the sequence when processing later parts.

Example:
- "The movie was great" → positive sentiment
- "The movie was not great" → negative sentiment  
Here, the word "not" changes the meaning — RNNs can capture that dependency.

---

## 📖 Encodings, Embeddings, and RNNs

### **1. Encodings**
Before feeding text to a model, we convert words into numbers.  
A common way is **integer encoding**:
"I love movies" → [12, 87, 45]


### **2. Embeddings**
Instead of treating words as arbitrary integers, embeddings map them to **dense vectors** in a continuous space, capturing semantic relationships:
"good" and "great" → vectors close together in space
"good" and "bad" → vectors far apart


In Keras, `Embedding(input_dim, output_dim)` handles this automatically.

### **3. How RNN Works**
RNN processes sequences **step-by-step**:
1. Takes the first word vector, updates its hidden state.
2. Passes this state to the next timestep along with the next word vector.
3. Repeats until the entire sequence is processed.
4. Final hidden state contains context for the whole sequence (used for classification).

---

## 🔍 RNN Processing Diagram

Here’s a visual of how information flows in an RNN:

x1 ---> [ RNN Cell ] ---> h1
↑
|
x2 ---> [ RNN Cell ] ---> h2
↑
|
x3 ---> [ RNN Cell ] ---> h3 ---> Output


Where:
- **x1, x2, x3** → word embeddings at different timesteps.
- **h1, h2, h3** → hidden states carrying information forward.

---

## 📂 Project Overview
In this repo, we:
- Load the **IMDb dataset** from `keras.datasets`.
- Preprocess the reviews into integer sequences.
- Use an **Embedding layer** to learn word representations.
- Pass embeddings into a **SimpleRNN** for sequence modeling.
- Output sentiment: **positive (1)** or **negative (0)**.

---

## 🚀 Quick Start

```python
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load IMDb dataset
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)

# Pad sequences to fixed length
x_train = pad_sequences(x_train, maxlen=50)
x_test = pad_sequences(x_test, maxlen=50)

# Build RNN model
model = Sequential([
    Embedding(input_dim=10000, output_dim=2, input_length=50),
    SimpleRNN(32, return_sequences=False),
    Dense(1, activation='sigmoid')
])

# Compile & train
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))