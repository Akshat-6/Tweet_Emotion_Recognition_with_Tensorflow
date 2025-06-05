# Tweet_Emotion_Recognition_with_Tensorflow
Emotion classification of tweets using a deep learning model built with TensorFlow and NLP preprocessing techniques.


💬 Tweet Emotion Recognition using TensorFlow
This project demonstrates emotion recognition from tweets using deep learning techniques in TensorFlow. It involves natural language preprocessing, building a text classification model, and predicting one of several emotional classes based on tweet content.

🎯 Objective
To develop a TensorFlow-based model that can:

Preprocess tweet text (cleaning, tokenizing, padding)

Classify tweets into emotions like:

Happy, Angry, Sad, Fear, Love, Surprise, etc.

📁 Dataset
Source: Tweet Emotion Dataset - NLP

Contains:

tweet_text — raw text of tweets

emotion — labeled emotion category for each tweet

🧠 Workflow
1. Data Preprocessing
Removed special characters, mentions, hashtags

Tokenized and padded tweet sequences

Encoded emotion labels numerically

2. Model Building (TensorFlow/Keras)
Used Embedding layer for word vectorization

LSTM / Bidirectional LSTM for sequence modeling

Dense + Softmax output layer for multi-class classification

3. Training
Optimizer: Adam

Loss: Categorical Crossentropy

Metrics: Accuracy, Precision, Recall

4. Evaluation
Confusion matrix

Classification report

Accuracy and loss curves

📈 Results
Achieved high validation accuracy (>85%)

Accurately classified multiple emotions

Model generalizes well on unseen tweet samples

🔧 Tech Stack
Python

TensorFlow / Keras

NLTK / SpaCy (for preprocessing, optional)

Scikit-learn (for evaluation)

Matplotlib / Seaborn (for visualizations)

💡 Future Work
Use pre-trained embeddings (GloVe, Word2Vec)

Deploy model with Flask or Streamlit

Try transformer-based models like BERT

🧪 Evaluation Metrics
Accuracy

Precision / Recall

F1-score

Confusion Matrix

