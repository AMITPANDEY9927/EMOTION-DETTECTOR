# EMOTION-DETTECTOR

**Overview:**
the project aimed to identify a emotion of a person based on the text input given to it.

**Dataset:**
Contains text labeled with emotions like anger, love, hate, worry, neutral, etc.
File used: emotion_sentimen_dataset.csv
url: https://www.kaggle.com/datasets/simaanjali/emotion-analysis-based-on-text

**Steps:**
1️. Data Preprocessing
Removed user mentions (@username), URLs, numbers, punctuation
Converted text to lowercase
Balanced dataset by selecting equal samples for each emotion

2️. Feature Extraction
Used TF-IDF Vectorization (max 2500 features)

3️. Model Training
Algorithm: Random Forest Classifier
Train-Test Split: 80% training, 20% testing
Accuracy: ~0.96

4️. Prediction
Users can enter text to get an emotion prediction

Running the Project
Install dependencies:

nginx
pip install pandas numpy scikit-learn matplotlib seaborn
Run the script:
python EmotionDetector.py
