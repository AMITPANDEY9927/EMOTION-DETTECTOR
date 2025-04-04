**EMOTION-DETECTOR**  

**Overview:**  
This project aims to identify the emotion of a person based on the given text input.  

**Dataset:**  
- Contains text labeled with emotions like anger, love, hate, worry, neutral, etc.  
- File used: `emotion_sentimen_dataset.csv`  
- Source: [Kaggle Dataset](https://www.kaggle.com/datasets/simaanjali/emotion-analysis-based-on-text)  

**Steps:**  

**1. Data Preprocessing**  
- Removed user mentions (@username), URLs, numbers, and punctuation.  
- Converted text to lowercase.  
- Balanced the dataset by selecting equal samples for each emotion.  

**2. Feature Extraction**  
- Used **TF-IDF Vectorization** with a max of 2500 features.  

**3. Model Training**  
- Algorithm: **Random Forest Classifier**  
- Train-Test Split: **80% training, 20% testing**  
- Accuracy: **~0.89 - 0.91**  

**4. Prediction**  
- Users can enter a sentence, and the model predicts the emotion.  

**Running the Project:**  

**Install Dependencies:**  
pip install pandas numpy scikit-learn matplotlib seaborn neattext joblib

Run the Script:
python EmotionDetector.py

