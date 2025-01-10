import os
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Function to load text files from folder
def load_text_files(folder_path, encoding='utf-8'):
    texts = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r', encoding=encoding, errors='ignore') as file:
            texts.append(file.read())
    return texts

# Define folder paths and their corresponding labels
folder_paths = ['D:\image_to_text - Copy/email', 'D:\image_to_text - Copy/questionnaire']
labels = [0, 1]  # Assign unique labels to each folder

# Load text data and corresponding labels
X = []
y = []
for folder_path, label in zip(folder_paths, labels):
    texts = load_text_files(folder_path)
    X.extend(texts)
    y.extend([label] * len(texts))

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define pipeline with LDA and Random Forest classifier
pipeline = Pipeline([
    ('vectorizer', CountVectorizer(max_df=0.95, min_df=2, stop_words='english')),
    ('lda', LatentDirichletAllocation(n_components=10, random_state=42)),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train pipeline
pipeline.fit(X_train, y_train)

#Save the trained model
joblib.dump(pipeline, 'text_classification_model.joblib')

# Load the saved model
loaded_model = joblib.load('text_classification_model.joblib')

# Function to classify new text file
def classify_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        text = file.read()
    predicted_label = loaded_model.predict([text])[0]
    return predicted_label

# Example usage: Classify a new text file
new_file_path = 'D:/image_to_text - Copy/email/All-About-Me-Questions-Printable.txt'
predicted_folder = classify_text_file(new_file_path)
print("Predicted folder:", predicted_folder)