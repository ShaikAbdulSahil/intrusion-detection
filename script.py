import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import pickle
import os

# ---------------------------
# Define reusable functions
# ---------------------------

# Initializes label encoders for key categorical features
label_encoders = {
    'protocol_type': LabelEncoder(),
    'service': LabelEncoder(),
    'flag': LabelEncoder()
}

def preprocess_dataset(df):
    # Drop unnecessary columns
    columns_to_drop = ['unnamed: 0']
    # Drop columns that start with 'l' except 'label'
    columns_to_drop.extend([col for col in df.columns if col.startswith('l') and col != 'label'])
    for col in columns_to_drop:
        if col in df.columns:
            df = df.drop(columns=[col])
    return df

def load_and_preprocess_data(file_path):
    try:
        # Ignore lines that start with "/" (e.g., the file path comment)
        df = pd.read_csv(file_path, comment='/')
        # Remove extra whitespace and convert column names to lowercase
        df.columns = df.columns.str.strip().str.lower()
        df = preprocess_dataset(df)
        print("Columns loaded:", list(df.columns))
        return df
    except FileNotFoundError:
        print(f"Error: File {file_path} not found!")
        return None
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

# ---------------------------
# TRAINING PHASE
# ---------------------------
# Load training dataset from kddcup99_csv.csv
df = load_and_preprocess_data('kddcup99_csv.csv')
if df is None:
    exit(1)

# Explore the dataset
print("\nDataset shape:", df.shape)
print("\nFirst 5 records:")
print(df.head())
print("\nColumns:")
print(list(df.columns))
print("\nDataset info:")
print(df.info())
print("\nMissing values per feature:")
print(df.isna().sum())

# Print category counts for object (categorical) features
print('\nData set categories:')
for col_name in df.columns:
    if df[col_name].dtype == 'object':
        unique_cat = len(df[col_name].unique())
        print("Feature '{}' has {} categories".format(col_name, unique_cat))
print()
if 'service' in df.columns:
    print("Distribution of categories in service:")
    print(df['service'].value_counts().sort_values(ascending=False).head())

# Encode categorical features using LabelEncoder
for col in ['protocol_type', 'service', 'flag']:
    if col in df.columns:
        df[col] = label_encoders[col].fit_transform(df[col])

# Map the label column into numerical values.
# For example, let 0 = 'normal' and (1 or 2) = attack types.
if 'label' in df.columns:
    print("\nOriginal label distribution:")
    print(df['label'].value_counts())
    mapping = {
        'normal': 0,
        'smurf': 1, 'neptune': 1, 'back': 1, 'teardrop': 1, 'pod': 1, 'land': 1,
        'satan': 2, 'ipsweep': 2, 'portsweep': 2, 'warezclient': 2, 'nmap': 2,
        'guess_passwd': 2, 'buffer_overflow': 2, 'warezmaster': 2, 'imap': 2,
        'rootkit': 2, 'loadmodule': 2, 'ftp_write': 2, 'multihop': 2, 'phf': 2,
        'perl': 2, 'spy': 2
    }
    df['label'] = df['label'].replace(mapping)
    print("\nMapped label distribution:")
    print(df['label'].value_counts())
else:
    print("Error: No 'label' column found in training data.")
    exit(1)

# Save the processed training data for later reference
df.to_csv('New_Data.csv', index=False)
print("\nProcessed training data saved as 'New_Data.csv'.")

# Plot label distribution and save heatmap
plt.figure(figsize=(6,4))
sns.countplot(x='label', data=df)
plt.title("Label Distribution in Training Set")
plt.savefig('training_label_distribution.png')
plt.show()

# Prepare features and labels
X = df.drop(columns=['label'])
y = df['label']

# Split data into training and test sets
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.33, random_state=42)

# ---------------------------
# Model Training and Evaluation
# ---------------------------
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": svm.LinearSVC(random_state=20),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(random_state=42)
}

accuracy_scores = {}

for model_name, model in models.items():
    try:
        model.fit(xtrain, ytrain)
        predictions = model.predict(xtest)
        acc = accuracy_score(ytest, predictions)
        accuracy_scores[model_name] = acc
        print(f"\n{model_name} trained successfully. Accuracy: {acc:.5f}")
        
        # Show confusion matrix with heatmap, save figure
        cnf = confusion_matrix(ytest, predictions, normalize='true')
        print(f"{model_name} Confusion Matrix:\n", cnf)
        labels_list = ['0', '1', '2']
        plt.figure(figsize=(6,4))
        sns.heatmap(cnf, annot=True, cmap="YlGnBu", fmt=".3f", xticklabels=labels_list, yticklabels=labels_list)
        plt.title(f"{model_name} Confusion Matrix")
        plt.savefig(f'{model_name.lower().replace(" ", "_")}_confusion_matrix.png')
        plt.show()
        
    except Exception as e:
        print(f"Error training {model_name}: {str(e)}")

# Save the label encoders for future use
try:
    with open('label_encoders.pkl', 'wb') as f:
        pickle.dump(label_encoders, f)
except Exception as e:
    print(f"Error saving label encoders: {str(e)}")

# Plot a comparison bar chart for all models and save it
objects = list(accuracy_scores.keys())
performance = [accuracy_scores[model] for model in objects]
y_pos = np.arange(len(objects))

fig, ax = plt.subplots()
ax.bar(y_pos, performance, align='center', alpha=0.8)
ax.set_ylabel('Accuracy Score')
ax.set_xticks(y_pos)
ax.set_xticklabels(objects)
for i, v in enumerate(performance):
    ax.text(i, v + 0.01, "{:.5f}".format(v), ha='center', va='bottom')
plt.title("Model Accuracy Comparison")
plt.savefig('algorithm_comparison.png')
plt.show()

# ---------------------------
# Intrusion Detection (Prediction) Phase
# ---------------------------
def detect_intrusion(new_data_file):
    print("\n🔍 Detecting Intrusions in New Data...")
    new_data = load_and_preprocess_data(new_data_file)
    if new_data is None:
        return

    try:
        # Ensure all training columns are present in new data; add with default of 0 if missing
        required_columns = X.columns.tolist()
        missing_columns = set(required_columns) - set(new_data.columns)
        if missing_columns:
            print(f"Warning: Missing required columns: {missing_columns}. Adding them with default value 0.")
            for col in missing_columns:
                new_data[col] = 0

        # Transform categorical features using saved label encoders
        for col in label_encoders.keys():
            if col in new_data.columns:
                new_data[col] = new_data[col].map(
                    dict(zip(label_encoders[col].classes_, 
                             label_encoders[col].transform(label_encoders[col].classes_)))
                ).fillna(-1)
        new_data = new_data[required_columns]

        # Predict using the best model, here we use Random Forest as an example
        best_model = models["Random Forest"]
        predictions = best_model.predict(new_data)
        new_data['prediction'] = predictions
        new_data['prediction'] = new_data['prediction'].map({0: 'Safe', 1: 'Attack', 2: 'Attack'})
        new_data.to_csv('detection_results.csv', index=False)
        print("\n✅ Detection Complete! Results saved in 'detection_results.csv'")
    
    except Exception as e:
        print(f"Error during intrusion detection: {str(e)}")

# Run our detection on the prediction dataset
detect_intrusion('inputdata.csv')
