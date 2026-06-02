import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Column names for NSL-KDD dataset
columns = [
    'duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent',
    'hot','num_failed_logins','logged_in','num_compromised','root_shell','su_attempted','num_root',
    'num_file_creations','num_shells','num_access_files','num_outbound_cmds','is_host_login',
    'is_guest_login','count','srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate',
    'same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count',
    'dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate',
    'dst_host_rerror_rate','dst_host_srv_rerror_rate','label','difficulty'
]

DATASET_PATH = 'dataset/KDDTrain+.txt'
MODEL_PATH = 'models/intrusion_model.pkl'
ENCODER_PATH = 'models/label_encoders.pkl'
TARGET_ENCODER_PATH = 'models/target_encoder.pkl'


def load_dataset():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            "Dataset not found. Please put KDDTrain+.txt inside the dataset folder."
        )
    data = pd.read_csv(DATASET_PATH, names=columns)
    return data


def preprocess_data(data):
    # Convert labels into two classes: normal or attack
    data['label'] = data['label'].apply(lambda x: 'normal' if x == 'normal' else 'attack')

    X = data.drop(['label', 'difficulty'], axis=1)
    y = data['label']

    label_encoders = {}
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le

    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(y)

    return X, y, label_encoders, target_encoder


def train_model():
    print('Loading dataset...')
    data = load_dataset()

    print('Preprocessing data...')
    X, y, label_encoders, target_encoder = preprocess_data(data)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print('Training Random Forest model...')
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print('\nModel Accuracy:', round(accuracy * 100, 2), '%')
    print('\nClassification Report:\n')
    print(classification_report(y_test, y_pred, target_names=target_encoder.classes_))

    os.makedirs('models', exist_ok=True)
    os.makedirs('images', exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(label_encoders, ENCODER_PATH)
    joblib.dump(target_encoder, TARGET_ENCODER_PATH)

    # Accuracy chart
    plt.figure(figsize=(6, 4))
    plt.bar(['Accuracy'], [accuracy * 100])
    plt.ylim(0, 100)
    plt.ylabel('Percentage')
    plt.title('Model Accuracy')
    plt.savefig('images/accuracy_chart.png')
    plt.close()

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=target_encoder.classes_, yticklabels=target_encoder.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('images/confusion_matrix.png')
    plt.close()

    print('\nModel saved successfully in models folder.')
    print('Graphs saved successfully in images folder.')


if __name__ == '__main__':
    train_model()
