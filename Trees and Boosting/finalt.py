import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from DecesionTree import DecisionTree
from RandomForest import RandomForest
from Boosting import AdaBoost

# File paths
training_data_path = "C:/Users/kaish/Downloads/p/train.csv"
testing_data_path = "C:/Users/kaish/Downloads/p/test.csv"

# Load and display initial data information
df = pd.read_csv(training_data_path)
print(df.head())
print(df.describe())
print(df.info())

# Calculate and display missing data information
missing_count = df.isnull().sum()
missing_percentage = (missing_count / len(df)) * 100
missing_data = pd.DataFrame({
    'Missing Values Count': missing_count,
    'Missing Values Percentage': missing_percentage
})
print(missing_data)

# Define relevant features and target variable
relevant_features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
target_variable = "Survived"

def acquire_data(train_path, test_path):
    """Load training and testing data from CSV files."""
    raw_train_data = pd.read_csv(train_path)
    raw_test_data = pd.read_csv(test_path)
    return raw_train_data, raw_test_data

def clean_and_transform(dataframe, is_training_set=True):
    """Clean and preprocess the dataset."""
    cleaned_df = dataframe.copy()
    cleaned_df["Sex"] = cleaned_df["Sex"].map({"male": 0, "female": 1})
    cleaned_df["Embarked"] = cleaned_df["Embarked"].map({"C": 0, "Q": 1, "S": 2})
    cleaned_df["Age"].fillna(cleaned_df["Age"].median(), inplace=True)
    cleaned_df["Fare"].fillna(cleaned_df["Fare"].median(), inplace=True)
    cleaned_df["Embarked"].fillna(2, inplace=True)
    if is_training_set:
        cleaned_df["Survived"] = cleaned_df["Survived"].astype(int)
    return cleaned_df.drop(columns=["PassengerId"], errors="ignore") if not is_training_set else cleaned_df

def exclude_anomalies(dataframe, columns_to_check, z_score_threshold=3):
    """Remove outliers based on Z-score."""
    cleaned_df = dataframe.copy()
    for column in columns_to_check:
        z_scores = np.abs((cleaned_df[column] - cleaned_df[column].mean()) / cleaned_df[column].std())
        cleaned_df = cleaned_df[z_scores < z_score_threshold]
    return cleaned_df

def partition_dataset(dataframe, feature_columns, target_column, test_size=0.2):
    """Split the dataset into training and validation sets."""
    np.random.seed(42)
    mask = np.random.rand(len(dataframe)) < (1 - test_size)
    train = dataframe[mask]
    test = dataframe[~mask]
    train_features = train[feature_columns].values
    train_targets = train[target_column].values
    test_features = test[feature_columns].values
    test_targets = test[target_column].values
    return train_features, test_features, train_targets, test_targets

def compute_sample_weight(y):
    """Compute balanced sample weights."""
    class_weights = 1. / np.bincount(y)
    return class_weights[y]

def build_model(model_class, feature_matrix, target_vector, **model_params):
    """Initialize and train a machine learning model."""
    ml_model = model_class(**model_params)
    sample_importance = compute_sample_weight(target_vector)
    if isinstance(ml_model, AdaBoost):
        ml_model.fit(feature_matrix, target_vector, sample_weight=sample_importance)
    else:
        ml_model.fit(feature_matrix, target_vector)
    return ml_model

def compute_confusion_matrix(true_labels, predictions):
    """Compute the confusion matrix."""
    classes = np.unique(true_labels)
    matrix = np.zeros((len(classes), len(classes)), dtype=int)
    for i, true_class in enumerate(classes):
        for j, pred_class in enumerate(classes):
            matrix[i, j] = np.sum((true_labels == true_class) & (predictions == pred_class))
    return matrix

def compute_classification_report(true_labels, predictions):
    """Compute precision, recall, F1-score, and support for each class."""
    classes = np.unique(true_labels)
    report = {}
    for class_label in classes:
        true_positives = np.sum((true_labels == class_label) & (predictions == class_label))
        false_positives = np.sum((true_labels != class_label) & (predictions == class_label))
        false_negatives = np.sum((true_labels == class_label) & (predictions != class_label))
        true_negatives = np.sum((true_labels != class_label) & (predictions != class_label))
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        support = np.sum(true_labels == class_label)  # Number of true instances for each class
        
        report[class_label] = {
            'precision': precision,
            'recall': recall,
            'f1-score': f1_score,
            'support': support
        }
    return report

def assess_model(ml_model, feature_matrix, true_labels):
    """Evaluate the model's performance."""
    predictions = ml_model.predict(feature_matrix)
    accuracy = np.mean(predictions == true_labels)
    print(f"{ml_model.__class__.__name__} Accuracy: {accuracy:.4f}")
    
    # Compute and display confusion matrix
    conf_matrix = compute_confusion_matrix(true_labels, predictions)
    print("\nConfusion Matrix:")
    print(conf_matrix)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {ml_model.__class__.__name__}')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    
    # Compute and display classification report
    class_report = compute_classification_report(true_labels, predictions)
    print("\nClassification Report:")
    for class_label, metrics in class_report.items():
        print(f"Class {class_label}:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")
        print()

if __name__ == "__main__":
    # Data acquisition and preprocessing
    raw_train, raw_test = acquire_data(training_data_path, testing_data_path)
    processed_train = clean_and_transform(raw_train, is_training_set=True)
    processed_test = clean_and_transform(raw_test, is_training_set=False)
    
    # Anomaly removal
    cleaned_train = exclude_anomalies(processed_train, ["Age", "Fare"])
    
    # Dataset partitioning
    train_features, val_features, train_targets, val_targets = partition_dataset(cleaned_train, relevant_features, target_variable)
    
    # Model training with optimized parameters
    model_collection = {
        "Decision Tree": build_model(DecisionTree, train_features, train_targets, max_depth=10),
        "Random Forest": build_model(RandomForest, train_features, train_targets, num_trees=100, min_features=2),
        "AdaBoost": build_model(AdaBoost, train_features, train_targets, num_learners=100, learning_rate=0.01, weak_learner=DecisionTree)
    }
    
    # Model evaluation
    for model_name, trained_model in model_collection.items():
        print(f"\nEvaluating {model_name}:")
        assess_model(trained_model, val_features, val_targets)
