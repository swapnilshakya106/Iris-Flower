Iris classifier · PY
"""
Iris Flower Classifier (Enhanced)
-----------------------------------
A beginner-to-intermediate Machine Learning project that trains a model to
recognize the species of an Iris flower based on its measurements.
 
New features added in this version:
1. Choice of multiple ML models (KNN, Logistic Regression, Decision Tree,
   Random Forest, SVM) instead of just one.
2. Cross-validation to get a more reliable accuracy estimate.
3. A confusion matrix and a full classification report (precision/recall/F1).
4. Data visualization: a scatter plot of the features, and a heatmap of the
   confusion matrix (saved as PNG files).
5. Prediction confidence: shows the probability for each species, not just
   the single predicted label.
6. Feature importance chart for tree-based models.
7. Save/load the trained model to disk with joblib, so you don't have to
   retrain every time.
8. A simple command-line menu so you can explore all of the above.
 
Concepts used:
- Loading a dataset
- Train/test split
- Training and comparing classification models
- Cross-validation
- Confusion matrix & classification report
- Data visualization
- Model persistence (saving/loading)
 
Libraries needed:
    pip install scikit-learn matplotlib seaborn joblib
"""
 
import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)
 
MODEL_FILE = "iris_model.joblib"
 
# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------
 
def load_data():
    """
    Loads the Iris dataset.
    - data.data   -> the measurements (features): sepal length, sepal width,
                      petal length, petal width
    - data.target -> the species (labels): 0 = setosa, 1 = versicolor, 2 = virginica
    """
    return load_iris()
 
 
# ---------------------------------------------------------------------------
# 2. Model zoo — pick any classifier by name
# ---------------------------------------------------------------------------
 
def get_available_models():
    """
    Returns a dictionary of model name -> untrained model instance.
    Feel free to tweak the hyperparameters here.
    """
    return {
        "1": ("K-Nearest Neighbors", KNeighborsClassifier(n_neighbors=3)),
        "2": ("Logistic Regression", LogisticRegression(max_iter=200)),
        "3": ("Decision Tree", DecisionTreeClassifier(random_state=42)),
        "4": ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42)),
        "5": ("Support Vector Machine", SVC(probability=True)),
    }
 
 
def choose_model():
    """Lets the user pick which model to train from a menu."""
    models = get_available_models()
    print("\nChoose a model to train:")
    for key, (name, _) in models.items():
        print(f"  {key}. {name}")
    choice = input("Enter choice (default 1): ").strip() or "1"
    name, model = models.get(choice, models["1"])
    print(f"-> Using {name}")
    return name, model
 
 
# ---------------------------------------------------------------------------
# 3. Training, cross-validation, evaluation
# ---------------------------------------------------------------------------
 
def train_model(model, X_train, y_train):
    """Trains the given (already-instantiated) model."""
    model.fit(X_train, y_train)
    return model
 
 
def cross_validate_model(model, X, y, folds=5):
    """
    Runs k-fold cross-validation on the FULL dataset to get a more robust
    sense of how well the model generalizes (rather than relying on a
    single train/test split).
    """
    scores = cross_val_score(model, X, y, cv=folds)
    print(f"\nCross-validation scores ({folds}-fold): {np.round(scores, 3)}")
    print(f"Average CV accuracy: {scores.mean() * 100:.2f}% (+/- {scores.std() * 100:.2f}%)")
    return scores
 
 
def evaluate_model(model, X_test, y_test, target_names):
    """
    Predicts on the test set and prints:
    - overall accuracy
    - a confusion matrix
    - a full classification report (precision, recall, F1-score per class)
    """
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"\nModel accuracy on test data: {accuracy * 100:.2f}%")
 
    cm = confusion_matrix(y_test, predictions)
    print("\nConfusion matrix:")
    print(cm)
 
    print("\nClassification report:")
    print(classification_report(y_test, predictions, target_names=target_names))
 
    plot_confusion_matrix(cm, target_names)
    return accuracy, predictions
 
 
# ---------------------------------------------------------------------------
# 4. Visualization
# ---------------------------------------------------------------------------
 
def plot_confusion_matrix(cm, target_names, filename="confusion_matrix.png"):
    """Saves a heatmap image of the confusion matrix."""
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=target_names, yticklabels=target_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved confusion matrix plot to '{filename}'")
 
 
def plot_feature_scatter(data, filename="feature_scatter.png"):
    """
    Saves a scatter plot of petal length vs petal width, colored by species.
    This is usually the most visually distinctive pair of features in the
    Iris dataset.
    """
    X = data.data
    y = data.target
    plt.figure(figsize=(6, 5))
    for i, name in enumerate(data.target_names):
        plt.scatter(X[y == i, 2], X[y == i, 3], label=name)
    plt.xlabel("Petal length (cm)")
    plt.ylabel("Petal width (cm)")
    plt.title("Iris Species by Petal Measurements")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved feature scatter plot to '{filename}'")
 
 
def plot_feature_importance(model, feature_names, model_name, filename="feature_importance.png"):
    """
    If the model exposes `feature_importances_` (tree-based models), plots
    which measurements mattered most for its decisions.
    """
    if not hasattr(model, "feature_importances_"):
        print(f"({model_name} does not support feature importance — skipping chart)")
        return
    importances = model.feature_importances_
    plt.figure(figsize=(6, 4))
    plt.barh(feature_names, importances, color="teal")
    plt.xlabel("Importance")
    plt.title(f"Feature Importance ({model_name})")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved feature importance plot to '{filename}'")
 
 
# ---------------------------------------------------------------------------
# 5. Saving / loading the trained model
# ---------------------------------------------------------------------------
 
def save_model(model, path=MODEL_FILE):
    joblib.dump(model, path)
    print(f"Model saved to '{path}'")
 
 
def load_model(path=MODEL_FILE):
    if not os.path.exists(path):
        print(f"No saved model found at '{path}'.")
        return None
    print(f"Loaded model from '{path}'")
    return joblib.load(path)
 
 
# ---------------------------------------------------------------------------
# 6. Predicting new flowers (now with probabilities)
# ---------------------------------------------------------------------------
 
def predict_new_flower(model, target_names):
    """
    Asks the user to enter measurements for a new flower, then uses the
    trained model to predict its species AND shows the model's confidence
    for each possible species (if the model supports probabilities).
    """
    print("\nEnter the flower's measurements (in cm):")
    sepal_length = float(input("Sepal length: "))
    sepal_width = float(input("Sepal width: "))
    petal_length = float(input("Petal length: "))
    petal_width = float(input("Petal width: "))
 
    new_flower = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = model.predict(new_flower)
    predicted_species = target_names[prediction[0]]
    print(f"\nPredicted species: {predicted_species}")
 
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(new_flower)[0]
        print("Confidence per species:")
        for name, prob in zip(target_names, probabilities):
            print(f"  {name:12s}: {prob * 100:5.1f}%")
 
 
# ---------------------------------------------------------------------------
# 7. Main program with a simple menu
# ---------------------------------------------------------------------------
 
def main():
    print("=== Iris Flower Classifier (Enhanced) ===")
 
    data = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
 
    model = None
 
    while True:
        print("\n--- Menu ---")
        print("1. Train a new model")
        print("2. Evaluate current model (accuracy, confusion matrix, report)")
        print("3. Run cross-validation on current model")
        print("4. Visualize the dataset (feature scatter plot)")
        print("5. Show feature importance (tree-based models only)")
        print("6. Predict a new flower")
        print("7. Save current model")
        print("8. Load a previously saved model")
        print("9. Quit")
        choice = input("Choose an option: ").strip()
 
        if choice == "1":
            name, chosen = choose_model()
            model = train_model(chosen, X_train, y_train)
            model.name_ = name  # tag it so we can print it later
        elif choice == "2":
            if model is None:
                print("Train or load a model first (option 1 or 8).")
                continue
            evaluate_model(model, X_test, y_test, data.target_names)
        elif choice == "3":
            if model is None:
                print("Train or load a model first (option 1 or 8).")
                continue
            cross_validate_model(model, data.data, data.target)
        elif choice == "4":
            plot_feature_scatter(data)
        elif choice == "5":
            if model is None:
                print("Train or load a model first (option 1 or 8).")
                continue
            plot_feature_importance(model, data.feature_names, getattr(model, "name_", "model"))
        elif choice == "6":
            if model is None:
                print("Train or load a model first (option 1 or 8).")
                continue
            predict_new_flower(model, data.target_names)
        elif choice == "7":
            if model is None:
                print("Nothing to save — train a model first.")
                continue
            save_model(model)
        elif choice == "8":
            loaded = load_model()
            if loaded is not None:
                model = loaded
        elif choice == "9":
            print("Thanks for using the Iris Flower Classifier!")
            break
        else:
            print("Invalid choice, please try again.")
 
 
if __name__ == "__main__":
    main()
