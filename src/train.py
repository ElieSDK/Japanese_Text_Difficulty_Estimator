import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

def train_model(X, y):
    """
    Train a logistic regression model on the feature matrix X and target y,
    including data splitting, scaling, training, evaluation, and model saving.
    """
    # Split data into training and test sets (80% train, 20% test),
    # stratified to keep class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Define a pipeline with a scaler and logistic regression model
    pipeline = Pipeline([
        ('scaler', StandardScaler(with_mean=False)),  # Scale features without centering (sparse input)
        ('logreg', LogisticRegression(
            max_iter=20000,         # Increase max iterations for convergence
            solver='saga',          # Solver that supports sparse data and l1/l2 penalties
            class_weight='balanced',# Handle imbalanced classes automatically
            random_state=42
        ))
    ])

    # Train the pipeline on the training data
    pipeline.fit(X_train, y_train)

    # Predict on the test data
    y_pred = pipeline.predict(X_test)

    # Print evaluation metrics
    print("Logistic Regression Results")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Save the trained pipeline to a file for later use
    with open("logreg_pipeline.pkl", "wb") as f:
        pickle.dump(pipeline, f)
