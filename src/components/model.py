import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from src.components.evaluate import plot_evaluation_metrics

def models(df_titanic):

    X = df_titanic.drop(columns=['Survived', 'PassengerId'])
    y = df_titanic['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "RandomForest": RandomForestClassifier(),
        "LogisticRegression": LogisticRegression(solver='liblinear'),
        "GradientBoosting": GradientBoostingClassifier(),
        "SVM": SVC(probability=True),
        "NaiveBayes": GaussianNB()
    }

    params = {
        "RandomForest": {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]},
        "LogisticRegression": {'C': [0.01, 0.1, 1, 10, 100]},
        "GradientBoosting": {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1]},
        "SVM": {'C': [0.01, 0.1, 1, 10, 100]},
        "NaiveBayes": {}
    }

    best_model = None
    best_f1 = -np.inf

    for model_name, model in models.items():
        print(f"Evaluating {model_name}...")
        clf = GridSearchCV(model, params[model_name], cv=5, scoring='accuracy')
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        f1 = plot_evaluation_metrics(y_test, y_pred, clf, X_test, model_name )

        if f1 > best_f1:
            best_f1 = f1
            best_model = clf

    print(f"Best Model: {best_model.best_estimator_}")
    print(f"Best F1-Score: {best_f1}")
