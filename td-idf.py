import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

data_dir = Path(__file__).parent # we can change this later if we want to move the data into its own folder
documents = []
ratings = []
   
for file_path in sorted(data_dir.glob("Health_and_Personal_Care.part-*.jsonl")):
    with file_path.open(encoding="utf-8") as file:
        for line in file:
            review = json.loads(line)
            rating = int(float(review["rating"]))
            
            if rating == 3:
                continue

            title = review.get("title") or ""
            text = review.get("text") or ""
            document = f"{title.strip()} {text.strip()}".strip() # combining title and text for td-idf vectorization

            if document:
                documents.append(document) # combined txt
                ratings.append(0 if rating <= 2 else 1) # convert rating to class

X_train, X_test, y_train, y_test = train_test_split(
    documents,
    ratings,
    test_size=0.2,
    random_state=42,
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

baseline_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", sublinear_tf=True)),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))
])

baseline_scores = cross_val_score(baseline_pipeline, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1)

print(f"Baseline Cross-Val F1 Score (Default Params): {np.mean(baseline_scores):.4f}")

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=0))
])

param_distributions = [
    {
        "tfidf__max_features": [10000, 25000, 50000],
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__sublinear_tf": [True, False],
        "clf__penalty": ["l2"],
        "clf__C": [0.01, 0.1, 1.0, 10.0, 100.0],
        "clf__solver": ["lbfgs"]
    },
    {
        "tfidf__max_features": [10000, 25000, 50000],
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__sublinear_tf": [True, False],
        "clf__penalty": ["l1"],
        "clf__C": [0.01, 0.1, 1.0, 10.0, 100.0],
        "clf__solver": ["liblinear"]
    }
]

search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_distributions,
    n_iter=15,
    cv=cv,
    scoring="f1_macro",
    random_state=0,
    n_jobs=-1
)
search.fit(X_train, y_train)

print("\nBest Parameters:", search.best_params_)
print("Best Estimator:", search.best_estimator_)
print("Mean CV Scores across sampled combinations:\n", np.round(search.cv_results_["mean_test_score"], 4))

best_model = search.best_estimator_
y_pred = best_model.predict(X_test)

target_names = ["Negative (1-2)", "Positive (4-5)"]
print("\n--- Test Set Classification Report ---")
print(classification_report(y_test, y_pred, target_names=target_names))

print("--- Test Set Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred))

best_vectorizer = best_model.named_steps["tfidf"]
best_clf = best_model.named_steps["clf"]

feature_names = np.array(best_vectorizer.get_feature_names_out())
coefs = best_clf.coef_

top_neg_indices = np.argsort(coefs)[:15]
top_pos_indices = np.argsort(coefs)[-15:][::-1]

print("\nTop 15 Words Predicting NEGATIVE Reviews:")
print(feature_names[top_neg_indices])

print("\nTop 15 Words Predicting POSITIVE Reviews:")
print(feature_names[top_pos_indices])