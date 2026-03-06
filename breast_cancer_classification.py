from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix


class BreastCancerProject:
    def __init__(self):
        self.X = None
        self.y = None

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.scaler = StandardScaler()
        self.results = []

    def load_data(self):
        data = load_breast_cancer()
        self.X = data.data
        self.y = data.target

    def split_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=0.2,
            random_state=42,
            stratify=self.y
        )

    def scale_data(self):
        # Fit scaler on training data only, then apply to both train and test
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

    def evaluate(self, name, model):
        model.fit(self.X_train, self.y_train)

        y_pred = model.predict(self.X_test)
        y_prob = model.predict_proba(self.X_test)[:, 1]

        acc = accuracy_score(self.y_test, y_pred)
        prec = precision_score(self.y_test, y_pred)
        rec = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        auc = roc_auc_score(self.y_test, y_prob)
        cm = confusion_matrix(self.y_test, y_pred)

        self.results.append({
            "name": name,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc": auc,
            "confusion_matrix": cm
        })

    def run_models(self):
        # Models that benefit from scaling (LogReg + KNN)
        self.evaluate("Logistic Regression", LogisticRegression(max_iter=5000, random_state=42))
        self.evaluate("KNN (k=7)", KNeighborsClassifier(n_neighbors=7))

        # Random Forest does NOT need scaling, so use original (unscaled) data for it
        # We'll temporarily reload the split before scaling to keep it simple.
        X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
            self.X,
            self.y,
            test_size=0.2,
            random_state=42,
            stratify=self.y
        )

        rf = RandomForestClassifier(n_estimators=300, random_state=42)
        rf.fit(X_train_raw, y_train_raw)
        y_pred = rf.predict(X_test_raw)
        y_prob = rf.predict_proba(X_test_raw)[:, 1]

        acc = accuracy_score(y_test_raw, y_pred)
        prec = precision_score(y_test_raw, y_pred)
        rec = recall_score(y_test_raw, y_pred)
        f1 = f1_score(y_test_raw, y_pred)
        auc = roc_auc_score(y_test_raw, y_prob)
        cm = confusion_matrix(y_test_raw, y_pred)

        self.results.append({
            "name": "Random Forest",
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc": auc,
            "confusion_matrix": cm
        })

    def print_results(self):
        for r in self.results:
            print(r["name"])
            print("Accuracy:", round(r["accuracy"], 4))
            print("Precision:", round(r["precision"], 4))
            print("Recall:", round(r["recall"], 4))
            print("F1:", round(r["f1"], 4))
            print("ROC AUC:", round(r["roc_auc"], 4))
            print("Confusion Matrix:")
            print(r["confusion_matrix"])
            print()

    def best_model(self):
        # Pick best by ROC AUC
        best = self.results[0]
        for r in self.results:
            if r["roc_auc"] > best["roc_auc"]:
                best = r
        return best


def main():
    project = BreastCancerProject()

    project.load_data()
    project.split_data()

    # Scale (for Logistic Regression + KNN)
    project.scale_data()

    # Train + evaluate all models
    project.run_models()

    # Print results
    project.print_results()

    # Print best
    best = project.best_model()
    print("Best Model (by ROC AUC):", best["name"])
    print("ROC AUC:", round(best["roc_auc"], 4))


if __name__ == "__main__":
    main()