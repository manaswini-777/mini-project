import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Simulate dataset (500 records)
np.random.seed(42)
n = 500
data = {
    'C1_transactions': np.random.randint(1, 10, size=n),
    'C2_avg_value': np.random.uniform(20, 500, size=n),
    'C3_complaints': np.random.randint(0, 3, size=n),
    'C4_days_since_last_txn': np.random.randint(1, 100, size=n),
    'C5_inventory_pct': np.random.uniform(0.1, 1.0, size=n),
    'C6_margin': np.random.uniform(0.1, 0.5, size=n),
    'C7_storage_cost': np.random.choice(['low', 'medium', 'high'], size=n),
    'C8_competitor_price': np.random.choice(['low', 'medium', 'high'], size=n),
    'C9_seasonal': np.random.choice([0, 1], size=n),
    'C10_demand': np.random.choice(['low', 'medium', 'high'], size=n),
    'Label': np.random.choice([-1, 0, 1], size=n, p=[0.3, 0.4, 0.3])  # -1: decrease, 0: same, 1: increase
}
df = pd.DataFrame(data)

# Step 2: Preprocessing
label_cols = ['C7_storage_cost', 'C8_competitor_price', 'C10_demand']
le = LabelEncoder()
for col in label_cols:
    df[col] = le.fit_transform(df[col])

# Feature scaling
scaler = MinMaxScaler()
features = df.drop('Label', axis=1)
X = scaler.fit_transform(features)
y = df['Label']

# Step 3: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Step 4: Train models
models = {
    'Naive Bayes': GaussianNB(),
    'Linear SVM': SVC(kernel='linear'),
    'Nonlinear SVM': SVC(kernel='rbf'),
    'Decision Tree': DecisionTreeClassifier(),
    'KNN': KNeighborsClassifier()
}

print("Model performance:\n")
best_model = None
best_score = 0

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"{name}: Accuracy = {acc:.4f}")
    if acc > best_score:
        best_score = acc
        best_model = (name, model)

# Step 5: Final evaluation
print("\nBest Model:", best_model[0])
final_preds = best_model[1].predict(X_test)
print("Classification Report:\n", classification_report(y_test, final_preds))
print("Confusion Matrix:\n", confusion_matrix(y_test, final_preds))