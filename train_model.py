import requests
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import time

print("Initiating API extraction from PokeAPI...")
# Fetch list of the first 300 Pokemon to build our dataset
url = "https://pokeapi.co/api/v2/pokemon?limit=300"
response = requests.get(url).json()
results = response['results']

raw_data = []
print(f"Fetching combat statistics for {len(results)} Pokémon...")

# Data Engineering: Unpacking deeply nested JSON
for i, p in enumerate(results):
    res = requests.get(p['url']).json()
    stats = {stat['stat']['name']: stat['base_stat'] for stat in res['stats']}
    ptype = res['types'][0]['type']['name']
    
    row = {
        'name': res['name'],
        'type': ptype,
        'hp': stats.get('hp', 0),
        'attack': stats.get('attack', 0),
        'defense': stats.get('defense', 0),
        'special-attack': stats.get('special-attack', 0),
        'special-defense': stats.get('special-defense', 0),
        'speed': stats.get('speed', 0)
    }
    raw_data.append(row)
    if (i + 1) % 50 == 0:
        print(f"Mined {i + 1} records...")
        time.sleep(0.1) # Prevent API rate limiting

# 1. Save Raw Data
raw_df = pd.DataFrame(raw_data)

print("Executing Feature Engineering and Data Cleaning...")
cleaned_df = raw_df.dropna().copy()

# Feature Engineering: Create a 'total_stats' column
cleaned_df['total_stats'] = cleaned_df[['hp', 'attack', 'defense', 'special-attack', 'special-defense', 'speed']].sum(axis=1)

# Target Filtering: Focus on 5 primary types to avoid extreme minority classes
top_types = ['water', 'normal', 'grass', 'bug', 'fire']
cleaned_df = cleaned_df[cleaned_df['type'].isin(top_types)]

# 2. Train/Test Split
features = ['hp', 'attack', 'defense', 'special-attack', 'special-defense', 'speed', 'total_stats']
X = cleaned_df[features]
y = cleaned_df['type']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training multiple classification models for comparison...")
# Model A: Random Forest (Best Model)
rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_preds)

# Model B: Logistic Regression (Baseline)
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)
lr_acc = accuracy_score(y_test, lr_preds)

# 3. Generate Evaluation Artifacts
metrics_df = pd.DataFrame({
    "Model": ["Random Forest", "Logistic Regression"],
    "Accuracy": [rf_acc, lr_acc],
    "Status": ["Selected (Best)", "Baseline"]
})

cm = confusion_matrix(y_test, rf_preds, labels=rf_model.classes_)

# 4. Package and Export All Artifacts
artifacts = {
    "raw_df": raw_df,
    "cleaned_df": cleaned_df,
    "metrics_df": metrics_df,
    "cm": cm,
    "classes": rf_model.classes_,
    "feature_importances": rf_model.feature_importances_,
    "feature_names": features
}

joblib.dump(rf_model, "pokemon_classifier.pkl")
joblib.dump(artifacts, "pokemon_artifacts.pkl")
print("✅ Success! Master model and dashboard artifacts saved.")