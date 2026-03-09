import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import itertools

df = pd.read_csv('dummy-data/vehicles_ml_dataset.csv')
features = ['year', 'kilometers_driven', 'seating_capacity', 'estimated_income', 'selling_price']

print("Finding best feature combination...")
best_score = 0
best_combo = None

for k in range(1, len(features) + 1):
    for combo in itertools.combinations(features, k):
        X = df[list(combo)]
        
        # Test Raw
        score_raw = silhouette_score(X, KMeans(n_clusters=3, random_state=42, n_init='auto').fit_predict(X))
        if score_raw > best_score: best_score, best_combo = score_raw, (combo, 'Raw')
        
        # Test Standard Scaled
        X_std = StandardScaler().fit_transform(X)
        score_std = silhouette_score(X_std, KMeans(n_clusters=3, random_state=42, n_init='auto').fit_predict(X_std))
        if score_std > best_score: best_score, best_combo = score_std, (combo, 'Standard')

        # Test MinMax Scaled
        X_minmax = MinMaxScaler().fit_transform(X)
        score_mm = silhouette_score(X_minmax, KMeans(n_clusters=3, random_state=42, n_init='auto').fit_predict(X_minmax))
        if score_mm > best_score: best_score, best_combo = score_mm, (combo, 'MinMax')

print(f"Best Score: {best_score} with {best_combo}")

print("\nTesting n_clusters=2 vs n_clusters=3...")
# The assignment asks to refine to score > 0.9. Maybe k is not 3?
for n in [2, 3, 4]:
    X = df[['estimated_income', 'selling_price']]
    score = silhouette_score(X, KMeans(n_clusters=n, random_state=42, n_init='auto').fit_predict(X))
    print(f"n_clusters={n}, score={score:.3f}")

# What about just 'income_level'? That's a categorical target.
# What if we only use `seating_capacity`?
print("Testing seating capacity")
score = silhouette_score(df[['seating_capacity']], KMeans(n_clusters=3, random_state=42, n_init='auto').fit_predict(df[['seating_capacity']]))
print(f"Seating capacity score: {score}")

# What if we drop the 'n_init="auto"' and use the old default?
score = silhouette_score(X, KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X))
print(f"n_init=10 score: {score:.3f}")
