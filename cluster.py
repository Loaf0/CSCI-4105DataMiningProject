from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('cleaned_mentalHealth.csv')
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df = df.drop(columns=['record_id', 'gaming_addiction_risk_level'])

features = [
    'daily_gaming_hours', 
    'sleep_hours',
    #'grades_gpa', 
    #'exercise_hours_weekly', 
    #'face_to_face_social_hours_weekly'
    ]

imputer = SimpleImputer(strategy='median')
X = imputer.fit_transform(df[features])

scaler = StandardScaler()
XScaled = scaler.fit_transform(X)

# 26-36 determines the optimal number of clusters using silhouette score
best_k = None
best_score = -1

for k in range(2, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(XScaled)
    score = silhouette_score(XScaled, labels)
    if score > best_score:
        best_score = score
        best_k = k

print("Best k:", best_k, "Silhouette:", round(best_score, 3))

kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
yKMeans = kmeans.fit_predict(XScaled)
    
plt.scatter(XScaled[:, 0], XScaled[:, 1], c=yKMeans, cmap="viridis")
plt.title("K-Means Clustering")
plt.show()