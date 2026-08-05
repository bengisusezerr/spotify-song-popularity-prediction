import pandas as pd
import numpy as np


df = pd.read_csv("data/spotify-tracks-dataset.csv")


print(df.head())


print("Veri Boyutu:", df.shape)


print("Sütunlar:")
print(df.columns)


print(df.info())


print(df.isnull().sum())


print(df.describe())

print(df.columns.tolist())

df = df.drop(columns=[
    'Unnamed: 0.1',
    'Unnamed: 0',
    'track_id',
    'album_name',
    'track_name'

])
print(df.columns.tolist())

print(df.shape)

import seaborn as sns
import matplotlib.pyplot as plt


numeric_df = df.select_dtypes(include=[np.number])


corr_matrix = numeric_df.corr()


plt.figure(figsize=(12,8))


sns.heatmap(corr_matrix, cmap='coolwarm')


plt.title("Correlation Heatmap")


plt.show()

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


features = [
    'popularity',
    'duration_ms',
    'danceability',
    'energy',
    'loudness',
    'speechiness',
    'acousticness',
    'instrumentalness',
    'liveness',
    'valence',
    'tempo'
]

X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)

principal_components = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(
    data=principal_components,
    columns=['PC1', 'PC2']
)

print(pca_df.head())

plt.figure(figsize=(8,6))

plt.scatter(
    pca_df['PC1'],
    pca_df['PC2'],
    alpha=0.5
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Visualization")

plt.show()

from sklearn.cluster import KMeans


kmeans = KMeans(
    n_clusters=5,
    random_state=42
)


kmeans.fit(X_scaled)


clusters = kmeans.predict(X_scaled)


pca_df['Cluster'] = clusters

plt.figure(figsize=(10,7))

scatter = plt.scatter(
    pca_df['PC1'],
    pca_df['PC2'],
    c=pca_df['Cluster'],
    cmap='viridis',
    alpha=0.6
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("KMeans Clustering on Spotify Songs")

plt.colorbar(scatter)

plt.show()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


y = df['popularity']


X = df[
    [
        'danceability',
        'energy',
        'loudness',
        'speechiness',
        'acousticness',
        'instrumentalness',
        'liveness',
        'valence',
        'tempo'
    ]
]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()


model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


df['hit_song'] = (df['popularity'] > 70).astype(int)


X = df[
    [
        'danceability',
        'energy',
        'loudness',
        'speechiness',
        'acousticness',
        'instrumentalness',
        'liveness',
        'valence',
        'tempo'
    ]
]


y = df['hit_song']


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


clf = DecisionTreeClassifier(random_state=42)


clf.fit(X_train, y_train)


y_pred = clf.predict(X_test)


print("Accuracy:", accuracy_score(y_test, y_pred))


print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nYeni bir şarkı için değer giriniz:")

danceability = float(input("Danceability (0-1): "))
energy = float(input("Energy (0-1): "))
loudness = float(input("Loudness (-60 ile 0 arası): "))
speechiness = float(input("Speechiness (0-1): "))
acousticness = float(input("Acousticness (0-1): "))
instrumentalness = float(input("Instrumentalness (0-1): "))
liveness = float(input("Liveness (0-1): "))
valence = float(input("Valence (0-1): "))
tempo = float(input("Tempo: "))

new_song = pd.DataFrame({
    'danceability': [danceability],
    'energy': [energy],
    'loudness': [loudness],
    'speechiness': [speechiness],
    'acousticness': [acousticness],
    'instrumentalness': [instrumentalness],
    'liveness': [liveness],
    'valence': [valence],
    'tempo': [tempo]
})

predicted_popularity = model.predict(new_song)

print("\nTahmin edilen popularity skoru:",
      predicted_popularity[0])

hit_prediction = clf.predict(new_song)
hit_probability = clf.predict_proba(new_song)

print("Hit tahmini:", hit_prediction[0])
print("Hit olma olasılığı:",
      hit_probability[0][1])

if hit_prediction[0] == 1:
    print("Sonuç: Bu şarkı hit olabilir.")
else:
    print("Sonuç: Bu şarkı hit olmayabilir.")