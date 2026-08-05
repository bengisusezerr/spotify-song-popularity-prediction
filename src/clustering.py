import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


df = pd.read_csv("data/spotify-tracks-dataset.csv")

df = df.drop(columns=[
    'Unnamed: 0.1',
    'Unnamed: 0',
    'track_id',
    'album_name',
    'track_name'
])

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
    principal_components,
    columns=['PC1', 'PC2']
)

kmeans = KMeans(
    n_clusters=5,
    random_state=42
)

kmeans.fit(X_scaled)

clusters = kmeans.predict(X_scaled)

pca_df['Cluster'] = clusters

plt.figure(figsize=(10, 7))

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