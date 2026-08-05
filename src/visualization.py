import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def create_visualizations(df):
    numeric_df = df.select_dtypes(include=[np.number])

    corr_matrix = numeric_df.corr()

    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, cmap='coolwarm')
    plt.title("Correlation Heatmap")
    plt.show()

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

    plt.figure(figsize=(8, 6))
    plt.scatter(
        pca_df['PC1'],
        pca_df['PC2'],
        alpha=0.5
    )

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("PCA Visualization")
    plt.show()

    return X_scaled, pca_df