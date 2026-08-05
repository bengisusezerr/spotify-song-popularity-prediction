import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


df = pd.read_csv("data/spotify-tracks-dataset.csv")

df = df.drop(columns=[
    'Unnamed: 0.1',
    'Unnamed: 0',
    'track_id',
    'album_name',
    'track_name'
])

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