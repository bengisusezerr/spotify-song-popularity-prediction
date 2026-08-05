import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier


df = pd.read_csv("data/spotify-tracks-dataset.csv")

df = df.drop(columns=[
    'Unnamed: 0.1',
    'Unnamed: 0',
    'track_id',
    'album_name',
    'track_name'
])

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

# Regression modeli
y_reg = df['popularity']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_reg,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

# Classification modeli
df['hit_song'] = (df['popularity'] > 70).astype(int)

y_cls = df['hit_song']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_cls,
    test_size=0.2,
    random_state=42
)

clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

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