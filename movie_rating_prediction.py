import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load the dataset (replace 'movies.csv' with your dataset path)
data = pd.read_csv('movies.csv')

# Display the first few rows
print(data.head())

# Check for missing values
print(data.isnull().sum())

# Drop rows with missing target values
data = data.dropna(subset=['rating'])

# Fill missing values in other columns (if any)
data.fillna('', inplace=True)

# Feature engineering
# Combine genre, director, and actors into a single text column
data['combined_features'] = data['genre'] + ' ' + data['director'] + ' ' + data['actors']

# Convert text features into numerical features using CountVectorizer
vectorizer = CountVectorizer(max_features=5000, stop_words='english')
X_text_features = vectorizer.fit_transform(data['combined_features']).toarray()

# Normalize numerical features (if any)
# Assuming 'budget' is a numerical feature
if 'budget' in data.columns:
    scaler = StandardScaler()
    X_numeric_features = scaler.fit_transform(data[['budget']])
else:
    X_numeric_features = np.array([]).reshape(len(data), 0)

# Combine text and numerical features
X = np.hstack((X_text_features, X_numeric_features))

# Target variable
y = data['rating']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Predict the rating of a new movie
new_movie = {
    'genre': 'Action',
    'director': 'Christopher Nolan',
    'actors': 'Christian Bale, Michael Caine, Liam Neeson',
    'budget': 150000000
}

# Combine features for the new movie
new_movie_features = vectorizer.transform([new_movie['genre'] + ' ' + new_movie['director'] + ' ' + new_movie['actors']]).toarray()
new_movie_budget = scaler.transform([[new_movie['budget']]])
new_movie_combined = np.hstack((new_movie_features, new_movie_budget))

# Predict rating
predicted_rating = model.predict(new_movie_combined)
print(f"Predicted Rating: {predicted_rating[0]}")