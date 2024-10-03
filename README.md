# Movie Recommender System
## Overview
This project is a movie recommender system built using data from two Kaggle datasets: movies.csv and credits.csv. The system leverages text data and similarity metrics to provide movie recommendations based on user input. The implementation is done in a Jupyter Notebook.

## Datasets
1. movies.csv: Contains information about movies such as title, genres, and overview.
2. credits.csv: Contains information about the cast and crew of each movie.
## Setup and Installation
1. Install required libraries:
Make sure you have the necessary Python libraries installed. You can install them using pip:

```
pip install pandas numpy scikit-learn nltk
```
2. Jupyter Notebook:
Ensure you have Jupyter Notebook installed and running. You can install Jupyter Notebook using pip if you don't have it:

```
pip install notebook
```
## Download datasets:
Download movies.csv and credits.csv from Kaggle and place them in your working directory.

## Implementation
The project involves the following key steps:

1. Loading the Data:
Read the movies.csv and credits.csv files into pandas DataFrames.

```
import pandas as pd
data_movies = pd.read_csv('movies.csv')
data_credits = pd.read_csv('credits.csv')
```
2. Preprocessing:
Perform necessary preprocessing steps such as merging datasets, cleaning text data, and handling missing values.

3. Feature Extraction:
Utilize the CountVectorizer from scikit-learn to convert text data into numerical feature vectors.
```
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5500, stop_words='english')
vectors = cv.fit_transform(data['tag']).toarray()
```

4. Similarity Calculation:
Compute similarity between movies using cosine similarity or another similarity metric.

5. Recommendation Function:
Define a function recommend that takes a movie title as input and returns the top 5 recommended movies based on similarity scores.

```
def recommend(movie):
    index = 0
    top5_movies = []
    movie_index = data[data['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[movie_index])), reverse=True, key=lambda x: x[1])[0:10]
    for i, j in distances:
        top5_movies.append(data.iloc[i].title)
    return top5_movies
```
```
recommend(avatar)
```
And we got our answer. CHeck out this project.
