from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import main
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Collaborative Filtering Using k-Nearest Neighbors (kNN)
# find clusters of similar users based on common book ratings,
# and make predictions using the average rating of top-k nearest neighbors.

combine_book_rating = pd.merge(main.ratings, main.books, on='ISBN')
columns = ['yearOfPublication', 'publisher',
           'bookAuthor', 'imageUrlS', 'imageUrlM', 'imageUrlL']
combine_book_rating = combine_book_rating.drop(columns, axis=1)
# group by book titles and create a new column for total rating count.
combine_book_rating = combine_book_rating.dropna(subset=['bookTitle'])

book_ratingCount = (combine_book_rating.
                    groupby(['bookTitle'])['bookRating'].
                    count().
                    reset_index().
                    rename(columns={'bookRating': 'totalRatingCount'})
                    [['bookTitle', 'totalRatingCount']]
                    )
print(book_ratingCount.head())

# We combine the rating data with the total rating count data, this gives us exactly what we need to find out which books are popular and filter out lesser-known books.
rating_with_totalRatingCount = combine_book_rating.merge(
    book_ratingCount, left_on='bookTitle', right_on='bookTitle', how='left')
# statistics of total rating count
pd.set_option('display.float_format', lambda x: '%.3f' % x)
print('Book Total Rating Count: \n',
      book_ratingCount['totalRatingCount'].describe())
# the top of the distribution
print('Top distribution of book total rating count: \n',
      book_ratingCount['totalRatingCount'].quantile(np.arange(.9, 1, .01)))

# About 1% of the books received 50 or more ratings. Because we have so many books in our data,
# we will limit it to the top 1%, and this will give us 2713 unique books.
popularity_threshold = 50
rating_popular_book = rating_with_totalRatingCount.query(
    'totalRatingCount >= @popularity_threshold')

# Filter to users in US and Canada only
combined = rating_popular_book.merge(
    main.users, left_on='userID', right_on='userID', how='left')

us_canada_user_rating = combined[combined['Location'].str.contains(
    "usa|canada")]
us_canada_user_rating = us_canada_user_rating.drop('Age', axis=1)


us_canada_user_rating = us_canada_user_rating.drop_duplicates(
    ['userID', 'bookTitle'])
us_canada_user_rating_pivot = us_canada_user_rating.pivot(
    index='bookTitle', columns='userID', values='bookRating').fillna(0)
# convert to sparse matrix (CSR method)
us_canada_user_rating_matrix = csr_matrix(us_canada_user_rating_pivot.values)

# Implementing kNN

model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(us_canada_user_rating_matrix)


query_index = np.random.choice(us_canada_user_rating_pivot.shape[0])
distances, indices = model_knn.kneighbors(
    us_canada_user_rating_pivot.iloc[query_index, :].reshape(1, -1), n_neighbors=6)

# Recommend sililar books:
for i in range(0, len(distances.flatten())):
    if i == 0:
        print('Recommendations for {0}:\n'.format(
            us_canada_user_rating_pivot.index[query_index]))
    else:
        print('{0}: {1}, with distance of {2}:'.format(
            i, us_canada_user_rating_pivot.index[indices.flatten()[i]], distances.flatten()[i]))
