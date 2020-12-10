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
us_canada_user_rating.head()
