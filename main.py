from os import sep
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

books = pd.read_csv('BX-CSV-Dump/BX-Books.csv', sep=';',
                    error_bad_lines=False, encoding='latin-1')
books.columns = ['ISBN', 'bookTitle', 'bookAuthor', 'yearOfPublication',
                 'publisher', 'imageUrlS', 'imageUrlM', 'imageUrlL']
users = pd.read_csv('BX-CSV-Dump/BX-Users.csv', sep=';',
                    error_bad_lines=False, encoding="latin-1")
users.columns = ['userID', 'Location', 'Age']
ratings = pd.read_csv('BX-CSV-Dump/BX-Book-Ratings.csv', sep=';',
                      error_bad_lines=False, encoding="latin-1")
ratings.columns = ['userID', 'ISBN', 'bookRating']

# Ratings Distribution
# The ratings are very unevenly distributed, and the vast majority of ratings are 0.
plt.rc("font", size=15)
ratings.bookRating.value_counts(sort=False).plot(kind='bar')
plt.title('Rating Distribution\n')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.savefig('ratings_distribution.png', bbox_inches='tight')
plt.show()

# Age Distribution
# The most active users are among those in their 20–30s.
users.Age.hist(bins=[0, 10, 20, 30, 40, 50, 100])
plt.title('Age Distribution\n')
plt.xlabel('Age')
plt.ylabel('Count')
plt.savefig('age_distribution.png', bbox_inches='tight')
plt.show()

# Recommendations based on rating counts
rating_count = pd.DataFrame(ratings.groupby('ISBN')['bookRating'].count())
print(rating_count.sort_values('bookRating', ascending=False).head())
most_rated_books = pd.DataFrame(['0971880107', '0316666343', '0385504209',
                                 '0060928336', '0312195516'], index=np.arange(5), columns=['ISBN'])
most_rated_books_summary = pd.merge(most_rated_books, books, on='ISBN')
print('Most rated books summary: \n', most_rated_books_summary)
# The recommender suggests that novels are popular and likely receive more ratings.
# And if someone likes “The Lovely Bones: A Novel”, we should probably also recommend to him(or her) “Wild Animus”.


average_rating = pd.DataFrame(ratings.groupby('ISBN')['bookRating'].mean())
average_rating['ratingCount'] = pd.DataFrame(
    ratings.groupby('ISBN')['bookRating'].count())
print(average_rating.sort_values('ratingCount', ascending=False).head())
# In this data set, the book that received the most rating counts was not highly rated at all.
# As a result, if we were to use recommendations based on rating counts, we would definitely make mistakes here.
# So, we need to have a better system.


# Recommendations based on correlations
# We use Pearsons’R correlation coefficient to measure the linear correlation between two variables, in our case, the ratings for two books.
# To ensure statistical significance, users with less than 200 ratings,
# and books with less than 100 ratings are excluded.
counts_user = ratings['userID'].value_counts()
ratings = ratings[ratings['userID'].isin(
    counts_user[counts_user >= 200].index)]
counts = ratings['bookRating'].value_counts()
ratings = ratings[ratings['bookRating'].isin(counts[counts >= 100].index)]

# Rating matrix
# We convert the ratings table to a 2D matrix.
# The matrix will be sparse because not every user rated every book.
ratings_pivot = ratings.pivot(index='userID', columns='ISBN').bookRating
userID = ratings_pivot.index
ISBN = ratings_pivot.columns
print(ratings_pivot.head())

# Which books are correlated with the 2nd most rated book "The Lovely Bones: A Novel"
bones_ratings = ratings_pivot['0316666343']
similar_to_bones = ratings_pivot.corrwith(bones_ratings)
corr_bones = pd.DataFrame(similar_to_bones, columns=['pearsonR'])
corr_bones.dropna(inplace=True)
corr_summary = corr_bones.join(average_rating['ratingCount'])
print(corr_summary[corr_summary['ratingCount'] >= 300].sort_values(
    'pearsonR', ascending=False).head(10))
books_corr_to_bones = pd.DataFrame(['0312291639', '0316601950', '0446610038', '0446672211', '0385265700', '0345342968', '0060930535', '0375707972', '0684872153'],
                                   index=np.arange(9), columns=['ISBN'])
corr_books = pd.merge(books_corr_to_bones, books, on='ISBN')
print(corr_books)
# These three books sound like they would be highly correlated with “The Lovely Bones”.
# It seems our correlation recommender system is working.
