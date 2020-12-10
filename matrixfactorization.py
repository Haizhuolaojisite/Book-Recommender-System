import warnings
from sklearn.decomposition import TruncatedSVD
import sklearn
import knn
import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)
us_canada_user_rating_pivot2 = knn.us_canada_user_rating.pivot(
    index='userID', columns='bookTitle', values='bookRating').fillna(0)
X = us_canada_user_rating_pivot2.values.T

# dimensionality reduction, 12 latent variables
SVD = TruncatedSVD(n_components=12, random_state=17)
matrix = SVD.fit_transform(X)
print('After fit to SVD : ', matrix.shape)

# To compare this with the results from kNN, we pick the same book
# “The Green Mile: Coffey’s Hands (Green Mile Series)” to find the books that have high correlation coefficients (between 0.9 and 1.0) with it.
# Pearson’s R correlation coefficient, rowwise is true by default
corr = np.corrcoef(matrix)
us_canada_book_title = us_canada_user_rating_pivot2.columns
us_canada_book_list = list(us_canada_book_title)
coffey_hands = us_canada_book_list.index(
    "The Green Mile: Coffey's Hands (Green Mile Series)")
print('Index of target book: ', coffey_hands)

corr_coffey_hands = corr.iloc[:, coffey_hands]
recommend_bookidx = corr_coffey_hands[corr_coffey_hands >= 0.9].index.tolist
recommend_book = [us_canada_book_title[i] for i in recommend_bookidx]
print(recommend_book)
