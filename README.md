## Titanic survivor prediction

The legendary [Kaggle Titanic competition](https://www.kaggle.com/c/titanic) is the first dive into their machine learning education for a lot of learners. For me, this was the starting point as well. 

Recently I went back to my solution, cleaned the code and used some of the new(er) sklearn methods. 

The highest increase in performance, was due improving the imputation method for the Age of the passengers. Rather than using  simple mean or median imputation of all passengers, I calculated the mean age based on the title of each person. To do so, I created an AgeImputer based on the provided BaseEstimator and TransformerMixin. 

Using Random Forest Classifier I could train a robust model getting an accuracy > 80%.  
