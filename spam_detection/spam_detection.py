# Spam detection using Naive Bayes 

import pandas as pd 
from sklearn.cross_validation import train_test_split 
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.naive_bayes import MultinomialNB 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score  

# Dataset from - https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection 
# Step 1.1 Understanding dataset
df = pd.read_table('smsspamcollection/SMSSpamCollection',
                   sep='\t', 
                   header = None, 
                   names=['label', 'sms_message']) 
# print df.head() 

# Data preprocessing 
df['label'] = df.label.map({'ham':0, 'spam':1}) 
# df['label'] = pd.Categorical(df.label).codes 
# print df.head() 
# print df.shape 
 
# Training and testing sets 
X_train, X_test, y_train, y_test = train_test_split(df['sms_message'],
                                                    df['label'],
                                                    random_state=1) 
# Apply Bag of words processing to dataset 
count_vector = CountVectorizer() 
training_data = count_vector.fit_transform(X_train) 
testing_data = count_vector.transform(X_test) 

# Naive Bayes 
naive_bayes = MultinomialNB() 
naive_bayes.fit(training_data, y_train) 
predictions = naive_bayes.predict(testing_data) 
print predictions 

# Evaluate model 
print('Accuracy score: ', format(accuracy_score(y_test, predictions)))
print('Precision score: ', format(precision_score(y_test, predictions))) 
print('Recall score: ', format(recall_score(y_test, predictions))) 
print('F1 score: ', format(f1_score(y_test, predictions))) 



