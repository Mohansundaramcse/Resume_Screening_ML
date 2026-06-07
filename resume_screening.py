import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score
#Load the Dataset
df=pd.read_csv("UpdatedResumeDataSet.csv")

#head will show first 5 rows
print(df.head())

#info will give datatypes ,no of rows, no of columns
print(df.info())

#check column 
print(df.columns)


#check unique categories
print(df["Category"].unique())

#count each category
print(df["Category"].value_counts())

#one resume
print(df["Resume"][0])

#clean using regex

def clean_Resume(text):
    #remove urls
    text=re.sub(r"http\S+"," ",text)
    #remove special character and numbers
    text=re.sub(r"[^a-zA-Z]"," ",text)
    text=text.lower()
     # Remove extra spaces
    text=" ".join(text.split())
    return text


#testing the function
print("before cleaning")
print(df["Resume"][0])

print("after cleaning")
print(clean_Resume(df["Resume"][0]))

df["cleaned_resume"]=df["Resume"].apply(clean_Resume)

print("--------------------------")

print(df[["Category","cleaned_resume"]].head())

#remove words like is,the,was...
tfidf=TfidfVectorizer(stop_words='english',ngram_range=(1,2))
x=tfidf.fit_transform(df["cleaned_resume"])

print("-----------------------------------")
print(x.shape)


#set the target converts into numbers
#tfidf converts words into number for input
#labelEncoder converts categories into number for output

le=LabelEncoder()
y=le.fit_transform(df["Category"])
print("-------------------------------")
print(le.classes_)
print(y)

#train-test splitting (80,20)rule:

print("-------training data and testing data--------")

x_train,x_test,y_train,y_test=train_test_split(
    x,y,test_size=0.2,random_state=42
)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


#train the model using navie bayes 97%, with ngram 98%

# model=MultinomialNB()
# model.fit(x_train,y_train)

#train model using logistic regression 99%
print("-------------logistic regression-----")
model=LogisticRegression(max_iter=1000)
model.fit(x_train,y_train)

# print("--------svm----") 99%
# model=LinearSVC()
# model.fit(x_train,y_train)



#combine multiple model 99%
# nb=MultinomialNB()
# lr=LogisticRegression(max_iter=1000)
# svm=LinearSVC()
# voting_model=VotingClassifier(
#     estimators=[
#         ('nb',nb),
#         ('lr',lr),
#         ('svm',svm)
#     ],
# voting='hard'
# )

# voting_model.fit(x_train,y_train)




#train using cross value score  97%

# scores = cross_val_score(
#     model,
#     x,
#     y,
#     cv=5
# )

# print("Accuracy for cross value:",scores.mean())


#predict

y_pred=model.predict(x_test)
# y_pred=voting_model.predict(x_test)



#evaluate using accuracy_Score
accuracy_value=accuracy_score(y_test,y_pred)

print("------accuracy score is:-----------")
print(accuracy_value)


print("-----classification report is:---------")
print(classification_report(y_test, y_pred))
