import pandas as pd
import os
from sklearn import preprocessing
from sklearn import ensemble
from sklearn import model_selection

#changes working directory
os.chdir("D:/Data Science/Data/")

titanic_train = pd.read_csv("titanic_train.csv")
titanic_train.shape
titanic_train.info()

titanic_test = pd.read_csv('titanic_test.csv')
titanic_test.shape
titanic_test.info()
titanic_test.Survived = None

#it gives the same never of levels for all the categorical variables
titanic = pd.concat([titanic_train, titanic_test])

#create title column from name
def extract_title(name):
     return name.split(',')[1].split('.')[0].strip()
titanic['Title'] = titanic['Name'].map(extract_title)

#create an instance of Imputer class with required arguments
mean_imputer = preprocessing.Imputer()
#compute mean of age and fare respectively
mean_imputer.fit(titanic_train[['Age','Fare']])
#fill up the missing data with the computed means 
titanic[['Age','Fare']] = mean_imputer.transform(titanic[['Age','Fare']])

#creaate categorical age column from age
def convert_age(age):
    if(age >= 0 and age <= 10): 
        return 'Child'
    elif(age <= 25): 
        return 'Young'
    elif(age <= 50): 
        return 'Middle'
    else: 
        return 'Old'
titanic['Age1'] = titanic['Age'].map(convert_age)

titanic['FamilySize'] = titanic['SibSp'] +  titanic['Parch'] + 1
def convert_familysize(size):
    if(size == 1): 
        return 'Single'
    elif(size <=3): 
        return 'Small'
    elif(size <= 6): 
        return 'Medium'
    else: 
        return 'Large'
titanic['FamilySize1'] = titanic['FamilySize'].map(convert_familysize)

#convert categorical columns to one-hot encoded columns
titanic1 = pd.get_dummies(titanic, columns=['Sex','Pclass','Embarked', 'Age1', 'Title', 'FamilySize1'])
titanic1.shape
titanic1.info()

titanic2 = titanic1.drop(['PassengerId','Name','Age','Ticket','Cabin','Survived'], axis=1, inplace=False)
titanic2.shape

X_train = titanic2[0:titanic_train.shape[0]]
X_train.shape
X_train.info()
y_train = titanic_train['Survived']

gbm_estimator = ensemble.GradientBoostingClassifier(random_state=2017)
#loss function to be optimized = ‘deviance’ refers to deviance (= logistic regression) for classification with probabilistic outputs. 
#n_estimators: The number of boosting stages to perform.
gbm_grid = {'loss': ['deviance'], 'n_estimators':[50, 100], 'learning_rate':[0.001,0.01,0.2,0.3]}
grid_gbm_estimator = model_selection.GridSearchCV(gbm_estimator, gbm_grid, cv=10,n_jobs=1)
grid_gbm_estimator.fit(X_train, y_train)
print(grid_gbm_estimator.grid_scores_)
print(grid_gbm_estimator.best_score_)
print(grid_gbm_estimator.best_params_)
print(grid_gbm_estimator.score(X_train, y_train))

#exlore feature importances calculated by GradientBoostingClassifier algorithm
features = X_train.columns
importances = grid_gbm_estimator.best_estimator_.feature_importances_
fe_df = pd.DataFrame({'feature':features, 'importance': importances})

X_test = titanic2[titanic_train.shape[0]:]
X_test.shape
X_test.info()
titanic_test['Survived'] = grid_gbm_estimator.predict(X_test)

titanic_test.to_csv('submission.csv', columns=['PassengerId','Survived'],index=False)
