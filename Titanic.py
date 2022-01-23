import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier

data = pd.read_csv("train.csv", index_col = 0)

x = data.iloc[ : , 1 : ]


x.drop(['Name', 'Ticket', 'Cabin'], inplace = True, axis = 1)

x['Embarked'].replace('S', 1, inplace = True)
x['Embarked'].replace('C', 2, inplace = True)
x['Embarked'].replace('Q', 3, inplace = True)


y = data.iloc[ : , 0 : 1]
x = np.array(x)
y = np.array(y)
ct = ColumnTransformer([('encoder', OneHotEncoder(), [1])], remainder = "passthrough")
x = np.array(ct.fit_transform(x))
sm = SimpleImputer(missing_values = np.nan, strategy = 'mean')
sm.fit(x)
x = sm.transform(x)
sc = StandardScaler()
x[:, 2:] = sc.fit_transform(x[:, 2:])

gbc = GradientBoostingClassifier(learning_rate = 0.5, max_depth = 5, n_estimators = 150)
gbc.fit(x, y.ravel())

test = pd.read_csv('test.csv', index_col = 0)
y1 = pd.read_csv('gender_submission.csv', index_col = 0)
x1 = test
x1.drop(['Name', 'Ticket', 'Cabin'], inplace = True, axis = 1)

x1['Embarked'].replace('S', 1, inplace = True)
x1['Embarked'].replace('C', 2, inplace = True)
x1['Embarked'].replace('Q', 3, inplace = True)
x1 = np.array(x1)
y1 = np.array(y1)
ct = ColumnTransformer([('encoder', OneHotEncoder(), [1])], remainder = 'passthrough')

x1 = np.array(ct.fit_transform(x1))
sm = SimpleImputer(missing_values = np.nan, strategy = 'mean')
sm.fit(x1)
x1 = sm.transform(x1)
sc = StandardScaler()
x1 = sc.fit_transform(x1)
predict = gbc.predict(x1)
test = pd.read_csv('test.csv')

predict_list = pd.Series(predict, name = 'Survived').astype('int')
predict_list2 = pd.concat([test['PassengerId'], predict_list], axis = 1)
predict_list2.to_csv('prediction', index = False)
print(gbc.score(x1, y1))







