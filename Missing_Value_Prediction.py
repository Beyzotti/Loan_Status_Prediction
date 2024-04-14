import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler, OneHotEncoder
from sklearn.svm import SVC
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df = pd.read_csv("Dataset/loan_data.csv")
df = df.drop(["Loan_ID"], axis=1)

# ML ile Eksik Veri Analizi
## Gender
df_pre = df.dropna()

model = CatBoostClassifier()

x = df_pre.drop(columns=['Gender'])
y = df_pre['Gender']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)

cat_features = x.select_dtypes(include='object').columns.to_list()

model = CatBoostClassifier(iterations=500, depth=5, learning_rate=0.05, cat_features=cat_features)
model.fit(x_train, y_train, eval_set=(x_test, y_test), plot=True)

df_Gender = df[df['Gender'].isna()].drop(labels='Gender', axis=1)
y_predict = model.predict(df_Gender)

y_predict  # Male

## Dependents
df_pre = df.dropna()

model = CatBoostClassifier()

x = df_pre.drop(columns=['Dependents'])
y = df_pre['Dependents']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)

cat_features = x.select_dtypes(include='object').columns.to_list()

model = CatBoostClassifier(iterations=500, depth=5, learning_rate=0.05, cat_features=cat_features)
model.fit(x_train, y_train, eval_set=(x_test, y_test), plot=True)

df_Dependents = df[df['Dependents'].isna()].drop(labels='Dependents', axis=1)
y_predict = model.predict(df_Dependents)

y_predict  # 0

## Self_Employed

df_pre = df.dropna()

model = CatBoostClassifier()

x = df_pre.drop(columns=['Self_Employed'])
y = df_pre['Self_Employed']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)

cat_features = x.select_dtypes(include = 'object').columns.to_list()

model = CatBoostClassifier(iterations=500, depth=5, learning_rate=0.05, cat_features=cat_features)
model.fit(x_train, y_train, eval_set=(x_test, y_test), plot=True)

df_Self_Employed = df[df['Self_Employed'].isna()].drop(labels = 'Self_Employed', axis = 1)
y_predict = model.predict(df_Self_Employed)

y_predict # No

## Loan_Amount_Term

df_pre = df.dropna()

model = LogisticRegression()

df_pre = pd.get_dummies(df_pre)

x = df_pre.drop(columns=['Loan_Amount_Term'])
y = df_pre['Loan_Amount_Term']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model.fit(x_train, y_train)

df_dummy = pd.get_dummies(df)
df_Loan_Amount_Term = df_dummy[df_dummy['Loan_Amount_Term'].isna()].drop(labels = 'Loan_Amount_Term', axis = 1)
y_predict = model.predict(df_Loan_Amount_Term)

y_predict # 360

## Credit_History

df_pre = df.dropna()

model = LogisticRegression()

df_pre = pd.get_dummies(df_pre)

x = df_pre.drop(columns=['Credit_History'])
y = df_pre['Credit_History']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model.fit(x_train, y_train)

df_dummy = pd.get_dummies(df)
df_Credit_History = df_dummy[df_dummy['Credit_History'].isna()].drop(labels = 'Credit_History', axis = 1)
y_predict = model.predict(df_Credit_History)

y_predict # 1