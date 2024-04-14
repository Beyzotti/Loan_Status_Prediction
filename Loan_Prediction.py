import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import math
from sklearn.exceptions import ConvergenceWarning
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, RocCurveDisplay
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler, OneHotEncoder
from sklearn.svm import SVC
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
import joblib

with warnings.catch_warnings():
    warnings.filterwarnings("ignore")

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df= pd.read_csv("C:/Users/90535/OneDrive/Masaüstü/data_analysis_with_python/loan_data.csv")
df = df.drop(["Loan_ID"], axis=1)
df.head()

def check_df(dataframe):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(3))
    print("##################### Tail #####################")
    print(dataframe.tail(3))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())


def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)


def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")


def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")


def correlation_matrix(df, cols):
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    fig = sns.heatmap(df[cols].corr(), annot=True, linewidths=0.5, annot_kws={'size': 12}, linecolor='w', cmap='RdBu')
    plt.show(block=True)


def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    # print(f"Observations: {dataframe.shape[0]}")
    # print(f"Variables: {dataframe.shape[1]}")
    # print(f'cat_cols: {len(cat_cols)}')
    # print(f'num_cols: {len(num_cols)}')
    # print(f'cat_but_car: {len(cat_but_car)}')
    # print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car


check_df(df)

df.describe().T

# Eksik veri kontrolü
df.isnull().sum()

# Fill missing values with predicted values.
df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(360)
df['Credit_History'] = df['Credit_History'].fillna(1)
df['Gender'] = df['Gender'].fillna('Male')
df['Dependents'] = df['Dependents'].fillna('0')
df['Self_Employed'] = df['Self_Employed'].fillna('No')

# Convert data types to a more convenient type for processing.
df['Credit_History'] = df['Credit_History'].astype(str)
df['ApplicantIncome'] = df['ApplicantIncome'].astype(float)

# Değişken türlerinin ayrıştırılması
cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=5, car_th=20)

# Kategorik değişkenlerin incelenmesi
for col in cat_cols:
    cat_summary(df, col)

# Sayısal değişkenlerin incelenmesi
df[num_cols].describe().T

# Sayısal değişkenkerin birbirleri ile korelasyonu
correlation_matrix(df, num_cols)

# Görselleştirme
## Numerik Değişkenlerin Görselleştirilmesi
def plot_distributions(df, columns, target_variable=None):

    # Determine the number of subplots required
    num_columns = len(columns)
    num_subplots = math.ceil(num_columns / 2)

    # Create subplots
    fig, axes = plt.subplots(num_subplots, 2, figsize=(12, 4 * num_subplots))
    axes = axes.flatten()

    # Plot distribution plots for each numeric column
    for i in range(num_columns):
        sns.histplot(data=df, x=columns[i], hue=target_variable, stat='percent', common_norm=False, kde=True, ax=axes[i])
        axes[i].set_title(f'Distribution of {columns[i]}')

    # Adjust layout
    plt.tight_layout()
    plt.show()
plot_distributions(df, num_cols, target_variable='Loan_Status')


## Kategorik Değişkenlerin Görselleştirilmesi
def plot_categorical_distributions(df, categorical_columns, target_variable=None):

    # Determine the number of subplots required
    num_categorical_columns = len(categorical_columns)
    num_subplots = math.ceil(num_categorical_columns / 2)

    # Create subplots
    fig, axes = plt.subplots(num_subplots, 2, figsize=(12, 4 * num_subplots))
    axes = axes.flatten()

    # Plot bar plots for categorical columns
    for i in range(num_categorical_columns):
        sns.countplot(data=df, x=categorical_columns[i], hue=target_variable, ax=axes[i])
        axes[i].set_title(f'Distribution of {categorical_columns[i]}')

    # Adjust layout
    plt.tight_layout()
    plt.show()
plot_categorical_distributions(df, cat_cols, target_variable='Loan_Status')


# Data'nın incelenmesi
def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


for i in num_cols:
    replace_with_thresholds(df, i)

df.head()

df = one_hot_encoder(df, cat_cols, drop_first=True)

#Standartlaştırma
X_scaled = StandardScaler().fit_transform(df[num_cols])
df[num_cols] = pd.DataFrame(X_scaled, columns=df[num_cols].columns)



y = df["Loan_Status_Y"]
X = df.drop(["Loan_Status_Y"], axis=1)

X_Train, X_Test, y_train, y_test=train_test_split(X,y, test_size=30,random_state=15)



def base_models(X, y, scoring="roc_auc"):
    print("Base Models....")
    classifiers = [('LR', LogisticRegression()),
                   ('KNN', KNeighborsClassifier()),
                   ("SVC", SVC()),
                   ("CART", DecisionTreeClassifier(error_score='raise')),
                   ("RF", RandomForestClassifier()),
                   ('Adaboost', AdaBoostClassifier()),
                   ('GBM', GradientBoostingClassifier()),
                   ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
                   ('LightGBM', LGBMClassifier(verbose=-1)),
                   ('CatBoost', CatBoostClassifier(verbose=False))
                   ]

    for name, classifier in classifiers:
        cv_results = cross_validate(classifier, X, y, cv=5, scoring=scoring)
        print(f"{scoring}: {round(cv_results['test_score'].mean(), 4)} ({name}) ")






lr_params = {"max_iter" : range(0,200,25),
             "solver" : ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
             "n_jobs": [-1,1,None]}


knn_params = {"n_neighbors": range(2, 50)}

cart_params = {'max_depth': range(1, 20),
               "min_samples_split": range(2, 30)}

rf_params = {"max_depth": [3,5, 8, 15, None],
             "min_samples_split": [15, 20,45,50],
             "n_estimators": [100, 200, 300]}

xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [5, 8],
                  "n_estimators": [100, 200],
                  "colsample_bytree": [0.5, 1]}

lightgbm_params = {"learning_rate": [0.01, 0.1],
                   "n_estimators": [300, 500],
                   "colsample_bytree": [0.7, 1],
                   "verbose": [-1]}

catboost_params = {"iterations": [700,800,900],
                   "learning_rate": [0.01, 0.1,0.03],
                   "depth": [3,6]}



classifiers = [('LR',LogisticRegression(), lr_params),
               ('KNN', KNeighborsClassifier(), knn_params),
               ("CART", DecisionTreeClassifier(), cart_params),
               ("RF", RandomForestClassifier(), rf_params),
               ('XGBoost', XGBClassifier(use_label_encoder=False), xgboost_params),
               ('LightGBM', LGBMClassifier(verbose=-1), lightgbm_params),
               ('CatBoost',CatBoostClassifier(verbose=False),catboost_params)]


def hyperparameter_optimization(X, y, cv=5, scoring="roc_auc"):
    print("Hyperparameter Optimization....")
    best_models = {}
    for name, classifier, params in classifiers:
        print(f"########## {name} ##########")
        cv_results = cross_validate(classifier, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (Before): {round(cv_results['test_score'].mean(), 4)}")

        gs_best = GridSearchCV(classifier, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
        final_model = classifier.set_params(**gs_best.best_params_)

        cv_results = cross_validate(final_model, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (After): {round(cv_results['test_score'].mean(), 4)}")
        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
        best_models[name] = final_model
    return best_models


def voting_classifier(best_models, X, y):
    print("Voting Classifier...")
    voting_clf = VotingClassifier(estimators=[('XGBoost', best_models["XGBoost"]), ('RF', best_models["RF"]),
                                              ('LightGBM', best_models["LightGBM"]), ('CatBoost', best_models["CatBoost"])],
                                  voting='soft').fit(X, y)
    cv_results = cross_validate(voting_clf, X, y, cv=3, scoring=["accuracy", "f1", "roc_auc"])
    print(f"Accuracy: {cv_results['test_accuracy'].mean()}")
    print(f"F1Score: {cv_results['test_f1'].mean()}")
    print(f"ROC_AUC: {cv_results['test_roc_auc'].mean()}")
    return voting_clf


hyperparameter_optimization(X, y, cv=5, scoring="roc_auc")

base_models(X, y)
best_models = hyperparameter_optimization(X_Train, y_train)
voting_clf = voting_classifier(best_models, X_Train, y_train)


##Train Hatası

y_pred=voting_clf.predict(X_Train)
y_prob=voting_clf.predict_proba(X_Train)[:,1]
print(classification_report(y_train,y_pred))
roc_auc_score(y_train,y_prob)

##Test Hatası

y_pred=voting_clf.predict(X_Test)
y_prob=voting_clf.predict_proba(X_Test)[:,1]
print(classification_report(y_test ,y_pred))
roc_auc_score(y_test,y_prob)

################################################
# LightGBM
################################################

lgbm_model = LGBMClassifier(random_state=15)

lgbm_params = {"learning_rate": [0.01, 0.1, 0.001],
               "n_estimators": [100, 300, 500, 1000],
               "colsample_bytree": [0.5, 0.7, 1]}

lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X_Train, y_train)

lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=17).fit(X_Train, y_train)

cv_results = cross_validate(lgbm_final, X_Train, y_train, cv=10, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

"""
cv_results['test_accuracy'].mean()
Out[61]: 0.8347619047619048
cv_results['test_f1'].mean()
Out[62]: 0.8934882971675424
cv_results['test_roc_auc'].mean()
Out[63]: 0.8089636363636362
"""


""""
Test Sonuç:
"""

y_pred=lgbm_final.predict(X_Test)
y_prob=lgbm_final.predict_proba(X_Test)[:,1]
print(classification_report(y_test ,y_pred))
roc_auc_score(y_test,y_prob)
## ROC_AUC 0.936


################################################
# XGBoost
################################################

xgboost_model = XGBClassifier(random_state=15)

xgboost_params = {"learning_rate": [0.1, 0.01, 0.001],
                  "max_depth": [5, 8, 12, 15, 20],
                  "n_estimators": [100, 500, 1000],
                  "colsample_bytree": [0.5, 0.7, 1]}

xgboost_best_grid = GridSearchCV(xgboost_model, xgboost_params, cv=5, n_jobs=-1, verbose=True).fit(X_Train, y_train)

xgboost_final = xgboost_model.set_params(**xgboost_best_grid.best_params_, random_state=17).fit(X_Train, y_train)

cv_results = cross_validate(xgboost_final, X_Train, y_train, cv=10, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

""" 
Train Sonuç:
cv_results['test_accuracy'].mean()
Out[79]: 0.8147619047619047
cv_results['test_f1'].mean()
Out[80]: 0.8765314576080392
cv_results['test_roc_auc'].mean()
Out[81]: 0.8094121212121212
"""
""""
Test Sonuç:
"""

y_pred=xgboost_final.predict(X_Test)
y_prob=xgboost_final.predict_proba(X_Test)[:,1]
print(classification_report(y_test ,y_pred))
roc_auc_score(y_test,y_prob)
## ROC_AUC 0.928

################################################
# CatBoost
################################################

catboost_model = CatBoostClassifier(random_state=15, verbose=False)

catboost_params = {"iterations": [700,800,900],
                   "learning_rate": [0.01, 0.1,0.03],
                   "depth": [3,6]}

catboost_best_grid = GridSearchCV(catboost_model, catboost_params, cv=5, n_jobs=-1, verbose=True).fit(X_Train, y_train)

catboost_final = catboost_model.set_params(**catboost_best_grid.best_params_, random_state=17).fit(X_Train, y_train)

cv_results = cross_validate(catboost_final, X_Train, y_train, cv=10, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

"""
Train Sonuç:
cv_results['test_accuracy'].mean()
Out[99]: 0.8262698412698413
cv_results['test_f1'].mean()
Out[100]: 0.8825518757971587
cv_results['test_roc_auc'].mean()
Out[101]: 0.8303575757575757
"""

"""
Test Sonuç:
"""
y_pred=catboost_final.predict(X_Test)
y_prob=catboost_final.predict_proba(X_Test)[:,1]
print(classification_report(y_test ,y_pred))
roc_auc_score(y_test,y_prob)
##ROC_AUC  0.9680000000000001

################################################
# Random Forests
################################################

rf_model = RandomForestClassifier(random_state=15)

rf_params = {"max_depth": [5, 8, None],
             "max_features": [3, 5, 7, "auto"],
             "min_samples_split": [2, 5, 8, 15, 20],
             "n_estimators": [100, 200, 500]}

rf_best_grid = GridSearchCV(rf_model, rf_params, cv=5, n_jobs=-1, verbose=True).fit(X_Train, y_train)

rf_best_grid.best_params_

rf_best_grid.best_score_

rf_final = rf_model.set_params(**rf_best_grid.best_params_, random_state=17).fit(X_Train, y_train)


cv_results = cross_validate(rf_final, X_Train, y_train, cv=10, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

"""
Train Sonuç:
cv_results['test_accuracy'].mean()
Out[111]: 0.8348412698412698
cv_results['test_f1'].mean()
Out[112]: 0.8930986867779319
cv_results['test_roc_auc'].mean()
Out[113]: 0.8139363636363637
"""

"""
Test Sonuç:
"""
y_pred=rf_final.predict(X_Test)
y_prob=rf_final.predict_proba(X_Test)[:,1]
print(classification_report(y_test ,y_pred))
roc_auc_score(y_test,y_prob)
##ROC_AUC  0.928


################################################
# Feature Importance
################################################

def plot_importance(model, features, num=len(X_Test), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(rf_final, X_Test)
plot_importance(xgboost_final, X_Test)
plot_importance(lgbm_final, X_Test)
plot_importance(catboost_final,X_Test)

################################################
# Voting Classifier
################################################


voting_clf = VotingClassifier(estimators=[("rf",rf_final),("xgb",xgboost_final),
                                          ("lg",lgbm_final), ("cb",catboost_final)],
                              voting='soft').fit(X_Train, y_train)

cv_results = cross_validate(voting_clf, X_Train, y_train, cv=5, scoring=["accuracy", "f1", "roc_auc"])
print(f"Accuracy: {cv_results['test_accuracy'].mean()}")
print(f"F1Score: {cv_results['test_f1'].mean()}")
print(f"ROC_AUC: {cv_results['test_roc_auc'].mean()}")

"""
Train Sonuç:
print(f"Accuracy: {cv_results['test_accuracy'].mean()}")
Accuracy: 0.8346881287726358
print(f"F1Score: {cv_results['test_f1'].mean()}")
F1Score: 0.8913734102519149
print(f"ROC_AUC: {cv_results['test_roc_auc'].mean()}")
ROC_AUC: 0.8353974732750242
"""

"""
Test Sonuç:
"""

y_pred=voting_clf.predict(X_Test)
y_prob=voting_clf.predict_proba(X_Test)[:,1]
print(classification_report(y_test ,y_pred))
roc_auc_score(y_test,y_prob)
##ROC_AUC   0.9440000000000001
