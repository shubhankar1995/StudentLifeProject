import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier,VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeClassifier
from sklearn.grid_search import GridSearchCV


from sklearn.datasets import load_digits
from sklearn.feature_selection import SelectKBest, chi2
import statsmodels.api as sm

# Feature Selection
data = pd.read_csv("merge_all_x_y.csv")

data["date_greets"]= data["date_greets"].replace(to_replace = "day$",value=0,regex=True)
data["date_greets"]= data["date_greets"].replace(to_replace = "evening$",value=1,regex=True)
data["date_greets"]= data["date_greets"].replace(to_replace = "night$",value=2,regex=True)
data["travelstate_mode"]= data["travelstate_mode"].replace(to_replace = "stationary",value=0,regex=True)
data["travelstate_mode"]= data["travelstate_mode"].replace(to_replace = "moving",value=1,regex=True)

data = data.drop(['student_id', 'uid_x','uid_y'], axis=1)

# data = data.drop(['student_id', 'uid_x','uid_y'], axis=1)

#normalize and define X and Y
normalizer = preprocessing.MinMaxScaler()
data[data.columns[3:15]] = normalizer.fit_transform(data[data.columns[3:15]])

y_pos = data[data.columns[-3]]
y_neg = data[data.columns[-2]]
y_fs = data[data.columns[-1]]

X = data[data.columns[0:15]]

# Feature selection for Flourishing

X_90p = sm.add_constant(X.astype(float))

Linearmodel = sm.OLS(y_fs, X_90p).fit()
print(Linearmodel.summary())

df_coeffs_bin_my_avg = pd.DataFrame({"Coefficients": Linearmodel.params[0:], "p": Linearmodel.pvalues[0:]})
df_coeffs_bin_my_avg.reindex(df_coeffs_bin_my_avg["p"].sort_values().index).style.bar(subset = ["Coefficients", "p"], align='mid', color=['#d65f5f', '#5fba7d'])

# Feature selection for Positive Panas

X_90p = sm.add_constant(X.astype(float))

Linearmodel = sm.OLS(y_pos, X_90p).fit()
print(Linearmodel.summary())

df_coeffs_bin_my_avg = pd.DataFrame({"Coefficients": Linearmodel.params[0:], "p": Linearmodel.pvalues[0:]})
df_coeffs_bin_my_avg.reindex(df_coeffs_bin_my_avg["p"].sort_values().index).style.bar(subset = ["Coefficients", "p"], align='mid', color=['#d65f5f', '#5fba7d'])

# Feature selection for Negative Panas

X_90p = sm.add_constant(X.astype(float))

Linearmodel = sm.OLS(y_neg, X_90p).fit()
print(Linearmodel.summary())

df_coeffs_bin_my_avg = pd.DataFrame({"Coefficients": Linearmodel.params[0:], "p": Linearmodel.pvalues[0:]})
df_coeffs_bin_my_avg.reindex(df_coeffs_bin_my_avg["p"].sort_values().index).style.bar(subset = ["Coefficients", "p"], align='mid', color=['#d65f5f', '#5fba7d'])

# Executing the algorithms on the selected features

# input the flourishing scale file
data = pd.read_csv("fs.csv")

# drop the indexed column
data = data.drop("Unnamed: 0",axis=1)

array = data.values
X = array[:,0:7]
Y = array[:,-1]

ABC = MLPClassifier()

# run grid search
grid_search_ABC = GridSearchCV(estimator = ABC, param_grid = param_grid, cv = 3, n_jobs = -1, verbose = 2)

grid_search_ABC.fit(X,Y)

print(grid_search_ABC.best_params_)

best_grid = grid_search_ABC.best_estimator_

# k-fold evaluation on MLP classifier
k_fold = KFold(len(Y), shuffle=True, random_state=42)
mlp = MLPClassifier(activation = 'tanh', learning_rate = 'invscaling', shuffle = True, solver = 'lbfgs')
score = cross_val_score(mlp, X, Y, cv=k_fold, n_jobs=1,scoring="accuracy")
print("scores: ",score)
print("Mean score: ",score.mean())


param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}

rf = RandomForestClassifier()


DTC = DecisionTreeClassifier(random_state = 11, max_features = "auto", max_depth = None)

ABC = AdaBoostClassifier(base_estimator = DTC)

# run grid search
grid_search_ABC = GridSearchCV(estimator = rf, param_grid = param_grid, cv = 3, n_jobs = -1, verbose = 2)

print(grid_search_ABC.best_params_)

# k-fold evaluation on decision tree
DTC = DecisionTreeClassifier(random_state = 11, max_features = "auto",splitter="random",criterion="entropy")
clf = AdaBoostClassifier(base_estimator=RandomForestClassifier(),n_estimators=50)
k_fold = KFold(len(Y), n_folds=3, shuffle=True, random_state=42)
score = cross_val_score(clf, X, Y, cv=k_fold, n_jobs=1,scoring="accuracy")
print("scores: ",score)
print("Mean score: ",score.mean())

# k-fold evaluation on boosted decision tree
clf = AdaBoostClassifier(base_estimator=best_grid,n_estimators=50)
k_fold = KFold(len(Y), n_folds=3, shuffle=True, random_state=42)
score2 = cross_val_score(clf, X, Y, cv=k_fold, n_jobs=1,scoring="accuracy")
print("scores: ",score2)
print("Mean score: ",score2.mean())
print("Improved Accuracy: ",score2.mean()-score.mean())

# Comparision on multiple folds
fold = np.array([i for i in range(1,4)])
acc = np.array(score)
plt.scatter(x=fold,y=acc)
plt.xlabel("Folds")
plt.ylabel("Accuracy")
plt.show()


data = pd.read_csv("merge_all_x_y.csv")

data["date_greets"]= data["date_greets"].replace(to_replace = "day$",value=0,regex=True)
data["date_greets"]= data["date_greets"].replace(to_replace = "evening$",value=1,regex=True)
data["date_greets"]= data["date_greets"].replace(to_replace = "night$",value=2,regex=True)
data["travelstate_mode"]= data["travelstate_mode"].replace(to_replace = "stationary",value=0,regex=True)
data["travelstate_mode"]= data["travelstate_mode"].replace(to_replace = "moving",value=1,regex=True)

data = data.drop(['student_id', 'uid_x','uid_y'], axis=1)

# data = data.drop(['student_id', 'uid_x','uid_y'], axis=1)

#normalize and define X and Y
normalizer = preprocessing.MinMaxScaler()
data[data.columns[3:15]] = normalizer.fit_transform(data[data.columns[3:15]])

y_pos = data[data.columns[-3]]
y_neg = data[data.columns[-2]]
y_fs = data[data.columns[-1]]

X = data[data.columns[0:15]]

from sklearn.model_selection import train_test_split

data_90p, data_10p = train_test_split(data, test_size=0.1, random_state=0)

y_fs_90p = data_90p[data_90p.columns[-1]]

X_90p = data_90p[data_90p.columns[0:15]]


y_fs_10p = data_10p[data_10p.columns[-1]]

X_10p = data_10p[data_10p.columns[0:15]]

from sklearn.model_selection import GridSearchCV

tuned_parameters = {"fit_intercept" : [0,1], "normalize" : [0,1], "copy_X" : [0,1]}

# Grid search
clf = GridSearchCV(LinearRegression(), tuned_parameters, cv=5)
clf.fit(X_90p, y_fs_90p)
print(clf.best_params_)

from sklearn.linear_model import LinearRegression

# Executing linear regression
reg = LinearRegression(copy_X= 0, fit_intercept= 1, normalize= 0).fit(X_90p, y_fs_90p)

# Evaluating linear Regression
X_10p["Prediction_score"] = reg.predict(X_10p)

(y_fs_10p - X_10p['Prediction_score']).mean()



# input the Positive Panas scale file
data = pd.read_csv("pos_panas.csv")

# drop the indexed column
data = data.drop("Unnamed: 0",axis=1)

array = data.values
X = array[:,0:7]
Y = array[:,-1]

ABC = MLPClassifier()

# run grid search
grid_search_ABC = GridSearchCV(estimator = ABC, param_grid = param_grid, cv = 3, n_jobs = -1, verbose = 2)

grid_search_ABC.fit(X,Y)

print(grid_search_ABC.best_params_)

best_grid = grid_search_ABC.best_estimator_

# k-fold evaluation on MLP classifier
k_fold = KFold(len(Y), shuffle=True, random_state=42)
mlp = MLPClassifier(activation = 'tanh', learning_rate = 'invscaling', shuffle = True, solver = 'lbfgs')
score = cross_val_score(mlp, X, Y, cv=k_fold, n_jobs=1,scoring="accuracy")
print("scores: ",score)
print("Mean score: ",score.mean())


param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}

rf = RandomForestClassifier()


DTC = DecisionTreeClassifier(random_state = 11, max_features = "auto", max_depth = None)

ABC = AdaBoostClassifier(base_estimator = DTC)

# run grid search
grid_search_ABC = GridSearchCV(estimator = rf, param_grid = param_grid, cv = 3, n_jobs = -1, verbose = 2)

print(grid_search_ABC.best_params_)

# k-fold evaluation on decision tree
DTC = DecisionTreeClassifier(random_state = 11, max_features = "auto",splitter="random",criterion="entropy")
clf = AdaBoostClassifier(base_estimator=RandomForestClassifier(),n_estimators=50)
k_fold = KFold(len(Y), n_folds=3, shuffle=True, random_state=42)
score = cross_val_score(clf, X, Y, cv=k_fold, n_jobs=1,scoring="accuracy")
print("scores: ",score)
print("Mean score: ",score.mean())

# k-fold evaluation on boosted decision tree
clf = AdaBoostClassifier(base_estimator=best_grid,n_estimators=50)
k_fold = KFold(len(Y), n_folds=3, shuffle=True, random_state=42)
score2 = cross_val_score(clf, X, Y, cv=k_fold, n_jobs=1,scoring="accuracy")
print("scores: ",score2)
print("Mean score: ",score2.mean())
print("Improved Accuracy: ",score2.mean()-score.mean())

# Comparision on multiple folds
fold = np.array([i for i in range(1,4)])
acc = np.array(score)
plt.scatter(x=fold,y=acc)
plt.xlabel("Folds")
plt.ylabel("Accuracy")
plt.show()


data = pd.read_csv("merge_all_x_y.csv")

data["date_greets"]= data["date_greets"].replace(to_replace = "day$",value=0,regex=True)
data["date_greets"]= data["date_greets"].replace(to_replace = "evening$",value=1,regex=True)
data["date_greets"]= data["date_greets"].replace(to_replace = "night$",value=2,regex=True)
data["travelstate_mode"]= data["travelstate_mode"].replace(to_replace = "stationary",value=0,regex=True)
data["travelstate_mode"]= data["travelstate_mode"].replace(to_replace = "moving",value=1,regex=True)

data = data.drop(['student_id', 'uid_x','uid_y'], axis=1)

# data = data.drop(['student_id', 'uid_x','uid_y'], axis=1)

#normalize and define X and Y
normalizer = preprocessing.MinMaxScaler()
data[data.columns[3:15]] = normalizer.fit_transform(data[data.columns[3:15]])

y_pos = data[data.columns[-3]]
y_neg = data[data.columns[-2]]
y_fs = data[data.columns[-1]]

X = data[data.columns[0:15]]

from sklearn.model_selection import train_test_split

data_90p, data_10p = train_test_split(data, test_size=0.1, random_state=0)

y_fs_90p = data_90p[data_90p.columns[-1]]

X_90p = data_90p[data_90p.columns[0:15]]


y_fs_10p = data_10p[data_10p.columns[-1]]

X_10p = data_10p[data_10p.columns[0:15]]

from sklearn.model_selection import GridSearchCV

tuned_parameters = {"fit_intercept" : [0,1], "normalize" : [0,1], "copy_X" : [0,1]}

# Grid search
clf = GridSearchCV(LinearRegression(), tuned_parameters, cv=5)
clf.fit(X_90p, y_fs_90p)
print(clf.best_params_)

from sklearn.linear_model import LinearRegression

# Executing linear regression
reg = LinearRegression(copy_X= 0, fit_intercept= 1, normalize= 0).fit(X_90p, y_fs_90p)

# Evaluating linear Regression
X_10p["Prediction_score"] = reg.predict(X_10p)

(y_fs_10p - X_10p['Prediction_score']).mean()


# input the Negative Panas scale file
data = pd.read_csv("neg_panas.csv")

# drop the indexed column
data = data.drop("Unnamed: 0",axis=1)

array = data.values
X = array[:,0:7]
Y = array[:,-1]

ABC = MLPClassifier()

# run grid search
grid_search_ABC = GridSearchCV(estimator = ABC, param_grid = param_grid, cv = 3, n_jobs = -1, verbose = 2)

grid_search_ABC.fit(X,Y)

print(grid_search_ABC.best_params_)

best_grid = grid_search_ABC.best_estimator_

# k-fold evaluation on MLP classifier
k_fold = KFold(len(Y), shuffle=True, random_state=42)
mlp = MLPClassifier(activation = 'tanh', learning_rate = 'invscaling', shuffle = True, solver = 'lbfgs')
score = cross_val_score(mlp, X, Y, cv=k_fold, n_jobs=1,scoring="accuracy")
print("scores: ",score)
print("Mean score: ",score.mean())


param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}

rf = RandomForestClassifier()


DTC = DecisionTreeClassifier(random_state = 11, max_features = "auto", max_depth = None)

ABC = AdaBoostClassifier(base_estimator = DTC)

# run grid search
grid_search_ABC = GridSearchCV(estimator = rf, param_grid = param_grid, cv = 3, n_jobs = -1, verbose = 2)

print(grid_search_ABC.best_params_)

# k-fold evaluation on decision tree
DTC = DecisionTreeClassifier(random_state = 11, max_features = "auto",splitter="random",criterion="entropy")
clf = AdaBoostClassifier(base_estimator=RandomForestClassifier(),n_estimators=50)
k_fold = KFold(len(Y), n_folds=3, shuffle=True, random_state=42)
score = cross_val_score(clf, X, Y, cv=k_fold, n_jobs=1,scoring="accuracy")
print("scores: ",score)
print("Mean score: ",score.mean())

# k-fold evaluation on boosted decision tree
clf = AdaBoostClassifier(base_estimator=best_grid,n_estimators=50)
k_fold = KFold(len(Y), n_folds=3, shuffle=True, random_state=42)
score2 = cross_val_score(clf, X, Y, cv=k_fold, n_jobs=1,scoring="accuracy")
print("scores: ",score2)
print("Mean score: ",score2.mean())
print("Improved Accuracy: ",score2.mean()-score.mean())

# Comparision on multiple folds
fold = np.array([i for i in range(1,4)])
acc = np.array(score)
plt.scatter(x=fold,y=acc)
plt.xlabel("Folds")
plt.ylabel("Accuracy")
plt.show()


data = pd.read_csv("merge_all_x_y.csv")

data["date_greets"]= data["date_greets"].replace(to_replace = "day$",value=0,regex=True)
data["date_greets"]= data["date_greets"].replace(to_replace = "evening$",value=1,regex=True)
data["date_greets"]= data["date_greets"].replace(to_replace = "night$",value=2,regex=True)
data["travelstate_mode"]= data["travelstate_mode"].replace(to_replace = "stationary",value=0,regex=True)
data["travelstate_mode"]= data["travelstate_mode"].replace(to_replace = "moving",value=1,regex=True)

data = data.drop(['student_id', 'uid_x','uid_y'], axis=1)

# data = data.drop(['student_id', 'uid_x','uid_y'], axis=1)

#normalize and define X and Y
normalizer = preprocessing.MinMaxScaler()
data[data.columns[3:15]] = normalizer.fit_transform(data[data.columns[3:15]])

y_pos = data[data.columns[-3]]
y_neg = data[data.columns[-2]]
y_fs = data[data.columns[-1]]

X = data[data.columns[0:15]]

from sklearn.model_selection import train_test_split

data_90p, data_10p = train_test_split(data, test_size=0.1, random_state=0)

y_fs_90p = data_90p[data_90p.columns[-1]]

X_90p = data_90p[data_90p.columns[0:15]]


y_fs_10p = data_10p[data_10p.columns[-1]]

X_10p = data_10p[data_10p.columns[0:15]]

from sklearn.model_selection import GridSearchCV

tuned_parameters = {"fit_intercept" : [0,1], "normalize" : [0,1], "copy_X" : [0,1]}

# Grid search
clf = GridSearchCV(LinearRegression(), tuned_parameters, cv=5)
clf.fit(X_90p, y_fs_90p)
print(clf.best_params_)

from sklearn.linear_model import LinearRegression

# Executing linear regression
reg = LinearRegression(copy_X= 0, fit_intercept= 1, normalize= 0).fit(X_90p, y_fs_90p)

# Evaluating linear Regression
X_10p["Prediction_score"] = reg.predict(X_10p)

(y_fs_10p - X_10p['Prediction_score']).mean()