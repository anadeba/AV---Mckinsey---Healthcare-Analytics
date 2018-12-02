import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm
from imblearn.under_sampling import RandomUnderSampler
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt

data_train_processed = pd.read_csv(r'X:\Hackathon\Av - Mckinsey - Healthcare Analytics\data_train_processed.csv')
data_test_processed_X = pd.read_csv(r'X:\Hackathon\Av - Mckinsey - Healthcare Analytics\data_test_processed.csv')

data_train_processed_Y = data_train_processed['stroke'].values
data_train_processed_X = np.array(data_train_processed.drop(labels = ['stroke'], axis = 1))

sm = SMOTE(ratio = 0.1,random_state=42)
data_train_processed_X_res, data_train_processed_Y_res = sm.fit_sample(data_train_processed_X, data_train_processed_Y)

rus = RandomUnderSampler(return_indices=True)
data_train_processed_X_res, data_train_processed_Y_res, idx_resampled = rus.fit_sample(data_train_processed_X, data_train_processed_Y)

#### RF - classifier
clfr = RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_split = 2, random_state=0, max_features = 3, n_jobs =-1, verbose = 2, oob_score=True)
clfr.fit(data_train_processed_X_res, data_train_processed_Y_res)
data_test_predicted_Y = clfr.predict(data_test_processed_X)
pd.DataFrame(data_test_predicted_Y).to_csv(r'X:\Hackathon\Av - Mckinsey - Healthcare Analytics\predicted.csv')

#### Adaboost - classifier

clfr = AdaBoostClassifier(n_estimators=800, random_state=100)
scores = cross_val_score(clfr, data_train_processed_X_res, data_train_processed_Y_res, cv=5 ,verbose=5, n_jobs=-1)
clfr.fit(data_train_processed_X_res, data_train_processed_Y_res)
data_train_predicted_Y =clfr.predict(data_train_processed_X)
data_test_predicted_Y = clfr.predict(data_test_processed_X)
pd.DataFrame(data_test_predicted_Y).to_csv(r'X:\Hackathon\Av - Mckinsey - Healthcare Analytics\predicted.csv')


#### SVM - classifier

clfr = svm.SVC(kernel='rbf', C = 10, gamma = 0.1)
scores = cross_val_score(clfr, data_train_processed_X_res, data_train_processed_Y_res, cv=5, verbose = 5, n_jobs = -1)
clfr.fit(data_train_processed_X_res, data_train_processed_Y_res)
data_test_predicted_Y = clfr.predict(data_test_processed_X)
pd.DataFrame(data_test_predicted_Y).to_csv(r'X:\Hackathon\Av - Mckinsey - Healthcare Analytics\predicted.csv')


### KNN - Classifier

neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(data_train_processed_X_res, data_train_processed_Y_res)
data_test_predicted_Y = neigh.predict(data_test_processed_X)


### PCA

from sklearn.decomposition import PCA
pca = PCA(n_components=4)
X_r = pca.fit(data_train_processed_X_res).transform(data_train_processed_X_res)
target_names = ['stroke', 'no-stroke']
colors = ['navy', 'turquoise']
plt.figure()
for color, i, target_name in zip(colors, [0, 1], target_names):
    plt.scatter(X_r[data_train_processed_Y_res == i, 0], X_r[data_train_processed_Y_res == i, 1], color=color, alpha=.8, lw=2, label = target_name)

plt.show()

