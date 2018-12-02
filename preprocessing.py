import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt
from sklearn import svm

data_train = pd.read_csv(r'X:\Hackathon\Av - Mckinsey - Healthcare Analytics\train_ajEneEa.csv')
data_train.drop(labels = ['id'], inplace= True, axis = 1)

data_test = pd.read_csv(r'X:\Hackathon\Av - Mckinsey - Healthcare Analytics\test_v2akXPA.csv')
data_test.drop(labels = ['id'], inplace= True, axis = 1)

def check_missing_values(df):
   df2 = pd.DataFrame()
   for field in list(df):
       df2 = df2.append(df[[field]].isnull().sum().reset_index())
   df2[0] = (df2[0]/df.shape[0])*100
   df2.columns = ['Fields', '% of missing values']
   df2.reset_index(drop = True, inplace = True)
   return df2.sort_values(by = '% of missing values', ascending = False)

def numerical_imputer(df, list_to_impute,strtgy = 'median'):
    for field in list_to_impute:
        num_imputer = Imputer(strategy=strtgy)
        df[field] = num_imputer.fit_transform(df[field].reshape(-1,1))
        return df

def check_class_imbalance(df, predictor):
    x = (df.groupby(predictor)[predictor].count() / len(df[predictor]))*100
    return x


def categorical_variable_encoding(df):
   return pd.get_dummies(df, columns=list(df))



### Model preparation Training Set :
check_missing_values(data_train) ### Check % missing values

data_train = numerical_imputer(data_train, ['bmi'], 'mean')   ### Impute numerical columns
data_train['smoking_status'].fillna(value = 'MISSING', inplace = True) ### Treat missing values as separate class 'MISSING' for categorical

data_train_string = data_train.select_dtypes(include = 'object')
data_train_numeric = data_train.select_dtypes(exclude = 'object')
data_train_string_converted = categorical_variable_encoding(data_train_string) ### Categorical encoding

data_train_processed = pd.concat([data_train_numeric, data_train_string_converted], axis = 1)
#data_train_processed.to_csv(r'X:\Hackathon\Av - Mckinsey - Healthcare Analytics\data_train_processed.csv', index = False)



### Model preparation Test Set :
check_missing_values(data_test)
### Impute numerical columns
num_imputer = Imputer(strategy='mean')
num_imputer.fit(data_train['bmi'].reshape(-1,1))
data_test['bmi'] = num_imputer.transform(data_test['bmi'].reshape(-1,1))
data_test['smoking_status'].fillna(value = 'MISSING', inplace = True) ### Treat missing values as separate class 'MISSING' for categorical

data_test_string = data_test.select_dtypes(include = 'object')
data_test_numeric = data_test.select_dtypes(exclude = 'object')
data_test_string_converted = categorical_variable_encoding(data_test_string) ### Categorical encoding

data_test_processed_X = pd.concat([data_test_numeric, data_test_string_converted], axis = 1)
#data_test_processed.to_csv(r'X:\Hackathon\Av - Mckinsey - Healthcare Analytics\data_test_processed.csv', index = False)




####### Model : SVM - Classifier

data_train_processed_Y = data_train_processed['stroke'].values
data_train_processed_X = np.array(data_train_processed.drop(labels = ['stroke'], axis = 1))

sm = SMOTE(ratio = 0.1,random_state=42)
data_train_processed_X_res, data_train_processed_Y_res = sm.fit_sample(data_train_processed_X, data_train_processed_Y)

clfr = svm.SVC(kernel='rbf', C = 10, gamma = 0.1)
scores = cross_val_score(clfr, data_train_processed_X_res, data_train_processed_Y_res, cv=5, verbose = 5, n_jobs = -1)
clfr.fit(data_train_processed_X_res, data_train_processed_Y_res)
data_test_predicted_Y = clfr.predict(data_test_processed_X)
#pd.DataFrame(data_test_predicted_Y).to_csv(r'X:\Hackathon\Av - Mckinsey - Healthcare Analytics\predicted.csv')


