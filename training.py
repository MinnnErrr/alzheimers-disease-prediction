#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import libraries
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from ydata_profiling import ProfileReport
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import joblib
import plotly.express as px


# In[2]:


#import dataset
ds=pd.read_csv('alzheimers_disease_data.csv')


# In[3]:


#EDA
ds.info()


# In[4]:


ds.shape


# In[5]:


ds.dtypes


# In[6]:


ds.head()


# In[7]:


ds.describe()


# In[8]:


#drop unnecessary columns
ds=ds.drop(columns=['PatientID', 'DoctorInCharge'])


# In[9]:


#profile=ProfileReport(ds,title="alzheimers disease data")
#profile.to_notebook_iframe()


# In[10]:


numeric = [col for col in ds.columns if ds[col].nunique()>4]
binary_categorical = [col for col in ds.columns if col not in numeric and col != 'diagnosis' and ds[col].nunique()<3]
nominal_categorical = [col for col in ds.columns if col not in numeric and col not in binary_categorical and col != 'diagnosis']
print('numeric =', numeric)
print('binary_categorical =', binary_categorical)
print('nominal_categorical =', nominal_categorical)


# In[11]:


for col in numeric:
    sns.boxplot(x=ds[col])
    plt.title(f'Boxplot of {col}')
    plt.show()


# In[12]:


corr = ds.corr()['Diagnosis'].drop('Diagnosis').sort_values(ascending=False)
corr_df = pd.DataFrame({'Feature': corr.index, 'Correlation': corr.values})

fig = px.bar(
    corr_df,
    x='Correlation',
    y='Feature',
    title='Correlation of Features with Diagnosis'
)

fig.update_layout(yaxis=dict(autorange="reversed"), height=25*len(corr_df))  
fig.show()


# In[13]:


sns.boxplot(x='Diagnosis', y='MMSE', data=ds)
plt.title('MMSE Distribution by Diagnosis')
plt.show()


# In[14]:


sns.boxplot(x='Diagnosis', y='ADL', data=ds)
plt.title('MMSE Distribution by ADL')
plt.show()


# In[15]:


sns.boxplot(x='Diagnosis', y='FunctionalAssessment', data=ds)
plt.title('MMSE Distribution by FunctionalAssessment')
plt.show()


# In[16]:


#check null values
ds.isnull().sum()


# In[17]:


null = ds[["SleepQuality", "Disorientation"]].isnull()
missing = ds[null.any(axis=1)]
missing


# In[18]:


#handle missing value for SleepQuality
ds['SleepQuality'].fillna(ds['SleepQuality'].mean(), inplace=True)

#handle missing value for Disorientation
ds['Disorientation'].fillna(ds['Disorientation'].mode()[0], inplace=True)


# In[19]:


ds.isnull().sum()


# In[20]:


#split train test
x=ds.drop(columns=['Diagnosis'])
y=ds['Diagnosis']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
#stratify ensures both train and test sets have balanced y distribution 


# In[21]:


#preprocessing
preprocessor=ColumnTransformer(transformers=[
    ('encoder', OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore'), nominal_categorical),
    ('scaler', StandardScaler(), numeric),
], remainder='passthrough')


# In[22]:


#pipeline for random forest
rf_pipeline=Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

svm_pipeline=Pipeline([
    ('preprocessor', preprocessor), 
    ('classifier', SVC())
])


# In[23]:


#training random forest
rf_pipeline.fit(x_train, y_train)
y_pred_rf=rf_pipeline.predict(x_test)
y_pred_rf


# In[24]:


x_train_transformed = rf_pipeline.named_steps['preprocessor'].transform(x_train)
feature_names = rf_pipeline.named_steps['preprocessor'].get_feature_names_out()
x_train_preprocessed = pd.DataFrame(x_train_transformed, columns=feature_names, index=x_train.index)
x_train_preprocessed.head()


# In[25]:


#training svm
svm_pipeline.fit(x_train, y_train)
y_pred_svm=svm_pipeline.predict(x_test)
y_pred_svm


# In[26]:


x_train_transformed = svm_pipeline.named_steps['preprocessor'].transform(x_train)
feature_names = svm_pipeline.named_steps['preprocessor'].get_feature_names_out()
x_train_preprocessed = pd.DataFrame(x_train_transformed, columns=feature_names, index=x_train.index)
x_train_preprocessed.head()


# In[27]:


#evaluate results with confusion matrix (random forest)
cm_rf=confusion_matrix(y_test, y_pred_rf)
cm_rf


# In[28]:


#evaluate results with confusion matrix (SVM)
cm_svm=confusion_matrix(y_test, y_pred_svm)
cm_svm


# In[29]:


#evalutate random forest model
accuracy_rf=accuracy_score(y_test, y_pred_rf)
precision_rf=precision_score(y_test, y_pred_rf)
recall_rf=recall_score(y_test, y_pred_rf)
f1_rf=f1_score(y_test, y_pred_rf)
roc_auc_rf = roc_auc_score(y_test, y_pred_rf)

print(f"random forest evaluation")
print(f"accuracy: {accuracy_rf}")
print(f"precision: {precision_rf}")
print(f"recall: {recall_rf}")
print(f"F1 score: {f1_rf}")
print(f"ROC AUC: {roc_auc_rf}")


# In[30]:


#evaluate SVM model
accuracy_svm = accuracy_score(y_test, y_pred_svm)
precision_svm = precision_score(y_test, y_pred_svm)
recall_svm = recall_score(y_test, y_pred_svm)
f1_svm = f1_score(y_test, y_pred_svm)
roc_auc_svm = roc_auc_score(y_test, y_pred_svm)

print(f"SVM evaluation")
print(f"accuracy: {accuracy_svm}")
print(f"precision: {precision_svm}")
print(f"recall: {recall_svm}")
print(f"F1 score: {f1_svm}")
print(f"ROC AUC: {roc_auc_svm}")


# In[31]:


#feature selection and hyperparameter tuning for random forest

rf_classifier=RandomForestClassifier(random_state=42)

#feature selection using model importance
rf_selector=SelectFromModel(rf_classifier, threshold='mean')

#hyperparameter tuning grid
param_grid_rf = {
    'tuning__n_estimators': [100, 200, 300],
    'tuning__max_depth': [None, 10, 20],
    'tuning__min_samples_split': [2, 5, 10],
    'tuning__min_samples_leaf': [1, 2, 4],
    'tuning__class_weight': [None, 'balanced']
}

#pipeline for the processes
pipeline_rf = Pipeline([
    ('preprocessor', preprocessor),
    ('feature_selection', rf_selector),
    ('tuning', rf_classifier)
])

#perform grid search
grid_search_rf = GridSearchCV(estimator=pipeline_rf, param_grid=param_grid_rf, cv=StratifiedKFold(5), scoring='f1', n_jobs=-1)

#fit model
grid_search_rf.fit(x_train, y_train)


# In[32]:


print("Best random forest Params:", grid_search_rf.best_params_)
print("Best cross-validation score: ", grid_search_rf.best_score_)


# In[33]:


#show selected features

#extract selected features
rf_features=grid_search_rf.best_estimator_.named_steps['preprocessor'].get_feature_names_out()
rf_selected_mask=grid_search_rf.best_estimator_.named_steps['feature_selection'].get_support()
rf_selected_features= rf_features[rf_selected_mask]

#extract feature importances
best_rf_model = grid_search_rf.best_estimator_.named_steps['tuning']
rf_importances = best_rf_model.feature_importances_

rf_importance_df=pd.DataFrame({
    'Feature': rf_selected_features,
    'Importance': rf_importances
}).sort_values(by='Importance', ascending=False)

print("Selected features for random forest:")
print(rf_importance_df)


# In[34]:


y_pred_rf_after=grid_search_rf.predict(x_test)
y_pred_rf_after


# In[35]:


cm_rf_after=confusion_matrix(y_test, y_pred_rf_after)
cm_rf_after


# In[36]:


#evaluate random forest model after feature selection and tuning
accuracy_rf_after = accuracy_score(y_test, y_pred_rf_after)
precision_rf_after = precision_score(y_test, y_pred_rf_after)
recall_rf_after = recall_score(y_test, y_pred_rf_after)
f1_rf_after = f1_score(y_test, y_pred_rf_after)
roc_auc_rf_after = roc_auc_score(y_test, y_pred_rf_after)

print(f"random forest evaluation after improvement")
print(f"accuracy: {accuracy_rf_after}")
print(f"precision: {precision_rf_after}")
print(f"recall: {recall_rf_after}")
print(f"F1 score: {f1_rf_after}")
print(f"ROC AUC: {roc_auc_rf_after}")


# In[37]:


#feature selection and hyperparameter tuning for SVM

#RFECV for feature selection
svm_selector=RFECV(estimator=SVC(kernel='linear'), cv=StratifiedKFold(5), scoring='f1')

#svm model for tuning
svc_classifier=SVC(kernel='rbf')

#hyperparameter grid for tuning
param_grid_svm = {
    'tuning__C': [0.1, 1, 10],
    'tuning__gamma': ['scale', 'auto', 0.1],
    'tuning__class_weight': [None, 'balanced']
}

#pipeline to process feature selection first then tuning
pipeline_svm = Pipeline([
    ('preprocessor', preprocessor),
    ('feature_selection', svm_selector),
    ('tuning', svc_classifier)
])

#perform grid search
grid_search_svm = GridSearchCV(estimator=pipeline_svm, param_grid=param_grid_svm, cv=StratifiedKFold(5), scoring='f1', n_jobs=-1)

#fit the improved model on dataset
grid_search_svm.fit(x_train, y_train)


# In[38]:


print("Best SVM Params:", grid_search_svm.best_params_)
print("Best cross-validation score: ", grid_search_svm.best_score_)


# In[90]:


#show selected features

#extract selected features
svm_features=grid_search_svm.best_estimator_.named_steps['preprocessor'].get_feature_names_out()
svm_selected_mask=grid_search_svm.best_estimator_.named_steps['feature_selection'].get_support() #mask for selected features only
svm_selected_features=svm_features[svm_selected_mask]

#extract feature importances
svm_ranking = grid_search_svm.best_estimator_.named_steps['feature_selection'].ranking_[svm_selected_mask]

svm_ranking_df=pd.DataFrame({
    'Feature': svm_selected_features,
    'Ranking': svm_ranking
})

print("Selected features for SVM:")
print(svm_ranking_df)


# In[40]:


y_pred_svm_after=grid_search_svm.predict(x_test)
y_pred_svm_after


# In[41]:


cm_svm_after=confusion_matrix(y_test, y_pred_svm_after)
cm_svm_after


# In[42]:


#evaluate SVM model after feature selection and tuning
accuracy_svm_after = accuracy_score(y_test, y_pred_svm_after)
precision_svm_after = precision_score(y_test, y_pred_svm_after)
recall_svm_after = recall_score(y_test, y_pred_svm_after)
f1_svm_after = f1_score(y_test, y_pred_svm_after)
roc_auc_svm_after = roc_auc_score(y_test, y_pred_svm_after)

print(f"SVM evaluation after fine tuning")
print(f"accuracy: {accuracy_svm_after}")
print(f"precision: {precision_svm_after}")
print(f"recall: {recall_svm_after}")
print(f"F1 score: {f1_svm_after}")
print(f"ROC AUC: {roc_auc_svm_after}")


# In[92]:


#random forest model chosen
joblib.dump(grid_search_rf.best_estimator_, "final_model.pkl")


# In[ ]:




