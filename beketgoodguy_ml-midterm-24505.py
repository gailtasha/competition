# # Midterm by Manabayev Beket (24505)
# 
# # P1: K-Nearest Neighbors


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

train_df = pd.read_csv("C:/Users/MBK/Desktop/Machine Learning/santander-customer-transaction-prediction/train.csv")
test_df = pd.read_csv("C:/Users/MBK/Desktop/Machine Learning/santander-customer-transaction-prediction/test.csv")


print(train_df.sample(1))

print(test_df.sample(1))


train_df.info()


train_df.describe()


train_df.corr()


import seaborn as sns

sns.countplot(x='target', data=train_df)


train_df[train_df.columns[1:]].std().plot('hist')


train_df[train_df.columns[1:]].mean().plot('hist')


sns.distplot(train_df.var_0, kde=False) #blue
sns.distplot(train_df.var_5, kde=False) #orange
sns.distplot(train_df.var_10, kde=False) #green
sns.distplot(train_df.var_15, kde=False) #pink
sns.distplot(train_df.var_20, kde=False) #purple


corr = train_df.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(train_df.columns),1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(train_df.columns)
ax.set_yticklabels(train_df.columns)
plt.show()


#corr_matrix = train_df.corr(train_df.target)
a=train_df.corr()

with pd.option_context('display.max_rows', None, 'display.max_columns', None): 
  print(a["target"].sort_values(ascending=False))


train_df_new = train_df.drop(['var_40','var_184','var_78','var_170','var_191','var_94','var_67','var_18','var_173','var_164','var_118','var_147','var_91','var_89','var_95','var_155','var_35','var_71','var_106','var_162','var_157','var_48','var_163','var_180','var_5','var_119','var_145','var_167','var_49','var_32','var_130','var_90','var_24','var_195','var_125','var_135','var_52','var_151','var_137','var_128','var_70','var_111','var_51','var_105','var_199','var_112','var_196','var_66','var_11','var_82','var_175','var_144','var_74','var_8','var_138','var_15','var_134','var_55','var_140','var_159','var_97','var_187','var_171','var_168','var_62','var_181','var_25','var_84','var_19','var_65','var_3','var_4','var_189','var_47','var_69','var_16','var_37','var_79','var_176','var_61','var_60','var_46','var_29','var_124','var_161','var_96','var_117','var_100','var_126','var_38','var_17','var_30','var_185','var_27','var_41','var_103','var_10','var_7','var_136','var_158','var_98','var_39','var_160','var_183','var_129','var_14','var_73','var_153','var_182','var_42','var_101','var_59','var_152','var_120','var_143','var_68','var_72','var_113','var_64','var_50','var_63','var_57','var_54','var_77','var_193','var_20','var_102','var_142','var_178','var_45','var_83','var_88','var_156','var_194','var_116','var_28','var_58','var_132','var_85','var_23','var_31','var_150','var_114','var_104','var_43','var_141','var_186','var_131','var_188','var_56','var_93','var_197','var_87','var_177','var_172','var_75','var_36','var_127','var_86','var_121','var_107','var_123','var_122','var_9','var_192','var_33','var_108','var_154','var_92','var_149','var_169','var_44','var_109'], axis = 1)


train_df_new.sample(1)


print (train_df_new.isna().sum())
print (train_df_new.isnull().any().any())


X, y = train_df_new.iloc[:,2:], train_df_new.iloc[:,1]


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state = 47, stratify = y)


from sklearn.neighbors import KNeighborsClassifier


knn = KNeighborsClassifier(11)


knn.fit(X_train, y_train)


X_test1 = np.nan_to_num(X_test)


y_preds = knn.predict(X_test1)


from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report, confusion_matrix


accuracy_score(y_test, y_preds)


cm = confusion_matrix(y_test,y_preds)
cm


print(classification_report(y_test, y_preds))


X.shape


knn_res = KNeighborsClassifier(n_jobs=-1, n_neighbors=7)


knn_res.fit(X_train, y_train)


y_preds_res = knn_res.predict(X)


submission_knn = pd.DataFrame({
    "ID_code": test_df["ID_code"],
    "target": y_preds_res
})
submission_knn.to_csv('submission_knn.csv', index=False)


# # P2: Logistic Regression


import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelBinarizer
from sklearn import metrics


train_df = pd.read_csv("C:/Users/MBK/Desktop/Machine Learning/santander-customer-transaction-prediction/train.csv")
test_df = pd.read_csv("C:/Users/MBK/Desktop/Machine Learning/santander-customer-transaction-prediction/test.csv")


train_df.sample(1)


X, y = train_df_new.iloc[:,2:], train_df_new.iloc[:,1]


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=0)


logreg = LogisticRegression()
logreg.fit(X_train,y_train)


y_pred=logreg.predict(X_test)


cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred, average = 'macro'))
print("Recall:",metrics.recall_score(y_test, y_pred, average = 'macro'))


metrics.precision_recall_fscore_support(y_test, y_pred, average='weighted')


y_preds_res = logreg.predict(X)


accuracy_score(y, y_preds_res)


cm = confusion_matrix(y,y_preds_res)
cm


submission_logereg = pd.DataFrame({
    "ID_code": test_df["ID_code"],
    "target": y_preds_res
})
submission_logereg.to_csv('submission_logereg.csv', index=False)


# # P3: Support Vector Machines


import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelBinarizer
from sklearn import metrics


train_df = pd.read_csv("C:/Users/MBK/Desktop/Machine Learning/santander-customer-transaction-prediction/train.csv")
test_df = pd.read_csv("C:/Users/MBK/Desktop/Machine Learning/santander-customer-transaction-prediction/test.csv")


X, y = train_df_new.iloc[:,2:], train_df_new.iloc[:,1]


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25)


from sklearn.svm import SVC
svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)


# # Unfortunately, both linear and non-linear SVM get stuck here (been waiting for 2.5+ hours, but only wasted time)


# # P4: Naive Bayes


#Import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB

#Create a Gaussian Classifier
gnb = GaussianNB()

#Train the model using the training sets
gnb.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = gnb.predict(X_test)


from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report, confusion_matrix


accuracy_score(y_test, y_pred)


cm = confusion_matrix(y_test,y_pred)
cm


print(classification_report(y_test, y_pred))


y_preds_res = gnb.predict(X)


submission_nb = pd.DataFrame({
    "ID_code": test_df["ID_code"],
    "target": y_preds_res
})
submission_nb.to_csv('submission_nb.csv', index=False)


# # P5: Decision Tree


from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

clf = tree.DecisionTreeClassifier(max_depth=10)
clf = clf.fit(X, y)


y_preds_clf = clf.predict(X)


accuracy_score(y, y_preds_clf)


cm = confusion_matrix(y, y_preds_clf)
cm


print(classification_report(y, y_preds_clf))


submission_tree = pd.DataFrame({
    "ID_code": test_df["ID_code"],
    "target": y_preds_clf
})
submission_tree.to_csv('submission_tree.csv', index=False)


# # P6: Random Forest


from sklearn.ensemble import RandomForestClassifier


clf = RandomForestClassifier(n_jobs=-1, max_depth=30, random_state=0)


clf.fit(X_train, y_train)


y_pred = clf.predict(X_test)


accuracy_score(y_test, y_pred)


cm = confusion_matrix(y_test,y_pred)
cm


print(classification_report(y_test, y_pred))


y_preds = clf.predict(X)


accuracy_score(y, y_preds_res)


cm = confusion_matrix(y, y_preds_res)
cm


print(classification_report(y, y_preds_res))


submission_forest = pd.DataFrame({
    "ID_code": test_df["ID_code"],
    "target": y_preds_res
})
submission_forest.to_csv('submission_forest.csv', index=False)


# # P7: XGBoost 


from sklearn.multiclass import OneVsRestClassifier
from xgboost import XGBClassifier

clf = OneVsRestClassifier(XGBClassifier(n_jobs=-1, max_depth=30))


clf.fit(X_train, y_train)


y_pred = clf.predict(X_test)


accuracy_score(y_test, y_pred)


cm = confusion_matrix(y_test,y_pred)
cm


print(classification_report(y_test, y_pred))


y_preds_res = clf.predict(X)


accuracy_score(y, y_preds_res)


cm = confusion_matrix(y, y_preds_res)
cm


print(classification_report(y, y_preds_res))


submission_xgboost = pd.DataFrame({
    "ID_code": test_df["ID_code"],
    "target": y_preds_res
})
submission_xgboost.to_csv('submission_xgboost.csv', index=False)



