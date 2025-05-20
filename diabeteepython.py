import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale, StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score
from sklearn.metrics import confusion_matrix,accuracy_score,mean_squared_error,r2_score,roc_auc_score,roc_curve,classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import KFold
import warnings
warnings.simplefilter(action='ignore')
sns.set()
plt.style.use("ggplot")


df= pd.read_csv('diabetes.csv')
df.head()
df.info()

# independent feature->'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
#        'BMI', 'DiabetesPedigreeFunction', 'Age'
# dependent feature-> outcome

# descriptive statistics of the dataset
df.describe()

df.shape
#Abbiamo 768 Righe e 9 colonne

df["Outcome"].value_counts()*100/len(df)
#Calcoliamo la distribuzione di soggetti sani e malati
#0 = NON MALATO      1 = MALATO

#Istogramma dell'età
plt.figure(figsize = (10,10))
plt.xlabel("Age",fontsize=10)
plt.ylabel("Count",fontsize=10)
df["Age"].hist(edgecolor="black")

print(df["Age"].max(),df["Age"].min())


#density graph
fig,ax = plt.subplots(4,2, figsize=(20,20))
sns.distplot(df.Pregnancies, bins=20, ax=ax[0,0], color="red")
sns.distplot(df.Glucose, bins=20, ax=ax[0,1], color="red")
sns.distplot(df.BloodPressure, bins=20, ax=ax[1,0], color="red")
sns.distplot(df.SkinThickness, bins=20, ax=ax[1,1], color="red")
sns.distplot(df.Insulin, bins=20, ax=ax[2,0], color="red")
sns.distplot(df.BMI, bins=20, ax=ax[2,1], color="red")
sns.distplot(df.DiabetesPedigreeFunction, bins=20, ax=ax[3,0], color="red")
sns.distplot(df.Age, bins=20, ax=ax[3,1], color="red")

#Raggruppiamo il df in base all'Outcome e calcoliamo la media delle gravidanze e del glucosio
pregnancies_mean = df.groupby("Outcome").agg({'Pregnancies': 'mean'})
glucose_mean = df.groupby("Outcome").agg({'Glucose': 'mean'})

# Unione orizzontale (per indice, cioè "Outcome")
pd.concat([pregnancies_mean, glucose_mean], axis=1)


df.groupby("Outcome").agg({'Glucose': "max"})


f, ax = plt.subplots(1, 2, figsize=(18, 8))

# Pie chart
df['Outcome'].value_counts().plot.pie(
    explode=[0, 0.1],
    autopct="%1.1f%%",
    ax=ax[0],
    shadow=True
)
ax[0].set_title('Target')
ax[0].set_ylabel('')

# Count plot
sns.countplot(x='Outcome', data=df, ax=ax[1])
ax[1].set_title('Outcome')

plt.show()



df.corr()

f,ax = plt.subplots(figsize=[10,5])
sns.heatmap(df.corr(), annot=True, fmt = '.2f', ax=ax, cmap='magma')
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()

#Sostituiamo "Nan" values con 0
df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age']] = df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age']].replace(0, np.NaN)

# Vediamo quanti missing values ci sono
df.isnull().sum()

import missingno as msno
msno.bar(df,color='red')

#riempiamio i valori nulli con la mediana
def median_target(var):
    temp = df[df[var].notnull()]
    temp = temp[[var, 'Outcome']].groupby(['Outcome'])[[var]].median().reset_index()
    return temp

columns = df.columns
columns = columns.drop("Outcome")  # correzione qui: usa sempre "columns"

for i in columns:
    temp = median_target(i)
    df.loc[(df['Outcome'] == 0) & (df[i].isnull()), i] = temp.loc[temp['Outcome'] == 0, i].values[0]
    df.loc[(df['Outcome'] == 1) & (df[i].isnull()), i] = temp.loc[temp['Outcome'] == 1, i].values[0]



df.head()

df.isnull().sum()

p = sns.pairplot(df, hue="Outcome", height=2)


#detectiamo gli outliers mediante il IQR
for feature in df:
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3-Q1
    lower = Q1-1.5*IQR
    upper = Q3+1.5*IQR
    if df[(df[feature]>upper)].any(axis=None):
        print(feature, "yes")
    else:
        print(feature, "no")

plt.figure(figsize=(8,7))
sns.boxplot(x=df["Insulin"], color="red")

Q1= df.Insulin.quantile(0.25)
Q3=df.Insulin.quantile(0.75)
IQR = Q3-Q1
lower = Q1-1.5*IQR
upper = Q3+1.5*IQR
df.loc[df['Insulin']>upper, "Insulin"] = upper


#lOCAL OUTLIER FACTOR
from sklearn.neighbors import LocalOutlierFactor
lof = LocalOutlierFactor(n_neighbors=10)
lof.fit_predict(df)

df.head()

plt.figure(figsize=(8,7))
sns.boxplot(x= df["Pregnancies"], color="red")

df_scores = lof.negative_outlier_factor_
np.sort(df_scores)[0:20]

thresold = np.sort(df_scores)[7]
outlier = df_scores>thresold
df = df[outlier]

df.head()

df.shape

plt.figure(figsize=(8,7))
sns.boxplot(x= df["Pregnancies"], color="red")

NewBMI = pd.Series(["Underweight","Normal", "Overweight","Obesity 1", "Obesity 2", "Obesity 3"], dtype = "category")


NewBMI

df['NewBMI'] = NewBMI
df.loc[df["BMI"]<18.5, "NewBMI"] = NewBMI[0]
df.loc[(df["BMI"]>18.5) & df["BMI"]<=24.9, "NewBMI"] = NewBMI[1]
df.loc[(df["BMI"]>24.9) & df["BMI"]<=29.9, "NewBMI"] = NewBMI[2]
df.loc[(df["BMI"]>29.9) & df["BMI"]<=34.9, "NewBMI"] = NewBMI[3]
df.loc[(df["BMI"]>34.9) & df["BMI"]<=39.9, "NewBMI"] = NewBMI[4]
df.loc[df["BMI"]>39.9, "NewBMI"] = NewBMI[5]

df.tail()

def set_insuline(row):
    if row["Insulin"]>=16 and row["Insulin"]<=166:
        return "Normal"
    else:
        return "Abnormal"

df = df.assign(NewInsulinScore=df.apply(set_insuline, axis=1))

df.head()

# Some intervals were determined according to the glucose variable and these were assigned categorical variables.
# Definisco le categorie
categories = ["Low", "Normal", "Prediabetes", "Diabetes"]

# Inizializzo la colonna "NewGlucose" come categoria con valori NaN
df["NewGlucose"] = pd.Series(pd.Categorical([None] * len(df), categories=categories))

# Assegno le categorie in base ai valori della colonna "Glucose"
df.loc[df["Glucose"] <= 70, "NewGlucose"] = "Low"
df.loc[(df["Glucose"] > 70) & (df["Glucose"] <= 99), "NewGlucose"] = "Normal"
df.loc[(df["Glucose"] >= 100) & (df["Glucose"] <= 125), "NewGlucose"] = "Prediabetes"
df.loc[df["Glucose"] >= 126, "NewGlucose"] = "Diabetes"

df.head()


print([col for col in df.columns if col.startswith("NewGlucose")])


categorical_df = df[['NewBMI_Obesity 1', 'NewBMI_Obesity 2', 'NewBMI_Obesity 3', 'NewBMI_Overweight',
                     'NewBMI_Underweight', 'NewInsulinScore_Normal',
                     'NewGlucose_Normal', 'NewGlucose_Prediabetes', 'NewGlucose_Diabetes']]


categorical_df = categorical_df.astype(int)
categorical_df.head()

y = df['Outcome']

X = df.drop(['Outcome', 'NewBMI_Obesity 1', 'NewBMI_Obesity 2', 'NewBMI_Obesity 3',
             'NewBMI_Overweight', 'NewBMI_Underweight', 'NewInsulinScore_Normal',
             'NewGlucose_Normal', 'NewGlucose_Prediabetes', 'NewGlucose_Diabetes'], axis=1)


cols = X.columns
index = X.index

X.head()

from sklearn.preprocessing import RobustScaler
transformer = RobustScaler().fit(X)
X=transformer.transform(X)
X=pd.DataFrame(X, columns = cols, index = index)

X = pd.concat([X, categorical_df], axis=1)

X.head()

X_train, X_test, y_train , y_test = train_test_split(X,y, test_size=0.2, random_state=0)

scaler =StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

DT = DecisionTreeClassifier()
DT.fit(X_train, y_train)
y_pred = DT.predict(X_test)
print(accuracy_score(y_train, DT.predict(X_train)))

print(accuracy_score(y_test, DT.predict(X_test)))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# hyperparameter tuning of dt
grid_param = {
    'criterion':['gini','entropy'],
    'max_depth' :  [3,5,7,10],
    'splitter' : ['best','radom'],
    'min_samples_leaf':[1,2,3,5,7],
    'min_samples_split':[1,2,3,5,7],
    'max_features':['auto','sqrt','log2']
}
grid_search_dt = GridSearchCV(DT, grid_param, cv=50, n_jobs=-1, verbose = 1)
grid_search_dt.fit(X_train, y_train)

rand_clf = RandomForestClassifier(criterion = 'entropy', max_depth = 15, max_features = 0.75, min_samples_leaf = 2, min_samples_split = 3, n_estimators = 130)
rand_clf.fit(X_train, y_train)

y_pred = rand_clf.predict(X_test)

y_pred = rand_clf.predict(X_test)
print(accuracy_score(y_train, rand_clf.predict(X_train)))
rand_acc = accuracy_score(y_test, rand_clf.predict(X_test))
print(accuracy_score(y_test, rand_clf.predict(X_test)))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

