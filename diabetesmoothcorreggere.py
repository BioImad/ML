import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib
import warnings
warnings.simplefilter(action='ignore')
sns.set()
plt.style.use("ggplot")

# Funzione per generare un report con figure
def create_figure(fig_num, title):
    plt.figure(figsize=(10, 6))
    plt.suptitle(title, fontsize=16)
    return plt

# Caricamento del dataset
print("Caricamento del dataset...")
df = pd.read_csv(r"C:\Users\imada\Desktop\MACHINE_LEARNING\models\diabetes.csv")
print(f"Dimensioni del dataset: {df.shape}")


print("\n1. ESPLORAZIONE DEI DATI")

# Visualizzazione delle prime righe
print("\nPrime 5 righe del dataset:")
print(df.head())

# Informazioni sul dataset
print("\nInformazioni sul dataset:")
df.info()

# Statistiche descrittive
print("\nStatistiche descrittive:")
df.describe()


# Distribuzione della variabile target
outcome_counts = df["Outcome"].value_counts()
print(f"\nDistribuzione della variabile target:")
print(f"Non diabetici (0): {outcome_counts[0]} ({outcome_counts[0]*100/len(df):.1f}%)")
print(f"Diabetici (1): {outcome_counts[1]} ({outcome_counts[1]*100/len(df):.1f}%)")

# Distribuzione grafica della variabile target
fig = create_figure(1, "Distribuzione dei casi di diabete")
plt.subplot(1, 2, 1)
df['Outcome'].value_counts().plot.pie(
    explode=[0, 0.1],
    autopct="%1.1f%%",
    shadow=True,
    labels=['Non diabetico', 'Diabetico']
)
plt.ylabel('')

plt.subplot(1, 2, 2)
sns.countplot(x='Outcome', data=df)
plt.xticks([0, 1], ['Non diabetico', 'Diabetico'])
plt.xlabel('Outcome')
plt.ylabel('Conteggio')
plt.savefig('distribuzione_target.png')


# Visualizzazione della distribuzione delle variabili
fig = create_figure(2, "Distribuzione delle variabili numeriche")
cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
for i, col in enumerate(cols):
    plt.subplot(4, 2, i+1)
    sns.histplot(df[col], kde=True, color="red")
    plt.xlabel(col)
    plt.ylabel('Frequenza')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('distribuzione_variabili.png')

# Matrice di correlazione
fig = create_figure(3, "Matrice di correlazione")
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='magma')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('matrice_correlazione.png')


# Differenze tra gruppi
print("\nDifferenze tra diabetici e non diabetici:")
group_comparison = df.groupby("Outcome").agg({
    'Pregnancies': ['mean', 'median'],
    'Glucose': ['mean', 'median'],
    'BMI': ['mean', 'median'],
    'Age': ['mean', 'median']
})
print(group_comparison.to_string())

print("\n2. PREPROCESSING DEI DATI")

# Identificazione di valori sospetti (0) in colonne dove non dovrebbero esistere
print("\nConteggio di valori 0 nelle colonne (potenziali valori mancanti):")
for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
    zero_count = (df[col] == 0).sum()
    print(f"{col}: {zero_count} ({zero_count*100/len(df):.1f}%)")

# Sostituzione degli 0 con NaN per le colonne rilevanti
print("\nSostituzione degli 0 con NaN per le colonne dove 0 non è un valore plausibile...")
cols_to_replace = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_to_replace] = df[cols_to_replace].replace(0, np.NaN)

# Conteggio dei valori mancanti
print("\nValori mancanti dopo la sostituzione:")
print(df.isnull().sum())


# Imputazione avanzata con Random Forest
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

# Applichiamo Random Forest Imputation
rf_imputer = IterativeImputer(
    estimator=RandomForestRegressor(n_estimators=10, random_state=0),
    max_iter=10,
    random_state=0
)
X = df.drop(columns="Outcome")  # cambia "Outcome" con il nome corretto della tua colonna target, se diverso

X_imputed = rf_imputer.fit_transform(X)

# Se X era un DataFrame, convertiamo di nuovo
X_imputed = pd.DataFrame(X_imputed, columns=X.columns, index=X.index)

print("\nRilevamento degli outlier con il metodo IQR:")
# Salta colonne non numeriche o categoriali
numeric_features = df.select_dtypes(include=['float64', 'int64']).columns
for feature in numeric_features:
    # Salta la colonna Outcome e is_outlier
    if feature in ['Outcome', 'is_outlier']:
        continue
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    n_outliers = sum((df[feature] < lower) | (df[feature] > upper))
    if n_outliers > 0:
        print(f"{feature}: {n_outliers} outlier rilevati ({n_outliers*100/len(df):.1f}%)")


 #Trattamento degli outlier per Insulin (solo come esempio)
print("\nTrattamento degli outlier per Insulin...")
fig = create_figure(4, "Boxplot di Insulin prima del trattamento degli outlier")
sns.boxplot(x=df["Insulin"], color="red")
plt.xlabel("Insulin")
plt.savefig('insulin_prima.png')

Q1 = df.Insulin.quantile(0.25)
Q3 = df.Insulin.quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
print(f"Limite superiore per Insulin: {upper}")
df.loc[df['Insulin'] > upper, "Insulin"] = upper

fig = create_figure(5, "Boxplot di Insulin dopo il trattamento degli outlier")
sns.boxplot(x=df["Insulin"], color="green")
plt.xlabel("Insulin")
plt.savefig('insulin_dopo.png')


# Imputazione avanzata con Random Forest
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

# Applichiamo Random Forest Imputation
rf_imputer = IterativeImputer(
    estimator=RandomForestRegressor(n_estimators=10, random_state=0),
    max_iter=10,
    random_state=0
)

X_imputed = rf_imputer.fit_transform(X)

# Se X era un DataFrame, convertiamo di nuovo
X_imputed = pd.DataFrame(X_imputed, columns=X.columns, index=X.index)



from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

# Colonne numeriche usate per l'imputazione
numeric_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                   'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

# Separiamo le feature numeriche e la target
X_num = df[numeric_columns]

# Imputazione avanzata
rf_imputer = IterativeImputer(
    estimator=RandomForestRegressor(n_estimators=10, random_state=0),
    max_iter=10,
    random_state=0
)
X_imputed = rf_imputer.fit_transform(X_num)

# Ricreiamo il DataFrame numerico imputato
X_imputed_df = pd.DataFrame(X_imputed, columns=X_num.columns, index=X_num.index)

# Reinseriamo tutte le altre colonne non numeriche (categoriche + target)
df = pd.concat([X_imputed_df, df.drop(columns=numeric_columns)], axis=1)




print("\nCreazione di categorie di BMI...")
df['BMI_Category'] = pd.cut(
    df['BMI'],
    bins=[0, 18.5, 24.9, 29.9, 34.9, 39.9, float('inf')],
    labels=['Underweight', 'Normal', 'Overweight', 'Obesity 1', 'Obesity 2', 'Obesity 3']
)

# Creazione di categorie per Insulin
print("\nCreazione di categorie per Insulin...")
df['Insulin_Category'] = pd.cut(
    df['Insulin'],
    bins=[0, 25, 50, float('inf')],
    labels=['Normal', 'Elevated', 'High']
)

# Creazione di categorie per Glucose
print("\nCreazione di categorie per Glucose...")
df['Glucose_Category'] = pd.cut(
    df['Glucose'],
    bins=[0, 70, 100, 126, float('inf')],
    labels=['Low', 'Normal', 'Prediabetes', 'Diabetes']




print("\nDistribuzione delle nuove categorie (%):")

categorical_columns = ['BMI_Category', 'Insulin_Category', 'Glucose_Category']

for col in categorical_columns:
    print(f"\n{col}:")
    counts = df[col].value_counts(normalize=True) * 100
    # Trasforma in DataFrame per visualizzazione tabellare pulita
    counts_df = pd.DataFrame(counts).reset_index()
    counts_df.columns = [col, 'Percentuale']
    counts_df['Percentuale'] = counts_df['Percentuale'].map("{:.2f}%".format)
    display(counts_df)


print("\n5. PREPARAZIONE DEI DATI PER IL MODELLO")

# Definizione delle colonne di input e target
print("\nPreparazione delle variabili di input e target...")
y = df['Outcome']

# Separazione delle colonne numeriche e categoriche
numeric_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
categorical_columns = ['BMI_Category', 'Insulin_Category', 'Glucose_Category']

# Divisione in train e test set
print("\nDivisione in train e test set...")
X_train, X_test, y_train, y_test = train_test_split(
    df[numeric_columns + categorical_columns],
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Preprocessing separato prima di SMOTE
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(drop='first', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_columns),
        ('cat', categorical_transformer, categorical_columns)
    ]
)

# Fit + transform su X_train
X_train_encoded = preprocessor.fit_transform(X_train)
X_test_encoded = preprocessor.transform(X_test)  # per dopo

# ✅ Ora possiamo usare SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_encoded, y_train)

print("Dati bilanciati con SMOTE:")
print(pd.Series(y_train_resampled).value_counts())

print("\n6. ADDESTRAMENTO DEL MODELLO RANDOM FOREST")

# Pipeline completa con Random Forest
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Cross-validation per valutare la performance del modello base
print("\nValutazione del modello con cross-validation...")
cv = KFold(n_splits=3, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf_pipeline, X_train, y_train, cv=cv, scoring='accuracy')
print(f"Accuracy con 3-fold CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Ottimizzazione degli iperparametri
print("\nOttimizzazione degli iperparametri...")
param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [10, 15, 20],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4],
    'classifier__max_features': ['sqrt', 'log2']
}
print("\nIperparametri ottimizzati")
from imblearn.over_sampling import SMOTE

grid_search = GridSearchCV(
    rf_pipeline,
    param_grid=param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

# Migliori iperparametri
print("\nMigliori iperparametri trovati:")
print(grid_search.best_params_)
print(f"Miglior accuracy in CV: {grid_search.best_score_:.4f}")

# Modello finale con i migliori iperparametri
best_rf = grid_search.best_estimator_




print("\n7. VALIDAZIONE DEL MODELLO")

# Validazione con k-fold cross validation
print("\nValidazione con 5-fold cross validation...")
cv = KFold(n_splits=5, shuffle=True, random_state=42)
scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

for metric in scoring_metrics:
    cv_scores = cross_val_score(best_rf, X_train, y_train, cv=cv, scoring=metric)
    print(f"{metric.capitalize()}: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Valutazione del modello sul test set
print("\nValutazione del modello sul test set:")
y_pred = best_rf.predict(X_test)
y_prob = best_rf.predict_proba(X_test)[:, 1]

# Metriche di performance
accuracy = accuracy_score(y_test, y_pred)
from sklearn.metrics import precision_score, recall_score, f1_score
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC AUC Score: {roc_auc:.4f}")


# Matrice di confusione
print("\nMatrice di confusione:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Visualizzazione della matrice di confusione
fig = create_figure(6, "Matrice di confusione")
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Non diabetico', 'Diabetico'],
            yticklabels=['Non diabetico', 'Diabetico'])
plt.xlabel('Predetto')
plt.ylabel('Effettivo')
plt.tight_layout()
plt.savefig('confusion_matrix.png')

print("\nReport di classificazione:")
print(classification_report(y_test, y_pred))

# Curva ROC
fig = create_figure(7, "Curva ROC")
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr, label=f'Random Forest (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.savefig('roc_curve.png')


print("\nCalcolo dell'importanza delle feature con analisi delle permutazioni...")

from sklearn.inspection import permutation_importance

# Preprocessing manuale del test set
X_test_transformed = best_rf.named_steps['preprocessor'].transform(X_test)

# Calcola importanza permutazioni sul classificatore (senza preprocessore)
perm_importance = permutation_importance(best_rf.named_steps['classifier'],
                                        X_test_transformed,
                                        y_test,
                                        n_repeats=10,
                                        random_state=42)
preprocessor = best_rf.named_steps['preprocessor']
num_features = preprocessor.transformers_[0][2]
cat_features = preprocessor.transformers_[1][2]
ohe = preprocessor.transformers_[1][1].named_steps['onehot']
cat_feature_names = ohe.get_feature_names_out(cat_features)
all_features = list(num_features) + list(cat_feature_names)
# Controllo lunghezza feature
if len(all_features) != len(perm_importance.importances_mean):
    raise ValueError(f"Mismatch: {len(all_features)} features names vs {len(perm_importance.importances_mean)} permutation importances")

perm_importance_df = pd.DataFrame({
    'Feature': all_features,
    'Importance': perm_importance.importances_mean,
    'Std': perm_importance.importances_std
}).sort_values('Importance', ascending=False)

print("\nTop 10 feature secondo l'analisi delle permutazioni:")
print(perm_importance_df.head(10))

fig = create_figure(9, "Importanza delle feature (analisi delle permutazioni)")
plt.errorbar(
    x=perm_importance_df['Importance'].head(10),
    y=range(10),
    xerr=perm_importance_df['Std'].head(10),
    fmt='o'
)
plt.yticks(range(10), perm_importance_df['Feature'].head(10))
plt.xlabel('Importanza (riduzione della performance)')
plt.title('Importanza delle feature (analisi delle permutazioni)')
plt.tight_layout()
plt.savefig('permutation_importance.png')



# PDPs (Partial Dependence Plots) per le feature più importanti
try:
    print("\nCreazione dei Partial Dependence Plots (PDP) per le feature più importanti...")
    from sklearn.inspection import PartialDependenceDisplay

    # Troviamo le feature numeriche più importanti
    top_numeric_features = [f for f in feature_importance_df['Feature']
                           if f in numeric_columns][:3]

    fig, ax = plt.subplots(figsize=(12, 6))
    PartialDependenceDisplay.from_estimator(
        best_rf,
        X_test,
        top_numeric_features,
        kind="average",
        ax=ax
    )
    plt.tight_layout()
    plt.savefig('pdp_plots.png')
except Exception as e:
    print(f"Impossibile creare i PDP plots: {e}")



print("\n10. FUNZIONE DI PREVISIONE PER NUOVI DATI")

def predict_diabetes(model, new_data):
    """
    Effettua previsioni su nuovi dati.

    Args:
        model: Modello addestrato
        new_data: DataFrame con i dati del paziente

    Returns:
        Probabilità di diabete e predizione
    """
    # Assicurarsi che tutte le colonne necessarie siano presenti
    required_cols = numeric_columns + categorical_columns
    for col in required_cols:
        if col not in new_data.columns:
            raise ValueError(f"Colonna mancante: {col}")

    # Effettuare la previsione
    prob = model.predict_proba(new_data[required_cols])[0, 1]
    pred = 1 if prob >= 0.5 else 0

    # Calcolo del rischio
    if prob < 0.2:
        risk_level = "Basso"
    elif prob < 0.5:
        risk_level = "Moderato"
    elif prob < 0.7:
        risk_level = "Alto"
    else:
        risk_level = "Molto alto"

    return {
        "probability": prob,
        "prediction": pred,
        "risk_level": risk_level
         }


# Esempio di utilizzo della funzione di previsione in Jupyter Notebook

print("\nEsempio di utilizzo della funzione di previsione:")

sample_patient = X_test.iloc[0:1].copy()  # Primo paziente del test set come esempio

result = predict_diabetes(best_rf, sample_patient)

print("Dati del paziente:")
display(sample_patient[numeric_columns].T)  # Trasposto per visualizzazione verticale pulita

print(f"Probabilità di diabete: {result['probability']:.4f}")
print(f"Predizione: {'Diabetico' if result['prediction'] == 1 else 'Non diabetico'}")
print(f"Livello di rischio: {result['risk_level']}")
print(f"Valore reale: {'Diabetico' if y_test.iloc[0] == 1 else 'Non diabetico'}")
