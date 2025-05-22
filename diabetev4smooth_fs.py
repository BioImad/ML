import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve, roc_auc_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from imblearn.over_sampling import SMOTE
from sklearn.inspection import permutation_importance
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

# ========================================
# 1. CARICAMENTO E ESPLORAZIONE DEI DATI
# ========================================
print("Caricamento del dataset...")
df = pd.read_csv(r"C:\Users\imada\Desktop\MACHINE_LEARNING\models\diabetes.csv")
print(f"Dimensioni del dataset: {df.shape}")

print("\n1. ESPLORAZIONE DEI DATI")
print("\nInformazioni sul dataset:")
df.info()

print("\nStatistiche descrittive:")
print(df.describe())

# Distribuzione della variabile target
outcome_counts = df["Outcome"].value_counts()
print(f"\nDistribuzione della variabile target:")
print(f"Non diabetici (0): {outcome_counts[0]} ({outcome_counts[0]*100/len(df):.1f}%)")
print(f"Diabetici (1): {outcome_counts[1]} ({outcome_counts[1]*100/len(df):.1f}%)")

# Visualizzazioni
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
plt.tight_layout()
plt.savefig('distribuzione_target.png')
plt.show()

# Matrice di correlazione
fig = create_figure(2, "Matrice di correlazione")
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='magma')
plt.tight_layout()
plt.savefig('matrice_correlazione.png')
plt.show()

# ========================================
# 2. PREPROCESSING DEI DATI
# ========================================
print("\n2. PREPROCESSING DEI DATI")

# Identificazione di valori sospetti (0)
print("\nConteggio di valori 0 nelle colonne (potenziali valori mancanti):")
cols_to_replace = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in cols_to_replace:
    zero_count = (df[col] == 0).sum()
    print(f"{col}: {zero_count} ({zero_count*100/len(df):.1f}%)")

# Sostituzione degli 0 con NaN
print("\nSostituzione degli 0 con NaN...")
df[cols_to_replace] = df[cols_to_replace].replace(0, np.NaN)

# IMPUTAZIONE UNICA (eliminando le ripetizioni)
print("\nImputazione dei valori mancanti con Random Forest...")
numeric_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                   'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

X_num = df[numeric_columns]
rf_imputer = IterativeImputer(
    estimator=RandomForestRegressor(n_estimators=10, random_state=0),
    max_iter=10,
    random_state=0
)
X_imputed = rf_imputer.fit_transform(X_num)
X_imputed_df = pd.DataFrame(X_imputed, columns=X_num.columns, index=X_num.index)

# Ricostruzione del DataFrame
df = pd.concat([X_imputed_df, df[['Outcome']]], axis=1)

# Trattamento outlier per Insulin (semplificato)
print("\nTrattamento degli outlier per Insulin...")
Q1 = df.Insulin.quantile(0.25)
Q3 = df.Insulin.quantile(0.75)
IQR = Q3 - Q1
upper = Q3 + 1.5 * IQR
df.loc[df['Insulin'] > upper, "Insulin"] = upper

# Creazione di categorie semplificate
print("\nCreazione di categorie...")
df['BMI_Category'] = pd.cut(
    df['BMI'],
    bins=[0, 25, 30, float('inf')],
    labels=['Normal', 'Overweight', 'Obese']
)

df['Glucose_Category'] = pd.cut(
    df['Glucose'],
    bins=[0, 100, 126, float('inf')],
    labels=['Normal', 'Prediabetes', 'Diabetes']
)

# ========================================
# 3. PREPARAZIONE E MODELLAZIONE
# ========================================
print("\n3. PREPARAZIONE E MODELLAZIONE")

# Definizione variabili
y = df['Outcome']
categorical_columns = ['BMI_Category', 'Glucose_Category']

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    df[numeric_columns + categorical_columns],
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Training set: {X_train.shape[0]} campioni")
print(f"Test set: {X_test.shape[0]} campioni")

# Pipeline di preprocessing
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(drop='first', sparse_output=False)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_columns),
        ('cat', categorical_transformer, categorical_columns)
    ]
)

# Pipeline completa
rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# MIGLIORAMENTO 1: Ottimizzazione degli iperparametri focalizzata sul recall
print("\nOttimizzazione degli iperparametri (focus su recall)...")
param_grid = {
    'classifier__n_estimators': [200, 300],
    'classifier__max_depth': [15, 20, None],
    'classifier__min_samples_split': [2, 5],
    'classifier__min_samples_leaf': [1, 2],
    'classifier__class_weight': ['balanced', {0: 1, 1: 2}]  # Peso maggiore alla classe minoritaria
}

# Grid search con F1 score per bilanciare precision e recall
grid_search = GridSearchCV(
    rf_pipeline,
    param_grid=param_grid,
    cv=3,
    scoring='f1',  # Cambiato da accuracy a f1
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print("\nMigliori iperparametri:")
for param, value in grid_search.best_params_.items():
    print(f"{param}: {value}")
print(f"Miglior F1 score in CV: {grid_search.best_score_:.4f}")

best_rf = grid_search.best_estimator_

# ========================================
# 4. VALUTAZIONE DEL MODELLO
# ========================================
print("\n4. VALUTAZIONE DEL MODELLO")

# Predizioni
y_pred = best_rf.predict(X_test)
y_prob = best_rf.predict_proba(X_test)[:, 1]

# MIGLIORAMENTO 2: Ottimizzazione della soglia di decisione
print("\nOttimizzazione della soglia di decisione...")
from sklearn.metrics import precision_recall_curve

precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
f1_scores = 2 * (precision * recall) / (precision + recall)
best_threshold_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_threshold_idx]

print(f"Migliore soglia trovata: {best_threshold:.3f}")
print(f"F1 score con nuova soglia: {f1_scores[best_threshold_idx]:.4f}")

# Predizioni con nuova soglia
y_pred_optimized = (y_prob >= best_threshold).astype(int)

# Metriche finali
print("\n=== RISULTATI FINALI ===")
print("\nCon soglia standard (0.5):")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")

print(f"\nCon soglia ottimizzata ({best_threshold:.3f}):")
print(f"Accuracy: {accuracy_score(y_test, y_pred_optimized):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_optimized):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_optimized):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred_optimized):.4f}")

print("\nReport di classificazione ottimizzato:")
print(classification_report(y_test, y_pred_optimized))

# Matrice di confusione
fig = create_figure(3, "Confronto Matrici di Confusione")

plt.subplot(1, 2, 1)
cm1 = confusion_matrix(y_test, y_pred)
sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Non diabetico', 'Diabetico'],
            yticklabels=['Non diabetico', 'Diabetico'])
plt.title('Soglia 0.5')

plt.subplot(1, 2, 2)
cm2 = confusion_matrix(y_test, y_pred_optimized)
sns.heatmap(cm2, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Non diabetico', 'Diabetico'],
            yticklabels=['Non diabetico', 'Diabetico'])
plt.title(f'Soglia {best_threshold:.3f}')

plt.tight_layout()
plt.savefig('confusion_matrices_comparison.png')
plt.show()

# Importanza delle feature
print("\nImportanza delle feature:")
feature_importance = best_rf.named_steps['classifier'].feature_importances_

# Ottieni i nomi delle feature dopo la trasformazione
num_features = numeric_columns
cat_features = best_rf.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out(categorical_columns)
all_features = list(num_features) + list(cat_features)

feature_importance_df = pd.DataFrame({
    'Feature': all_features,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)

print("\nTop 10 feature più importanti:")
print(feature_importance_df.head(10))

# Visualizzazione importanza
fig = create_figure(4, "Importanza delle Feature")
top_features = feature_importance_df.head(10)
plt.barh(range(len(top_features)), top_features['Importance'])
plt.yticks(range(len(top_features)), top_features['Feature'])
plt.xlabel('Importanza')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()

# ========================================
# 5. FUNZIONE DI PREVISIONE
# ========================================
def predict_diabetes(model, new_data, threshold=0.5):
    """
    Effettua previsioni su nuovi dati.
    
    Args:
        model: Modello addestrato
        new_data: DataFrame con i dati del paziente
        threshold: Soglia di decisione (default 0.5)
    
    Returns:
        Dizionario con probabilità, predizione e livello di rischio
    """
    required_cols = numeric_columns + categorical_columns
    
    # Controllo colonne
    missing_cols = [col for col in required_cols if col not in new_data.columns]
    if missing_cols:
        raise ValueError(f"Colonne mancanti: {missing_cols}")
    
    # Previsione
    prob = model.predict_proba(new_data[required_cols])[0, 1]
    pred = 1 if prob >= threshold else 0
    
    # Livello di rischio
    if prob < 0.3:
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

# Esempio di utilizzo
print("\n=== ESEMPIO DI UTILIZZO ===")
sample_patient = X_test.iloc[0:1].copy()
result = predict_diabetes(best_rf, sample_patient, threshold=best_threshold)

print("Dati del paziente:")
for col in numeric_columns:
    print(f"{col}: {sample_patient[col].iloc[0]:.2f}")

print(f"\nProbabilità di diabete: {result['probability']:.4f}")
print(f"Predizione: {'Diabetico' if result['prediction'] == 1 else 'Non diabetico'}")
print(f"Livello di rischio: {result['risk_level']}")
print(f"Valore reale: {'Diabetico' if y_test.iloc[0] == 1 else 'Non diabetico'}")

# Salvataggio del modello
print("\nSalvataggio del modello...")
joblib.dump(best_rf, 'diabetes_model.pkl')
joblib.dump(best_threshold, 'optimal_threshold.pkl')
print("Modello salvato come 'diabetes_model.pkl'")
print("Soglia ottimale salvata come 'optimal_threshold.pkl'")