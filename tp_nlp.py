# tp_nlp.py - Classification de Textes
# Copie-colle ce code ENTIER

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings
warnings.filterwarnings('ignore')

# NLTK
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Vérifier les données NLTK
try:
    nltk.data.find('tokenizers/punkt')
except:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except:
    nltk.download('wordnet')

# ============================================
# CHARGEMENT DES DONNÉES
# ============================================
print("="*50)
print("TP CLASSIFICATION DE TEXTES")
print("="*50)

print("\n1. Chargement des données...")
df = pd.read_csv('new_dataset.csv')
print(f"    {len(df)} articles chargés")
print(f"    Catégories: {df['categorie'].unique()}")
print(f"\n   Distribution:")
print(df['categorie'].value_counts())

# ============================================
# PRÉTRAITEMENT
# ============================================
print("\n2. Prétraitement des textes...")

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Minuscules
    text = text.lower()
    # Supprimer ponctuation et chiffres
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenisation
    tokens = word_tokenize(text)
    # Supprimer stop words et lemmatiser
    tokens = [lemmatizer.lemmatize(token) for token in tokens 
              if token not in stop_words and len(token) > 2]
    return ' '.join(tokens)

df['text_clean'] = df['text'].apply(preprocess_text)

print(f"    Prétraitement terminé")
print(f"\n   Exemple:")
print(f"   Original: {df['text'].iloc[0]}")
print(f"   Nettoyé:  {df['text_clean'].iloc[0]}")

# ============================================
# VECTORISATION
# ============================================
print("\n3. Vectorisation TF-IDF...")

tfidf = TfidfVectorizer(ngram_range=(1,2), max_features=3000)
X = tfidf.fit_transform(df['text_clean'])
y = df['categorie']

print(f"    Matrice créée: {X.shape[0]} lignes, {X.shape[1]} features")

# ============================================
# SÉPARATION TRAIN/TEST
# ============================================
print("\n4. Séparation Train/Test...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"    Train: {X_train.shape[0]} articles")
print(f"    Test: {X_test.shape[0]} articles")

# ============================================
# ENTRAÎNEMENT DES MODÈLES
# ============================================
print("\n5. Entraînement des modèles...")

models = {
    'Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'SVM': LinearSVC(random_state=42, max_iter=2000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

results = []

for name, model in models.items():
    print(f"\n    Entraînement de {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    results.append({'Modèle': name, 'Accuracy': round(acc, 4), 'F1-Score': round(f1, 4)})
    print(f"    Accuracy: {acc:.4f} | F1-Score: {f1:.4f}")

# ============================================
# RÉSULTATS
# ============================================
print("\n" + "="*50)
print("RÉSULTATS FINAUX")
print("="*50)

results_df = pd.DataFrame(results)
print("\n Tableau comparatif:")
print(results_df.to_string(index=False))

best_model_row = results_df.loc[results_df['Accuracy'].idxmax()]
print(f"\n MEILLEUR MODÈLE: {best_model_row['Modèle']} (Accuracy: {best_model_row['Accuracy']})")

# Meilleur modèle
best_model_name = best_model_row['Modèle']
best_model = models[best_model_name]

# Réentraîner le meilleur modèle
best_model.fit(X_train, y_train)
y_pred_best = best_model.predict(X_test)

# Matrice de confusion
print("\n Matrice de confusion:")
cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=best_model.classes_, 
            yticklabels=best_model.classes_)
plt.title(f'Matrice de confusion - {best_model_name}')
plt.xlabel('Prédit')
plt.ylabel('Réel')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("    Sauvegardée: confusion_matrix.png")

# Rapport de classification
print("\n Rapport de classification:")
print(classification_report(y_test, y_pred_best))

# ============================================
# TEST SUR NOUVEAUX TEXTES
# ============================================
print("\n" + "="*50)
print("TEST SUR NOUVEAUX TEXTES")
print("="*50)

def predict_category(text):
    text_clean = preprocess_text(text)
    vec = tfidf.transform([text_clean])
    return best_model.predict(vec)[0]

test_texts = [
    "The company announced record profits this quarter",
    "New smartphone with 5G capabilities launched today",
    "The football team won the match 3-0",
    "The president signed a new law for education",
    "The movie broke box office records"
]

for text in test_texts:
    pred = predict_category(text)
    print(f"\n Texte: {text}")
    print(f"    Prédiction: {pred}")

print("\n" + "="*50)
print(" TP TERMINÉ AVEC SUCCÈS!")
print("="*50)

# Afficher le graphique
plt.show()