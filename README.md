#  Classification de Textes NLP

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![NLTK](https://img.shields.io/badge/NLTK-3.8-green.svg)](https://www.nltk.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-orange.svg)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0-yellow.svg)](https://pandas.pydata.org/)
[![License](https://img.shields.io/badge/License-MIT-red.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/Code%20Style-PEP%208-blueviolet.svg)](https://www.python.org/dev/peps/pep-0008/)
[![Status](https://img.shields.io/badge/Status-Completed-brightgreen.svg)]()
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

>  Un pipeline complet de classification automatique de textes utilisant des techniques classiques de NLP (Traitement Automatique du Langage Naturel)

##  Table des Matières

- [Aperçu](#-aperçu)
- [Fonctionnalités](#-fonctionnalités)
- [Architecture du Pipeline](#-architecture-du-pipeline)
- [Installation](#-installation)
- [Utilisation](#-utilisation)
- [Modèles Testés](#-modèles-testés)
- [Résultats](#-résultats)
- [Structure du Projet](#-structure-du-projet)
- [Technologies Utilisées](#-technologies-utilisées)
- [Auteur](#-auteur)
- [License](#-license)

##  Aperçu

Ce projet implémente un système complet de classification de textes capable de catégoriser automatiquement des articles de presse dans **5 catégories** différentes :

| Catégorie |  | Description |
|-----------|-------|-------------|
| Business |  | Actualités économiques et financières |
| Technology |  | Innovations et avancées technologiques |
| Sport |  | Compétitions et performances sportives |
| Politics |  | Actualités politiques et gouvernementales |
| Entertainment |  | Divertissement et culture populaire |

##  Fonctionnalités

-  **Prétraitement complet** des données textuelles
  - Tokenisation, lemmatisation, suppression des stop words
-  **Vectorisation** avec TF-IDF et Bag of Words
-  **4 modèles de Machine Learning** entraînés et comparés
-  **Optimisation** des hyperparamètres avec GridSearchCV
-  **Évaluation détaillée** (Accuracy, Precision, Recall, F1-Score)
-  **Analyse des erreurs** de classification
-  **Interprétabilité** des décisions du modèle
-  **Prédiction en temps réel** sur de nouveaux textes


##  Installation

### Prérequis

- Python 3.8 ou supérieur
- pip (gestionnaire de paquets Python)

### 1 Cloner le projet

```bash
git clone https://github.com/votre-username/classification-texte-nlp.git
cd classification-texte-nlp

### Installer les dépendances

pip install pandas numpy matplotlib seaborn scikit-learn nltk wordcloud joblib

### Utilisation
Exécution du programme principal
python tp_nlp.py

### Prédiction sur un nouveau texte
from tp_nlp import predict_category

# Exemple d'utilisation
texte = "The company announced record profits this quarter"
categorie = predict_category(texte)
print(f"Catégorie: {categorie}")  

###
Modèles Testés
Modèle	Type	Avantages
Naive Bayes	Probabiliste	Rapide, efficace sur petits datasets
Logistic Regression	Linéaire	Interprétable, performant
SVM (LinearSVC)	Marge maximale	Excellent pour textes de grande dimension
Random Forest	Ensemble	Robuste aux overfitting