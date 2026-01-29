#### **Projet :** Régression Immobilière & Pipeline IA (California Housing)

L'objectif de ce projet est de construire un modèle capable de prédire le prix des logements en Californie en utilisant le dataset officiel disponible via Scikit-Learn (ou via Kaggle). 

[Boston](https://www.kaggle.com/code/prasadperera/the-boston-housing-dataset)
[California](https://www.kaggle.com/datasets/camnugent/california-housing-prices)

Pour ce travail, vous devrez accorder une attention particulière à la "Data Quality" : 
* **identifiez les valeurs aberrantes** (outliers) et les corrélations fortes avant toute chose. 
* Le traitement des données ne doit pas se faire au coup par coup, mais via l'implémentation de **Pipelines** Scikit-Learn et de ColumnTransformer pour garantir que votre pré-traitement est reproductible et éviter le "Data Leakage" entre vos jeux d'entraînement et de test.


Concernant le Processing, voici quelques astuces cruciales à appliquer :

* **Gestion des Outliers :** Le dataset California possède des plafonds artificiels (notamment sur la valeur médiane des maisons) qu'il est souvent préférable de supprimer pour ne pas biaiser l'apprentissage.
* **Feature Engineering :** Ne vous contentez pas des variables brutes. Créez des ratios pertinents, comme le nombre de pièces par habitant ou le nombre de chambres par logement, qui sont souvent plus prédictifs que les valeurs isolées.
* **Normalisation :** Utilisez un StandardScaler ou un RobustScaler (si vous gardez des outliers), car les modèles de régression y sont très sensibles.
* **Géolocalisation :** Les coordonnées (Latitude/Longitude) ne doivent pas être traitées comme des nombres linéaires simples ; envisagez un regroupement (Clustering) ou utilisez-les pour votre visualisation finale.

Pour la partie technique, vous devez impérativement utiliser MLflow pour tracker vos expériences, comparer les métriques ($MSE$, $R^2$) et enregistrer vos hyperparamètres. Le projet doit être géré avec uv et un fichier pyproject.toml