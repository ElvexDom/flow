## 1. Classification : Prédiction du Churn (Désabonnement)

[telco-customer-churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
[churn-for-bank-customers](https://www.kaggle.com/datasets/mathchi/churn-for-bank-customers)

**Le scénario :** Une banque ou un opérateur télécom veut savoir quels clients vont résilier leur contrat.

* **Dataset :** *Telco Customer Churn* (Kaggle) ou *Bank Customer Churn*.
* **Challenge Pandas :** Gérer les types de données (certaines colonnes numériques sont lues comme du texte), traiter les variables binaires (Yes/No  0/1).
* **Challenge Scikit-Learn :** Utiliser la **Régression Logistique** et analyser la **Matrice de Confusion**.
* **Cible :** Churn / Exited
  
C'est un cas d'école de **dataset déséquilibré** (beaucoup plus de gens restent qu'ils ne partent). 

---

## 2. Régression : Estimation du prix de voitures d'occasion

[toyotacorollacsv](https://www.kaggle.com/datasets/klkwak/toyotacorollacsv)
[car-details-dataset](https://www.kaggle.com/datasets/akshaydattatraykhare/car-details-dataset)

**Le scénario :** Créer un estimateur pour un site comme "Le Bon Coin".

* **Dataset :** *Car Details Dataset* ou *Toyota Corolla dataset*.
* **Challenge Pandas :** Extraire l'année ou la marque à partir d'une chaîne de caractères, gérer les valeurs aberrantes (ex: une voiture à 0km ou 1 euro).
* **Challenge Scikit-Learn :** Comparer la **Régression Linéaire** et les **K-Nearest Neighbors (KNN)**. Mise en place obligatoire du `StandardScaler`.

* **Cible :** Price / selling_price

---

## 3. Clustering : Segmentation de clientèle

[customer-segmentation-tutorial-in-python](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)

**Le scénario :** Un centre commercial veut regrouper ses clients en 4 ou 5 profils pour envoyer des bons de réduction ciblés.

* **Dataset :** *Mall Customer Segmentation*.
* **Challenge Pandas :** Analyse exploratoire (EDA) avec Seaborn pour voir les groupes naturels (âge vs score de dépense).
* **Challenge Scikit-Learn :** Utiliser **K-Means**. Trouver le nombre de clusters optimal avec la **méthode du coude (Elbow Method)**.

**score** :  
* silhouette_score
* davies_bouldin_score
* calinski_harabasz_score

---

## 4. Projet "Données Sales" : Qualité de l'Air ou Météo

[air-quality-data-in-india](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india)
[weather-dataset-rattle-package](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package)

**Le scénario :** Prédire si demain il pleuvra ou quel sera le taux de pollution.

* **Dataset :** *Air Quality Data in India* ou *Rain in Australia*.
* **Challenge Pandas :** C'est le boss final du nettoyage. Il y a des tonnes de `NaN`, des séries temporelles (dates à gérer), et des capteurs défaillants (valeurs à 0 impossibles).
* **Challenge Scikit-Learn :** Pipeline complet : Imputation  Scaling  Classification.

**weather-dataset-rattle-package :**
* Classification sur `RainTomorrow` 

**air-quality-data-in-india :**
* Regression sur `AQI` (AQI est une simple combinaison linéaire des autres colonnes, ca arrive parfois ...)
* Classification sur `AQI_Bucket` 
  
