# 📦 Installation et lancement du projet avec `uv`

Ce projet utilise **exclusivement `uv`** pour la gestion de l’environnement virtuel **et** des dépendances Python.

Aucune commande `pip` n’est nécessaire.

---

## 🔧 Prérequis

* **Python ≥ 3.9** (recommandé : 3.11 ou 3.12)
* **uv** installé sur votre machine

### Installer `uv`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Redémarrez ensuite votre terminal ou rechargez votre shell.

Vérifier l’installation :

```bash
uv --version
```

---

## 📁 Récupération du projet

```bash
git clone <URL_DU_DEPOT>
cd <NOM_DU_PROJET>
```

---

## 🐍 Initialisation du projet

### Initialiser le projet (si nécessaire)

```bash
uv init
```

Cette commande crée le fichier `pyproject.toml` s’il n’existe pas encore.

---

## 📦 Installation des dépendances

### Créer l’environnement virtuel **et** installer les dépendances

```bash
uv sync
```

* L’environnement virtuel est créé automatiquement dans `.venv`
* Les dépendances sont installées depuis `pyproject.toml`
* Le fichier `uv.lock` garantit des installations reproductibles
* **scikit-learn**, **seaborn** et **mlflow** sont incluses comme dépendances pour le machine learning, le suivi d’expériences et la visualisation

---

## ➕ Ajouter des dépendances principales

### Suivi d’expériences ML (MLflow)

```bash
uv add mlflow
```

### Support Jupyter / Notebooks

```bash
uv add ipykernel
```

### Tests unitaires (pytest – dépendance de développement)

```bash
uv add --dev pytest
```

Puis synchroniser :

```bash
uv sync
```

---

## ▶️ Activer l’environnement virtuel (optionnel)

`uv` peut exécuter les commandes sans activation manuelle,
mais si nécessaire :

**Linux / macOS**

```bash
source .venv/bin/activate
```

**Windows (PowerShell)**

```powershell
.venv\Scripts\Activate.ps1
```

---

## ▶️ Lancer le projet

### Exécution directe avec `uv`

```bash
uv run python main.py
```

ou

```bash
uv run python -m nom_du_module
```

---

## 🧪 Tests

```bash
uv run pytest
```

---

## 📓 Utiliser Jupyter Notebook avec le venv `uv`

Après avoir ajouté `ipykernel` :

```bash
uv run python -m ipykernel install --user --name=mon_env_uv --display-name "Python (uv)"
```

Puis lancer Jupyter :

```bash
uv run jupyter lab
```

---

## ➕ Ajouter une dépendance

```bash
uv add nom_du_package
```

Ajouter une dépendance de développement :

```bash
uv add --dev nom_du_package
```

Pour ajouter manuellement des dépendances ML si besoin :

```bash
uv add scikit-learn seaborn mlflow
```

---

## 🧹 Commandes utiles

Mettre à jour les dépendances :

```bash
uv sync --upgrade
```

Supprimer l’environnement virtuel :

```bash
rm -rf .venv
```

---

## 📌 Bonnes pratiques

* Le dossier `.venv` doit être ajouté au `.gitignore`
* Ne pas modifier `uv.lock` manuellement
* Toujours utiliser `uv run` pour garantir l’environnement correct
* `uv` remplace `pip`, `virtualenv` et `pip-tools`
