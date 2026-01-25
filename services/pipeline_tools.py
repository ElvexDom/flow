from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
import pandas as pd

class PipelineTools:
    """
    Classe pour construire et gérer un pipeline sklearn avec prétraitement et modèle,
    avec méthode pour récupérer les paramètres du pipeline.
    """

    def __init__(self, num_features, cat_features, ordinal_features, class_weight=None, random_state=42):
        """
        Initialise le pipeline.
        Args:
            num_features (list): colonnes numériques
            cat_features (list): colonnes catégorielles
            ordinal_features (list): colonnes ordinales
            class_weight (dict): pour gérer le déséquilibre de classes
            random_state (int)
        """
        self.num_features = num_features
        self.cat_features = cat_features
        self.ordinal_features = ordinal_features
        self.class_weight = class_weight
        self.random_state = random_state
        self.pipeline = self._build_pipeline()

    def _build_pipeline(self) -> Pipeline:
        """
        Crée le pipeline avec prétraitement et modèle LogisticRegression.
        """
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self.num_features),
                ("cat", OneHotEncoder(drop="first", sparse_output=False), self.cat_features),
                ("sex_map", OrdinalEncoder(), self.ordinal_features)
            ]
        )

        pipe = Pipeline(
            steps=[
                ("preproc", preprocessor),
                ("clf", LogisticRegression(
                    max_iter=5000,
                    class_weight=self.class_weight,
                    random_state=self.random_state
                    # penalty supprimé pour éviter le warning
                ))
            ]
        )
        return pipe

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Entraîne le pipeline sur les données.
        """
        self.pipeline.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame):
        """
        Prédit les labels pour de nouvelles données.
        """
        return self.pipeline.predict(X)

    def predict_proba(self, X: pd.DataFrame):
        """
        Retourne les probabilités de classe.
        """
        return self.pipeline.predict_proba(X)

    def get_params(self):
        """
        Retourne un dictionnaire des paramètres importants du pipeline.
        Utile pour log MLflow.
        """
        return {
            "num_features": self.num_features,
            "cat_features": self.cat_features,
            "ordinal_features": self.ordinal_features,
            "class_weight": self.class_weight,
            "random_state": self.random_state,
            "model": type(self.pipeline.named_steps["clf"]).__name__
        }
