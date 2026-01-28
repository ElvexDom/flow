from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
import pandas as pd

from services.utils import monitor_performance

class PipelineTools:
    """
    Pipeline sklearn générique avec prétraitement et modèle configurable.
    """

    def __init__(self, num_features, cat_features, ordinal_features, model_class=None, model_params=None):
        """
        Args:
            num_features (list): colonnes numériques
            cat_features (list): colonnes catégorielles
            ordinal_features (list): colonnes ordinales
            model_class (sklearn estimator): instance de modèle ou classe (ex: LinearRegression)
            model_params (dict): dictionnaire des hyperparamètres pour le modèle
        """
        self.num_features = num_features
        self.cat_features = cat_features
        self.ordinal_features = ordinal_features
        self.model_class = model_class
        self.model_params = model_params or {}
        self.pipeline = self._build_pipeline()

    def _build_pipeline(self):
        # Prétraitement
        transformers = []

        if self.num_features:
            transformers.append(("num", StandardScaler(), self.num_features))
        if self.cat_features:
            transformers.append(("cat", OneHotEncoder(drop="first", sparse_output=False), self.cat_features))
        if self.ordinal_features:
            transformers.append(("ord", OrdinalEncoder(), self.ordinal_features))

        preprocessor = ColumnTransformer(transformers=transformers)

        # Modèle
        if self.model_class is None:
            raise ValueError("model_class doit être défini (ex: LinearRegression, KNeighborsRegressor)")
        model = self.model_class(**self.model_params)

        # Pipeline final
        pipe = Pipeline([
            ("preproc", preprocessor),
            ("model", model)
        ])
        return pipe

    @monitor_performance
    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.pipeline.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame):
        return self.pipeline.predict(X)

    def get_params(self):
        return {
            "num_features": self.num_features,
            "cat_features": self.cat_features,
            "ordinal_features": self.ordinal_features,
            "model_class": type(self.pipeline.named_steps["model"]).__name__,
            "model_params": self.model_params
        }
