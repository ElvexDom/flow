from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
from services.utils import monitor_performance

class PipelineTools:
    """
    Pipeline sklearn générique avec config dict
    - Split train/test interne
    - Transformation automatique dès le départ
    - Fit uniquement sur le modèle
    - Predict train/test simple et rapide
    """

    def __init__(self, X: pd.DataFrame, y: pd.Series, config: dict):
        self.X = X
        self.y = y

        # Config
        self.num_features = config.get("num_features", [])
        self.cat_features = config.get("cat_features", [])
        self.ordinal_features = config.get("ordinal_features", [])
        self.model_class = config.get("model_class", None)
        self.model_params = config.get("model_params", {})

        if self.model_class is None:
            raise ValueError("model_class doit être défini dans config")

        # Pipeline pour le preprocessing
        self.pipeline = self._build_pipeline()

        self.model = self.pipeline.named_steps["model"]

        # Split
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        # Mode pour monitor_performance
        self.mode = None

    # -------------------------------
    # Construction du pipeline
    # -------------------------------
    def _build_pipeline(self):
        transformers = []
        if self.num_features:
            transformers.append(("num", StandardScaler(), self.num_features))
        if self.cat_features:
            transformers.append(("cat", OneHotEncoder(drop="first", sparse_output=False), self.cat_features))
        if self.ordinal_features:
            transformers.append(("ord", OrdinalEncoder(), self.ordinal_features))

        preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
        model = self.model_class(**self.model_params)
        return Pipeline([("preproc", preprocessor), ("model", model)])

    # -------------------------------
    # Split + transformation + fit
    # -------------------------------
    @monitor_performance
    def train(self, test_size=0.2, random_state=42, stratify=None):
        # Split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, stratify=stratify
        )

        # Transformation immédiate
        preproc = self.pipeline.named_steps["preproc"]
        X_train_trans = preproc.fit_transform(self.X_train)
        X_test_trans = preproc.transform(self.X_test)

        # Récupération des noms de colonnes et nettoyage
        feature_names = preproc.get_feature_names_out()
        feature_names = [f.replace(" ", "_") for f in feature_names]

        # Stocker sous forme de DataFrame avec noms de colonnes
        self.X_train = pd.DataFrame(X_train_trans, columns=feature_names, index=self.X_train.index)
        self.X_test = pd.DataFrame(X_test_trans, columns=feature_names, index=self.X_test.index)

        # Fit uniquement le modèle
        self.model.fit(self.X_train, self.y_train)

        return self

    # -------------------------------
    # Predict train
    # -------------------------------
    @monitor_performance
    def predict_train(self):
        if self.X_train is None:
            raise RuntimeError("Le pipeline n'a pas encore été entraîné")
        self.mode = "pipeline_predict_train"
        return self.model.predict(self.X_train)

    # -------------------------------
    # Predict test
    # -------------------------------
    @monitor_performance
    def predict_test(self):
        if self.X_test is None:
            raise RuntimeError("Le pipeline n'a pas encore été entraîné")
        self.mode = "pipeline_predict_test"
        return self.model.predict(self.X_test)

    # -------------------------------
    # Params
    # -------------------------------
    def get_params(self):
        model = self.model
        return {
            "num_features": self.num_features,
            "cat_features": self.cat_features,
            "ordinal_features": self.ordinal_features,
            "model_class": f"{model.__class__.__module__}.{model.__class__.__name__}",
            "model_params": self.model_params
        }
