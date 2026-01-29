from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
from services.utils import monitor_performance

class PipelineTools:
    """
    Pipeline sklearn générique :
    - Split train/test interne
    - Fit
    - Predict train / test
    - Runtimes clairement nommés
    - Les données X et y sont fournies à l'instanciation
    """

    def __init__(self, X: pd.DataFrame, y: pd.Series,
                 num_features=None, cat_features=None, ordinal_features=None,
                 model_class=None, model_params=None):
        # Données
        self.X = X
        self.y = y

        # Features
        self.num_features = num_features or []
        self.cat_features = cat_features or []
        self.ordinal_features = ordinal_features or []

        # Modèle
        self.model_class = model_class
        self.model_params = model_params or {}

        # Pipeline
        self.pipeline = self._build_pipeline()

        # Datasets internes
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        # utilisé par monitor_performance
        self.mode = None

    # --------------------------------------------------
    # Construction du pipeline
    # --------------------------------------------------
    def _build_pipeline(self):
        transformers = []

        if self.num_features:
            transformers.append(("num", StandardScaler(), self.num_features))

        if self.cat_features:
            transformers.append(("cat", OneHotEncoder(drop="first", sparse_output=False), self.cat_features))

        if self.ordinal_features:
            transformers.append(("ord", OrdinalEncoder(), self.ordinal_features))

        preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

        if self.model_class is None:
            raise ValueError("model_class doit être défini")

        model = self.model_class(**self.model_params)

        return Pipeline([("preproc", preprocessor), ("model", model)])

    # --------------------------------------------------
    # TRAIN = split + fit
    # --------------------------------------------------
    @monitor_performance
    def train(self, test_size=0.2, random_state=42, stratify=None):
        """
        Split + fit du pipeline
        """
        if self.X is None or self.y is None:
            raise ValueError("Les données X et y doivent être fournies à l'instanciation")

        self.mode = "pipeline_fit_training"

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify
        )

        self.pipeline.fit(self.X_train, self.y_train)
        return self

    # --------------------------------------------------
    # PREDICT TRAIN
    # --------------------------------------------------
    @monitor_performance
    def predict_train(self):
        if self.X_train is None:
            raise RuntimeError("Le pipeline n'a pas encore été entraîné")
        self.mode = "pipeline_predict_train"
        return self.pipeline.predict(self.X_train)

    # --------------------------------------------------
    # PREDICT TEST
    # --------------------------------------------------
    @monitor_performance
    def predict_test(self):
        if self.X_test is None:
            raise RuntimeError("Le pipeline n'a pas encore été entraîné")
        self.mode = "pipeline_predict_test"
        return self.pipeline.predict(self.X_test)

    # --------------------------------------------------
    # PARAMS
    # --------------------------------------------------
    def get_params(self):
        model = self.pipeline.named_steps["model"]
        return {
            "num_features": self.num_features,
            "cat_features": self.cat_features,
            "ordinal_features": self.ordinal_features,
            "model_class": f"{model.__class__.__module__}.{model.__class__.__name__}",
            "model_params": self.model_params
        }
