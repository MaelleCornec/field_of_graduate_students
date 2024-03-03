from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer

categorical_data = [
    "annee",       
    "etablissement",
    "academie",
    "situation"
]

numerical_data = [
    "taux_dinsertion", 
    "emplois_cadre_ou_professions_intermediaires",
    "emplois_stables",
    "emplois_a_temps_plein",
    "salaire_net_median_des_emplois_a_temps_plein",
    "salaire_brut_annuel_estime",
    "de_diplomes_boursiers",
    "taux_de_chomage_regional",
    "salaire_net_mensuel_median_regional",
    "emplois_cadre",
    "emplois_exterieurs_a_la_region_de_luniversite",
    "femmes", 
    "salaire_net_mensuel_regional_1er_quartile",
    "salaire_net_mensuel_regional_3eme_quartile",
]

class Classifier(BaseEstimator):
    def __init__(self):
        self.numerical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        self.categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot", OneHotEncoder()),
            ]
        )
        self.preprocessing_pipeline = ColumnTransformer(
            transformers=[
                ('num', self.numerical_pipeline, numerical_data),
                ('cat', self.categorical_pipeline, categorical_data),
            ]
        )
        self.model = LogisticRegression(max_iter=1000)
        self.pipe = ImbPipeline([
            ("preprocessing", self.preprocessing_pipeline),
            ("model", self.model)
        ])

    def fit(self, X, y):
        self.pipe.fit(X, y)

    def predict(self, X):
        return self.pipe.predict(X)

    def predict_proba(self, X):
        return self.pipe.predict_proba(X)
