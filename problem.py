import os
import pandas as pd
import rampwf as rw
from sklearn.model_selection import StratifiedShuffleSplit

problem_title = "Field of graduate students prediction"
_target_column_name = "domaine"
_ignore_column_names = [
    'diplome',
    'discipline',
    'taux_d_emploi',
    'taux_d_emploi_salarie_en_france',
    'remarque',
    'etablissementactuel',
    'nombre_de_reponses', 
    'taux_de_reponse', 
    'id_paysage',
    'cle_etab',
    'cle_disc', 
    'numero_de_l_etablissement',
    'code_de_l_academie',
    'code_du_domaine', 
    'code_de_la_discipline',
    'poids_de_la_discipline'
]
_prediction_label_names = [
    "Droit, économie et gestion",
    "Lettres, langues, arts",
    "Masters enseignement",
    "Sciences humaines et sociales",
    "Sciences, technologies et santé"
]
# A type (class) which will be used to create wrapper objects for y_pred
Predictions = rw.prediction_types.make_multiclass(
    label_names = _prediction_label_names
)
# An object implementing the workflow
workflow = rw.workflows.Classifier()

score_types = [
    rw.score_types.BalancedAccuracy(name="bal_acc",
                                    precision=3,
                                    adjusted=False),
    rw.score_types.Accuracy(name="acc", precision=3),
]

def get_cv(X, y):
    cv = StratifiedShuffleSplit(n_splits=3, test_size=0.2, random_state=42)
    return cv.split(X, y)

def read_data(path, f_name):
    data = pd.read_csv(os.path.join(path, 'data', f_name))
    X = data.drop(columns=[_target_column_name] + _ignore_column_names, axis=1)
    y = data[_target_column_name]   # à revoir
    return X, y

def get_train_data(path='.'):
    f_name = 'train.csv'
    return read_data(path, f_name)

def get_test_data(path='.'):
    f_name = 'test.csv'
    return read_data(path, f_name)
