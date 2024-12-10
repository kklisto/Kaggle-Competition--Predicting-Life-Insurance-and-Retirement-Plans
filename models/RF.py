# Importations
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV

def get_features_A():
    ranked_features = [(1, 'Age'), (1, 'Assdec'), (1, 'Cel'), (1, 'Couple'), (1, 'Detlog'), (1, 'Dette'), (1, 'Detvo'), 
                    (1, 'Dip'), (1, 'Eplog'), (1, 'Epsal'), (1, 'Evtgrav'), (1, 'Herit'), (1, 'Livdf'), (1, 'Logt'), 
                    (1, 'Mere'), (1, 'Nat'), (1, 'Nbenf'), (1, 'Occ'), (1, 'Pea'), (1, 'Pel'), (1, 'Pep'), (1, 'Pere'), 
                    (1, 'Sexe'), (1, 'Statmat'), (1, 'Statpro'), (1, 'Terre'), (1, 'Urbani'), (1, 'Vmob'), (1, 'Work'), 
                    (1, 'Zres'), (2, 'Gpm'), (3, 'Livep'), (4, 'Gpp'), (5, 'Capi')]

    features_A = []

    rm = 2

    for r, n in ranked_features:
        if r <= rm:
            features_A.append(n)
    
    return features_A

def get_features_R():
    ranked_features = [(1, 'Age'), (1, 'Assdec'), (1, 'Cel'), (1, 'Couple'), (1, 'Detlog'), (1, 'Dette'), (1, 'Detvo'), 
                   (1, 'Dip'), (1, 'Epsal'), (1, 'Evtgrav'), (1, 'Herit'), (1, 'Livdf'), (1, 'Logt'), (1, 'Mere'), 
                   (1, 'Nbenf'), (1, 'Occ'), (1, 'Pea'), (1, 'Pel'), (1, 'Pep'), (1, 'Pere'), (1, 'Sexe'), (1, 'Statmat'), 
                   (1, 'Statpro'), (1, 'Terre'), (1, 'Urbani'), (1, 'Vmob'), (1, 'Work'), (1, 'Zres'), (2, 'Eplog'), 
                   (3, 'Gpm'), (4, 'Livep'), (5, 'Gpp'), (6, 'Nat'), (7, 'Capi')]

    features_R = []

    rm = 2

    for r, n in ranked_features:
        if r <= rm:
            features_R.append(n)
    
    return features_R


def Assvie(random_s, data):
    features_A = get_features_A()

    data_features = data[features_A]

    x = np.array(data_features)
    y = np.array(data['Assvie'])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=random_s)

    clf = RandomForestClassifier(n_estimators=400, max_depth=10, min_samples_split = 2,  max_features = 'sqrt',
                                                    random_state = random_s, n_jobs=-1)

    clf.fit(X_train, y_train)

    return clf


def Retraite(random_s, data):
    features_R = get_features_R()

    data_features = data[features_R]

    x = np.array(data_features)
    y = np.array(data['Retraite'])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=random_s)

    clf = RandomForestClassifier(n_estimators=400, max_depth=5, min_samples_split = 5,  max_features = 'sqrt',
                                                    random_state = random_s, n_jobs=-1)

    clf.fit(X_train, y_train)

    return clf

def calibrate_Assvie(clf, random_s, data):
    features_A = get_features_A()

    data_features = data[features_A]

    x = np.array(data_features)
    y = np.array(data['Assvie'])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=random_s)

    calibrated_clf = CalibratedClassifierCV(clf, method='sigmoid', cv='prefit')
    calibrated_clf.fit(X_train, y_train)

    return calibrated_clf

def calibrate_Retraite(clf, random_s, data):
    features_R = get_features_R()

    data_features = data[features_R]

    x = np.array(data_features)
    y = np.array(data['Retraite'])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=random_s)

    calibrated_clf = CalibratedClassifierCV(clf, method='sigmoid', cv='prefit')
    calibrated_clf.fit(X_train, y_train)

    return calibrated_clf

