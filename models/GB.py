# Importations
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV

def get_features_A():
    ranked_features = [(1, 'Age'), (1, 'Eplog'), (1, 'Herit'), (1, 'Livep'), (1, 'Pea'), (1, 'Vmob'), (1, 'Work'), (2, 'Dip'), 
                   (3, 'Occ'), (4, 'Pep'), (5, 'Zres'), (6, 'Urbani'), (7, 'Epsal'), (8, 'Pel'), (9, 'Statpro'), 
                   (10, 'Dette'), (11, 'Livdf'), (12, 'Nbenf'), (13, 'Statmat'), (14, 'Nat'), (15, 'Cel'), (16, 'Gpp'), 
                   (17, 'Sexe'), (18, 'Detvo'), (19, 'Capi'), (20, 'Assdec'), (21, 'Pere'), (22, 'Detlog'), (23, 'Couple'), 
                   (24, 'Evtgrav'), (25, 'Terre'), (26, 'Logt'), (27, 'Gpm'), (28, 'Mere')]

    features_A = []

    rm = 28

    for r, n in ranked_features:
        if r <= rm:
            features_A.append(n)

    return features_A

def get_features_R():
    ranked_features = [(1, 'Age'), (1, 'Assdec'), (1, 'Dip'), (1, 'Epsal'), (1, 'Occ'), (1, 'Pea'), (1, 'Pep'), (1, 'Statmat'), 
                   (1, 'Statpro'), (1, 'Urbani'), (1, 'Vmob'), (1, 'Work'), (1, 'Zres'), (2, 'Livep'), (3, 'Nbenf'), 
                   (4, 'Eplog'), (5, 'Capi'), (6, 'Dette'), (7, 'Nat'), (8, 'Pere'), (9, 'Cel'), (10, 'Mere'), (11, 'Couple'), 
                   (12, 'Pel'), (13, 'Herit'), (14, 'Evtgrav'), (15, 'Terre'), (16, 'Detlog'), (17, 'Sexe'), (18, 'Gpp'), 
                   (19, 'Gpm'), (20, 'Detvo'), (21, 'Livdf'), (22, 'Logt')]
    
    features_R = []

    rm = 22

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

    clf = GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate = 0.1, min_samples_split = 5, 
                                                       subsample = 0.5,  max_features = 'sqrt', random_state =random_s)
    
    clf.fit(X_train, y_train)

    return clf

def Retraite(random_s, data):
    features_R = get_features_R()

    data_features = data[features_R]

    x = np.array(data_features)
    y = np.array(data['Retraite'])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=random_s)

    clf = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate = 0.09, min_samples_split = 5, 
                                                       subsample = 1.0,  max_features = 'sqrt', random_state =random_s)
    
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