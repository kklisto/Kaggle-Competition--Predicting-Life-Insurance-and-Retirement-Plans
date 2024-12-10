# Importations
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score

from models import RF, GB

best_mod = {}
best_mod['Work'] = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate = 0.1, min_samples_split = 8, 
                                                        subsample = 0.8,  max_features = 'sqrt', random_state = 42)
best_mod['Gpm'] = GradientBoostingClassifier(n_estimators=100, max_depth=2, learning_rate = 0.2, min_samples_split = 5, 
                                                        subsample = 0.5,  max_features = 'sqrt', random_state = 42)
best_mod['Gpp'] = GradientBoostingClassifier(n_estimators=50, max_depth=4, learning_rate = 0.2, min_samples_split = 8, 
                                                        subsample = 0.8,  max_features = 'sqrt', random_state = 42)
best_mod['Pere'] = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate = 0.09, min_samples_split = 2, 
                                                        subsample = 1.0,  max_features = 'sqrt', random_state = 42)
best_mod['Dip'] = GradientBoostingClassifier(n_estimators=50, max_depth=2, learning_rate = 0.2, min_samples_split = 5, 
                                                        subsample = 0.5,  max_features = 'sqrt', random_state = 42)
best_mod['Mere'] = GradientBoostingClassifier(n_estimators=200, max_depth=2, learning_rate = 0.09, min_samples_split = 5, 
                                                        subsample = 0.5,  max_features = 'sqrt', random_state = 42)
best_mod['Evtgrav'] = GradientBoostingClassifier(n_estimators=50, max_depth=3, learning_rate = 0.09, min_samples_split = 2, 
                                                        subsample = 1.0,  max_features = 'sqrt', random_state = 42)
best_mod['Statpro'] = GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate = 0.1, min_samples_split = 5, 
                                                        subsample = 1.0,  max_features = 'sqrt', random_state = 42)

def preprocessing(filePath):
    # Ouverture du fichier train.csv
    data = pd.read_csv(filePath, sep=",")

    #drop Id columns cause it is irrelevant 
    data = data.drop(['Id'], axis = 1)

    if filePath == 'train.csv':
        ordinal = ['Zres', 'Urbani', 'Occ', 'Nat', 'Statmat','Assvie', 'Sexe', 'Couple', 'Herit', 'Livep', 'Eplog', 'Epsal', 'Vmob', 'Livdf', 'Pel', 
            'Cel', 'Pep', 'Assdec', 'Retraite', 'Capi', 'Pea', 'Logt', 'Terre', 'Dette', 'Detlog', 
            'Detvo'] 
    else:
        ordinal = ['Zres', 'Urbani', 'Occ', 'Nat', 'Statmat', 'Sexe', 'Couple', 'Herit', 'Livep', 'Eplog', 'Epsal', 'Vmob', 'Livdf', 'Pel', 
            'Cel', 'Pep', 'Assdec', 'Capi', 'Pea', 'Logt', 'Terre', 'Dette', 'Detlog', 
            'Detvo'] 

    #Ordinal variables are converted into discrete features
    for col in ordinal:
        v = list(data[col].values)
        data[col] = LabelEncoder().fit_transform(v)

    #The missing columns have to be delt separatly
    missing_col = ['Work', 'Gpm','Gpp', 'Pere', 'Dip', 'Mere', 'Evtgrav', 'Statpro']

    #GradientBoostingClassifier for all missing values
    for temp in missing_col:
        # Split data into train and test sets
        train_data = data.dropna(subset=[temp])
        test_data = data[data[temp].isna()]

        # Split features and target variable
        X_train = train_data.drop(columns=['Work', 'Gpm','Gpp', 'Pere', 'Dip', 'Mere', 'Evtgrav', 'Statpro'])
        v = list(train_data[temp].values)
        y_train = LabelEncoder().fit_transform(v)

        # Train random forest classifier
        clf = best_mod[temp]
        clf.fit(X_train, y_train)

        # Predict missing values
        X_test = test_data.drop(columns=['Work', 'Gpm','Gpp', 'Pere', 'Dip', 'Mere', 'Evtgrav', 'Statpro'])
        missing_values = clf.predict(X_test)

        # Replace missing values in original dataframe
        data.loc[data[temp].notna(), temp] = y_train
        data.loc[data[temp].isna(), temp] = missing_values

    return data

def predict_test(df, features_A, features_R, model_A, model_R):
    df_A = df[features_A]
    df_R = df[features_R]

    X_A = np.array(df_A)
    X_R = np.array(df_R)

    sub = pd.read_csv('sample_submission.csv')

    predictions = model_A.predict_proba(X_A)[:, 1]
    sub['Assvie'] = predictions

    predictions = model_R.predict_proba(X_R)[:, 1]
    sub['Retraite'] = predictions

    return sub

def mix(RF_pred, GB_pred):
    sub = pd.read_csv('sample_submission.csv')

    for i in range(RF_pred.shape[0]):
        sub.iloc[i,1] = (RF_pred.iloc[i,1] + GB_pred.iloc[i,1])/2
        sub.iloc[i,2] = (RF_pred.iloc[i,2] + GB_pred.iloc[i,2])/2

    return sub

def main ():
    data = preprocessing(filePath = 'train.csv')
    df = preprocessing(filePath = 'test.csv')

    random_s = 47
    RF_Assvie = RF.Assvie(random_s = random_s, data=data)
    RF_Assvie = RF.calibrate_Assvie(RF_Assvie, random_s = random_s, data=data)

    GB_Assvie = GB.Assvie(random_s = random_s, data=data)
    GB_Assvie = GB.calibrate_Assvie(GB_Assvie, random_s = random_s, data=data)

    random_s = 88
    RF_Retraite = RF.Retraite(random_s = random_s, data=data)
    RF_Retraite = RF.calibrate_Retraite(RF_Retraite, random_s = random_s, data=data)

    GB_Retraite = GB.Retraite(random_s = random_s, data=data)
    GB_Retraite = GB.calibrate_Retraite(GB_Retraite, random_s = random_s, data=data)

    RF_pred = predict_test(df, RF.get_features_A(), RF.get_features_R(), RF_Assvie, RF_Retraite)
    GB_pred = predict_test(df, GB.get_features_A(), GB.get_features_R(), GB_Assvie, GB_Retraite)

    submission = mix(RF_pred, GB_pred)

    submission.to_csv('CSV/calibrated_boom.csv', index=False)

    print("Done")

if __name__ == "__main__":
    main()