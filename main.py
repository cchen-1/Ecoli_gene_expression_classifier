import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest, RandomForestClassifier, VotingClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedStratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    training_data = pd.read_csv('Ecoli.csv')
    test_data = pd.read_csv('Ecoli_test.csv')
    random_state = 0
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=random_state)

    # class-specific imputation
    positive_data = training_data.loc[training_data["Target (Col 107)"] == 1]
    negative_data = training_data.loc[training_data["Target (Col 107)"] == 0]
    pos_num, pos_nom, neg_num, neg_nom = positive_data.iloc[:, :103], positive_data.iloc[:, 103:-1], negative_data.iloc[:, :103], negative_data.iloc[:, 103:-1]
    num_median_imputer = SimpleImputer(strategy='median')
    pos_num_imputed = pd.DataFrame(num_median_imputer.fit_transform(pos_num))
    neg_num_imputed = pd.DataFrame(num_median_imputer.fit_transform(neg_num))
    nom_imputer = SimpleImputer(strategy='most_frequent')
    pos_nom_imputed = pd.DataFrame(nom_imputer.fit_transform(pos_nom))
    neg_nom_imputed = pd.DataFrame(nom_imputer.fit_transform(neg_nom))
    pos_imputed = pd.concat([pos_num_imputed, pos_nom_imputed], axis=1, ignore_index=True)
    neg_imputed = pd.concat([neg_num_imputed, neg_nom_imputed], axis=1, ignore_index=True)
    pos_imputed['106'] = 1
    neg_imputed['106'] = 0
    df_imputed = pd.concat([pos_imputed, neg_imputed], ignore_index=True)
    X0 = df_imputed.iloc[:, :-1]
    y0 = df_imputed.iloc[:, -1]

    # outlier detection using isolation forest
    iforest = IsolationForest(contamination=0.05, max_features=29, max_samples=155,
                              n_estimators=265, random_state=1)
    anomaly_pred = iforest.fit_predict(X0)
    df_imputed['anomaly'] = anomaly_pred
    anomaly = df_imputed.loc[df_imputed['anomaly'] == -1]

    # remove outliers
    df_outlier_removed = df_imputed.drop(df_imputed[df_imputed['anomaly'] == -1].index)

    # Max-min normalisation
    scaler = MinMaxScaler()
    np_array = pd.DataFrame(df_outlier_removed.iloc[:, 103:-1].to_numpy())
    df = pd.DataFrame(scaler.fit_transform(df_outlier_removed.iloc[:, :103]))
    df = pd.concat([df, np_array], axis=1, ignore_index=True)
    df.iloc[:, -1] = df.iloc[:, -1].astype(int)

    # train test split
    X, y = df.iloc[:, :-1], df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, stratify=y, random_state=random_state)

    # classifier construction
    voting_clf = VotingClassifier(estimators=[
        ('knn', KNeighborsClassifier(n_neighbors=4)),
        ('dt', DecisionTreeClassifier(max_depth=2, random_state=random_state)),
        ('rf', RandomForestClassifier(n_estimators=82, max_features=46, max_depth=5, random_state=random_state)),
    ], voting='hard')

    # test data preprocessing
    dummy_array = pd.DataFrame(test_data.iloc[:, 103:].to_numpy())
    test_df = pd.DataFrame(scaler.fit_transform(test_data.iloc[:, :103]))
    test_df = pd.concat([test_df, dummy_array], axis=1, ignore_index=True)

    # prediction on scaled test data
    voting_clf = voting_clf.fit(X, y)
    test_pred = pd.DataFrame(voting_clf.predict(test_df))
    predictions = pd.concat([test_pred, test_pred], axis=1, ignore_index=True)
    predictions.iloc[:,-1] = None
    predictions.to_csv('s4765227.csv', header = 0, index = 0, na_rep = '')

    # calculate cross-validated F1 and accuracy on training data
    y_pred = voting_clf.fit(X_train, y_train).predict(X_test)
    f1 = f1_score(y_test, y_pred, pos_label=1, average='binary')
    f1 = "{:.3f}".format(f1)
    acc = accuracy_score(y_test, y_pred)
    acc = "{:.3f}".format(acc)
    measurements = pd.DataFrame(list([[acc,f1]]))
    measurements.to_csv('s4765227.csv', mode='a', header=0, index=0)







