import numpy as np
import pandas as pd
from sklearn import ensemble
from sklearn import preprocessing


def convert_date(df):
    df['year'] = df['Original_Quote_Date'].apply(lambda x: x[0:4]).astype('int')
    df['month'] = df['Original_Quote_Date'].apply(lambda x: x[5:7]).astype('int')
    df['day'] = df['Original_Quote_Date'].apply(lambda x: x[8:]).astype('int')


def convert_field10(df):
    df['Field10'] = df['Field10'].apply(lambda x: str(x).replace(',', '')).astype('int')


def fill_PersonalField84(df):
    df.loc[df['PersonalField84'].isnull(), 'PersonalField84'] = 2


def fill_PropertyField29(df):
    df.loc[df['PropertyField29'].isnull(), 'PropertyField29'] = df['PropertyField29'].mean()


def beat_over_fitting(selected_feature):
    f = open('data/feature_importance.txt', 'r')
    sample_size = 0
    for line in f:
        sample_size += 1
        selected_feature.append(line.strip())
    return selected_feature


if __name__ == '__main__':
    train_file = 'data/train.csv'
    test_file = 'data/test.csv'
    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)

    dtype = train.dtypes
    not_number = []
    for i, k in enumerate(train.keys()):
        if (str(dtype[i]) != 'float64') & (str(dtype[i]) != 'int64'):
            not_number.append(k)

    not_number.pop(not_number.index('Original_Quote_Date'))
    not_number.pop(not_number.index('Field10'))

    for k in not_number:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(np.unique(list(train[k].values) + list(test[k].values)))
        train[k] = lbl.transform(list(train[k].values))
        test[k] = lbl.transform(list(test[k].values))

    convert_date(train)
    convert_field10(train)
    convert_date(test)
    convert_field10(test)
    y = train['QuoteConversion_Flag']
    x = train.select_dtypes(include=['int16', 'int32', 'int64', 'float16', 'float32', 'float64'])
    xt = test.select_dtypes(include=['int16', 'int32', 'int64', 'float16', 'float32', 'float64'])
    x = x.drop('QuoteConversion_Flag', axis=1)

    fill_PersonalField84(x)
    fill_PersonalField84(xt)
    fill_PropertyField29(x)
    fill_PropertyField29(xt)

    #selected_feature = beat_over_fitting([])
    #x = x[selected_feature[0:23]]
    #xt = xt[selected_feature[0:23]]

    # forest = RandomForestClassifier(n_estimators=300)
    # forest = forest.fit(np.array(x), np.array(y))
    # yhat = forest.predict_proba(np.array(xt))[:, 1]

    params = {'n_estimators': 1000, 'max_leaf_nodes': 4, 'max_depth': None, 'random_state': 2,
              'min_samples_split': 5, 'learning_rate': 0.1, 'subsample': 0.5}
    gb = ensemble.GradientBoostingClassifier(**params)
    gb.fit(x, y)

    yhat = gb.predict_proba(xt)

    result_data = {'QuoteNumber': xt['QuoteNumber'], 'QuoteConversion_Flag': yhat[:, 1]}
    result = pd.DataFrame(result_data, columns=('QuoteNumber', 'QuoteConversion_Flag'))

    result.to_csv('data/result.csv', index=False)
