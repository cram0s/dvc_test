"""
Команда запуска скрипта из корневой директории проекта
(параметры использованны по умолчанию):
python src/train.py -x_train=data/x_train.csv -y_train=data/y_train.csv -path_pkl=models/model.pkl

Команда для создания этапа DVC pipeline:
dvc run -n prepare \
-d src/prepare.py \
-d data/train.csv \
-o data/x_train.csv \
-o data/y_train.csv \
-o data/x_test.csv \
-o data/y_test.csv \
python src/prepare.py \
-path_train='data/raw/titanic.csv'

"""
import os
from typing import Tuple
import argparse
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-path_train',
                 action="store",
                 dest="path_train",
                 required=True)
    args = parser.parse_args()
    return args


def make_train(path_file: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
   
    train = pd.read_csv(path_file)

    train_cat = list(train.select_dtypes(include='object'))
    train_num = list(train.select_dtypes(exclude='object'))

    train_missing_obj_col = []
    for col in train_cat:
        if train[col].isnull().any():
            train_missing_obj_col.append(col)
    # ['Cabin', 'Embarked']
    train_missing_num_col = []
    for col in train_num:
        if train[col].isnull().any():
            train_missing_num_col.append(col)
    # ['Age']
    temp = train[train_missing_num_col]
    train.drop(train_missing_num_col, inplace=True, axis=1)

    my_imputer = SimpleImputer()
    imputed_temp = pd.DataFrame(my_imputer.fit_transform(temp))
    imputed_temp.columns = temp.columns

    train = pd.concat([train, imputed_temp],axis=1)
    train['Embarked'].fillna(train['Embarked'].mode(),inplace=True)
    dummy1 = pd.get_dummies(train[['Sex', 'Embarked']])
    train.drop(train_missing_obj_col, axis=1, inplace=True)
    train = pd.concat([train, dummy1],axis=1)
    train.drop(['Name', 'PassengerId', 'Sex', 'Ticket'], axis=1, inplace=True)
    train['Age'] = np.log(train['Age']+1)
    train['Fare'] = np.log(train['Fare']+1)
    
    y=train['Survived']
    train.drop(['Survived'],axis=1,inplace=True)

    return train, y

if __name__ == "__main__":
    args = get_args()
    x_train, y_train = make_train(args.path_train)

    x_train,x_test,y_train,y_test=train_test_split(x_train, y_train, random_state=42)

    x_train.to_csv(f"data/processed/x_train.csv", index=False)
    y_train.to_csv(f"data/processed/y_train.csv", index=False)
    x_test.to_csv(f"data/processed/x_test.csv", index=False)
    y_test.to_csv(f"data/processed/y_test.csv", index=False)

