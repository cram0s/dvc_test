import argparse
import os
import pandas as pd

def get_args():
    """
    Считывание параметров
    path_in - путь получения датасета
    path_out - путь извлечения датасета
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-path_in", action="store", dest="path_in", required=True)
    parser.add_argument("-path_out", action="store", dest="path_out")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    train = pd.read_csv(args.path_in)
    new_train=train[(train['Sex']=='female') & (train['Survived']==1) & (train['Pclass']==3)]
    new_train.to_csv(f"{args.path_out}/titanic.csv", index=False)