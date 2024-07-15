import argparse
import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle
import yaml
from dvclive import Live
from warnings import filterwarnings
filterwarnings('ignore')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-x_train', action="store", dest="x_train", required=True)
    parser.add_argument('-y_train', action="store", dest="y_train", required=True)
    parser.add_argument('-path_pkl', action="store", dest="path_pkl")
    args = parser.parse_args()
    return args

def train_model(x_train, y_train, params):
    solver = params['solver']
    if params['penalty'] == "none":
        params['penalty'] = None

    logistic = LogisticRegression(max_iter=params['max_iter'],
                                  random_state=params['random_state'],
                                  n_jobs=params['n_jobs'],
                                  solver=params['solver'],
                                  penalty=params['penalty'],
                                  l1_ratio=0.5)

    with Live() as live:
        live.log_param("max_iter", params['max_iter'])
        live.log_param("solver", params['solver'])
        live.log_param("penalty", params['penalty'])
        live.log_param("l1_ratio", 0.5)

        for epoch in range(params['max_iter']):
            logistic.fit(x_train, y_train.to_numpy().ravel())
            accuracy = logistic.score(x_train, y_train)
            
            # Logging accuracy metric
            live.log_metric('accuracy', accuracy)
            live.next_step()

        # Save the model
        with open(args.path_pkl, 'wb') as fd:
            pickle.dump(logistic, fd)

if __name__ == "__main__":
    args = get_args()
    params = yaml.safe_load(open('params.yaml'))['train_logistic']
    x_train = pd.read_csv(args.x_train)
    y_train = pd.read_csv(args.y_train)

    train_model(x_train, y_train, params)
