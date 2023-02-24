import math
import os
import pickle

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


def main():
    path = os.path.join(os.path.dirname(__file__), "log/")

    log_files = os.listdir(path)
    data_set = []
    for file in log_files:
        with open(os.path.join(path, file), "rb") as f:
            data_set.append(pickle.load(f))

    features = []
    targets = []
    for i, data in enumerate(data_set):
        features += data["features"]
        targets += data["targets"]

    features = np.array(features)
    targets = np.array(targets)

    # Training
    model = DecisionTreeRegressor(
        criterion="squared_error", max_depth=8000, splitter="best"
    )
    model.fit(features, targets)

    # Save model
    if not os.path.exists(os.path.dirname(__file__) + "/save"):
        os.makedirs(os.path.dirname(__file__) + "/save")
    with open(
        os.path.join(os.path.dirname(__file__), "save", "model.pickle"), "wb"
    ) as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    main()
