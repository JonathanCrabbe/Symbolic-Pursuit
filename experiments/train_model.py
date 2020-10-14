from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor


def train_model(X_train, y_train, black_box="MLP"):
    # Fits a given black-box model to the training set
    if black_box == "MLP":
        model = MLPRegressor()
    elif black_box == "KNN":
        model = KNeighborsRegressor()
    elif black_box == "SVM":
        model = SVR()
    elif black_box == "XGB":
        model = XGBRegressor(objective='reg:squarederror')
    elif black_box == "Tree":
        model = DecisionTreeRegressor()
    elif black_box == "RF":
        model = RandomForestRegressor()
    else:
        raise NameError("black-box model type unknown")
    model.fit(X_train, y_train)
    return model
