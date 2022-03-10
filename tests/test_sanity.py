from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from symbolic_pursuit.models import SymbolicRegressor


def test_args() -> None:
    symbolic_model = SymbolicRegressor(
        loss_tol=1000,
        ratio_tol=2,
        random_seed=0,
        maxiter=10,
        eps=0.1,
    )

    assert symbolic_model.loss_tol == 1000
    assert symbolic_model.ratio_tol == 2
    assert symbolic_model.random_seed == 0
    assert symbolic_model.maxiter == 10
    assert symbolic_model.eps == 0.1


def test_sanity() -> None:
    model = LinearRegression()
    X, y = load_boston(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    model.fit(X_train, y_train)
    symbolic_model = SymbolicRegressor(
        loss_tol=1000,
        ratio_tol=2,
        random_seed=0,
        maxiter=10,
    )
    symbolic_model.fit(model.predict, X_test)
