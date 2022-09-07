import mpmath
import numpy as np
from scipy.optimize import minimize
from sympy import Symbol, sympify

import symbolic_pursuit.logger as log
from symbolic_pursuit.pysymbolic.models.special_functions import MeijerG

# Hyperparameters related functions:


def load_h():
    return {
        "hyper_1": (np.array([0.5, 0.0]), [1, 0, 0, 2]),  # Parent of sin , cos, sh, ch
        "hyper_2": (
            np.array([2.0, 2.0, 2.0, 1.0]),
            [0, 1, 3, 1],
        ),  # Parent of monomials
        "hyper_3": (
            np.array([0.3, 0.1, 0.1, 0.0, 0.3]),
            [2, 1, 2, 3],
        ),  # Parent of exp, Gamma, gamma
    }


# Miscellaneous functions


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def ReLU(x):
    return x * (x > 0)


"""
------------------------------------------------------------
Class for symbolic models associated to a regression problem
------------------------------------------------------------
"""


class SymbolicRegressor:

    # Adaptation of existing methods:

    def __init__(
        self,
        loss_tol=1.0e-3,
        ratio_tol=0.9,
        maxiter=100,
        eps=1.0e-5,
        random_seed=42,
        baselines=list(load_h().keys()),
        task_type="regression",
        patience: int = 10,
    ):
        self.terms_list = []  # List of all the terms in the model
        self.loss_list = []  # List of residual losses associated to each term
        self.n_points = 0  # Number of points
        self.current_resi = np.array(self.n_points)  # Array of residues
        self.dim_x = 0  # Number of features
        self.loss_tol = (
            loss_tol  # The tolerance for the loss under which the pursuit stops
        )
        self.ratio_tol = (
            ratio_tol  # A new term is added only if new_loss / old_loss < ratio_tol
        )
        self.maxiter = maxiter  # Maximum number of iterations for optimization
        self.eps = eps  # Small number used for numerical stability
        self.random_seed = random_seed  # Random seed for reproducibility
        self.baselines = baselines
        self.task_type = task_type
        self.patience = patience

        log.info(
            "Model created with the following hyperparameters :"
            + "\n loss_tol={} \n ratio_tol={} "
            "\n maxiter={} \n eps={} \n random_seed={}".format(
                loss_tol, ratio_tol, maxiter, eps, random_seed
            )
        )

    def __str__(self):
        expression = ""
        expression += str(self.get_expression())
        return expression

    # Optimizer

    def optimize_CG(self, loss, theta_0):
        # Encodes the parameters of the optimal parameters of an additional term inside theta_opt
        opt = minimize(
            loss,
            theta_0,
            method="CG",
            options={"maxiter": self.maxiter},
        )
        theta_opt = opt.x
        loss_ = opt.fun
        return theta_opt, loss_

    def _get_nonlin(self):
        if self.task_type == "regression":
            return ReLU
        elif self.task_type == "classification":
            return sigmoid
        else:
            raise RuntimeError(f"Unuspported task type {self.task_type}")

    def _get_nonlin_name(self) -> str:
        if self.task_type == "regression":
            return "ReLU"
        elif self.task_type == "classification":
            return "Sigmoid"
        else:
            raise RuntimeError(f"Unuspported task type {self.task_type}")

    # Extract information from the model

    def predict(self, X, exclude_term=False, exclusion_id=0):
        # Returns the evaluation of the model minus term # exclusion_id at the point in X
        result = np.zeros(len(X))
        index_list = [k for k in range(len(self.terms_list))]
        if exclude_term:
            index_list.pop(exclusion_id)
        for k in index_list:
            meijer_g, v, w = self.terms_list[k]
            result = result + w * meijer_g.evaluate(
                self._get_nonlin()(np.matmul(X, v))
                / (np.sqrt(self.dim_x) * np.linalg.norm(v))
            )
        return result

    def get_expression(self):
        # Returns the symbolic expression of the model
        expression = 0
        for k in range(len(self.terms_list)):
            meijer_gk, _, w_k = self.terms_list[k]
            argument_str = f"[{self._get_nonlin_name()}(P{k + 1})]"
            argument_symbol = Symbol(argument_str)
            expression += w_k * meijer_gk.expression(x=argument_symbol)
        return expression

    def get_projections(self):
        # Returns the projections appearing in the symbolic expression
        proj_list = []
        for k in range(len(self.terms_list)):
            _, vk, _ = self.terms_list[k]
            symbol_k = 0
            for j in range(self.dim_x):
                symbol_k += vk[j] * Symbol("X" + str(j + 1))
            proj_list.append(symbol_k)
        return proj_list

    def string_projections(self):
        # Returns a string containing all projections
        proj_str = ""
        proj_list = self.get_projections()
        for k in range(len(proj_list)):
            proj_str += "P" + str(k + 1) + " = " + str(proj_list[k]) + "\n"
        return proj_str

    def print_projections(self):
        # Prints the projections appearing in the symbolic expression
        print(self.string_projections())

    def get_taylor(self, x0, approx_order):
        # Returns the Taylor expansion around x0 of order approx_order for our model
        expression = 0
        symbol_list = [Symbol("X" + str(k)) for k in range(self.dim_x)]
        for k in range(len(self.terms_list)):
            g_k, v_k, w_k = self.terms_list[k]
            x_k = np.dot(v_k, x0) / (np.sqrt(self.dim_x) * np.linalg.norm(v_k))
            if x_k > 0:
                P_k = 0
                for n in range(self.dim_x):
                    P_k += (
                        v_k[n]
                        * symbol_list[n]
                        / (np.sqrt(self.dim_x) * np.linalg.norm(v_k))
                    )
                coef_k = mpmath.chop(mpmath.taylor(g_k.math_expr, x_k, approx_order))
                for n in range(len(coef_k)):
                    if n > 0:
                        expression += w_k * coef_k[n] * (P_k - x_k) ** n
                    else:
                        expression += w_k * coef_k[n]
        return expression

    def get_feature_importance(self, x0):
        # Returns the feature importance for a prediction at x0
        x0 = np.asarray(x0)

        importance_list = [self.eps for _ in range(self.dim_x)]
        for k in range(len(self.terms_list)):
            g_k, v_k, w_k = self.terms_list[k]
            x_k = np.dot(v_k, x0) / (np.sqrt(self.dim_x) * np.linalg.norm(v_k))
            if x_k > 0:
                coef_k = mpmath.chop(mpmath.taylor(g_k.math_expr, x_k, 1))
                for n in range(self.dim_x):
                    importance_list[n] += sympify(
                        w_k
                        * coef_k[1]
                        * v_k[n]
                        / (np.sqrt(self.dim_x) * np.linalg.norm(v_k))
                    )
        return importance_list

        # Change the model:

    def tune_new_term(self, X, g_order, theta_0):
        # Tunes a new term for the model for f with a Meijer G-function of order g_order

        def split_theta(theta):
            # Splits theta in the Meijer G-function part, the vector part and the weight part
            _, _, p, q = g_order
            theta_g = np.concatenate((theta[: p + q], np.array([1.0])))
            theta_v = theta[p + q : -1]
            theta_w = theta[-1]
            return theta_g, theta_v, theta_w

        def loss(theta):
            # Computes the loss for a new term of parameter theta
            residual_list = self.current_resi
            theta_g, v_, w_ = split_theta(theta)
            meijer_g_ = MeijerG(theta=theta_g, order=g_order)
            Y = w_ * meijer_g_.evaluate(
                self._get_nonlin()(
                    np.matmul(X, v_) / (np.sqrt(self.dim_x) * np.linalg.norm(v_))
                )
            )

            loss_ = np.mean((Y - residual_list) ** 2)
            return loss_

        new_theta, new_loss = self.optimize_CG(loss, theta_0)
        new_theta_meijer, new_v, new_w = split_theta(new_theta)
        new_meijerg = MeijerG(theta=new_theta_meijer, order=g_order)
        return new_meijerg, new_v.squeeze(), new_w, new_loss

    def fit(self, f, X_raw):
        # Fits a model for f via a projection pursuit strategy
        X = np.asarray(X_raw)

        self.dim_x = len(X[0])
        self.n_points = len(X)
        h_dic = load_h()
        loss_tol = self.loss_tol
        w0 = 1.0
        count = 0

        Y_target = np.asarray(f(X_raw)).squeeze()

        current_loss = np.mean(Y_target**2)

        self.loss_list.append(current_loss)
        while current_loss > loss_tol and count < self.patience:
            count += 1
            new_loss_list = []
            new_terms_list = []
            np.random.seed(self.random_seed)
            v0 = np.random.randn(self.dim_x)
            self.current_resi = Y_target - self.predict(X)

            log.info(100 * "%")
            log.info(f"Now working on term number {count}.")

            for key in self.baselines:
                log.info(100 * "=")
                log.info(f"Now working on hyperparameter tree number {key}.")

                theta_g0, g_order = h_dic[key]
                theta_0 = np.concatenate((theta_g0, v0, [w0]))
                new_meijer_g, new_v, new_w, new_loss = self.tune_new_term(
                    X, g_order, theta_0
                )
                new_loss_list.append(new_loss)
                new_terms_list.append([new_meijer_g, new_v, new_w])
                if new_loss < loss_tol:
                    log.info(100 * "=")
                    log.info(
                        "The algorithm stopped because the desired precision was achieved."
                    )
                    break
            best_index = np.argmin(np.array(new_loss_list))
            best_term = new_terms_list[int(best_index)]
            best_loss = new_loss_list[int(best_index)]
            if best_loss / current_loss < self.ratio_tol:
                self.terms_list.append(best_term)
                self.loss_list.append(best_loss)

                log.info(100 * "=")
                log.info(f"The tree number {best_index + 1} was selected as the best.")
                self.backfit(f, X_raw)
                current_loss = self.loss_list[-1]
            else:
                log.info(100 * "=")
                log.info(
                    "The algorithm stopped because it was unable to find "
                    f"a term that significantly decreases the loss. Loss ratio: {best_loss / current_loss}"
                )
                break
            log.info(100 * "=")
            log.info(100 * "=")
            log.info(f"The current model has the following expression: {self}")
            log.info(f"The current value of the loss is: {current_loss}.")

        log.info(100 * "-")
        log.info(100 * "-")
        log.info("The final model has the following expression:")
        log.info(self)

        log.info(self.string_projections())

        log.info(f"The number of terms inside the expansion is {len(self.terms_list)}.")
        log.info(f"The current loss is {self.loss_list[-1]}.")
        log.info(100 * "-")

    def backfit(self, f, X_raw):
        # The backfitting procedure invoked at each iteration of fit to correct the previous terms
        X = np.asarray(X_raw)
        for k in range(len(self.terms_list) - 1):
            log.info(100 * "=")
            log.info(f"Now backfitting term number {k + 1}.")

            target = np.asarray(f(X_raw)).squeeze()

            self.current_resi = target - self.predict(
                X, exclude_term=True, exclusion_id=k
            )
            meijer_g0, v0, w0 = self.terms_list[k]
            theta_meijer0 = meijer_g0.theta[:-1]
            theta0 = np.concatenate((theta_meijer0, v0, [w0]))
            g_order = meijer_g0.order
            new_meijerg, new_v, new_w, new_loss = self.tune_new_term(X, g_order, theta0)
            if new_loss < self.loss_list[-1]:
                self.terms_list[k] = [new_meijerg, new_v, new_w]
                self.loss_list[-1] = new_loss
        log.info(100 * "=")
        log.info("Backfitting complete.")
