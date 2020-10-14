from __future__ import absolute_import, division, print_function
import numpy as np
from sympy import *
from scipy.optimize import minimize
import warnings
import os, sys

warnings.filterwarnings("ignore")
if not sys.warnoptions:
    warnings.simplefilter("ignore")

from pysymbolic.classes.faithful_model import FaithfulModel
from pysymbolic.models.special_functions import MeijerG
from pysymbolic.utilities.performance import compute_Rsquared
from gplearn.genetic import SymbolicRegressor


def load_H():
    # Hyperparameters for Meijer G-functions

    H = {
        'hyper_1': (np.array([0.0, 0.1, 0.1, 0.0, 0.1, 1.0]), [2, 1, 2, 3]),  # Parent of exp, Gamma, gamma
        'hyper_2': (np.array([2.0, 2.0, 2.0, 1.0, 1.0]), [0, 1, 3, 1]),  # Parent of monomials
        'hyper_3': (np.array([0.5, 0.0, 1.0]), [1, 0, 0, 2]),  # Parent of sin , cos, sh, ch
        'hyper_4': (np.array([0.0, 0.0, 1.0, 0.0, 1.0]), [1, 1, 2, 2]),  # Parent of Step functions
        'hyper_5': (np.array([1.0, 1.2, 3.0, 3.3, 0.4, 1.5, 1.0]), [2, 2, 3, 3]),  # Parent of ln, arcsin, arctg
        'hyper_6': (np.array([1.1, 1.2, 1.3, 1.4, 1.0]), [2, 0, 1, 3])  # Parent of Bessel functions
    }

    return H


def load_hyperparameters():
    # Hyperparameters aother than for the Meijer G-function

    hyperparameters = {
        'lasso_cost': 0.1,
        'grad_tol': 1.0e-5,
        'loss_tol': 1.0e-5
    }
    return hyperparameters


def optimize(loss, theta_0, grad_tol=load_hyperparameters()['grad_tol']):
    opt = minimize(loss, theta_0, method='CG',
                   options={'gtol': grad_tol, 'disp': True, 'maxiter': 100})
    loss_ = opt.fun
    theta_opt = opt.x
    return theta_opt, loss_


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def unpack_theta(theta, g_in_order, g_out_order):
    # Unpack the group of parameters contained in theta

    m_in, n_in, p_in, q_in = g_in_order
    m_out, n_out, p_out, q_out = g_out_order
    theta_in = theta[:p_in + q_in + 1]
    theta_out = theta[p_in + q_in + 1: p_in + q_in + p_out + q_out + 2]
    eps = theta[-2]
    ksi = theta[-1]
    return theta_in, theta_out, eps, ksi


def symbolic_modeling(f, dim_x,
                      g_in_order, g_out_order, theta_in_0, theta_out_0,
                      eps_0=1, ksi_0=1, x_range=[0, 1], n_points=100):
    # Returns a symbolic metamodel for f

    def mse_loss(theta):
        # Computes the MSE loss

        theta_in, theta_out, eps, ksi = unpack_theta(theta, g_in_order, g_out_order)
        G_in = MeijerG(theta=theta_in, order=g_in_order)
        G_out = MeijerG(theta=theta_out, order=g_out_order)
        X_flat = X.flatten()
        Arg_flat = np.array([x + eps*i for i in range(2*dim_x+1) for x in X_flat])
        Im_flat = G_in.evaluate(sigmoid(Arg_flat))
        Im_inj = np.array([ksi**((j+1)%(dim_x+1)) * Im_flat[j]
                           for j in range(len(Im_flat))]).reshape(2*dim_x+1, n_points,  dim_x)  # i,n,j
        Im_in = np.sum(Im_inj, axis=2)
        Add_in = np.array([[i for _ in range(n_points)] for i in range(2*dim_x+1)])
        Arg_in = Im_in + Add_in
        Arg_flat = Arg_in.flatten()
        Im_flat = G_out.evaluate(sigmoid(Arg_flat))
        Y_in = Im_flat.reshape(2*dim_x+1, n_points)
        Y_est = np.sum(Y_in, axis=0)
        Y_true = np.apply_along_axis(f, 1, X)
        loss = np.mean((Y_est-Y_true)**2)
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        print("Loss : ", loss)
        print("Expression for G_in : ", G_in.expression())
        print("Expression for G_out : ", G_out.expression())
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        return loss

    X = np.random.uniform(x_range[0], x_range[1], dim_x * n_points)
    X = X.reshape((n_points, dim_x))
    g_tol = load_hyperparameters()['grad_tol']
    theta_0 = np.concatenate((theta_in_0, theta_out_0, [eps_0, ksi_0]))
    theta_opt, loss = optimize(mse_loss, theta_0, grad_tol=g_tol)
    faithful_model = FaithfulModel(dim_x, g_in_order, g_out_order, unpack_theta(theta_opt, g_in_order, g_out_order))
    return faithful_model, loss


def get_symbolic_model(f, dim_x, n_points=100, x_range=[0, 1]):
    H = load_H()
    loss_tol = load_hyperparameters()['loss_tol']
    faithful_models = []
    losses = []

    for k in range(len(H)):
        for l in range(len(H)):
            print("===================================================================================================")
            print("Testing Hyperparameter Configuration k =  ", k + 1 , " ; l = ", l+1)
            faithful_model, loss = symbolic_modeling(f, dim_x,
                                                      H['hyper_' + str(k + 1)][1], H['hyper_' + str(l + 1)][1],
                                                      H['hyper_' + str(k + 1)][0], H['hyper_' + str(l + 1)][0],
                                                      n_points=n_points, x_range=x_range)

            faithful_models.append(faithful_model)
            losses.append(loss)

            if losses[-1] <= loss_tol:
                print("==========The desired loss was achieved so the algorithm stopped==========")
                break

    best_model = np.argmin(np.array(losses))
    X = np.random.uniform(x_range[0], x_range[1], dim_x * n_points)
    X = X.reshape((n_points, dim_x))
    Y_true = f(X).reshape((-1, 1))
    Y_est = faithful_models[best_model].evaluate(X).reshape((-1, 1))
    R2_perf = compute_Rsquared(Y_true, Y_est)

    return faithful_models[best_model], R2_perf


def symbolic_regressor(f, npoints, xrange):
    X = np.linspace(xrange[0], xrange[1], npoints).reshape((-1, 1))
    y = f(X)

    est_gp = SymbolicRegressor(population_size=5000,
                               generations=20, stopping_criteria=0.01,
                               p_crossover=0.7, p_subtree_mutation=0.1,
                               p_hoist_mutation=0.05, p_point_mutation=0.1,
                               max_samples=0.9, verbose=1,
                               parsimony_coefficient=0.01, random_state=0)

    est_gp.fit(X, y)

    sym_expr = str(est_gp._program)

    converter = {
        'sub': lambda x, y: x - y,
        'div': lambda x, y: x / y,
        'mul': lambda x, y: x * y,
        'add': lambda x, y: x + y,
        'neg': lambda x: -x,
        'pow': lambda x, y: x ** y
    }

    x, X0 = symbols('x X0')
    sym_reg = simplify(sympify(sym_expr, locals=converter))
    sym_reg = sym_reg.subs(X0, x)

    Y_true = y.reshape((-1, 1))
    Y_est = np.array([sympify(str(sym_reg)).subs(x, X[k]) for k in range(len(X))]).reshape((-1, 1))

    R2_perf = compute_Rsquared(Y_true, Y_est)

    return sym_reg, R2_perf
