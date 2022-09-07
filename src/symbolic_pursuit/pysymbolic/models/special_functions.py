# Copyright (c) 2019, Ahmed M. Alaa
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from __future__ import absolute_import, division, print_function

import copy
import json
import sys
import warnings

import numpy as np
from mpmath import mp, taylor
from sympy import Symbol, hyperexpand, lambdify, meijerg, simplify

warnings.filterwarnings("ignore")
if not sys.warnoptions:
    warnings.simplefilter("ignore")

lambdify_cache = {}
approx_expression_cache = {}


def lambdify_wrapper(expr: str):
    if expr in lambdify_cache:
        fn = lambdify_cache[expr]
        return fn

    x = Symbol("x", real=True)
    evaluater_ = lambdify([x], expr)

    lambdify_cache[expr] = evaluater_

    return evaluater_


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class MeijerG:

    """
    The MeijerG class creates an instance of the Meijer G-functions. The the G-function was originally introduced by
    Cornelis Simon Meijer [1,2] as a very general class of special functions that encapsulates most of the known functional
    forms as particular cases. (There are other general function classes such as the generalized  hypergeometric function and
    the  MacRobert E-function, but Meijer's G-function includes those as special cases as well [3].)

    The MeijerG enables fast evaluation and manipulation of Meijer G-functions. It does so by building a wrapper
    over the "meijerg" function implementation of Sympy [4].

    References:
    -----------

    [1] CS Meijer. On the G-function. North-Holland, 1946.
    [2] CS Meijer. Uber whittakersche bezw. besselsche funktionen und deren produkte (english translation:
        About whittaker and bessel functions and their products). Nieuw Archief voor Wiskunde, 18(2):10–29, 1936.
    [3] Richard Beals and Jacek Szmigielski. Meijer G-functions: a gentle introduction. Notices of the AMS, 60(7):866–872, 2013.
    [4] Aaron Meurer, Christopher P. Smith, Mateusz Paprocki, Ondrej Certik, Sergey B. Kirpichev, Matthew Rocklin,
        AMiT Kumar, Sergiu Ivanov, Jason K. Moore, Sartaj Singh, Thilina Rath nayake, Sean Vig, Brian E. Granger,
        Richard P. Muller, Francesco Bonazzi, Harsh Gupta, Shivam Vats, Fredrik Johansson, Fabian Pedregosa,
        Matthew J. Curry, Andy R. Terrel, Stepan, Roucka, Ashutosh Saboo, Isuru Fernando, Sumith Kulal, Robert Cimrman,
        and Anthony Scopatz. Sympy: symbolic computing in python. PeerJ Computer Science, 3:e103, 2017.
    """

    def __init__(
        self,
        theta=[2, 2, 2, 1, 1],
        order=[0, 1, 3, 1],
        evaluation_mode="numpy",
        approximation_order=15,
        **kwargs
    ):

        """
        :param theta: contains the poles and zeros of the Meijer G-function G(a_1,...,a_n,...,a_p; b_1,...,b_p,...,b_n| c * x),
                      where theta = [a_1,..., a_n,..., a_p, b_1,..., b_p,..., b_n, c].
        :param order: The indexes m, n, p, q, respectively.
        :param evaluation_mode: The method used to evaluate the Meijer G function for a vector X. For direct symbolic evaluation
                                using Sympy's evalf() use evaluation_mode='eval'. For fast computation using Numpy, Cython
                                or Theano graphs set evaluation_mode='numpy', 'cython' and 'theano', respectively.
        :param approximation_order: Truncation order of the Taylor series approximation used to implement the fast computation
                                    functions with Cython.
        Default setting for theta=[2, 2, 2, 1, 1], order=[0, 1, 3, 1] corresponds to the identity function
        G(a_1,...,a_n,...,a_p; b_1,...,b_p,...,b_n| c * x) = x

        """

        self.theta = theta
        self.order = order
        self.evaluation_mode = evaluation_mode
        self.approximation_order = approximation_order

        self.set_G_parameters()

    def set_G_parameters(self):

        """
        Sets the poles and zeros of the Meijer G-function based on the input parameters
        """
        a_p_ = self.theta[0 : self.order[2]]
        b_q_ = self.theta[self.order[2] :][: self.order[3]]

        self.a_p = [a_p_[: self.order[1]], a_p_[self.order[1] :]]
        self.b_q = [b_q_[: self.order[0]], b_q_[self.order[0] :]]
        self._const = self.theta[-1]

    def expression(self, x=Symbol("x", real=True)):

        """
        Returns a symbolic expression for the Meijer G-function encapsulated in the class.
        """
        return hyperexpand(meijerg(self.a_p, self.b_q, self._const * x))

    def math_expr(self, x):

        """
        Returns a symbolic expression for the Meijer G-function that is
        compatible with data types used by the math library
        """
        a_p_ = [list(self.a_p[k]) for k in range(len(self.a_p))]
        b_q_ = [list(self.b_q[k]) for k in range(len(self.b_q))]

        return mp.meijerg(a_p_, b_q_, self._const * x)

    def approx_expression(self, midpoint=0.5):
        """
        Returns a polynomial approximate expression for the Meijer G-function using a Taylor series approximation
        """
        key_dict = {
            "a_p": self.a_p,
            "b_p": self.b_q,
            "const": self._const,
            "midpoint": midpoint,
            "approximation_order": self.approximation_order,
        }
        key = json.dumps(key_dict, cls=NumpyEncoder)

        if key in approx_expression_cache:
            return approx_expression_cache[key]
        x = Symbol("x", real=True)

        Taylor_poly_ = taylor(self.math_expr, midpoint, self.approximation_order)
        coeffp = Taylor_poly_[::-1]

        approx_expr = 0

        for k in range(self.approximation_order):

            approx_expr = approx_expr + coeffp[k] * (
                (x - midpoint) ** (self.approximation_order - k)
            )

        approx_expr = approx_expr + coeffp[-1]
        approx_expr = simplify(approx_expr)

        approx_expression_cache[key] = approx_expr

        return approx_expr

    def evaluate(self, X):
        """
        Evaluates the Meijer G function for the input vector X
        """
        if self.evaluation_mode == "eval":

            Y = np.array(
                list(
                    map(
                        lambda z: float(
                            meijerg(self.a_p, self.b_q, self._const * z).evalf()
                        ),
                        list(X),
                    )
                )
            )

        else:
            evaluater_ = lambdify_wrapper(self.approx_expression())
            Y = evaluater_(X)
        return np.real(Y)

    def gradients(self, x):

        h = 0.01
        grads = []

        for u in range(len(self.theta)):

            f_g = copy.deepcopy(self)
            f_g.theta[u] = self.theta[u] + h
            f_g.set_G_parameters()

            grads.append(((np.real(f_g.evaluate(x)) - np.real(self.evaluate(x))) / h))

        return grads
