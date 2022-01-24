import scipy.optimize as opt
import numpy as np

class Model:
    def __init__(self, dates, feature_data):
        self.dates = dates
        self.feature_data = feature_data
        self.model_function = None
        self.optimal_values = None
        self.covariance = None

        #Models description:
        self.ExponentialModel = "exponential_model"
        self.PolynomialModel = "polynomial_model"
        self.PeriodicExponential = "periodic_exponential"
        self.LogisticModel = "logistic_model"

        #Flag set to true if the curve fitting with a model was successful
        self.status = False



    def applyModel(self, model_type, initial_value=None):

        if model_type == self.ExponentialModel:
            self.model_function = self._exp_growth
        elif model_type == self.PolynomialModel:
            self.model_function = self._poly_fit
        elif model_type == self.PeriodicExponential:
            self.model_function = self._periodic_exp_growth
        elif model_type == self.LogisticModel:
            self.model_function = self._logistics
        else:
            raise Exception(f"Model {model_type} not supported")

        status = "Successful"
        try:
            self.optimal_values, self.covariance = opt.curve_fit(
                self.model_function, self.dates, self.feature_data,
                                           p0=initial_value)
            self.status = True
        except Exception as e:
            self.status = False
            status = "Failed: " + e


    def getTheoriticalValues(self):
        if not self.status:
            raise Exception("Model was not applied or model failed to apply")

        theoritical_values = self.model_function(self.dates, 
                                                 *self.optimal_values)
        return theoritical_values

    def getPredictions(self,indicies):
        retval = []
        for index in indicies:
            retval.append(self.model_function(index, *self.optimal_values))

        return retval

    '''
    Models functions list
    '''

    def _exp_growth(self, t, scale, growth):
        """ Computes exponential function with scale and growth as free 
        parameters"""
        f = scale * np.exp(growth * (t - 1990))

        return f


    def _logistics(self, t, a, k, t0):
        """ Computes logistics function with scale and incr as free 
        parameters"""
        f = a / (1.0 + np.exp(-k * (t - t0)))

        return f


    def _poly_fit(self, t,  a4, a3, a2, a1, a0):
        """ Computes 4 polynomial function"""
        poly = [  a4, a3, a2, a1, a0]
        x = t - 1990

        f = poly[0]
        for p in poly[1:]:
            f = f * x + p
        return f


    def _periodic_exp_growth(self, t, A1, G1, A2, period, bias):
        """ Similar to exponential except added bias and period """
        x = t - 1990
        f = A1 * np.exp(G1 * x) + A2 * x * np.sin(period * x) + bias
        return f