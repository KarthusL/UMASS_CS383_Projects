from scipy.special import gamma
import numpy as np
import matplotlib.pyplot as plt
from Data import Data

"""
CS383: Hw6
Instructor: Ian Gemp
TAs: Scott Jordan, Yash Chandak
University of Massachusetts, Amherst

README:

Feel free to make use of the function/libraries imported
You are NOT allowed to import anything else.

Following is a skeleton code which follows a Scikit style API.
Make necessary changes, where required, to get it correctly running.

Note: Running this empty template code might throw some error because 
currently some return values are not as per the required API. You need to
change them.

Good Luck!
"""

class Posterior:
    def __init__(self, limes, cherries, a=2, b=2):
        self.a = a
        self.b = b
        self.limes = limes          # shape: (N,)
        self.cherries = cherries    # scalar int
        self.N = np.shape(self.limes)[0]

    def get_MAP(self):
        """
        compute MAP estimate
        :return: MAP estimates for diff. values of lime; shape:(N,)
        """
        # WRITE the required CODE HERE and return the computed values
        result = np.zeros(self.N)
        c = self.cherries
        l = self.limes
        for i in list(range(self.N)):
            result[i] = 1 - (c + self.a - 1) / (c + l[i] + self.a + self.b - 2)
        return result

    def get_finite(self):
        """
        compute posterior with finite hypotheses
        :return: estimates for diff. values of lime; shape:(N,)
        """
        # WRITE the required CODE HERE and return the computed values

        result = np.zeros(self.N)
        N = list(range(self.N))
        for i in range(len(result)):
            result[i] = ((0.25) * (0.2) * (0.25 ** N[i])) + ((0.5) * (0.4) * (0.5 ** N[i])) + ((0.75) * (0.2) * (0.75 ** N[i])) + ((1) * (0.1) * (1 ** N[i]))
            notP = ((1) * (0.1) * (0 ** N[i])) + ((0.75) * (0.2) * (0.25 ** N[i])) + ((0.5) * (0.4) * (0.5 ** N[i])) + ((0.25) * (0.2) * (0.75 ** N[i]))
            alpha = 1 / (result[i] + notP)
            result[i] = result[i] * alpha
        return result



    def alpha(self, a, i, b, len):
        a = gamma(self.a + i + self.b) / gamma(self.a + i) * gamma(self.b)
        return a


    def get_infinite(self):
        """
        compute posterior with beta prior
        :return: estimates for diff. values of lime; shape:(N,)
        """
        # WRITE the required CODE HERE and return the computed values
        P = np.zeros(self.N)
        for i in range(self.N):
            P[i] = gamma(self.a + i + 1) * gamma(self.b) / gamma(self.a + i + self.b + 1)
            alpha = self.alpha(self.a, i, self.b, range(self.N))
            P[i] *= alpha
        return P

if __name__ == '__main__':
    # Get data
    data = Data()
    limes, cherries = data.get_bayesian_data()
    print("limes", limes, "cherries", cherries)


    # Create class instance
    posterior = Posterior(limes=limes, cherries=cherries)
    print(posterior.N)

    # PLot the results
    plt.plot(limes, posterior.get_MAP(), label='MAP')
    plt.plot(limes, posterior.get_finite(), label='5 Hypotheses')
    plt.plot(limes, posterior.get_infinite(), label='Bayesian with Beta Prior')
    plt.legend()
    plt.savefig('figures/Q4.png')
