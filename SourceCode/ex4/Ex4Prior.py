import math
import numpy as np
import matplotlib.pyplot as plt

sigma1 = 1.73#float(input("Enter sigma1: "))
sigma2 = 0.1/7#0.86/7#float(input("Enter sigma2: "))

normal_density = lambda sigma, mu, x :1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (x - mu)**2 / (2 * sigma**2))
uniform_density = lambda a, b, x: 1/(b-a) if (a <= x and x <= b) else 0
gamma_density = lambda k, theta, x : x**(k-1)*np.exp(-x/theta) / (math.gamma(theta) * theta**k) #note: theta is the scale = 1/rate

def piBeta(beta):
    """
    prior distribution of beta is set to uniform U(10^-10, 1) because of lacking knowlegde for covid-19 tranmission rate 
    """
    return uniform_density(10**-3, 6, beta)

def piGamma(gamma):
    """
    prior distribution of gamma is assumed to follow N(0.5, 0.1), where the mean 0.5 is the inverse of averaged recovery time for
    covid-19 (2 weeks) and the standard deviation 0.1 is estimate for the reported recovered period from ~ 1-4 weeks 
    """
    return normal_density(0.1/7, 0.5/7, gamma)
    #return uniform_density(10**-4, 3, gamma)

def logPDF(t):
    #pi(beta, gamma) = pi(beta) * pi(gamma)
    return np.log(piBeta(t[0]) * piGamma(t[1]))

def normalProposal(t):
    # tPrime ~ Normal(t, sigma^2)
    return [np.random.normal(t[0],sigma1,1)[0],np.random.normal(t[1],sigma2,1)[0]]

def logNormalProposalPDF(tPrime, t):
    return np.log(normal_density(sigma1,t[0],tPrime[0]) * normal_density(sigma2,t[1],tPrime[1]))

dummyPDF = lambda tPrime, t: 1 # we can use this instead of the proposalPDF if it's symmertric
