import math
import numpy as np
import matplotlib.pyplot as plt
import Ex3libMHSimplified as MH

m = int(input("Enter m: "))
sigma1 = float(input("Enter sigma1: ")) #1.73
sigma2 = float(input("Enter sigma2: ")) #0.1/7
burnIn = int(input("Enter burnIn: "))

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
    #return uniform_density(10**-2, 1, gamma)

def logPDF(t):
    #pi(beta, gamma) = pi(beta) * pi(gamma)
    return np.log(piBeta(t[0]) * piGamma(t[1]))

def normalProposal(t):
    # tPrime ~ Normal(t, sigma^2)
    return [np.random.normal(t[0],sigma1,1)[0],np.random.normal(t[1],sigma2,1)[0]]

def logNormalProposalPDF(tPrime, t):
    return np.log(normal_density(sigma1,t[0],tPrime[0]) * normal_density(sigma2,t[1],tPrime[1]))

dummyPDF = lambda tPrime, t: 1 # we can use this instead of the proposalPDF if it's symmertric

trace, ac, re, iac, ire = MH.metropolisHasting(m, logPDF, normalProposal, logNormalProposalPDF, [1, 0.1], burnIn)

sumg = 0
sumb = 0
betaTrace = []
gammaTrace = []

for b,g in trace:
    sumb += b
    sumg += g
    betaTrace.append(b)
    gammaTrace.append(g)

meanb = sumb/len(trace)
meang = sumg/len(trace)

betaAccepted = []
gammaAccepted= []
for b,g in ac:
    betaAccepted.append(b)
    gammaAccepted.append(g)

betaRejected = []
gammaRejected= []
for b,g in re:
    betaRejected.append(b)
    gammaRejected.append(g)

dataHist = plt.figure(figsize=(12,9))
ax = dataHist.add_subplot(2,1,1)
ax.plot(iac, betaAccepted, '+', color='blue', label="Accepted")
ax.plot(ire, betaRejected, 'x', color='red', label="Rejected")
ax.set_ylabel("Value")
ax.set_title("Figure 1: Sampling Plot of beta")
ax.legend()

ax = dataHist.add_subplot(2,1,2)
ax.plot(iac, gammaAccepted, '+', color='blue', label="Accepted")
ax.plot(ire, gammaRejected, 'x', color='red', label="Rejected")
ax.set_xlabel("Iteration")
ax.set_ylabel("Value")
ax.set_title("Figure 2: Sampling Plot of gamma")
ax.legend()


bTrace = plt.figure(figsize=(10,10))
ax = bTrace.add_subplot(2,2,1)
ax.plot(betaTrace)
ax.set_xlabel("Iteration")
ax.set_ylabel("Value")
ax.set_title("Figure 3: Trace of beta")

ax = bTrace.add_subplot(2,2,2)
ax.plot(gammaTrace)
ax.set_xlabel("Iteration")
ax.set_ylabel("Value")
ax.set_title("Figure 4: Trace of gamma")

ax = bTrace.add_subplot(2,2,3)
ax.hist(betaTrace, bins=50 ,)
ax.set_xlabel("Value")
ax.set_ylabel("Frequency")
ax.set_title("Figure 5: Histogram of beta")

ax = bTrace.add_subplot(2,2,4)
ax.hist(gammaTrace, bins=50 ,)
ax.set_xlabel("Value")
ax.set_ylabel("Frequency")
ax.set_title("Figure 6: Histogram of gamma")


print("Acceptance Rate = ", len(ac)/(len(ac)+len(re)))

print("Expected b = ", meanb)
print("Expected g = ", meang)

print("Done", flush = True)


plt.show()

input()