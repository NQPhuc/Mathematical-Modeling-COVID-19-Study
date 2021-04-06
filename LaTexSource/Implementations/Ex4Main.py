import Ex4Prior as pr
import Ex3libMHSimplified as MH
import csv
import numpy as np
import math
import scipy.special as sp


def loglikelihood_standard_normal(X): #for a continuous distribution, this gave a very very low number
    n = len(X)
    return -1/2*np.sum(np.power(X,2))-n*np.log(np.sqrt(2*np.pi))

bound = 1 #confidence range, should be positive
def loglikelihood_standard_normal_accept_ratio(X):
    return np.log(((X > -bound) & (X < bound )).sum() / len(X))

def loglikelihoodXinGammaDist(beta, gamma, X):#pdf of gammaDist, we don't use this, it's very imprecise
    n = len(X)
    return n*beta*np.log(gamma) - n*np.log(sp.gamma(beta)) + (beta-1)*np.sum(np.log(X)) - gamma*np.sum(X)


#calling ex3 sampling function for prior of beta and gamma, the sample is stored in the list 'trace'
m = 10000#int(input("Enter m: "))
burnIn = 3000#int(input("Enter burnIn: "))
trace, ac, re, iac, ire = MH.metropolisHasting(m, pr.logPDF, pr.normalProposal, pr.dummyPDF, [1, 0.1], burnIn)

#extracting and reprocess csv
with open('cumumlativedaily_confirmed.csv','rt')as f:
  fileContent = csv.reader(f)
  data1 = []
  for row in fileContent:
      data1.append(row)
with open('cumumlativedaily_death.csv','rt')as f:
  fileContent = csv.reader(f)
  data2 = []
  for row in fileContent:
      data2.append(row)
with open('cumumlativedaily_recovered.csv','rt')as f:
  fileContent = csv.reader(f)
  data3 = []
  for row in fileContent:
      data3.append(row)

numberOfDay = len(data1) - 1
numberOfWeek = int(numberOfDay / 7)
numberOfCountry = len(data1[0]) - 1 #will crash if data is empty

countryList = []
for i in range (0, numberOfCountry):
    countryList.append(data1[0][i+1])

offset = 0

IRList = [[0 for x in range(numberOfDay-offset)] for y in range(numberOfCountry)]
for i in range (0, numberOfCountry):
    for j in range (0, numberOfDay-offset):
        IRList[i][j] = int(data1[j+1+offset][i+1])
IRList = np.array(IRList) 

RList = [[0 for x in range(numberOfDay-offset)] for y in range(numberOfCountry)]
for i in range (0, numberOfCountry):
    for j in range (0, numberOfDay-offset):
        RList[i][j] = int(data2[j+1+offset][i+1]) + int(data3[j+1+offset][i+1])
RList = np.array(RList)

offset += 1        
IList = [[0 for x in range(numberOfDay-offset)] for y in range(numberOfCountry)]
for i in range (0, numberOfCountry):
    for j in range (0, numberOfDay-offset):
        IList[i][j] = IRList[i][j] - RList[i][j]
IList = np.array(IList)

dIdRList = [[0 for x in range(numberOfDay-offset)] for y in range(numberOfCountry)]
for i in range (0, numberOfCountry):
    for j in range (0, numberOfDay-offset):
        dIdRList[i][j] = IRList[i][j+1] - IRList[i][j]
        if dIdRList[i][j] <= 0 : dIdRList[i][j] = 0.001
dIdRList = np.array(dIdRList)

dRList = [[0 for x in range(numberOfDay-offset)] for y in range(numberOfCountry)]
for i in range (0, numberOfCountry):
    for j in range (0, numberOfDay-offset):
        dRList[i][j] = RList[i][j+1] - RList[i][j]
        if dRList[i][j] <= 0 : dRList[i][j] = 0.001
dRList = np.array(dRList)

#csv part done

#ContryOffset = [42,42,51,45,45,46,35,35,47,30,48,45,48,36]
ContryOffset = [42,45,51,45,45,46,40,40,49,37,50,45,56,45]
#gammaDelay =   [0,0, 0,0,0,0,0,0, 0,0,0,0,0,0]
#betaDelay =    [0,0, 0,0,0,0,0,0, 0,0,0,0,0,0]
PolicyStart =  [64,69,64,62,64,63,68,69,66,45,66,70,63,65] #lag time included
PolicyEnd = [102,137,120,88,167,127,131,117,134,102,128,133,130,125]
R0List = []
R0ListPolicy = []
for i in range (0, numberOfCountry, 1):
    dRL = []
    dIdRL = []
    IListGamma = []
    IListBeta = []
    #maxDelay = gammaDelay[i] if gammaDelay[i] > betaDelay[i] else betaDelay[i]

    #start = PolicyStart[i] if(PolicyStart[i] != -1) else len(dRList[i])-maxDelay      
    start = PolicyStart[i]    
    for j in range(ContryOffset[i], start+1):
        #dRL.append(dRList[i][j+gammaDelay[i]])
        #dIdRL.append(dIdRList[i][j+betaDelay[i]])
        dRL.append(dRList[i][j])
        dIdRL.append(dIdRList[i][j])
        IListGamma.append(IList[i][j])
        IListBeta.append(IList[i][j])
    
    dRL = np.array(dRL)
    dIdRL = np.array(dIdRL)
    IListGamma = np.array(IListGamma)
    IListBeta = np.array(IListBeta)

    dRLPolicy = []
    dIdRLPolicy = []
    IListPolicy = []
    #for k in range(start, len(dRList[i])-maxDelay):
        #dRLPolicy.append(dRList[i][k+gammaDelay[i]])
        #dIdRLPolicy.append(dIdRList[i][k+betaDelay[i]])
    for k in range(start, PolicyEnd[i]):
        dRLPolicy.append(dRList[i][k])
        dIdRLPolicy.append(dIdRList[i][k])
        IListPolicy.append(IList[i][k])

    dRLPolicy = np.array(dRLPolicy)
    dIdRLPolicy = np.array(dIdRLPolicy)
    IListPolicy = np.array(IListPolicy)

    sum = 0
    sumPolicy = 0
    for beta, gamma in trace:
        lamdaGamma = gamma*IListGamma
        lamdaBeta = beta*IListBeta
        likelihoodGamma = loglikelihood_standard_normal_accept_ratio((dRL-lamdaGamma) / np.sqrt(lamdaGamma))             
        likelihoodBeta = loglikelihood_standard_normal_accept_ratio((dIdRL-lamdaBeta) / np.sqrt(lamdaBeta))
        likelihood = likelihoodGamma + likelihoodBeta
        sum += np.exp(likelihood) * beta/gamma

        lamdaGamma = gamma*IListPolicy
        lamdaBeta = beta*IListPolicy
        likPolicyGamma = loglikelihood_standard_normal_accept_ratio((dRLPolicy-lamdaGamma) / np.sqrt(lamdaGamma))  
        likPolicyBeta = loglikelihood_standard_normal_accept_ratio((dIdRLPolicy-lamdaBeta) / np.sqrt(lamdaBeta))
        sumPolicy += np.exp(likPolicyBeta+ likPolicyGamma) * beta/gamma

    R0List.append(sum)
    R0ListPolicy.append(sumPolicy)

for i in range (0, numberOfCountry, 1):
    print(countryList[i], ": R0 before policy = ", R0List[i], " | R0 after policy = ", R0ListPolicy[i])

print("Done", flush = True)

input()