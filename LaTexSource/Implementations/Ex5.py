import numpy as np
import moduleRK4SIR as rk4
import csv
import matplotlib.pyplot as plt

def traniningLoss(lossFuntion, proposal, data, t0, randomness, randomnessScaleFactor, searchTimeBeforeScaling, m):   
    # lossFuntion: L(t, data): loss value of predicted data base on t and real data, it's up to you to decide the prediction outside
    # proposal(t, randomness) = q(t, randomess) : return a new t' from current t, should be sysmetric   
    # data: real world data, used to find loss
    # t0: inital parameter
    # randomness: the inital randomness to be passed to the proposal, the length of this list should be the same as t0
    # randomnessScaleFactor: scaling to increase randomness used to bump up randomness at searching for a long time
    # searchTimeBeforeScaling: Number of loop without new change in loss value before scaling up the randoomness. This number will grow as the search area grows, by scale to the power of dimension
    # m: number of accepted new t before stop

    #Return:
    #t: final [beta, gamma]

    #converting all list to array to prevent further complication of matrix operation later on, and also to boost performance
    randomness = np.array(randomness)
    t0 = np.array(t0)
    data = np.array(data)
    #end convert
   
    #E.g: if one dimension of t scale by a factor of 2, and the dimension is 2, the area will be quadruple, therefore, we'll spend 4 times more time looking inside it
    dimension = len(t0)
    timeScaleFactor = int(np.power(randomnessScaleFactor, dimension))

    t = t0.copy()
    rand = randomness.copy()
    searchTime = searchTimeBeforeScaling
    loss = lossFuntion(t, data)

    ac = 0
    scaleCounter = 0
    while ac < m:
        tNew = proposal(t, rand)
        newLoss = lossFuntion(tNew, data)
        
        if(newLoss < loss):          
            t = tNew
            loss = newLoss 
            
            print(ac, int(loss), t, rand, searchTime, flush=True)
            rand = randomness.copy() # reset randomness
            searchTime = searchTimeBeforeScaling # reset searchTime
            scaleCounter = 0 # reset counter
            ac += 1            
        else:
            scaleCounter += 1
            if(scaleCounter > searchTime):
                rand = randomnessScaleFactor*rand               
                searchTime = timeScaleFactor*searchTime

                print("BumpUp randomness: ", rand, searchTime, flush=True)
                scaleCounter = 0 # reset counter


        """
        r = 1 if loss > newLoss else 0.01
        a = np.random.uniform(0, 1)
        if a <= r:
            print(insideTol, int(loss), t)
            t = tNew
            loss = newLoss
        """

    return t, loss

def SIRlossFunction(t, data):
    predictI, predictR = rk4.RK4SIR(1000000000, I0, R0, t[0], t[1], time, 1)
    #dataI = np.array(data[0])
    #dataS = np.array(data[1])
    return (meanSquareError(data[0], predictI) + meanSquareError(data[1], predictR))/2

def meanSquareError(data, prediction):
    return np.sum(np.power(data - prediction, 2)) / len(data)

def normalProposal(t, randomness): #only propose non-negative for beta and gamma
    # tPrime ~ Normal(t, sigma)
    r = []
    i = 0
    for para in t:
        newT = np.random.normal(para, randomness[i], 1)[0]
        while newT <= 0: #ensure non-negative
            newT = np.random.normal(para, randomness[i], 1)[0]
        r.append(newT)
        i += 1
    return np.array(r)


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
    #IRList[i] = np.array(IRList[i])
IRList = np.array(IRList) 

RList = [[0 for x in range(numberOfDay-offset)] for y in range(numberOfCountry)]
for i in range (0, numberOfCountry):
    for j in range (0, numberOfDay-offset):
        RList[i][j] = int(data2[j+1+offset][i+1]) + int(data3[j+1+offset][i+1])
    #RList[i] = np.array(RList[i])
RList = np.array(RList)
        
IList = [[0 for x in range(numberOfDay-offset)] for y in range(numberOfCountry)]
for i in range (0, numberOfCountry):
    for j in range (0, numberOfDay-offset):
        IList[i][j] = IRList[i][j] - RList[i][j]
    #IList[i] = np.array(IList[i])
IList = np.array(IList)


#csv part done

print("Country List:")
count = 0
for s in countryList:
    print(count,". ",s)
    count += 1

countryID = int(input("Choose a country: "))
start = int(input("Choose start day index (0-168): "))
end = int(input("Choose end day index (startDay -> 168): "))

data = []
data.append(np.array(IList[countryID])[start:end])
data.append(np.array(RList[countryID])[start:end])

I0 = data[0][0]
R0 = data[1][0]
time = end - start - 1

#betaGamma, loss = traniningLoss(1.1, SIRlossFunction, normalProposal, data, [0, 0], 200)
generation = int(input("Number of generation: "))
betaGamma, loss = traniningLoss(SIRlossFunction, normalProposal, data, [0.1, 0.1], [0.0001, 0.0001], 2, 5, generation)

print("beta = ", betaGamma[0])
print("gamma = ", betaGamma[1])
print("Loss = ", loss)


#! Ploting Graph
I_predict, R_predict = rk4.RK4SIR(1000000000, I0, R0, betaGamma[0], betaGamma[1], time, 1)
x = []
for i in range(0, time + 1):
    x.append(i)

plt.figure(figsize=(12,7))
# Real Data
plt.subplot(121)
plt.plot(x, data[0], label="Infected", color='red')
plt.plot(x, data[1], label="Removed", color='blue')
plt.xlabel('Week')
plt.ylabel('People')

plt.yscale('linear')
plt.title('Real Covid Data')
plt.legend()

# Predicted Data
plt.subplot(122)
plt.plot(x, I_predict, label="Infected", color='red')
plt.plot(x, R_predict, label="Removed", color='blue')
plt.xlabel('Week')
plt.ylabel('People')

plt.yscale('linear')
plt.title('Predicted Covid Data')
plt.legend()

plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35)
plt.show()

input()