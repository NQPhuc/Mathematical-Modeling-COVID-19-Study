import matplotlib.pyplot as plt

# Input Values
BETA = float(input("Enter beta: "))
GAMMA = float(input("Enter gamma: "))
N = int(input("Enter the total number of people: "))
INIT_INFECTED = int(input("Enter number of initial infected cases: "))
INIT_RECOVERED = int(input("Enter number of initial recovered cases: "))
TIME = int(input("Enter the time to get the approximation: "))


# SIR Class
class SIRValue:
    def __init__(self, suspected_num, infected_num, recovered_num):
        self.suspected = (suspected_num, 0)[suspected_num < 0]
        self.infected = (infected_num, 0)[infected_num < 0]
        self.recovered = (recovered_num, 0)[recovered_num < 0]

    def showValue(self):
        suspected_str = str(self.suspected)
        infected_str = str(self.infected)
        recovered_str = str(self.recovered)

        print("Number of Suspected People: " + suspected_str)
        print("Number of Infected People: " + infected_str)
        print("Number of Recovered People: " + recovered_str)

    def getTotalPopulation(self):
        return self.suspected + self.infected + self.recovered

    def getValueByList(self):
        return [self.suspected, self.infected, self.recovered]


class SIRRatio:
    def __init__(self, infected_rate, recovered_rate):
        self.infected = infected_rate
        self.recovered = recovered_rate


class Slop:
    def __init__(self, slop_value):
        self.value = slop_value


class SIRRungeKuttaSlop:
    def __init__(self, slop_values):
        self.values = slop_values


# SIR Helper Functions
def calculateSuspected(N, sir_value):
    infected_num = sir_value.infected
    recovered_num = sir_value.recovered
    return N - infected_num - recovered_num


def printSIRValues(sir_values, dt):
    i = 0
    for value in sir_values:
        print("\nWeek " + str(i))
        value.showValue()
        i += dt


#  Derivative Functions

#  Calculate:
#      - dSdt: derivative of the suspected in the period of time
#      - dIdt: derivative of the infected in the period of time
#      - dRdt: derivative of the recovered in the period of time

#  dS = -(beta/N)*S*I
def dSdt(t, sir_value, sir_ratio):
    infected_num = sir_value.infected
    suspected_num = sir_value.suspected
    total_population = sir_value.getTotalPopulation()

    #  Infected ratio per person = beta / N
    infected_rate = sir_ratio.infected
    infected_ratio_per_person = infected_rate / total_population

    diffSuspected = -(infected_ratio_per_person *
                      (suspected_num * infected_num))
    return diffSuspected


#  dI = (beta/N)*S*I - gamma*R = -dS - dR
def dIdt(t, sir_value, sir_ratio):
    diffSuspected = dSdt(t, sir_value, sir_ratio)
    diffRecovered = dRdt(t, sir_value, sir_ratio)

    diffInfected = -diffSuspected - diffRecovered
    return diffInfected


#  dR = gamma*R
def dRdt(t, sir_value, sir_ratio):
    infected_num = sir_value.infected
    recovered_rate = sir_ratio.recovered

    diffRecovered = (recovered_rate * infected_num)
    return diffRecovered


#  Evaluate the values of the slop
#  return: list of diferential values
def calculateSlopValues(t, sir_value, sir_ratio):
    dS = dSdt(t, sir_value, sir_ratio)
    dI = dIdt(t, sir_value, sir_ratio)
    dR = dRdt(t, sir_value, sir_ratio)

    return [dS, dI, dR]


#  RungeKutta Helper Functions

#  Evaluate K1 ~ First slop of RK4
#  K1 = f(t, value(t))
#  return the list of Slops
def calculateInitialSlop(time, sir_value, sir_ratio):
    [KS, KI, KR] = calculateSlopValues(time, sir_value, sir_ratio)

    suspected_slop = Slop(KS)
    infected_slop = Slop(KI)
    recovered_slop = Slop(KR)

    K1Slops = [suspected_slop, infected_slop, recovered_slop]

    return SIRRungeKuttaSlop(K1Slops)


#  Evaluate K2/3 ~ Second and Third slop of RK4
#  K2 = f(t + dt/2, value(t + dt*K1/2))
#  K3 = f(t + dt/2, value(t + dt*K2/2))
#  return the list of Slops
def calculateMiddleSlop(time, dt, sir_value, sir_ratio, prev_slop):
    #  Get the current sir values
    [suspected_num, infected_num, recovered_num] = sir_value.getValueByList()

    #  Retrieve value from prev slop
    [prev_sus, prev_inf, prev_rec] = [slop.value for slop in prev_slop]

    #  Evaluate next SIR-value for counting the difference
    #  S_next = S_current + (dt * S_prev_slop/2)
    k_suspected_num = suspected_num + (dt*prev_sus / 2)
    k_infected_num = infected_num + (dt*prev_inf / 2)
    k_recovered_num = recovered_num

    k_sir_value = SIRValue(k_suspected_num, k_infected_num, k_recovered_num)

    #  Calculate the slop values
    [KS, KI, KR] = calculateSlopValues(time + dt/2, k_sir_value, sir_ratio)

    suspected_slop = Slop(KS)
    infected_slop = Slop(KI)
    recovered_slop = Slop(KR)

    KSlop_value = [suspected_slop, infected_slop, recovered_slop]

    return SIRRungeKuttaSlop(KSlop_value)


#  Evaluate K4 ~ Last slop of RK4
#  K4 = f(t + dt, value(t + dt*K3))
#  return the list of Slops
def calculateLastSlop(time, dt, sir_value, sir_ratio, prev_slop):
    #  Get the current sir values
    [suspected_num, infected_num, recovered_num] = sir_value.getValueByList()

    #  Retrieve value from prev slop
    [prev_sus, prev_inf, prev_rec] = [slop.value for slop in prev_slop]

    # TODO Refactor this code
    #  Evaluate next SIR-value for counting the difference
    #  S_next = S_current + (dt * S_prev_slop)
    k_suspected_num = suspected_num + (dt*prev_sus)
    k_infected_num = infected_num + (dt*prev_inf)
    k_recovered_num = recovered_num

    k_sir_value = SIRValue(k_suspected_num, k_infected_num, k_recovered_num)

    #  Calculate the slop values
    [KS, KI, KR] = calculateSlopValues(time + dt, k_sir_value, sir_ratio)

    suspected_slop = Slop(KS)
    infected_slop = Slop(KI)
    recovered_slop = Slop(KR)

    KSlop_value = [suspected_slop, infected_slop, recovered_slop]

    return SIRRungeKuttaSlop(KSlop_value)


def calculateNextValue(curr_value, dt, K):
    [K1, K2, K3, K4] = [slop.value for slop in K]

    #  Calculate the difference of the current value and next value
    difference = (dt/6) * (K1 + 2*K2 + 2*K3 + K4)

    #  next value = current value + difference
    return curr_value + difference


# RungeKutta SIR Model Function
def approximateRK4SIR(sir_value, sir_ratio, t, dt=1):
    suspected_num = sir_value.suspected
    infected_num = sir_value.infected
    recovered_num = sir_value.recovered

    sir_values = []
    curr_sir_value = SIRValue(suspected_num, infected_num, recovered_num)
    sir_values.append(curr_sir_value)

    prev_slop = [None]*3
    KS = [None]*4
    KI = [None]*4
    KR = [None]*4

    # TODO Refactor the code here too
    for i in range(0, t, dt):
        # K1
        slop = calculateInitialSlop(i, sir_values[i], sir_ratio)
        prev_slop = slop.values
        KS[0] = prev_slop[0]
        KI[0] = prev_slop[1]
        KR[0] = prev_slop[2]

        # K2
        slop = calculateMiddleSlop(i, dt, sir_values[i], sir_ratio, prev_slop)
        prev_slop = slop.values
        KS[1] = prev_slop[0]
        KI[1] = prev_slop[1]
        KR[1] = prev_slop[2]

        # K3
        slop = calculateMiddleSlop(i, dt, sir_values[i], sir_ratio, prev_slop)
        prev_slop = slop.values
        KS[2] = prev_slop[0]
        KI[2] = prev_slop[1]
        KR[2] = prev_slop[2]

        # K4
        slop = calculateLastSlop(i, dt, sir_values[i], sir_ratio, prev_slop)
        prev_slop = slop.values
        KS[3] = prev_slop[0]
        KI[3] = prev_slop[1]
        KR[3] = prev_slop[2]

        # Next SIR Value
        suspected_num = calculateNextValue(suspected_num, dt, KS)
        infected_num = calculateNextValue(infected_num, dt, KI)
        recovered_num = calculateNextValue(recovered_num, dt, KR)

        # Add to SIR Array
        curr_sir_value = SIRValue(suspected_num, infected_num, recovered_num)
        sir_values.append(curr_sir_value)

    return sir_values


# TODO Write function to strict the ratio of beta and gamma
def RK4SIR(total_population, I0, R0, beta, gamma, time, dt):
    sir_value = SIRValue(0, I0, R0)
    sir_value.suspected = calculateSuspected(total_population, sir_value)

    sir_ratio = SIRRatio(beta, gamma)

    approximated_values = approximateRK4SIR(sir_value, sir_ratio, time, dt)

    return approximated_values


# Testing SIR Model
sir_values = RK4SIR(N, INIT_INFECTED, INIT_RECOVERED, BETA, GAMMA, TIME, 1)
printSIRValues(sir_values, 1)

# Plot Graph
x = []
for i in range(0, TIME + 1):
    x.append(i)

Sus = []
Inf = []
Rec = []
for value in sir_values:
    Sus.append(value.suspected)
    Inf.append(value.infected)
    Rec.append(value.recovered)

plt.plot(x, Sus, label="Suspected")
plt.plot(x, Inf, label="Infected")
plt.plot(x, Rec, label="Recovered")

plt.xlabel('Week')
plt.ylabel('People')

plt.title('SIR-RK4 Approximation')

plt.legend()
plt.show()

#  input()
