import numpy as np

def metropolisHasting(m, logPdf, proposal, logProposalPDF, t0, burnIn = 0):   
    #  m: number of sample to draw
    # logPdf: h(t), the pdf that's atleast proportional to our desired distribution
    # proposal(t) = q(t) : return a new t' from current t
    # proposalPDF(tPrime, t) = q(tPrime | t) : the probability density function of q(t)
    # t0: inital parameter
    # burnIn: number of ignored first iteration

    #Return:
    #traceValue: The return trace of t at all iteration, this will be our returned sample
    #accepted: The accepted values, used for plotting
    #rejected: The rejected values, used for plotting
    #samplingPlotAc: The accepted values's iteration, used for plotting
    #samplingPlotRe: The rejected values's iteration, used for plotting

    t = t0;
    traceValue = []
    accepted = []
    rejected = []
    samplingPlotAc = []
    samplingPlotRe = []

    for j in range(burnIn):
            tPrime = proposal(t)

            p = logPdf(t) +logProposalPDF(tPrime, t)
            pPrime = logPdf(tPrime) + logProposalPDF(t, tPrime)
     
            if pPrime > p:              
                t = tPrime
            else:
                a = np.random.uniform(0, 1)
                if a < np.exp(pPrime - p):
                    t = tPrime

    for i in range(m):
         tPrime = proposal(t)

         p = logPdf(t) +logProposalPDF(tPrime, t)
         pPrime = logPdf(tPrime) + logProposalPDF(t, tPrime)
     
         if pPrime > p:
             t = tPrime
             traceValue.append(t)
             accepted.append(t)
             samplingPlotAc.append(i)
         else:
             a = np.random.uniform(0, 1)
             if a < np.exp(pPrime - p):
                 t = tPrime
                 traceValue.append(t)
                 accepted.append(t)                 
                 samplingPlotAc.append(i)
             else:
                 traceValue.append(t)
                 rejected.append(tPrime)                 
                 samplingPlotRe.append(i)                
    return traceValue, accepted, rejected, samplingPlotAc, samplingPlotRe