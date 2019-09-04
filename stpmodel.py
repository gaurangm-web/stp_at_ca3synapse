# -*- coding: utf-8 -*-

"""
Created on Tue Sep 3 14:57:17 2019
@author: Gaurang Mahajan

##################################################################################################################################
This program simulates transmission of presynaptic inputs as a sequence of AP-evoked transmitter release events governed by 
a reduced model of short-term plasticity, mimicking key properties of facilitating CA3 synapses 
(please refer to Mahajan & Nadkarni (2019), bioRxiv/748400 for details).

Input signal s(t) models random place field crossings (at mean rate r_s) + variable firing frequency (s_min to s_max)
associated with every individual pass (fixed duration delta_t).

Default values of various model parameters are listed below (under "Specifying synaptic model parameters"), and may be changed as required.

Model synapse is parametrized by the basal spike-evoked release probability per vesicle (pv0) and maximum RRP size (Nmax).

Usage:
>>>python2 stpmodel.py

Output of the code:
R_info: Fractional mutual information between binned release profile and s(t) (relative to the input entropy)
R_ves: Mean rate of release events (averaged over the full simulation time window)
E: Synaptic efficiency (~release events per bit transmitted per sec)

Estimates expected to be accurate when delta_t,tau_R << 1/r_s and tau_R << 1/r_n.
###################################################################################################################################
"""


#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter
from collections import Counter,OrderedDict
import math
from random import *

####################################################################
#### Estimating discretized entropy rate of the presynaptic signal:
####################################################################

def signal_entropy(s_seq):
    
    plist = Counter(s_seq.values()).values()
    total = np.sum(plist)
    plist = [p/float(total) for p in plist]

    return plist, np.sum([-p*math.log(p,2) for p in plist])

####################################################################

######################################################################################
#### Estimating total entropy rate of the response (binned release profile):
#####################################################################################

def resp_entropy(releaseTimes):

    nReleases = [0]*int((tmax-tmin)/delta_t)
    
    for r in releaseTimes:
        bin = int(r/delta_t)
        nReleases[bin] += 1

    plist = Counter(nReleases).values()
    total = np.sum(plist)
    plist = [p/float(total) for p in plist] 

    return plist, np.sum([-p*math.log(p,2) for p in plist])

######################################################################################

######################################################################
#### Estimating noise entropy rate of the response profile:
######################################################################

def noise_entropy(s_seq, releaseTimes):
    
    s2r = {s:[] for s in s_seq.values()}

    nReleases = [0]*int((tmax-tmin)/delta_t)
    
    for r in releaseTimes:
        bin = int(r/delta_t)
        nReleases[bin] += 1
    
    for k in s_seq:
        s2r[s_seq[k]].append(nReleases[k])
        
    nbins = len(s_seq)
    nlist = Counter(s_seq.values())
    e = 0
    for s in s2r:
        rlist = Counter(s2r[s]).values()
        plist = [float(c)/len(s2r[s]) for c in rlist]
        e_s = np.sum([-p*math.log(p,2) for p in plist])
        e += (float(nlist[s])/nbins) * e_s
        
    return e

######################################################################


#########################################################################
#### Specifying synaptic model parameters:
#########################################################################

af = .03  ## STF/gain parameter (set to zero for a synapse lacking facilitation)

pv0 = .03  ## Initial per-vesicle fusion probability
Nmax = 8  ## Max RRP size for synapse (1-15 in the paper)

r_n = 0.1  ## Background (noise) spike rate (Hz)
r_s = 0.1  ## Mean rate of burst occurrences (Hz)
delta_t = 0.5  ## Time step in simulation; also, (fixed) width of individual PF passes (sec)

tmin = 0    ## start time (sec)
tmax = 100  ## end time (sec); tmax-tmin = 3 x 10^4 sec used in the paper

tau_F = 0.15  ## Facilitation time constant (sec)
tau_R = 2.0   ## Per-vesicle recovery time constant (sec)

s_min, s_max = [6,60]  ## Min/max limits of the dynamic range of inputs (uniform distribution of spiking frequencies in individual PF passes) (Hz)

nlevels = 20  ## Number of input levels (discrete set of frequencies spanning the s_min-s_max range)

######################################################################


######################################################################
#### Generating random temporal sequence of presynaptic inputs:
######################################################################

nSteps = int((tmax-tmin)/delta_t)
inputTimes = sample(range(nSteps), int(r_s*(tmax-tmin)))  ## Random occurrences of bursts (PF passes) 

s_seq = OrderedDict()		                    
spikeTimes = []
                    
for k in range(nSteps):
                        
    if k in inputTimes:
                            
    	s = sample(np.linspace(s_min,s_max,nlevels),1)[0]  ## AP frequency assigned to every burst
    	nspikes = np.random.poisson(delta_t*s) 
	spikeTimes.extend(np.random.uniform(k*delta_t,(k+1)*delta_t,nspikes))
        s_seq[k] = s
    
    else:
                            
        nspikes = np.random.poisson(r_n*delta_t)
        spikeTimes.extend(np.random.uniform(k*delta_t,(k+1)*delta_t,nspikes))
        s_seq[k] = 0

spikeTimes = sorted(spikeTimes)

print '-> Generated input sequence'

######################################################################
      

###############################################################################################        
#### Generating sequence of per-vesicle release probabilities (per spike) governed by STP:
###############################################################################################

pv_vec = [pv0]  ## Synapse initialized in the resting state

for i in range(1,len(spikeTimes)):
	pv = pv0*(1-np.exp(-(spikeTimes[i]-spikeTimes[i-1])/tau_F)) + (pv_vec[-1] + af*(1-pv_vec[-1]))*np.exp(-(spikeTimes[i]-spikeTimes[i-1])/tau_F)
	pv_vec.append(pv)
pv_vec = dict(zip(spikeTimes,pv_vec))

print '-> Generated vector of pv values per spike'

###############################################################################################


#######################################################
#### Generating sequence of stochastic releases:
#######################################################

releaseTimes = []
ps_vec = []
N_rrp = Nmax  ## Synapse initialized in the resting state

for i in range(len(spikeTimes)):
        
	if N_rrp < Nmax:
		N_rrp += np.random.binomial(Nmax-N_rrp, 1-np.exp(-(spikeTimes[i]-spikeTimes[i-1])/tau_R))
        
	release = 0
	ps = 1 - (1 - pv_vec[spikeTimes[i]])**N_rrp
        if N_rrp > 0:
		if random() < ps: release = 1  ## Implements univesicular release; for multivesicular release: release = np.random.binomial(N_rrp, pv_vec[spikeTimes[i]])
                else: release = 0
        N_rrp = N_rrp-release
        releaseTimes.extend(release*[spikeTimes[i]])
	ps_vec.append(ps)

print '-> Generated release events sequence'

#######################################################


#######################################################
#### Estimating entropies:
#######################################################

pS, SE = signal_entropy(s_seq)                      
pR, RE = resp_entropy(releaseTimes)
NE = noise_entropy(s_seq, releaseTimes)
       
R_info = (RE-NE)/SE
R_ves = float(len(releaseTimes))/(tmax-tmin)
E = R_ves/R_info 

print '\nR_info (~mutual info/input entropy) =',round(R_info,4)
print 'R_ves (# release events per sec) =',round(R_ves,4)
print 'E (~releases per bit per sec) =',round(E,4)

#######################################################


########################################################
#### Plotting input time-trace and event rasters:
########################################################

plt.figure()

s_vec = []
for k in range(nSteps):
	s_vec.append((k*delta_t,s_seq[k]))
	s_vec.append(((k+1)*delta_t,s_seq[k]))
plt.subplot(4,1,1)
plt.plot([t for t,s in s_vec],[s for t,s in s_vec],'g-',linewidth=1.2)
plt.ylim([-1,s_max+5])
plt.xlim([tmin,tmax])
plt.ylabel('Input, s(t) Hz',size=14)

plt.subplot(4,1,2)
for s in spikeTimes: plt.plot([s,s],[-1,1],'b-',linewidth=1)
plt.ylim([-5,5]) 
plt.xlim([tmin,tmax])
plt.yticks([],[])
plt.ylabel('Spikes',size=14)

ax1 = plt.subplot(4,1,3)
ax1.plot(spikeTimes,[pv_vec[s] for s in spikeTimes],'o-',color='#1f77b4',markersize=3,linewidth=1)
ax1.set_ylabel('Instantaneous p$_v$\n(at every spike)',size=14,color='#1f77b4')
plt.ylim([0,1])
ax2=ax1.twinx()
ax2.plot(spikeTimes,ps_vec,'o-',color='#ff7f0e',markersize=3,linewidth=1)
ax2.set_ylabel('Instantaneous P$_s$\n(at every spike)',size=14,color='#ff7f0e',rotation = -90)
plt.ylim([0,1])
plt.xlim([tmin,tmax])


plt.subplot(4,1,4)
for r in releaseTimes: plt.plot([r,r],[-1,1],'r-',linewidth=1)  
plt.ylim([-5,5])
plt.xlim([tmin,tmax])
plt.yticks([],[])
plt.ylabel('Release events',size=14)

plt.xlabel('Time (s)',size=16)

plt.subplot(4,1,1)
plt.title('r$_s$ = '+str(r_s)+' Hz, r$_n$ = '+str(r_n)+' Hz\np$_v$$^0$ = '+str(pv0)+', N$_{max}$ ='+str(Nmax)+', P$_s$$^0$ = '+str(round(1-(1-pv0)**Nmax,3))+', $\\alpha$$_f$ = '+str(af),size=14)

plt.show()

####################################################### 
    




