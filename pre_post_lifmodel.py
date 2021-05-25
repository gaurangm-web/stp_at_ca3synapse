#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  15 10:37:12 2020
@author: Gaurang Mahajan
"""

import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter
from collections import Counter,OrderedDict
from random import *
from scipy.integrate import odeint
from sklearn.feature_selection import mutual_info_regression


#### Global constants: ####
rtol = 1e-6
atol = 1e-6
###########################

def gen_spikes_withrefrac(n,ti,tf,refrac):

	### Generates spike timnes b/w ti and tf with abs refractory period 'refrac'
	tvec = []
	for i in range(n):
		c = 0
		while c<1:
			ts = np.random.uniform(ti,tf,1)[0]
			if all(np.abs(ts-j)>refrac for j in tvec): c=1
		tvec.append(ts)
	return sorted(tvec)

def T(t):

    n = len([r for r in releaseTimes if (t-r >= 0) and (t-r <= 0.001)])
    return n*0.001  ## Molar

def B(u):

    return 1.0/(1 + np.exp(-0.062*u)*(1.0/3.57))

def hhmodel(x, t):

    u, ra, rn = x

    ra_eq = alpha_a * T(t) * (1 - ra) - beta_a * ra
    
    rn_eq = alpha_n * T(t) * (1 - rn) - beta_n * rn

    u_eq = -(1.0/C) * (gL*(u - E_L) + ga*ra*(u - E_a) + gn*B(u)*rn*(u - E_n))

    return [u_eq, ra_eq, rn_eq]

#################################################################################
#### Parameters for postsynaptic membrane potential response:

C = 0.1e-9 ## F
gL = 5e-9  ## S
ga = .5e-9  ## S
gn = .2e-9  ## S

E_L = -70.  ## mV
E_a = 0
E_n = 0

alpha_a = 1.1e6
beta_a = 190.
alpha_n = 7.2e4
beta_n = 6.6


#################################################################################
#### Specifying STP model parameters:

delta_t = 0.5  ## Time step in simulation; also, (fixed) width of individual PF passes (sec)

tmin = 0    ## start time (sec)
tmax = 100  ## end time (sec); tmax-tmin = 3 x 10^4 sec used in the paper

tau_F = 0.15  ## Facilitation time constant (sec)
tau_R = 2.0   ## Per-vesicle recovery time constant (sec)

s_min, s_max = [6.0,60.0]  ## Min/max limits of the dynamic range of inputs (uniform distribution of spiking frequencies in individual PF passes) (Hz)

nlevels = 20  ## Number of input levels (discrete set of frequencies spanning the s_min-s_max range)

af, pv0, Nmax, r_n, r_s = [0.03, 0.03, 8, 0.1, 0.1]  ##map(float,sys.argv[1].split('_'))

#################################################################################

nSims = 1  ## No. of independent trials


f2psp = []

for nSim in range(nSims):

	print('Run #',nSim)

	######################################################################
	#### Generating random temporal sequence of presynaptic inputs:
	######################################################################

	nSteps = int((tmax-tmin)/delta_t)

	inputTimes = sorted(sample(range(nSteps), int(r_s*(tmax-tmin))))  ## Random occurrences of bursts (PF passes) 
	#print(inputTimes)

	s_seq = OrderedDict()		                    
	spikeTimes = []
	                    
	for k in range(nSteps):
	                        
	    if k in inputTimes:
	                            
	       s = sample(list(np.linspace(s_min,s_max,nlevels)),1)[0]  ## AP frequency assigned to every burst
	       nspikes = np.random.poisson(delta_t*s)
	       #spikeTimes.extend(np.random.uniform(k*delta_t,(k+1)*delta_t,nspikes))
	       spikeTimes.extend(gen_spikes_withrefrac(nspikes,k*delta_t,(k+1)*delta_t,0.001))
	       s_seq[k] = s
	    
	    else:
	                            
	       nspikes = np.random.poisson(r_n*delta_t)
	       #spikeTimes.extend(np.random.uniform(k*delta_t,(k+1)*delta_t,nspikes))
	       spikeTimes.extend(gen_spikes_withrefrac(nspikes,k*delta_t,(k+1)*delta_t,0.001))
	       s_seq[k] = 0

	spikeTimes = sorted(spikeTimes)

	print('-> Generated presynaptic input sequence')

	######################################################################
	      

	###############################################################################################        
	#### Generating sequence of per-vesicle release probabilities (one per spike) governed by STP:
	###############################################################################################

	pv_vec = [pv0]  ## Synapse initialized in the resting state

	for i in range(1,len(spikeTimes)):
	    pv = pv0*(1-np.exp(-(spikeTimes[i]-spikeTimes[i-1])/tau_F)) + (pv_vec[-1] + af*(1-pv_vec[-1]))*np.exp(-(spikeTimes[i]-spikeTimes[i-1])/tau_F)
	    pv_vec.append(pv)
	pv_vec = dict(zip(spikeTimes,pv_vec))

	print('-> Generated vector of pv values per spike')

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
	        if random() < ps: release = 1 #np.random.binomial(N_rrp, pv_vec[spikeTimes[i]]) #1  ## Implements univesicular release; for multivesicular release: release = np.random.binomial(N_rrp, pv_vec[spikeTimes[i]])
	        else: release = 0
	    N_rrp = N_rrp-release
	    releaseTimes.extend(release*[spikeTimes[i]])
	    ps_vec.append(ps)

	print('-> Generated release events sequence')


	########################################################
	#### Post-synaptic membrane potential time trace:
	########################################################

	nmax = 1000

	xinit = [E_L, 0, 0]


	t, V, ra, rn = [[],[],[],[]]
	times = sorted([tmin,tmax] + list(releaseTimes))

	for s in range(len(times)-1):

		if s==0: ti=times[0]
		else: ti = times[s]
		if s == len(times)-2: tf=times[-1]
		else: tf = times[s+1]
		trange = np.linspace(ti,tf,nmax)
		t.extend(trange[0:-1])
		sol = odeint(hhmodel, xinit,trange, atol=atol,rtol=rtol)
		xinit = sol[-1,:]
		V.extend(list(sol[0:-1,0]))
		ra.extend(list(sol[0:-1,1]))
		rn.extend(list(sol[0:-1,2]))

	Vdict = {k:v for k,v in zip(t,V)}
	
	for k in range(nSteps): f2psp.append((round(s_seq[k],2),round(np.max([Vdict[i] for i in t if ((i>k*delta_t) and (i < (k+1)*delta_t))]),4)))

	f2psp = np.array(f2psp)
	mi = mutual_info_regression(f2psp, f2psp[:,1], n_neighbors = 3, discrete_features=np.array([0]))
	print('Mutual info estimate =', round(mi[0],3))

	q1 = np.trapz([gL*(u - E_L) for u in V],t)
	q2 = np.trapz([ga*r*(u - E_a) for r,u in zip(ra,V)],t)
	q3 = np.trapz([gn*B(u)*r*(u - E_n) for r,u in zip(rn,V)],t)
	print('Net charge flow over time (leak/AMPAR/NMDAR) (mC) =',np.abs(q1),np.abs(q2),np.abs(q3))


########################################################
#### Plotting input time-trace and event rasters:
########################################################

plt.figure()

s_vec = []
for k in range(nSteps):
	s_vec.append((k*delta_t,s_seq[k]))
	s_vec.append(((k+1)*delta_t,s_seq[k]))
plt.subplot(5,1,1)
plt.plot([t for t,s in s_vec],[s for t,s in s_vec],'g-',linewidth=1.2)
plt.ylim([-1,s_max+5])
plt.xlim([tmin,tmax])
plt.ylabel('Input,\ns(t) Hz',size=14)

plt.subplot(5,1,2)
for s in spikeTimes: plt.plot([s,s],[-1,1],'b-',linewidth=1)
plt.ylim([-5,5]) 
plt.xlim([tmin,tmax])
plt.yticks([],[])
plt.ylabel('Spikes',size=14)

ax1 = plt.subplot(5,1,3)
ax1.plot(spikeTimes,[pv_vec[s] for s in spikeTimes],'o-',color='#1f77b4',markersize=3,linewidth=1)
ax1.set_ylabel('p$_v$',size=14,color='#1f77b4')
plt.ylim([0,1])
ax2=ax1.twinx()
ax2.plot(spikeTimes,ps_vec,'o-',color='#ff7f0e',markersize=3,linewidth=1)
ax2.set_ylabel('P$_s$',size=14,color='#ff7f0e')
plt.ylim([0,1])
plt.xlim([tmin,tmax])

plt.subplot(5,1,4)
for r in releaseTimes: plt.plot([r,r],[-1,1],'r-',linewidth=1)  
plt.ylim([-5,5])
plt.xlim([tmin,tmax])
plt.yticks([],[])
plt.ylabel('Release\nevents',size=14)

plt.subplot(5,1,5)
plt.plot(t,[-E_L+u for u in V],'k',linewidth=1)
plt.ylabel('PSP (mV)',size=14)
plt.xlim([tmin,tmax])

plt.xlabel('Time (s)',size=16)

plt.subplot(5,1,1)
plt.title('p$_v$$^0$ = '+str(pv0)+', N$_{max}$ ='+str(Nmax)+', P$_s$$^0$ = '+str(round(1-(1-pv0)**Nmax,3))+', $\\alpha$$_f$ = '+str(af),size=14)


#### Plotiing distribution of presyn frequency vs. postsyn voltage response (peak values) as a scatter plot on which MI is estimated:
plt.figure()

#for k in f2psp.keys(): plt.plot(k,-E_L+np.mean(f2psp[k]),'bo')
for l in f2psp: plt.plot(l[0],-E_L + l[1],'bo')
plt.ylabel('PSP size (mV)',size=14)
plt.xlabel('Pre-syn input frequency (Hz)',size=14)

plt.show()

#######################################################
