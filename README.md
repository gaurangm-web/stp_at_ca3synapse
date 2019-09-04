# stp_at_ca3synapse

The program "stpmodel.py" simulates transmission of presynaptic inputs as a sequence of AP-evoked transmitter release events governed by 
a reduced model of short-term plasticity, mimicking key properties of facilitating CA3 synapses (please refer to Mahajan & Nadkarni (2019), bioRxiv/748400 for details).

Input signal s(t) models random place field crossings (at mean rate r_s) + variable firing frequency (s_min to s_max)
associated with every individual pass (fixed duration delta_t).

Default values of various model parameters are listed below (under "Specifying synaptic model parameters"), and may be changed as required.

Model synapse is parametrized by the basal spike-evoked release probability per vesicle (pv0) and maximum RRP size (Nmax).

Usage:
>python2 stpmodel.py

Output of the code:<br/>
R_info: Fractional mutual information between binned release profile and s(t) (relative to the input entropy)<br/>
R_ves: Mean rate of release events (averaged over the full simulation time window)<br/>
E: Synaptic efficiency (~release events per bit transmitted per sec)

Estimates expected to be accurate when delta_t,tau_R << 1/r_s and tau_R << 1/r_n.
