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


---------------
Added Dec 2020:

The code "pre_post_lifmodel.py" extends the STP model to include a description of the postsynatic voltage response evoked by neurotransmitter binding to postsynaptic receptors. The postsynaptic membrane potential (V) is regulated by AMPAR, NMDAR, and leak currents, and the time profile of V(t) is tracked in response to a sequence of vesicular release events. Parameters and model details are adopted from Destexhe et al., Kinetic models of synaptic transmission (1998).

For each PF pass, the corresponding pre-syn input frequency (f) and peak of the post-syn voltage response (PSP size) are recorded. MI over the simulated time window for this joint distribution of discrete and continuous variables is estimated using the non-parametric method implemented in scikit-learn (sklearn.feature_selection.mutual_info_regression) which is based on B. C. Ross, “Mutual Information between Discrete and Continuous Data Sets”, PLoS ONE 9(2), 2014.

Outputs of the model:
Time traces of various quantities of interest;
MI estimate (per trial);
Total charge flow across the postsynaptic membrane per trial (separately for the leak, AMPAR and NMDAR currents) as proxy for postsynaptic energy usage (e.g. Harris et al., Current Biol. 25 (2015))
