import numpy as np
import gurobipy as gb
import pandas as pd
import defaults
import pickle
# from myhelpers import symmetrize_dict
try:
    # Python 2
    from itertools import izip
except ImportError:
    # Python 3
    izip = zip
from collections import defaultdict
# from load_fnct import load_network, load_generators, load_hvdc_links

####
#  Class to do the nodal day-ahead dispatch.
#  Init: Load network, load initial data, build model.
#  Optimize: Optimize the model.
#  Load_new_data: Takes new blocks of wind, solar and load data as input.
#                   Inserts them into the model.
####


# Class which can have attributes set.
class expando(object):
    pass


# Optimization class
class Optimize_Market:
    '''
        initial_(wind,load,solar) are (N,t) arrays where
        N is the number of nodes in the network, and
        t is the number of timesteps to optimize over.
        Note, that t is fixed by this inital assignment
    '''
    def __init__(self, wind_scenarios, load_signal):
        self.data = expando()
        self.variables = expando()
        self.constraints = expando()
        self.results = expando()
        self.data = expando()
        self._load_data(wind_scenarios, load_signal)
        self._build_model()

    def optimize(self):
        self.model.optimize()

    def load_new_data(self, wind_scenarios, load_signal):
        self._add_new_data(wind_scenarios, load_signal)
        self._update_constraints()

    ###
    #   Loading functions
    ###

    def _load_data(self, wind_scenarios, load_signal):
        self._load_network()
        self._load_generator_data()
        self._load_intial_data(wind_scenarios, load_signal)
        self.data.mybuses = ['n5']

    def _load_network(self):
        self.data.nodedf = pd.read_csv(defaults.nodefile, index_col=0)
        self.data.linedf = pd.read_csv(defaults.linefile, index_col=[0, 1])
        # # Node and edge ordering
        self.data.nodeorder = self.data.nodedf.index.tolist()
        self.data.lineorder = [tuple(x) for x in self.data.linedf.index]
        # # Line limits
        self.data.linelimit = self.data.linedf.limit.to_dict()

        def zero_to_inf(x):
            if x > 0.0001:
                return x
            else:
                return gb.GRB.INFINITY
        
        self.data.linelimit = {k: zero_to_inf(v) for k, v in self.data.linelimit.items()}
        self.data.lineadmittance = self.data.linedf.Y.to_dict()

    def _load_generator_data(self):
        self.data.generatorinfo = pd.read_csv(defaults.generatorfile, index_col=0)
        self.data.generators = self.data.generatorinfo.index.tolist()
        self.data.generatorsfornode = defaultdict(list)
        origodict = self.data.generatorinfo['origin']
        for gen, n in origodict.iteritems():
            self.data.generatorsfornode[n].append(gen)

    def _load_intial_data(self, wind_scenarios, load_signal):
        self.data.scenarios = wind_scenarios.items.tolist()
        self.data.taus = wind_scenarios.major_axis.tolist()
        self.data.mytaus = self.data.taus[1:1+4]
        self.data.wind_scenarios = wind_scenarios
        self.data.load = load_signal
        self.data.scenarioprob = {s: 1.0/len(self.data.scenarios) for s in self.data.scenarios}
        
        self.data.daprice = {t: p for t, p in zip(self.data.mytaus, [30,35,40,33])}
        self.data.bupprice = {t: p for t, p in zip(self.data.mytaus, [35,50,45,48])}
        self.data.bdownprice = {t: p for t, p in zip(self.data.mytaus, [10,30,15,23])}

    ###
    #   Model Building
    ###
    def _build_model(self):
        self.model = gb.Model()
        self._build_variables()
        self._build_objective()
        self._build_constraints()
        self.model.update()

    def _build_variables(self):
        mytaus = self.data.mytaus
        scenarios = self.data.scenarios
        nodes = self.data.nodeorder
        wind = self.data.wind_scenarios

        m = self.model
        
        # Day ahead wind energy at time t
        eday = {}
        for t in mytaus:
            eday[t] = m.addVar(lb=0.0, ub=100.0)
        self.variables.eday = eday
        
        # Balancing wind energy at time t
        ebalance = {}
        for t in mytaus:
            for s in scenarios:
                ebalance[s, t] = m.addVar(lb=-100.0, ub=100.0)
        self.variables.ebalance = ebalance
        
        # Beta at time t
        beta = {}
        for t in mytaus:
            for s in scenarios:
                beta[s, t] = m.addVar(lb=-5000.0, ub=5000.0)
        self.variables.beta = beta

        m.update()

        # Slack bus setting

    def _build_objective(self):
        mytaus = self.data.mytaus
        nodes = self.data.nodeorder
        scenarios = self.data.scenarios
#        generators = self.data.generators
#        gendata = self.data.generatorinfo.T.to_dict()

        m = self.model
        
        m.setObjective(
            gb.quicksum(
            self.data.scenarioprob[s]*(
            self.variables.eday[t]*self.data.daprice[t] +
            self.variables.beta[s, t]) for s in scenarios for t in mytaus
            ),
            gb.GRB.MAXIMIZE)
            
    def _build_constraints(self):
        mytaus = self.data.mytaus
        scenarios = self.data.scenarios
        mybuses = self.data.mybuses
#        generators = self.data.generators
#        gendata = self.data.generatorinfo.T.to_dict()
        nodes = self.data.nodeorder
        edges = self.data.lineorder
        wind = self.data.wind_scenarios
#        load = self.data.load.to_dict()

        m = self.model
        eday, ebalance, beta = self.variables.eday, self.variables.ebalance, self.variables.beta

        # Added wind balance constraint
        windpower_balance = {}
        for t in mytaus:
            for s in scenarios:
                windpower_balance[s, t] = m.addConstr(
                    gb.quicksum(wind[s, t, b] for b in mybuses)
                    - eday[t],
                    gb.GRB.EQUAL,
                    ebalance[s, t])
        self.constraints.windpower_balance = windpower_balance
        
        # Added beta constraints upper price range
        beta_upper = {}
        for t in mytaus:
            for s in scenarios:
                beta_upper[s, t] = m.addConstr(
                    beta[s, t],
                    gb.GRB.LESS_EQUAL,
                    ebalance[s, t] * self.data.bupprice[t])
        self.constraints.beta_upper = beta_upper
        
        # Added beta constraints lower price range
        beta_lower = {}
        for t in mytaus:
            for s in scenarios:
                beta_lower[s, t] = m.addConstr(
                    beta[s, t],
                    gb.GRB.LESS_EQUAL,
                    ebalance[s, t] * self.data.bdownprice[t])
        self.constraints.beta_lower = beta_lower