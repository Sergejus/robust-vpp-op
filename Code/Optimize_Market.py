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
        self.data.myscenarios = ['s0']

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
        self.data.batttaus = self.data.taus[0:0+5]
        self.data.wind_scenarios = wind_scenarios
        self.data.load = load_signal
        self.data.scenarioprob = {s: 1.0/len(self.data.scenarios) for s in self.data.scenarios}
        
        self.data.daprice = {t: p for t, p in zip(self.data.mytaus, [30,35,40,33])}
        self.data.bupprice = {t: p for t, p in zip(self.data.mytaus, [35,50,45,48])}
        self.data.bdownprice = {t: p for t, p in zip(self.data.mytaus, [10,30,15,23])}
        
        self.data.bprices = {t: p for t, p in zip(self.data.mytaus, [[35,10],[50,30],[45,15],[48,23]])}

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
        batttaus = self.data.batttaus
#        scenarios = self.data.scenarios
        scenarios = self.data.myscenarios
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
        
        beta_beta = {}
        for t in mytaus:
            for s in scenarios:
                beta_beta[s, t] = m.addVar(lb=-5000.0, ub=5000.0)
        self.variables.beta_beta = beta_beta
                
        
        # Beta at time t
        beta = {}
        time_steps = 4
        Vert_idx = ['v{0}'.format(v+1) for v in range(int(2**time_steps))]
        for v in Vert_idx:
            for t in mytaus:
                for s in scenarios:
                    beta[s, t, v] = m.addVar(lb=-5000.0, ub=5000.0)
            self.variables.beta = beta
            
            
        
        pin = {}
        for t in batttaus:
            for s in scenarios:
                pin[s, t] = m.addVar(lb=0.0, ub=20.0)
        self.variables.pin = pin
        
        pout = {}
        for t in batttaus:
            for s in scenarios:
                pout[s, t] = m.addVar(lb=0.0, ub=20.0)
        self.variables.pout = pout

        blevel = {}
        for t in batttaus:
            for s in scenarios:
                blevel[s, t] = m.addVar(lb=0.0, ub=60.0)
        self.variables.blevel = blevel


        m.update()

        # Slack bus setting

    def _build_objective(self):
        mytaus = self.data.mytaus
        nodes = self.data.nodeorder
#        scenarios = self.data.scenarios
        scenarios = self.data.myscenarios
        
        time_steps = 4
        Vert_idx = ['v{0}'.format(v+1) for v in range(int(2**time_steps))]        
        
        m = self.model
        
        m.setObjective(
            gb.quicksum(
            self.data.scenarioprob[s]*(
            self.variables.eday[t]*self.data.daprice[t] +
            self.variables.beta_beta[s, t]) for s in scenarios for t in mytaus
            ),
            gb.GRB.MAXIMIZE)
            
    def _build_constraints(self):
        mytaus = self.data.mytaus
        batttaus = self.data.batttaus
#        scenarios = self.data.scenarios
        scenarios = self.data.myscenarios
        mybuses = self.data.mybuses
        nodes = self.data.nodeorder
        edges = self.data.lineorder
        wind = self.data.wind_scenarios

        m = self.model
        eday, ebalance, beta, pin, pout, blevel, beta_beta = self.variables.eday, self.variables.ebalance, self.variables.beta, self.variables.pin, self.variables.pout, self.variables.blevel, self.variables.beta_beta

        # Added wind balance constraint
        windpower_balance = {}
        for t in mytaus:
            for s in scenarios:
                windpower_balance[s, t] = m.addConstr(
                    gb.quicksum(wind[s, t, b] for b in mybuses)
                    - eday[t],
                    gb.GRB.EQUAL,
                    ebalance[s, t] - pout[s, t] + pin[s, t])
        self.constraints.windpower_balance = windpower_balance
        
#        # Added beta constraints upper price range
#        beta_upper = {}
#        for t in mytaus:
#            for s in scenarios:
#                beta_upper[s, t] = m.addConstr(
#                    beta[s, t],
#                    gb.GRB.LESS_EQUAL,
#                    ebalance[s, t] * self.data.bupprice[t])
#        self.constraints.beta_upper = beta_upper
#        
#        # Added beta constraints lower price range
#        beta_lower = {}
#        for t in mytaus:
#            for s in scenarios:
#                beta_lower[s, t] = m.addConstr(
#                    beta[s, t],
#                    gb.GRB.LESS_EQUAL,
#                    beta_new[])
#        self.constraints.beta_lower = beta_lower
                
        time_steps = 4
        Time_idx = ['t{0}'.format(t+1) for t in range(time_steps)]
        Vert_idx = ['v{0}'.format(v+1) for v in range(int(2**time_steps))]
         
        lambda_d_list=[30,35,40,33]
        lambda_d = {Time_idx[t]:lambda_d_list[t] for t in range(time_steps)}
         
        psi_up = [5,15,5,15]
        psi_dw = [-20,-5,-25,-10]
         
        df=pd.DataFrame({Time_idx[t]:([psi_up[t]]*int(8/2**t)+[psi_dw[t]]*int(8/2**t))*int(2**t) for t in range(time_steps)},index=Vert_idx)
        lambda_b={}
        for t in Time_idx:
            for v in Vert_idx:
                lambda_b[t,v] = df.ix[v,t]+lambda_d[t]
                
        beta_new = {}
        for v in Vert_idx:
            for t in mytaus:
                for s in scenarios:
                    beta_new[s, t, v] = m.addConstr(
                        beta[s, t, v],
                        gb.GRB.LESS_EQUAL,
                        ebalance[s, t] * lambda_b[t, v])
            self.constraints.beta_new = beta_new  
        
        beta_beta_new = {}
        for v in Vert_idx:
            for t in mytaus:
                for s in scenarios:
                    beta_beta_new[s, t, v] = m.addConstr(
                        beta_beta[s, t],
                        gb.GRB.LESS_EQUAL,
                        beta[s, t, v])
        self.constraints.beta_beta_new = beta_beta_new      
        
        # Battery level constraint
        batt_level = {}
        eff = 0.9
        for t in mytaus:
            for s in scenarios:
                batt_level[s, t] = m.addConstr(
                    blevel[s, t],
                    gb.GRB.EQUAL,
                    blevel[s, batttaus[mytaus.index(t)]] + eff*pin[s,t] - (1/eff)*pout[s,t])
        self.constraints.batt_level = batt_level
        
        
        batt_level_init = {}
        for s in scenarios:
            batt_level_init[s, t] = m.addConstr(
                blevel[s, 't0'],
                gb.GRB.EQUAL,
                30.0)
            self.constraints.batt_level_init = batt_level_init
        
        batt_level_final = {}
        for s in scenarios:
            batt_level_final[s] = m.addConstr(
                blevel[s, 't4'],
                gb.GRB.EQUAL,
                30.0)
        self.constraints.batt_level_final = batt_level_final
        