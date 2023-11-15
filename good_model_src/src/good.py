import numpy as np
import pandas as pd 
from pyomo.environ import pyomo
from pyomo.opt import SolverFactory



class model_opt():

    def __init__(self,inputs={}):

        self.inputs=inputs

        if self.inputs:

			self.Build()

    def Solve(self,solver_kwargs={}):

		solver=pyomo.SolverFactory(**solver_kwargs)
		solver.solve(self.model)

		self.solution=self.Solution()

    def Build(self):

		#Pulling the keys from the inputs dict
		keys=self.inputs.keys()

		#Initializing the model as a concrete model (as in one that has fixed inputted values)
		self.model=pyomo.ConcreteModel()
		
        #Adding sets
        self.Set()
	
        #Adding parameters
        self.Param()

		#Adding variables
        self.Variables()

		#Adding the objective function
		self.Objective()
	
        #constraint 1: genToDemand(t, r) with evload constraint
        self.genToDemand()
	
        #constraint 2: Generation Limits
        self.genLimits()

        #constraint 3: transmission limits
        self.transLimits()

        #constraint 4: storage limits (r,t)
        self.storLimits()

        #constraint 5: storage state-of-charge (r,t)
        self.storSCO()

        #constraint 6: storage flow-in limits (charging)
        self.storFlowIn()

        #constaint 7: storage flow out limits (discharging)
        self.storFlowOut()

        #constraint 8: solar resource capacity limits
        self.solarCapLimits()

        #constraint 9: wind resource capacity limts
        self.windCapLimits()

        #constraint 10: electricity import limits
        self.importLimits()


    def Set(self): 
	
        self.model.T=pyomo.Set(initialize=[idx for idx in range(self.inputs['n_t'])])
        self.model.R=pyomo.Set(initialize=[idx for idx in range(self.inputs['n_r'])])
		
        self.model.g = Set(initialize=gen_dispatch) 
        self.model.src = Set(initialize=gen_solar_rc)
        self.model.wrc = Set(initialize=gen_wind_rc)
        self.model.cc = Set(initialize =cost_class)
        self.model.t = Set(initialize=hour_periods) # time period, hour
        self.model.d = Set(initialize=day_periods) # time period, day
        self.model.r = Set(initialize=regions) # region, defaulting to IPM regions
        self.model.s = Set() # set of regions used in the model corresponding to states (to account for policy constraints)
        self.model.gtor = Set(within=self.model.g * self.model.r) # generator to region mapping
        self.model.ttod = Set(within=self.model.t * self.model.d) # hour to day mapping

        # alias sets: when the model requires use of the same set within a single equation
        self.model.o = Set(initialize=self.model.r) 
        self.model.p = Set(initialize=self.model.r)
	
    def Param(self):
	
        self.model.c_gencost = pyomo.Param(self.model.g, initialize=cost_gen_dispatch)
        self.model.c_solarCost = pyomo.Param(self.model.r, self.model.src, self.model.cc, initialize=solarCost)
        self.model.c_windCost = pyomo.Param(self.model.r,self.model.wrc, self.model.cc, initialize=windCost)

        # demand pyomo.Parameters
        self.model.c_demandLoad = pyomo.Param(self.model.r, self.model.t, initialize=demand_region)
        self.model.c_evLoad = pyomo.Param(self.model.r, self.model.t, initialize=evLoad)
        self.model.c_fixedevLoad = pyomo.Param(self.model.r, self.model.d, initialize=fixedevLoad)

        # generation pyomo.Parameters
        self.model.c_maxGen = pyomo.Param(self.model.g, initialize=maxGen_dispatch)

        # renewable generation pyomo.Parameters
        self.model.c_solarCap = pyomo.Param(self.model.r, self.model.src, self.model.cc, initialize=solarCap)
        self.model.c_windCap = pyomo.Param(self.model.r, self.model.wrc, self.model.cc, initialize=windCap)
        self.model.c_solarCF = pyomo.Param(self.model.r, self.model.t, self.model.src, self.model.cc, initialize=solarCF)
        self.model.c_windCF = pyomo.Param(self.model.r, self.model.t, self.model.wrc, self.model.cc, initialize=windCF)
        self.model.c_solarMax = pyomo.Param(self.model.r, self.model.src, self.model.cc, initialize=solarMax)
        self.model.c_windMax = pyomo.Param(self.model.r, self.model.wrc, self.model.cc, initialize=windMax)

        # transmission pyomo.Parameters
        self.model.c_transCap = pyomo.Param(self.model.r, self.model.o, initialize=transCap)
        self.model.c_transCost = pyomo.Param(self.model.r, self.model.o, initialize=transCost)
        self.model.c_windTransCost = pyomo.Param(self.model.r, self.model.wrc, initialize=windTransCost)

        # energy storage pyomo.Parameters
        self.model.c_storCap = pyomo.Param(self.model.r, initialize=storCap)

        # policy pyomo.Parameters
        self.model.c_rps = pyomo.Param(self.model.r, initialize=rps)

        # define model scalars
        self.model.c_transLoss = pyomo.Param(initialize=0.972)
        self.model.c_storEff = pyomo.Param(initialize=0.7)
        self.model.c_storCost = pyomo.Param(initialize=10000) 
        self.model.c_importLimit = pyomo.Param(self.model.r, initialize=800000000)
        self.model.c_storFlowCap = pyomo.Param(initialize = 0.85)  


    def Variables(self):
		
        self.model.x_generation = pyomo.Var(self.model.g, self.model.t, within=(pyomo.NonNegativeReals), 
                   bounds=lambda m, g, t: (1e-08, self.model.c_maxGen[g]))
		self.model.x_trans = pyomo.Var(self.model.r, self.model.t, self.model.o, within=(pyomo.NonNegativeReals), 
                 bounds=lambda m, r, t, o: (1e-08, self.model.c_transCap[r, o]))
		self.model.x_solarNew = pyomo.Var(self.model.r, self.model.src, self.model.cc, within=(pyomo.NonNegativeReals))
		self.model.x_windNew = pyomo.Var(self.model.r, self.model.wrc, self.model.cc, within=(pyomo.NonNegativeReals))
		self.model.x_storSOC = pyomo.Var(self.model.r, self.model.t, within=(pyomo.NonNegativeReals))
		self.model.x_storIn = pyomo.Var(self.model.r, self.model.t, within=(pyomo.NonNegativeReals))
		self.model.x_storOut = pyomo.Var(self.model.r, self.model.t, within=(pyomo.NonNegativeReals))


		self.model.u_tp=pyomo.Var(self.model.T,domain=pyomo.Reals,
			bounds=(0,self.inputs['tp_max']))

		self.model.u_ts=pyomo.Var(self.model.T,domain=pyomo.Reals,
			bounds=(0,self.inputs['ts_max']))


    def Objective(self):

            def obj_func_rule(m):
                return (
                    sum(self.model.x_generation[g, t] * self.model.c_gencost[g] for g in self.model.g for t in self.model.t) +
                    sum(self.model.x_trans[r, t, o] * self.model.c_transCost[r, o] for r in self.model.r for t in self.model.t for o in self.model.o) +
                    sum(self.model.c_solarCost[r, s, c] * self.model.x_solarNew[r, s,c] for r in self.model.r for s in self.model.src for c in self.model.cc) +
                    sum((self.model.c_windCost[r,w,c] + self.model.c_windTransCost[r, w]) * self.model.x_windNew[r,w,c] for r in self.model.r for w in self.model.wrc for c in self.model.cc)
                )

            self.model.obj_func = pyomo.Objective(rule=obj_func_rule, sense=pyomo.minimize)

    #constraint 1: generation-demand balancing
    def genToDemand(self):

        self.model.gen_to_demand_rule = pyomo.ConstraintList()
        
        for t in self.model.t: 
            for r in self.model.r: 
                constraint_expr = (
                        sum(self.model.x_generation[g, t] for g in gtor[r]) + 
                        sum((self.model.c_solarCap[r, s, c] + self.model.x_solarNew[r, s, c]) * self.model.c_solarCF[r, t, s, c] for s in self.model.src for c in self.model.cc) +
                        sum((self.model.c_windCap[r, w,c] + self.model.x_windNew[r, w, c]) * self.model.c_windCF[r, t, w, c] for w in self.model.wrc for c in self.model.cc) + 
                        (self.model.x_storIn[r,t] - self.model.x_storOut[r,t]) -
                        sum(self.model.x_trans[o, t, r] * self.model.c_transLoss for o in self.model.o) - 
                        sum(self.model.x_trans[r, t, p] for p in self.model.p) - 
                        (self.model.c_demandLoad[r, t] + self.model.c_evLoad[r, t])
                        ) == 0
				
            self.model.gen_to_demand_rule.add(constraint_expr)	
				
    #constraint 2: Generation Limits   
    def genLimits(self):
	
        self.self.model.gen_limits_rule = pyomo.ConstraintList() 

        for g in self.model.g: 
            for t in self.model.t: 
                constraint_expr = ( 
                    self.model.c_maxGen[g] - self.model.x_generation[g,t]
                    ) >= 0
                
            self.model.gen_limits_rule.add(constraint_expr)
	
        

    #constraint 3: transmission limits
    def transLimits(self):
	
        self.model.trans_limits_rule = ConstraintList()

        for r in self.model.r: 
            for o in self.model.o: 
                for t in self.model.t: 
                    constraint_expr = (
                        self.model.c_transCap[r, o] - self.model.x_trans[r, t, o]
                        ) >=0
                    
                self.model.trans_limits_rule.add(constraint_expr)
	


    #constraint 4: storage limits (r,t)
    def storLimits(self):
	
        self.model.maxStorage = ConstraintList()

        for r in self.model.r: 
            for t in self.model.t: 
                constraint_expr = ( 
                    self.model.c_storCap[r] - self.model.x_storSOC[r, t]
                    ) >= 0
            
                self.model.maxStorage.add(constraint_expr)

    #constraint 5: storage state-of-charge (r,t)
    def storSOC(self):
	
        self.model.storageSOC_rule = ConstraintList()

        for r in self.model.r:
            for t in self.model.t:
                if t == min(self.model.t):
                    # For the first time step, set storSOC to 0
                    self.model.storageSOC_rule.add(self.model.x_storSOC[r, t] == 0)
                else:
                    # For subsequent time steps, apply the constraint
                    constraint_expr = (
                        self.model.x_storSOC[r, t] - self.model.x_storSOC[r, t - 1] - self.model.x_storIn[r, t - 1] * self.model.c_storEff + self.model.x_storOut[r,t-1]
                    ) == 0
                    
                self.model.storageSOC_rule.add(constraint_expr)

    #constraint 6: storage flow-in limits (charging)
    def storFlowIn(self): 
    
        self.model.stor_flowIN_rule = ConstraintList() 

        for r in self.model.r: 
            for t in self.model.t: 
                constraint_expr = (
                    self.model.c_storCap[r] * self.model.c_storFlowCap - self.model.x_storIn[r,t]
                    ) >= 0 
                
            self.model.stor_flowIN_rule.add(constraint_expr)

    #constaint 7: storage flow out limits (discharging)
    def storFlowOut(self):

        self.model.stor_flowOUT_rule = ConstraintList()

        for r in self.model.r: 
            for t in self.model.t: 
                constraint_expr = (
                    self.model.c_storCap[r] * self.model.c_storFlowCap - self.model.x_storOut[r,t]
                    ) >=0
                
            self.model.stor_flowOUT_rule.add(constraint_expr)


    #constraint 8: solar resource capacity limits
    def solarCapLimits(self):
    
        self.model.solar_cap_limits_rule = pyomo.ConstraintList()

        for r in self.model.r: 
            for s in self.model.src: 
                for c in self.model.cc:
                    constraint_expr = (
                        (self.model.x_solarNew[r,s,c] + self.model.c_solarCap[r,s,c]) - (self.model.c_solarMax[r,s,c])
                    ) == 0
                
                self.model.solar_cap_limits_rule.add(constraint_expr)  


    #constraint 9: wind resource capacity limts
    def windCapLimits(self):

        self.model.wind_cap_limits_rule = pyomo.ConstraintList()

        for r in self.model.r: 
            for w in self.model.wrc: 
                for c in self.model.cc:
                    constraint_expr = (
                        (self.model.x_windNew[r,w,c] + self.model.c_windCap[r,w,c]) - (self.model.c_windMax[r,s,c]) 
                    ) == 0
                
                self.model.wind_cap_limits_rule.add(constraint_expr)  

    #constraint 10: electricity import limits
    def importLimits(self):
    
        self.model.import_limit_rule = ConstraintList()

        for r in self.model.r: 
            constraint_expr = (
                self.model.c_importLimit[r] - sum(self.model.x_trans[r, t, o] for t in self.model.t for o in self.model.o)
                ) == 0
            self.model.import_limit_rule.add(constraint_expr)


	def Solution(self):
		'''
		From StackOverflow
		https://stackoverflow.com/questions/67491499/
		how-to-extract-indexed-variable-information-in-pyomo-model-and-build-pandas-data
		'''
		model_vars=self.model.component_map(ctype=pyomo.Var)

		serieses=[]   # collection to hold the converted "serieses"
		for k in model_vars.keys():   # this is a map of {name:pyo.Var}
			v=model_vars[k]

			# make a pd.Series from each    
			s=pd.Series(v.extract_values(),index=v.extract_values().keys())

			# if the series is multi-indexed we need to unstack it...
			if type(s.index[0])==tuple:# it is multi-indexed
				s=s.unstack(level=1)
			else:
				s=pd.DataFrame(s) # force transition from Series -> df

			# multi-index the columns
			s.columns=pd.MultiIndex.from_tuples([(k,t) for t in s.columns])

			serieses.append(s)

		self.solution=pd.concat(serieses,axis=1)