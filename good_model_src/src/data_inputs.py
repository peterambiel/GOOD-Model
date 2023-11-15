import numpy as np
import pandas as pd 

hour_periods = np.arange(0,8759, 1)
day_periods = np.arange(1, 366, 1)
regions = ['caiso', 'ercot', 'spp']

# demand
demand_region = {(region, hour): np.random.randint(15000,20000) for region in regions for hour in hour_periods} 
evLoad = {(region, hour): np.random.randint(0,1000) for region in regions for hour in hour_periods} 
fixedevLoad = {(region, day): np.random.randint(0,2500) for region in regions for day in day_periods}

# dispatchable generators
gen_dispatch = ['nat_gas', 'coal', 'hydro']
cost_gen_dispatch = {gen: np.random.randint(25,100) for gen in gen_dispatch}
maxGen_dispatch = {gen: np.random.randint(5000,20000) for gen in gen_dispatch}
genstartupCost = 10000

# renewable resources
cost_class = [1,2,3,4] # technology type which will dictate cost per mwh for wind or solar
gen_solar_rc = [1,2,3,4] # bad, good, better, best solar sources within a region
gen_wind_rc = [1,2,3,4]

solarCost = {(region, resource, cost): np.random.uniform(10, 35) for region in regions for resource in gen_solar_rc for cost in cost_class}
windCost = {(region,resource, cost): np.random.uniform(5, 45) for region in regions for resource in gen_wind_rc for cost in cost_class}

solarCap = {(region, resource, cost): np.random.randint(3000, 7500) for region in regions for resource in gen_solar_rc for cost in cost_class} # exisiting solar resources in a region                  
windCap = {(region, resource, cost): np.random.randint(3000, 7500) for region in regions for resource in gen_wind_rc for cost in cost_class} 

solarCF = {(region, hour, gen, resource):np.random.uniform(0.05,0.30) for region in regions  for hour in hour_periods for gen in cost_class for resource in gen_solar_rc}
windCF = {(region, hour, gen, resource): np.random.uniform(0.25, 0.55) for region in regions for hour in hour_periods for gen in cost_class for resource in gen_wind_rc}

solarMax = {(region, resource, cost): np.random.randint(1500000,2000000) for region in regions for resource in gen_solar_rc for cost in cost_class}
windMax = {(region, resource, cost): np.random.randint(1500000,2000000) for region in regions for resource in gen_wind_rc for cost in cost_class}

# transmission
transCap = {(region_origin, region_dest): 0 if region_origin == region_dest 
            else np.random.randint(75000000, 100000000) for region_origin in regions for region_dest in regions}
transCost = {(region_origin, region_dest): 10000000 if region_origin == region_dest 
             else np.random.randint(7500, 15000) for region_origin in regions for region_dest in regions} 
windTransCost = {(region, gen): np.random.randint(2500,3500) for region in regions for gen in cost_class}

# storage
storCap = {region: np.random.randint(1500, 2500) for region in regions}

# policy
rps = {'caiso': 0.35, 'ercot': 0.15, 'spp': 0.20}

# generator to region mapping
gtor = {region: [gen for gen in gen_dispatch] for region in regions}