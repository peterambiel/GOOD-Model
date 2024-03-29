def extract_sets(dict): 

    def gen_fuel_type_extraction(dict): 
        unique_keys = {'region': [], 'gen_type': [], 'gen_fuel': [], 'emissions_code':[]}  # Initialize unique_keys dictionary

        for tuples in dict['Plants_Dic']:
            for idx_val, idx_name in enumerate(tuples):
                if idx_val == 0:
                    if idx_name not in unique_keys['region']:
                        unique_keys['region'].append(idx_name)  # Assign idx_name to 'region' key in unique_keys
                elif idx_val == 1:
                    if idx_name not in unique_keys['gen_type']:
                        unique_keys['gen_type'].append(idx_name)  # Assign idx_name to 'generator' key in unique_keys
                elif idx_val == 2: 
                    if idx_name not in unique_keys['gen_fuel']:
                        unique_keys['gen_fuel'].append(idx_name)
                elif idx_val == 3: 
                    if idx_name not in unique_keys['emissions_code']:
                        unique_keys['emissions_code'].append(idx_name)

        return unique_keys

    hold_sets = {
        "resource_class": list(range(1, 12)),
        "cost_class": list(range(1, 7)),
        "hours": list(range(0, 8760)),
        "gen_heat_rate": list(range(1, 5)),
        "days": list(range(1, 366)),
        "states": ['AL','AK','AZ','AR','CA','CO','CT','DE','DC','FL','GA','HI','ID','IL',
        'IN','IA','KS','KY','LA','ME','MD','MA','MI','MN','MS','MO','MT','NE','NV','NH',
        'NJ','NM','NY','NC','ND','OH','OK','OR','PA','RI','SC','SD','TN','TX','UT',
        'VT','VA','WA','WV','WI','WY'],
        "regions": [key for key in dict['Transmission_index'].keys()],
        "gen_regions": gen_fuel_type_extraction(dict)['region'],
        "gen_fuel_type": gen_fuel_type_extraction(dict)['gen_fuel'],
        "gen_type": gen_fuel_type_extraction(dict)['gen_type'],
        "emissions_code": gen_fuel_type_extraction(dict)['emissions_code']
        }

    return hold_sets


def extract_params(dict): 
    
    hold_params = {'generator_cost': {(i[0], i[2]): d[1] for i, d in dict['Plants_Dic'].items()}, 
                'generator_capacity': dict['Plant_capacity_dic'],
                'solar_capex': {(i[0], i[2], i[3]): d for i,d in dict['Solar_capital_cost_photov_final'].items()}, 
                'solar_CF': {(i[0], i[2], i[3]): d for i,d in dict['Solar_capacity_factor_final'].items()}, 
                'solar_max_capacity': {(i[0], i[2], i[3]): d for i,d in dict['Solar_regional_capacity_final'].items()},
                'solar_installed_capacity': {i[0]: d for i,d in dict['Plant_capacity_dic'].items() if i[1] == 'Solar'},
                'wind_capex': {(i[0], i[2], i[3]): d for i,d in dict['Wind_capital_cost_final'].items()}, 
                'wind_CF': {(i[0], i[2], i[3]): d for i,d in dict['Wind_capacity_factor_final'].items()}, 
                'wind_max_capacity': {(i[0], i[2], i[3]): d for i,d in dict['Wind_onshore_capacity_final'].items()},
                'wind_installed_capacity': {i[0]: d for i, d in dict['Plant_capacity_dic'].items() if i[1] == 'Wind'},
                'wind_transmission_cost': {(i[0], i[2], i[3]): d for i,d in dict['Wind_trans_capital_cost_final'].items()},
                'transmission_cost': dict['Transmission_Cost_dic'], 
                'transmission_capacity': dict['Transmission_Capacity_dic'],
                'enerstor_installed_capacity': {i[0]: d for i, d in dict['Plant_capacity_dic'].items() if i[1] == 'EnerStor'},
                'load': dict['load_final']}
                
    return hold_params