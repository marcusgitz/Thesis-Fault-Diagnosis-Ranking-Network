"""Extracted BN Builder from BN_V2.ipynb (found in https://github.com/marcusgitz/Thesis-fault-diagnosis-BN) for cleaner code organization. The BN structure and CPDs are defined here, while inference and experiments are done in the experiments notebook."""
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD

def build_bn(verbose=False):

    #Define network structure
    model = DiscreteBayesianNetwork(
        [
            ('F_PSU_short', 'F_battery'),
            ('F_PSU_short', 'V_0'),
            ('F_PSU_short', 'M_PSU_short'),
            ('F_battery', 'V_0'),
            ('F_battery', 'O_PSU_LED'),
            ('F_battery', 'M_battery'),
            ('V_0', 'V_1'),
            ('V_1', 'V_2'),
            ('V_2', 'V_3'),
            ('V_3', 'V_4'),
            ('V_4', 'V_5'),
            ('V_5', 'V_6'),
            ('V_6', 'V_7'),
            ('V_7', 'V_8'),
            ('F_cable_1', 'V_1'),
            ('F_cable_2', 'V_2'),
            ('F_cable_3', 'V_3'),
            ('F_cable_4', 'V_4'),
            ('F_cable_5', 'V_5'),
            ('F_cable_6', 'V_6'),
            ('F_cable_7', 'V_7'),
            ('F_cable_8', 'V_8'),
            ('F_sw_1', 'V_1'),
            ('F_sw_2', 'V_2'),
            ('F_sw_3', 'V_3'),
            ('F_sw_4', 'V_4'),
            ('F_sw_5', 'V_5'),
            ('F_sw_6', 'V_6'),
            ('F_sw_7', 'V_7'),
            ('F_sw_8', 'V_8'),
            ('V_1', 'O_Ind_1'),
            ('V_2', 'O_Ind_2'),
            ('V_3', 'O_Ind_3'),
            ('V_4', 'O_Ind_4'),
            ('V_5', 'O_Ind_5'),
            ('V_6', 'O_Ind_6'),
            ('V_7', 'O_Ind_7'),
            ('V_8', 'O_Ind_8'),
            ('V_1', 'M_mod_1'),
            ('V_2', 'M_mod_2'),
            ('V_3', 'M_mod_3'),
            ('V_4', 'M_mod_4'),
            ('V_5', 'M_mod_5'),
            ('V_6', 'M_mod_6'),
            ('V_7', 'M_mod_7'),
            ('V_8', 'M_mod_8'),
            ('V_8', 'O_Lamp'),
            ('V_8', 'O_Lamp_indicator'),
            ('F_cable_load', 'O_Lamp'),
            ('F_cable_load', 'O_Lamp_indicator'),
            ('F_lamp', 'O_Lamp')
        ]
    )
    #Define CPDs

    cpd_F_PSU_short = TabularCPD(
        variable='F_PSU_short',
        variable_card=2,
        values=[[0.995],
                [0.005]],
        state_names={'F_PSU_short': ['no', 'yes']}
    )

    cpd_F_sw = {}
    for i in range(1, 9):
        cpd_F_sw[i] = TabularCPD(
            variable=f'F_sw_{i}',
            variable_card=2,
            values=[[0.97],
                    [0.03]],
            state_names={f'F_sw_{i}': ['ok', 'detached']}
        )

    cpd_F_cable = {}
    for i in range(1, 9):
        cpd_F_cable[i] = TabularCPD(
            variable=f'F_cable_{i}',
            variable_card=2,
            values=[[0.98],
                    [0.02]],
            state_names={f'F_cable_{i}': ['ok', 'broken']}
        )

    cpd_F_cable_load = TabularCPD(
        variable='F_cable_load',
        variable_card=2,
        values=[[0.98],
                [0.02]],
        state_names={'F_cable_load': ['ok', 'broken']}
    )

    cpd_F_lamp = TabularCPD(
        variable='F_lamp',
        variable_card=2,
        values=[[0.985],
                [0.015]],
        state_names={'F_lamp': ['ok', 'broken']}
    )

    cpd_F_battery = TabularCPD(
        variable='F_battery',
        variable_card=2,
        values=[[0.920, 0.050],
                [0.080, 0.950]],
        evidence=['F_PSU_short'],
        evidence_card=[2],
        state_names={
            'F_battery': ['good', 'exhausted'],
            'F_PSU_short': ['no', 'yes']
        }
    )

    cpd_V_0 = TabularCPD(
        variable='V_0',
        variable_card=2,
        values=[[1, 0, 0, 0],   # P(V_0=12V)
                [0, 1, 1, 1]],  # P(V_0=0V)
        evidence=['F_battery', 'F_PSU_short'],
        evidence_card=[2, 2],
        state_names={
            'V_0': ['12V', '0V'],
            'F_battery': ['good', 'exhausted'],
            'F_PSU_short': ['no', 'yes']
        }
    )

    # Column order 
    #   col 0: (12V, Ok,       Ok)      -> 12V passes through
    #   col 1: (12V, Ok,       Broken)  -> cable fault
    #   col 2: (12V, Detached, Ok)      -> switch fault
    #   col 3: (12V, Detached, Broken)  -> both faults
    #   col 4: (0V,  Ok,       Ok)      -> no upstream voltage
    #   col 5: (0V,  Ok,       Broken)
    #   col 6: (0V,  Detached, Ok)
    #   col 7: (0V,  Detached, Broken)

    cpd_V = {}
    prev = 'V_0'
    for i in range(1, 9):
        node = f'V_{i}'
        sw   = f'F_sw_{i}'
        cab  = f'F_cable_{i}'
        cpd_V[i] = TabularCPD(
            variable=node,
            variable_card=2,
            #        c0 c1 c2 c3 c4 c5 c6 c7
            values=[[1, 0, 0, 0, 0, 0, 0, 0],  # P(V_N=12V)
                    [0, 1, 1, 1, 1, 1, 1, 1]], # P(V_N=0V)
            evidence=[prev, sw, cab],
            evidence_card=[2, 2, 2],
            state_names={
                node: ['12V', '0V'],
                prev: ['12V', '0V'],
                sw:   ['ok', 'detached'],
                cab:  ['ok', 'broken']
            }
        )
        prev = node

    cpd_M_PSU_short = TabularCPD(
        variable='M_PSU_short',
        variable_card=2,
        values=[[1, 0],     # P(M_PSU_short=high)   
                [0, 1]],    # P(M_PSU_short=low)   
        evidence=['F_PSU_short'],
        evidence_card=[2],
        state_names={
            'M_PSU_short': ['high', 'low'],
            'F_PSU_short': ['no', 'yes']
        }
    )

    cpd_M_battery = TabularCPD(
        variable='M_battery',
        variable_card=2,
        values=[[1, 0],   
                [0, 1]],  
        evidence=['F_battery'],
        evidence_card=[2],
        state_names={
            'M_battery': ['12V', '0V'],
            'F_battery': ['good', 'exhausted']
        }
    )

    cpd_O_PSU_LED = TabularCPD(
        variable='O_PSU_LED',
        variable_card=2,
        values=[[1, 0],   
                [0, 1]], 
        evidence=['F_battery'],
        evidence_card=[2],
        state_names={
            'O_PSU_LED': ['on', 'off'],
            'F_battery': ['good', 'exhausted']
        }
    )

    cpd_O_Ind = {}
    for i in range(1, 9):
        node = f'V_{i}'
        obs  = f'O_Ind_{i}'
        cpd_O_Ind[i] = TabularCPD(
            variable=obs,
            variable_card=2,
            values=[[1, 0],   
                    [0, 1]],  
            evidence=[node],
            evidence_card=[2],
            state_names={
                obs:  ['on', 'off'],
                node: ['12V', '0V']
            }
        )

    cpd_M_mod = {}
    for i in range(1, 9):
        node = f'V_{i}'
        meas = f'M_mod_{i}'
        cpd_M_mod[i] = TabularCPD(
            variable=meas,
            variable_card=2,
            values=[[1, 0],   
                    [0, 1]], 
            evidence=[node],
            evidence_card=[2],
            state_names={
                meas: ['12V', '0V'],
                node: ['12V', '0V']
            }
        )

    cpd_O_Lamp = TabularCPD(
        variable='O_Lamp',
        variable_card=2,
        values=[
            # V_8=12V                          | V_8=0V
            # cab_load=Ok | cab_load=Broken    | cab_load=Ok | cab_load=Broken
            # lamp=Ok/Br  | lamp=Ok/Br         | lamp=Ok/Br  | lamp=Ok/Br
            [1, 0, 0, 0,    0, 0, 0, 0],   # P(O_Lamp=On)
            [0, 1, 1, 1,    1, 1, 1, 1],   # P(O_Lamp=Off)
        ],
        evidence=['V_8', 'F_cable_load', 'F_lamp'],
        evidence_card=[2, 2, 2],
        state_names={
            'O_Lamp': ['on', 'off'],
            'V_8': ['12V', '0V'],
            'F_cable_load': ['ok', 'broken'],
            'F_lamp': ['ok', 'broken']
        }
    )

    cpd_O_Lamp_indicator = TabularCPD(
        variable='O_Lamp_indicator',
        variable_card=2,
        values=[[1, 0, 0, 0],    # P(on)
                [0, 1, 1, 1]],   # P(off)
        evidence=['V_8', 'F_cable_load'],
        evidence_card=[2, 2],
        state_names={
            'O_Lamp_indicator': ['on', 'off'],
            'V_8': ['12V', '0V'],
            'F_cable_load': ['ok', 'broken']
        }
    )
    #Add all CPDs to the model

    all_cpds = [
        cpd_F_PSU_short,
        cpd_F_battery,
        cpd_F_cable_load,
        cpd_F_lamp,
        cpd_V_0,
        cpd_M_PSU_short,
        cpd_M_battery,
        cpd_O_PSU_LED,
        cpd_O_Lamp,
        cpd_O_Lamp_indicator,
    ]
    all_cpds += list(cpd_F_sw.values())
    all_cpds += list(cpd_F_cable.values())
    all_cpds += list(cpd_V.values())
    all_cpds += list(cpd_O_Ind.values())
    all_cpds += list(cpd_M_mod.values())

    model.add_cpds(*all_cpds)

    valid = model.check_model()
    if verbose: print("Model valid:", valid)
    return model
