"""
circuit_network.py

Builds the RankingNetwork for the 10-cube electrical circuit (toy example).
Source of truth: bn_full_cpt_specification_v_2.md and
bn_fault_priors_and_cpt_elicitation_for_10_cube_system.md. (These files are found in https://github.com/marcusgitz/Thesis-fault-diagnosis-BN)

Each Bayesian-network CPT is converted to a local kappa-table:
  - deterministic CPTs    -> rank 0 (consistent) and INF (impossible)
  - probabilistic priors  -> rank = round(-log10(P)) for the coarse resolution, normalised so min is 0
                             rank = round(-log_{sqrt(10)}(P)) = round(-2 * log10(P)) for the finer resolution, normalised so min is 0. The finer resolution doubles the number of ranks per order of magnitude, which should partially mitigate the issue of multiple faults having the same rank, which is a problem for the top-k fault list.
"""

import math
from itertools import product
from ranking_network_V2 import RankingNetwork, INF

# ---- Module-Level Helpers ----
def prob_to_rank(p: float) -> int:
    """Coarse order-of-magnitude mapping: kappa = round(-log10(p))."""
    if p <= 0:
        return INF # type: ignore
    if p >= 1:
        return 0
    return round(-math.log10(p))


def binary_prior(p_fail: float, ok="ok", fail="fail") -> dict:
    """Build the single column of a binary fault node with no parents."""
    r_fail = prob_to_rank(p_fail)
    r_ok   = prob_to_rank(1 - p_fail)
    base = min(r_fail, r_ok)
    return {ok: r_ok - base, fail: r_fail - base}

# prob to rank using finer rank mapping k = round(-log_{sqrt(10)}(p)), which doubles the resolution.
# Switches should end up rank 3 instead of 2, cables rank 3 etc. This should partially mitigate the issue of multiple faults having the same rank, which is a problem for the top-k fault list.
def binary_prior_fine(p_fail: float, ok="ok", fail="fail") -> dict:
    """Finer-resolution converson: kappa = round(-log_{sqrt(10)}(P))
    = round(-2 * log10(p)). Half an order of magnitude per rank step.
    Used for build_circuit_network_fine() instead of build_circuit_network()."""
    def to_rank(p):
        if p <= 0:
            return INF # type: ignore
        if p >= 1:
            return 0
        return round(-2 * math.log10(p))
    r_fail = to_rank(p_fail)
    r_ok   = to_rank(1 - p_fail)
    base = min(r_fail, r_ok)
    return {ok: r_ok - base, fail: r_fail - base}


def voltage_step(in_states, cable_states, switch_states):
    """V_out = 12V iff input is 12V AND cable is ok AND switch is ok.
    Otherwise V_out = 0V. Deterministic, so ranks are 0 and INF."""
    table = {}
    for v_in, cable, sw in product(in_states, cable_states, switch_states):
        passes = (v_in == "12V" and cable == "ok" and sw == "ok")
        if passes:
            table[(v_in, cable, sw)] = {"12V": 0, "0V": INF}
        else:
            table[(v_in, cable, sw)] = {"12V": INF, "0V": 0}
    return table


# ---- The Builder ----

def build_circuit_network() -> RankingNetwork:
    variables = {}
    parents = {}
    kappa_tables = {}

    # Pattern 1 — fault nodes with no parents     
    # (PSU short, eight switches, nine cables, lamp)
    # PSU short
    variables["F_PSU_short"] = ["no", "yes"]
    parents["F_PSU_short"]   = []
    kappa_tables["F_PSU_short"] = {(): binary_prior(0.005, "no", "yes")}

    # Eight switches in a loop
    for n in range(1, 9):
        name = f"F_sw_{n}"
        variables[name] = ["ok", "detached"]
        parents[name]   = []
        kappa_tables[name] = {(): binary_prior(0.030, "ok", "detached")}

    # Nine cables: F_cable_1..F_cable_8 and F_cable_load
    for n in range(1, 9):
        name = f"F_cable_{n}"
        variables[name] = ["ok", "broken"]
        parents[name]   = []
        kappa_tables[name] = {(): binary_prior(0.020, "ok", "broken")}
    
    # Load cable (not in the loop because it doesn't follow the F_cable_n naming pattern)
    variables["F_cable_load"] = ["ok", "broken"]
    parents["F_cable_load"]   = []
    kappa_tables["F_cable_load"] = {(): binary_prior(0.020, "ok", "broken")}

    # Lamp fault
    variables["F_lamp"] = ["ok", "broken"]
    parents["F_lamp"]   = []
    kappa_tables["F_lamp"] = {(): binary_prior(0.015, "ok", "broken")}

    
    # Pattern 2 — battery with parent F_PSU_short, parent ordering F_PSU_short = ["no", "yes"].
    variables["F_battery"] = ["good", "exhausted"]
    parents["F_battery"]   = ["F_PSU_short"]
    kappa_tables["F_battery"] = {
        ("no",):  binary_prior(0.080, "good", "exhausted"),
        ("yes",): binary_prior(0.950, "good", "exhausted"),
    }

    # Pattern 3 — voltage chain V_0..V_8
    # V_0 depends on F_battery and  F_PSU_short, parent ordering F_battery = ["good", "exhausted"] and F_PSU_short = ["no", "yes"].
    variables["V_0"] = ["12V", "0V"]
    parents["V_0"]   = ["F_battery", "F_PSU_short"]
    kappa_tables["V_0"] = {
        ("good", "no"):        {"12V": 0,   "0V": INF},
        ("good", "yes"):       {"12V": INF,   "0V": 0},
        ("exhausted", "no"): {"12V": INF, "0V": 0},
        ("exhausted", "yes"): {"12V": INF, "0V": 0},
    }

    # V_1 through V_8 with the chain helper function, parent ordering V_{n-1} = ["12V", "0V"], F_cable_n = ["ok", "broken"], F_sw_n = ["ok", "detached"].
    for n in range(1, 9):
        name      = f"V_{n}"
        prev_v    = f"V_{n-1}"
        cable     = f"F_cable_{n}"
        switch    = f"F_sw_{n}"
        variables[name] = ["12V", "0V"]
        parents[name]   = [prev_v, cable, switch]
        kappa_tables[name] = voltage_step(["12V", "0V"], ["ok", "broken"], ["ok", "detached"])

    # Pattern 4 — LED / lamp observables
    # PSU LED depends on battery, parent ordering F_battery = ["good", "exhausted"].        
    variables["O_PSU_LED"] = ["on", "off"]
    parents["O_PSU_LED"]   = ["F_battery"]
    kappa_tables["O_PSU_LED"] = {
        ("good",):        {"on": 0,   "off": INF},
        ("exhausted",): {"on": INF, "off": 0},
    }

    # Pattern 5 — eight switch indicator observables, parent ordering V_n = ["12V", "0V"].
    for n in range(1, 9):
        name = f"O_Ind_{n}"
        variables[name] = ["on", "off"]
        parents[name]   = [f"V_{n}"] 
        kappa_tables[name] = {
            ("12V",): {"on": 0, "off": INF},   
            ("0V",):  {"on": INF, "off": 0},   
    }
    
    # Lamp indicator observbale O_Lamp_indicator depends on V_8 and F_cable_load, parent ordering V_8 = ["12V", "0V"] and F_cable_load = ["ok", "broken"].
    variables["O_Lamp_indicator"] = ["on", "off"]
    parents["O_Lamp_indicator"]   = ["V_8", "F_cable_load"]
    kappa_tables["O_Lamp_indicator"] = {
        ("12V", "ok"): {"on": 0, "off": INF},
        ("12V", "broken"): {"on": INF, "off": 0},
        ("0V", "ok"): {"on": INF, "off": 0},
        ("0V", "broken"): {"on": INF, "off": 0},
    }
    # Lamp observable O_Lamp depends on V_8, F_cable_load, F_lamp, parent ordering V_8 = ["12V", "0V"], F_cable_load = ["ok", "broken"], F_lamp = ["ok", "broken"].
    variables["O_Lamp"] = ["on", "off"]
    parents["O_Lamp"]   = ["V_8", "F_cable_load", "F_lamp"]
    kappa_tables["O_Lamp"] = {
        ("12V", "ok", "ok"): {"on": 0, "off": INF},
        ("12V", "ok", "broken"): {"on": INF, "off": 0},
        ("12V", "broken", "ok"): {"on": INF, "off": 0},
        ("12V", "broken", "broken"): {"on": INF, "off": 0},
        ("0V", "ok", "ok"): {"on": INF, "off": 0},
        ("0V", "ok", "broken"): {"on": INF, "off": 0},
        ("0V", "broken", "ok"): {"on": INF, "off": 0},
        ("0V", "broken", "broken"): {"on": INF, "off": 0},
    }
  


    # Pattern 6 — multimeter observables 
    # multimieter battery voltage measurement M_battery depends on F_battery, parent ordering F_battery = ["good", "exhausted"].
    variables["M_battery"] = ["12V", "0V"]
    parents["M_battery"]   = ["F_battery"]
    kappa_tables["M_battery"] = {
        ("good",):        {"12V": 0,   "0V": INF},
        ("exhausted",): {"12V": INF, "0V": 0},
    }

    # Eight Multimeter module observables M_mod_n depends on V_n, parent ordering V_n = ["12V", "0V"].       
    for n in range(1, 9):
        name = f"M_mod_{n}"
        parent = f"V_{n}"
        variables[name] = ["12V", "0V"]
        parents[name]   = [parent]
        kappa_tables[name] = {
            ("12V",): {"12V": 0,   "0V": INF},
            ("0V",):  {"12V": INF, "0V": 0},
        }
    
    # Multimeter PSU short measurement M_PSU_short depends on F_PSU_short, parent ordering F_PSU_short = ["no", "yes"].
    variables["M_PSU_short"] = ["high", "low"]
    parents["M_PSU_short"]   = ["F_PSU_short"]
    kappa_tables["M_PSU_short"] = {
        ("no",):  {"high": 0,   "low": INF},
        ("yes",): {"high": INF, "low": 0},
    }    


    net = RankingNetwork(variables, parents, kappa_tables)
    net.validate()
    return net

# ---- The Builder V2 finer-grained mapping ----
def build_circuit_network_fine() -> RankingNetwork:
    """Same structure as build_circuit_network(), but with finer-grained rank mapping for the priors."""
    variables = {}
    parents = {}
    kappa_tables = {}

    # Pattern 1 — fault nodes with no parents     
    # (PSU short, eight switches, nine cables, lamp)
    # PSU short
    variables["F_PSU_short"] = ["no", "yes"]
    parents["F_PSU_short"]   = []
    kappa_tables["F_PSU_short"] = {(): binary_prior_fine(0.005, "no", "yes")}

    # Eight switches in a loop
    for n in range(1, 9):
        name = f"F_sw_{n}"
        variables[name] = ["ok", "detached"]
        parents[name]   = []
        kappa_tables[name] = {(): binary_prior_fine(0.030, "ok", "detached")}

    # Nine cables: F_cable_1..F_cable_8 and F_cable_load
    for n in range(1, 9):
        name = f"F_cable_{n}"
        variables[name] = ["ok", "broken"]
        parents[name]   = []
        kappa_tables[name] = {(): binary_prior_fine(0.020, "ok", "broken")}
    
    # Load cable (not in the loop because it doesn't follow the F_cable_n naming pattern)
    variables["F_cable_load"] = ["ok", "broken"]
    parents["F_cable_load"]   = []
    kappa_tables["F_cable_load"] = {(): binary_prior_fine(0.020, "ok", "broken")}

    # Lamp fault
    variables["F_lamp"] = ["ok", "broken"]
    parents["F_lamp"]   = []
    kappa_tables["F_lamp"] = {(): binary_prior_fine(0.015, "ok", "broken")}

    
    # Pattern 2 — battery with parent F_PSU_short, parent ordering F_PSU_short = ["no", "yes"].
    variables["F_battery"] = ["good", "exhausted"]
    parents["F_battery"]   = ["F_PSU_short"]
    kappa_tables["F_battery"] = {
        ("no",):  binary_prior_fine(0.080, "good", "exhausted"),
        ("yes",): binary_prior_fine(0.950, "good", "exhausted"),
    }

    # The rest of the network structure is the same as build_circuit_network(), with deterministic CPTs converted to 0/INF ranks using the same helper functions, so we can reuse the voltage_step() function and the rest of the code from build_circuit_network().
    # Pattern 3 — voltage chain V_0..V_8
    # V_0 depends on F_battery and  F_PSU_short, parent ordering F_battery = ["good", "exhausted"] and F_PSU_short = ["no", "yes"].
    variables["V_0"] = ["12V", "0V"]
    parents["V_0"]   = ["F_battery", "F_PSU_short"]
    kappa_tables["V_0"] = {
        ("good", "no"):        {"12V": 0,   "0V": INF},
        ("good", "yes"):       {"12V": INF,   "0V": 0},
        ("exhausted", "no"): {"12V": INF, "0V": 0},
        ("exhausted", "yes"): {"12V": INF, "0V": 0},
    }

    # V_1 through V_8 with the chain helper function, parent ordering V_{n-1} = ["12V", "0V"], F_cable_n = ["ok", "broken"], F_sw_n = ["ok", "detached"].
    for n in range(1, 9):
        name      = f"V_{n}"
        prev_v    = f"V_{n-1}"
        cable     = f"F_cable_{n}"
        switch    = f"F_sw_{n}"
        variables[name] = ["12V", "0V"]
        parents[name]   = [prev_v, cable, switch]
        kappa_tables[name] = voltage_step(["12V", "0V"], ["ok", "broken"], ["ok", "detached"])

    # Pattern 4 — LED / lamp observables
    # PSU LED depends on battery, parent ordering F_battery = ["good", "exhausted"].        
    variables["O_PSU_LED"] = ["on", "off"]
    parents["O_PSU_LED"]   = ["F_battery"]
    kappa_tables["O_PSU_LED"] = {
        ("good",):        {"on": 0,   "off": INF},
        ("exhausted",): {"on": INF, "off": 0},
    }

    # Pattern 5 — eight switch indicator observables, parent ordering V_n = ["12V", "0V"].
    for n in range(1, 9):
        name = f"O_Ind_{n}"
        variables[name] = ["on", "off"]
        parents[name]   = [f"V_{n}"] 
        kappa_tables[name] = {
            ("12V",): {"on": 0, "off": INF},   
            ("0V",):  {"on": INF, "off": 0},   
    }
    
    # Lamp indicator observbale O_Lamp_indicator depends on V_8 and F_cable_load, parent ordering V_8 = ["12V", "0V"] and F_cable_load = ["ok", "broken"].
    variables["O_Lamp_indicator"] = ["on", "off"]
    parents["O_Lamp_indicator"]   = ["V_8", "F_cable_load"]
    kappa_tables["O_Lamp_indicator"] = {
        ("12V", "ok"): {"on": 0, "off": INF},
        ("12V", "broken"): {"on": INF, "off": 0},
        ("0V", "ok"): {"on": INF, "off": 0},
        ("0V", "broken"): {"on": INF, "off": 0},
    }
    # Lamp observable O_Lamp depends on V_8, F_cable_load, F_lamp, parent ordering V_8 = ["12V", "0V"], F_cable_load = ["ok", "broken"], F_lamp = ["ok", "broken"].
    variables["O_Lamp"] = ["on", "off"]
    parents["O_Lamp"]   = ["V_8", "F_cable_load", "F_lamp"]
    kappa_tables["O_Lamp"] = {
        ("12V", "ok", "ok"): {"on": 0, "off": INF},
        ("12V", "ok", "broken"): {"on": INF, "off": 0},
        ("12V", "broken", "ok"): {"on": INF, "off": 0},
        ("12V", "broken", "broken"): {"on": INF, "off": 0},
        ("0V", "ok", "ok"): {"on": INF, "off": 0},
        ("0V", "ok", "broken"): {"on": INF, "off": 0},
        ("0V", "broken", "ok"): {"on": INF, "off": 0},
        ("0V", "broken", "broken"): {"on": INF, "off": 0},
    }
  


    # Pattern 6 — multimeter observables 
    # multimieter battery voltage measurement M_battery depends on F_battery, parent ordering F_battery = ["good", "exhausted"].
    variables["M_battery"] = ["12V", "0V"]
    parents["M_battery"]   = ["F_battery"]
    kappa_tables["M_battery"] = {
        ("good",):        {"12V": 0,   "0V": INF},
        ("exhausted",): {"12V": INF, "0V": 0},
    }

    # Eight Multimeter module observables M_mod_n depends on V_n, parent ordering V_n = ["12V", "0V"].       
    for n in range(1, 9):
        name = f"M_mod_{n}"
        parent = f"V_{n}"
        variables[name] = ["12V", "0V"]
        parents[name]   = [parent]
        kappa_tables[name] = {
            ("12V",): {"12V": 0,   "0V": INF},
            ("0V",):  {"12V": INF, "0V": 0},
        }
    
    # Multimeter PSU short measurement M_PSU_short depends on F_PSU_short, parent ordering F_PSU_short = ["no", "yes"].
    variables["M_PSU_short"] = ["high", "low"]
    parents["M_PSU_short"]   = ["F_PSU_short"]
    kappa_tables["M_PSU_short"] = {
        ("no",):  {"high": 0,   "low": INF},
        ("yes",): {"high": INF, "low": 0},
    }    


    net = RankingNetwork(variables, parents, kappa_tables)
    net.validate()
    return net

# ---- Entry Point ----
if __name__ == "__main__":
    net = build_circuit_network()
    print(net)            # uses  __repr__
    print(f"Variables: {len(net.variables)}")




