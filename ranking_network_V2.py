"""
ranking_network.py

Reusable kappa-function inference engine for ranking networks.

A RankingNetwork is a data structure (variables, parents, local kappa-tables).
The inference functions take the network as an argument, so the SAME functions
run on the 3-cube toy and on the 50-node DAG. Scaling = passing in a different
network; the engine is unchanged.

This version uses smart enumeration instead of the brute-force enumerator. It only
enumerates worlds over the probabilistic variables and propagates the deterministic
ones in topological order. For a network with many deterministic relations (voltage
chaines, Led observables, etc.) this is a huge speedup compared to the brute-force
version, since it avoids enumerating all 2^n worlds. Brute-force 'all_worlds' is
kept for testing and debugging.


Variable-elimination / Shenoy-Shafer inference will maybe be implemented later. 
If i have time for that, due to the small scope/timeframe i have.
Brute-force is fine for small toys; for 50 binary variables it which 2^50 worlds
it needs this msart enumartion, but prefarbly the shenoy-shafer or variable elimination 
approach, which is more efficient than enumeration.

Spohn semantics:
  combination     = integer addition
  marginalisation = minimisation
  rank 0          = unsurprising / believed
  higher rank     = more surprising
  INF             = impossible
"""

from dataclasses import dataclass
from itertools import product

INF = float("inf")


# --------------------------------------------------------------------------------
# Network as data in a separate file. NOT hardcoded inside the inference functions
# --------------------------------------------------------------------------------

@dataclass
class RankingNetwork:
    # var name -> list of states
    variables: dict
    # var name -> list of parent var names (in canonical order)
    parents: dict
    # var name -> {parent_value_tuple: {child_state: rank}}
    # Every parent column must have min rank = 0 (Spohn normalisation).
    kappa_tables: dict

    # ---- Helper functions ----
    def variable_states(self, var):
        """Return the list of states for the given variable."""
        return self.variables[var]
    
    def kappa(self, var, parent_values, state):
        """Return the local kappa for the given variable state and parent values."""
        return self.kappa_tables[var][parent_values][state]
    
    def get_parents(self, var):
        """Return the list of parent variable names for the given variable."""
        return self.parents[var]


    def validate(self):
        """Sanity-check the network. Raises ValueError on incorrect/invalid input."""
        for var in self.variables:
            if var not in self.parents:
                raise ValueError(f"{var}: missing parents entry")
            if var not in self.kappa_tables:
                raise ValueError(f"{var}: missing kappa_tables entry")
            for parent in self.parents[var]:
                if parent not in self.variables:
                    raise ValueError(f"{var}: unknown parent {parent}")
            parent_states = [self.variables[p] for p in self.parents[var]]
            combos = list(product(*parent_states)) if parent_states else [()]
            for combo in combos:
                col = self.kappa_tables[var].get(combo)
                if col is None:
                    raise ValueError(f"{var}: missing column for parents={combo}")
                if min(col.values()) != 0:
                    raise ValueError(
                        f"{var}: column {combo} has min rank != 0 "
                        f"(violates kappa normalisation)"
                    )
    def __repr__(self):
        return (f"RankingNetwork({len(self.variables)} variables, "
            f"{sum(len(parent) for parent in self.parents.values())} edges)")   

    def describe(self):
        """Print the network structure for inspection."""
        for var in self.variables:
            parents = self.parents[var] or ["(no parents)"]
            print(f"{var}  states={self.variables[var]}  parents={parents}")
         
    def print_joint(self):
        """Print every world with its kappa value. Useful for tiny networks."""
        for world in all_worlds(self):
            print(f"  {world}  ->  {joint_kappa(self, world)}")

# ---------------------------------------------------------------------------
# Topological sort and deterministic-variable detection.
# These are the new helpers for the smart enumeration version making scaling tractable.
# These are not needed for the brute-force version.
# --------------------------------------------------------------------------- 
    
def topological_sort(net):
    """Return a list of variables in topological order (parents before children).
    Raises ValueError if the graph has a cycle."""
    in_degree = {var: len(net.parents[var]) for var in net.variables}
    children = {var: [] for var in net.variables}
    for var in net.variables:
        for parent in net.parents[var]:
            children[parent].append(var)
    queue = [var for var, degree in in_degree.items() if degree == 0]
    order = []
    while queue:
        var = queue.pop(0)
        order.append(var)
        for child in children[var]:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)
    if len(order) != len(net.variables):
        raise ValueError("Graph has a cycle")
    return order


def _is_deterministic_var(net, var):
    """A variable is deterministic if every kappa column has exactly one state with 
    rank 0 and the rest INF. Such variables can be computed directly from their 
    parents without enumeration. Since their value is fixed by their parents."""
    for col in net.kappa_tables[var].values():
        zero_count = sum(1 for r in col.values() if r == 0)
        finite_count = sum(1 for r in col.values() if r != INF)
        if zero_count != 1 or finite_count != 1:
            return False
    return True


# ---------------------------------------------------------------------------
# Inference functions. Joint kappa, world enumeration, and posterior. These are the same for both the brute-force and smart enumeration versions; the difference is in how 'all_worlds' is implemented.
# ----------------------------------------------------------------------------


def parent_values(net, world, var):
    return tuple(world[p] for p in net.parents[var])


# Implements the ranking-network factorisation (Eq.5 in the paper): k(w) = sum of local conditional kappas. This is the "engine" that runs on both the 3-cube toy and the 50-node DAG.
def joint_kappa(net, world):
    """Joint kappa of a complete world = sum of local conditional kappas.
    Returns INF as soon as any contribution is INF, without needing to sum the rest."""
    total = 0
    for var in net.variables:
        col = net.kappa_tables[var][parent_values(net, world, var)]
        total += col[world[var]]
        if total == INF:
            return INF
    return total

# The set of all possible worlds (Ω) = Cartesian product of variable states. For small networks only. For binary variables, 2^n worlds; for 50 binary variables, 2^50 worlds = too big for brute-force.
def all_worlds(net):
    """Brute-force enumerator over every possible world (complete variable assignment). 
    Used for testing and debugging for tiny networks (n < 20). Use finite_worlds() for larger networks."""
    names = list(net.variables)
    for combo in product(*(net.variables[name] for name in names)):
        yield dict(zip(names, combo))

def finite_worlds(net):
    """Yield (world, kappa) pairs for all worlds with finite kappa.
    
    Only enumerates over the probabilistic vairables (those whose values isn't
    fixed by their parents) and propagates the deterministic variables in topological order. 
    For networks with many determinsitic relations (like the voltage chains and Led observables in the 50-node DAG) 
    this is much faster than all_worlds(), which enumerates over all 2^n worlds regardless of the deterministic structure."""
    order = topological_sort(net)
    deterministic = {var: _is_deterministic_var(net, var) for var in order}
    prob_vars = [var for var in order if not deterministic[var]]
    prob_states = [net.variables[var] for var in prob_vars]

    for combo in product(*prob_states):
        world = dict(zip(prob_vars, combo))
        total = 0
        finite = True
        for var in order:
            parent_vals = tuple(world[parent] for parent in net.parents[var])
            col = net.kappa_tables[var][parent_vals]
            if var in world:
                # Probabilistic variable: use the rank for the enumerated state.
                total += col[world[var]]
            else:
                # Deterministic variable: pick the unique rank 0 state
                for state, rank in col.items():
                    if rank == 0:
                        world[var] = state
                        break
                # Rank contribution is 0, no need to add it.
            if total == INF:
                finite = False
                break
        if finite:
            yield world, total

def consistent(world, evidence):
    """Checks if a world is consistent with the evidence (i.e. matches the observed variable states). 
    Used to filter the worlds when computing the posterior given evidence."""
    return all(world.get(var) == state for var, state in evidence.items())


# Implements the conditional ranking formula (Eq.4 in the paper): κ(q | e) = min_{w consistent with e} κ(w) - min_{w consistent with q,e} κ(w). This is the "engine" that runs on both the 3-cube toy and the 50-node DAG. Computes the base (denominator) as min k over worlds consistent with evidence, then for each query state, computes the min k over worlds consistent with both evidence and that query state, and subtracts the base to get the posterior rank. If the base is INF, raises an error because the evidence is impossible under the model.
def posterior(net, query, evidence, worlds=None):
    """Posterior kappa over the query variable's states given evidence.
    Renormalised so the minimum rank is 0.
    
    If 'worlds' is None, enumerates finite worlds on the fly. For repated
    queries on the same network, precompute once with
    'worlds = list(finite_worlds(net))' and pass it in to avoid redundant enumeration."""
    if query not in net.variables:
        raise KeyError(query)

    iter_source = worlds if worlds is not None else finite_worlds(net)

    base = INF
    state_min = {state: INF for state in net.variables[query]}

    for world, kappa in iter_source:
        if not consistent(world, evidence):
            continue
        if kappa < base:
            base = kappa
        state = world[query]
        if kappa < state_min[state]:
            state_min[state] = kappa

    if base == INF:
        raise ValueError("Evidence has rank INF: impossible under the model.")

    return {state: (kappa - base) if kappa != INF else INF for state, kappa in state_min.items()}


# ---------------------------------------------------------------------------
# Diagnostic helpers (engineer-readable output for inspecting the network and debugging)
# ---------------------------------------------------------------------------

def top_faults(net, evidence, k=5, worlds=None, fault_prefix="F_"):
    """Return the top-k most plausible faults hypotheses given the evidence.
    
    For each variable whose name start with 'fault_prefix' (default "F_"),
    treats the LAST states in net.variables[var] as the fault state. This 
    matches the spec convention of [healthy_state, fault_state] ordering.
    Computes the posterior rank of the fault state and returns the lowest k.
    
    returns a list of (vairable, fault_state, rank) tuples, sorted by increasing rank.
    Rank = 0 fully believed given the evidence. Higher = more surprising/less plausible."""
    results = []
    for var in net.variables:
        if not var.startswith(fault_prefix):
            continue
        states = net.variables[var]
        if len(states) < 2:
            continue
        fault_state = states[-1]
        post = posterior(net, var, evidence, worlds=worlds)
        results.append((var, fault_state, post[fault_state]))
    results.sort(key=lambda x: x[2]) # sort by rank
    return results[:k]

def explain_posterior(post, label=None):
    """Print a posterior dict in a human-readable way, sorted by rank ascending"""
    if label:
        print(f"{label}:")
    for state, rank in sorted(post.items(), key=lambda x: x[1]):
        if rank == INF:
            print(f"  {state}: impossible and ruled out")
        elif rank == 0:
            print(f"  {state}: rank 0, (most plausible)")
        else:
            print(f"  {state}: rank {rank}")


