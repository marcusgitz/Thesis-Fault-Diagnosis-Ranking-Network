# Ranking Networks for Fault Diagnosis

This repository contains the individual ranking-network code for my bachelor's thesis:

### Ranking Networks for Fault Diagnosis in Cyber-Physical Systems

The notebook compares a ranking network with a Bayesian network baseline on a ten-module electrical circuit. The goal is to show whether ordinal rank preserves the same broad diagnostic result as probabilities.

## Contents

- `experiments.ipynb` - Main notebook. Builds the models, runs the four thesis scenarios, checks admissibility, and compares the outputs.
- `ranking_network_V2.py` - Ranking-network data structure and inference functions.
- `circuit_network.py` - Builds the coarse and fine ranking networks for the circuit.
- `bn_network.py` - Pinned copy of the Bayesian network baseline builder. The BN is included here only for a comparison baseline.

## Baseline attribution

The Bayesian network baseline was developed as a shared project. The real version is in the BN repository:
https://github.com/marcusgitz/Thesis-fault-diagnosis-BN

The toy system and original scenario environment are here:
https://github.com/kataph/Diagnostic-Assistant-Demo

## How to run

Create a Python environment and install the requirements:

pip install -r requirements.txt

Then open the notebook:

jupyter notebook experiments.ipynb

Run all cells from top to bottom.

## Notes

The ranking-network engine is a prototype for the electrical toy circuit in this thesis. It enumerates the probabilistic fault variables and propagates the deterministic variables. It is exact for this model, but it is not a general industrial inference enginge.

The probability-to-rank conversion is used only for the comparison with the BN baseline. In a real ranking-network workflow, ranks should be elicited directly from experts.
