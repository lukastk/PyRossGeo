### Code architecture

The main principle informing the structure of the code is reducing sparsity in the equations. As the degrees of each node in the commuting network are - in general - several orders of magnitude smaller than the total number of nodes, this should be taken into account in the code.

The code can be broadly separated into two parts: The initialisation stage, and the simulation stage.

Talk about splitting up the simulation and initialization parts.

### Notes about simulation

- Infection dynamics for infection classes that have populations smaller than 1 is disabled.

```python
if _Ns[age_b] > 1: # No infecitons can occur if there are fewer than one person at node
    _lambdas[age_a][ui] += contact_matrices[cmat_i][age_a][age_b] * _Is[age_b] / _Ns[age_b]
```

This is because the infection dynamics can cause severe numerical instabilities in combination with transport. This is caused by the
division of `_Ns[age_b]`. When the entire population of a node leaves, `_Ns[age_b]` reaches zero with some numerical errors. These
numerical errors causes `_lambdas[age_a][ui]` to blow up.

- Linear dynamics is disabled when classes have negative populations. As can be easily verified by simply integrating single-node SIR,
negative values causes severe issues and divergences.

```python
if X_state[ si + u ] > 0: # Only allow interaction if the class is positive
    u = linear_terms[o][j]
    dX_state[si+o] += n.linear_coeffs[o][j] * X_state[ si + u ]
```
