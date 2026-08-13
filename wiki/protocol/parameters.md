# Parameters

`neo-params` owns the SuperNeo Appendix B.2 Goldilocks profile and the exact
shape census. `neo-fold-clean::paper::params::Params` is the protocol-layer
wrapper.

## Appendix B.2 core

| Symbol | Value |
|---|---:|
| q | 2^64 - 2^32 + 1 |
| eta | 81 |
| d | 54 |
| kappa | 18 |
| m | 2^30 |
| b | 2 |
| k_rho | 14 |
| B | 2^14 |
| T | 216 |
| s | 2 |
| lambda | 125 |

`Params::production()` returns these paper values unchanged.

## Shape-derived lambda

A concrete one-joint relation has an exact combined error census. It includes
the joint SumCheck term, the paper mixing term, and the coordinate-fork term.

`Params::for_r1cs_shape` and `Params::for_ccs_shape` keep the Appendix B.2
core and bind the strongest lambda supported by that exact census. There is no
repository-invented minimum or safety margin. The selected lambda is evidence
about the relation, not a product security target.

Callers that have an authoritative minimum can use the explicit `*_with`
constructors. Those constructors reject a shape that cannot meet the supplied
minimum and margin.

## Other checks

- The PiRLC guard enforces `(K + k) * T * (b - 1) < B`.
- The assignment and row domains must fit the Appendix B.2 bounds.
- The extension degree remains `s = 2`.
- The exact PiRLC sampler schedule is derived from the Appendix B.2 lambda.
- Poseidon2 parameters come from `neo_params::poseidon2_goldilocks`.
