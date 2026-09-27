# Parameters

The maintained Nightstream Goldilocks profile uses the following exact
values. It is not the SuperNeo Appendix B.2 reference profile.

| Parameter | Value |
|---|---:|
| Field modulus | 2^64 − 2^32 + 1 |
| Cyclotomic index / ring degree | 81 / 54 |
| Ajtai rank | 22 |
| Decomposition base `b` | 2 |
| `k_rho` | 16 |
| Norm bound `B` | 65536 |
| Expansion factor `T` | 216 |
| Extension degree | 2 |
| Sum-check rounds | 28 |
| Maximum key columns | 4,708,530 |

The selected package uses a prefix of the fixed key and zero extension to
the same domain. Its emitted setup gives the exact prefix length. The key
seed, rank, profile, and transcript schedule are bound to the package.

PiRLC combines 17 claims. Each challenge uses four transcript field elements
as one base-p integer, reduces modulo `5^54`, and decodes 54 digits in
`{−2,…,2}`. It cannot fail. Under uniform field draws, its statistical distance
from the uniform strong set is below `2^-132` per challenge. This is a sampler
bound, not a complete security estimate for concrete Poseidon2 execution.

Callers must choose a positive statistical-security minimum. The shape
estimator does not replace the explicit Fiat–Shamir and fixed-key MSIS
assumptions in the formal security argument.
