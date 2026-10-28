# neo-math

Field and ring arithmetic that every other crate uses.

- `Fq` (also `F`): the Goldilocks field, q = 2^64 − 2^32 + 1.
- `K`: the quadratic extension F_{q^2}.
- `Rq`: the cyclotomic ring F_q[X]/(Φ_81), of degree `D = 54`.
- `cf` and `cf_inv`: the coefficient maps between `Rq` and F_q^54.
- `SAction`: the left action of rotation matrices on vectors.
- `balanced` and `signed_sums`: balanced digits, and exact signed sums with a static overflow bound.
