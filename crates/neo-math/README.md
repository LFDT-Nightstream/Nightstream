# neo-math

Field and ring arithmetic that every other crate uses.

- `Fq` (also `F`): the prime field with q = 2^64 − 59. Each value is canonical.
- `K`: the quadratic extension F_q[u]/(u² − 7).
- `Rq`: the cyclotomic ring F_q[X]/(X^64 + 1), of degree `D = 64`.
- `cf` and `cf_inv`: the coefficient maps between `Rq` and F_q^64.
- `SAction`: the left action of rotation matrices on vectors.
- `balanced` and `signed_sums`: balanced digits, and exact signed sums with a static overflow bound.
