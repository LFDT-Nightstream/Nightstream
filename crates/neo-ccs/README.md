# neo-ccs

CCS frontend and relation objects used by reductions.

## Surface
- Types: `CcsStructure`, `McsInstance/Witness`, `MeInstance/Witness`.
- Loader for {M_j} and CCS polynomial f; public inputs `x ⊂ z`.
- Consistency checks: `c = L(Z)`, `y_j = Z M_j^T r^b`, `X = L_x(Z)`.
- Relaxed CCS knobs: slack `u` and factor `e` (defaults `u=0, e=1`).

## Note
- The **batched polynomial Q** and the **single sum‑check** live in `neo-fold`; this crate only shapes the claims.

## Tests
- Toy CCS instances (satisfiable/unsatisfiable).
- Public‑input projection edge cases.