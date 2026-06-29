# Base-2 DEC Authorization

This spec instantiates the abstract DEC authorization rule with SuperNeo's
existing base-2 decomposition surface.

## Imported SuperNeo Surface

The module reuses:

- `splitBase2Coeffs`
- `recomposeBase2Coeffs`
- `recomposeBase2Coeffs_splitBase2Coeffs_eq_of_val_lt_pow`

from `formal/superneo-lean`.

## Canonical Membership

For this bridge, child membership means the private children are exactly the
canonical rows returned by `splitBase2Coeffs parent.z k`.

This is a sufficient concrete instantiation of the abstract uniqueness
obligation. A fuller CE-level theorem can later refine this by deriving
canonical child equality from production constraints. It must not rely on signed
low-norm bounds alone, because signed base-2 decompositions are not unique.

## Theorem Target

For canonical base-2 split children, two accepted authorizations of the same
bound parent must feed the same next `Pi_CCS` inputs.
