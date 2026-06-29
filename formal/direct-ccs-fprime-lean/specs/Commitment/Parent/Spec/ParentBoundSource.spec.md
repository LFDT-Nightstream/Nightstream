# Parent-Bound Source Specification

## Mathematical Target

This module gives a concrete reduced-source shape for the direct CCS/F'
authorization theorem. The source carries the parent residue vector as
authoritative verifier-bound data.

The source binding relation is:

```text
SourceBindsParent(source, parent) := source.parentResidues = parent
```

This binding is functional: a single source authorizes at most one parent
residue vector.

## Required Theorems

- The concrete parent-binding relation is functional.
- Accepted Goldilocks child-table authorizations for the same concrete source
  have equal next `Pi_CCS` inputs.
- A deterministic challenge over the same concrete source cannot authorize
  different next inputs.

## Soundness Role

This module turns the abstract functional-binding precondition into an
implementation target. The Rust source passed to Fiat-Shamir must bind the
parent residue vector with this level of authority. A digest may compress the
source, but the digest itself is not authority unless the circuit/verifier
reconstructs it from this source data.
