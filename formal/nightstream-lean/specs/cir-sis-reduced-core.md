# CIR-SIS-REDUCED-CORE — model-level balanced-opening core

```text
property_id: CIR-SIS-REDUCED-CORE
claim:
  For canonical Goldilocks assignments with the verifier-fixed one column,
  the 123-gate model core consists of exactly 41 centered-unit cubic gates,
  41 negative-indicator definition rows, and 41 borrow-transition rows.

  If a separately checked lowering supplies SharedFieldDigitAlias, acceptance
  of that core is equivalent to satisfaction of the existing 124 canonical
  opening rows. The retained gates imply exact Digit semantics, negative
  indicator bitness and support, and internal borrow bitness. The explicit
  alias implies the reconstruction row. Every CanonicalWitness satisfies the
  reduced core and the alias.
assumptions:
  - Every assignment coordinate is a canonical Goldilocks residue.
  - Column zero is verifier-fixed to one.
  - Reconstruction removal requires SharedFieldDigitAlias from a separately
    checked lowering; the reduced gates do not establish this alias.
non_goals:
  - Production row selection, production row indices, or Rust conformance.
  - Authorization to remove rows from the production circuit.
  - Honest production witness materialization.
  - SIS commitment binding or a security reduction.
lean_theorems:
  - ShiftedTernaryReducedCore.gates_length
  - ShiftedTernaryReducedCore.Accepts.digit
  - ShiftedTernaryReducedCore.Accepts.negative_bitness_follows
  - ShiftedTernaryReducedCore.Accepts.negative_support_follows
  - ShiftedTernaryReducedCore.Accepts.borrow_bitness_follows
  - ShiftedTernaryReducedCore.reconstructionRow_holds_of_shared_alias
  - ShiftedTernaryReducedCore.reduced_iff_canonicalRows
  - ShiftedTernaryReducedCore.CanonicalWitness.reducedCore_complete
evidence_state: model-proved
route: theorem-first
```
