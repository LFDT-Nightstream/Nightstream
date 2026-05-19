# Reduced Source Necessity Specification

## Mathematical Target

This module records concrete counterexamples for reduced-source DEC
authorization. A reduced Fiat-Shamir source is not sound merely because a proof
about some hidden children verifies.

The source must bind the parent residues, and the next `Pi_CCS` accumulator
input must be wire-identical to the child table checked by the proof.

## Required Counterexamples

- If the same reduced source can bind different parent residues, then a
  deterministic challenge over that source can be identical while two accepted
  authorizations feed different child accumulators.
- If the proof-checked child table is not wired to the next `Pi_CCS` input, then
  the prover can verify one table and feed another.

## Soundness Role

This module is intentionally negative. It prevents the implementation from
treating a reduced source as authority by itself. Reduced sources are only
compression devices; authority comes from proof-checked parent binding,
child-table constraints, and wire identity.
