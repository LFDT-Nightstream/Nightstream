# F′ Stage 1 Per-Application Packages

**Status:** Accepted

## Problem

The HyperNova step relation contains an application transition `F`. A generic
package that lets the prover select `F`, its circuit, or its static key material
would not define one verifier-owned recursive relation.

## SuperNeo

SuperNeo v1.2 folds instances of one fixed CCS structure. HyperNova
Definitions 7 and 11 let the adversary or the IVC prover choose the
structure ([SPEC.md](../SPEC.md) section 8).

## Decision

The verifier owns the packages of each application. The prover cannot select
`F`, the packages, or any static key material.

1. Lean emits the light and refresh step packages of an application
   (`Export.BoundedDepth.StepPackage.Package`).
2. The package identity binds both packages
   (`Export.BoundedDepth.StepPackage.identity`).
3. The key digest of each circuit binds the identity and the circuit index
   (`Lifecycle.BoundedDepth.CircuitKeyDigest.digest`), and each state hash
   carries both key digests ([SPEC.md](../SPEC.md) sections 4.2 and 4.4).
4. The verifier pins the identity of the packages
   ([SPEC.md](../SPEC.md) Algorithm 2).

A carried digest is compression only. The verifier must recompute or pin
each authoritative value from verifier-owned inputs.

A Rust application that a caller compiles replaces the application region of
Lean's packages and gets its own identity ([SPEC.md](../SPEC.md) Algorithm 2,
step 3). Lean exports only its own packages, so the Lean results do not
cover that relation
([security model](../formal/nightstream-fprime/SECURITY_MODEL.md), Out of
scope).
