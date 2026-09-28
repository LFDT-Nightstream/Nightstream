# F′ Stage 1 Per-Application Packages

**Status:** Accepted

## Problem

The HyperNova step relation contains an application transition `F`. A generic
package that lets the prover select `F`, its circuit, or its static key material
would not define one verifier-owned recursive relation.

PiCCS, PiRLC, PiDEC, and the accumulator can supply phase-local evidence before
the final application exists. They cannot receive final conformance status
until the application, final package, verifier context, verification key, and
recursive fixed point are one closed object.

## Decision

Production uses verifier-owned, Lean-authored per-application packages.

For each production application `F`:

1. Lean assembles one canonical Stage 1 package for the exact application.
2. The package identity binds the exact application and all static key
   material.
3. The verifier-context digest binds the exact application and all static key
   material.
4. The verification key binds the exact application and all static key
   material.
5. The production verifier pins or allowlists the final package identity.
6. The prover cannot select `F`, the package, or any static key material.

The binding construction must be acyclic or have a proved recursive fixed
point. A carried digest is compression only. The verifier must recompute or pin
every authoritative value from verifier-owned inputs.

## Closure rule

No per-application package is Conformance-closed until all of the following
exist for that exact application:

- the concrete Lean application relation and circuit;
- the complete Lean Stage 1 assembler;
- the recursive fixed point;
- the final canonical package identity;
- the final verifier-context digest and verification key binding;
- a production verifier path that pins or allowlists the identity; and
- a complete rerun of every required Lean, exact-matrix, independent-assignment,
  nonzero-parity, mutation, loader, identity, and exact-cut external-review
  gate on that final identity.

This decision authorizes application, accumulator, terminal, and Stage 1
assembler work on phase-local evidence. It does not close any phase, authorize
Stage 2, authorize a proof backend, or make backend acceptance evidence for
Lean semantics or Rust assignment conformance.
