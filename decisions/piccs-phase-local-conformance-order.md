# PiCCS Phase-Local Conformance Order

**Status:** Accepted

## Decision

PiCCS phase-local conformance evidence can unlock PiRLC work.

PiCCS remains formally **status open**. It cannot be called
Compiler-closed, Conformance-closed, or Production-closed until this final
authority edge exists:

```text
final canonical package
  -> canonical 14-matrix LogicalRelation
  -> ProductionKey.key
  -> exact application F and StepHolds
  -> recursive fixed point
```

Before PiCCS or Stage 1 can be called Conformance-closed, every applicable
PiCCS gate must run again on that final package identity. The required gates
include exact Lean-to-Rust matrix equality, the independent Rust assignment
check against the Lean rows, complete nonzero Lean / Rust `paper_exact` / Rust
optimized parity, transcript and verifier-context checks, rejection tests,
mutation tests, package loading, and package identity checks.

## Scope

This decision changes owner order only. It does not weaken a PiCCS acceptance
criterion, convert a digest into authority, approve a proof backend, or make
backend acceptance evidence for the Lean relation or the Rust assignment.

The current digest-only PiCCS package and parity results are phase-local
evidence. They are sufficient to start the PiRLC sampler migration and PiRLC
phase work. They are not a final PiCCS closure claim.

This decision supplements the accepted
[PiCCS prior-state digest decision](piccs-prior-state-digest.md).
