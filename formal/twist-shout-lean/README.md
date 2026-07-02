# Twist/Shout Lean (Standalone)

This package is intentionally separate from [superneo-lean](../superneo-lean/README.md).
Its job is a paper-faithful, proof-complete Lean formalization of Setty-Thaler's
Twist/Shout paper, as a standalone mathematical artifact rather than a SuperNeo specialization.

## Design Rules

- The package mirrors the paper's theorem dependency graph before it mirrors any downstream integration.
- Each theorem-bearing component uses a 2-layer layout:
  - human spec: `specs/<Name>.spec.md`
  - implementation: `TwistShout/<Name>.lean`
  `<Name>Interface.lean` files exist only where they declare boundary content of
  their own; do not add per-module re-export façades.
- Specs are stateless. They describe the timeless mathematical target, not current proof status.
- Shared-core extraction from SuperNeo is deferred. Common modules such as `EqPoly`, `MLE`, and `SumCheck`
  are named and documented so they can move later without changing their theorem-facing intent.
- SuperNeo integration belongs in a separate bridge layer, not in this package.

## Layout

The package is organized around the paper's main sections.

| Barrel | Paper coverage | Re-exports |
|---|---|---|
| `TwistShout/Preliminaries.lean` | Section 3 | `EqPoly`, `MLE`, `SumCheck`, `OneHotEncoding`, `LessThanPoly` |
| `TwistShout/ShoutProtocol.lean` | Sections 4, 6, 7, Appendix C | `ShoutCore`, `ShoutOneHot`, `FastShoutSmallMemory`, `FastShoutStructuredMemory`, `ShoutLinearVariant` |
| `TwistShout/TwistProtocol.lean` | Sections 5, 8, Appendix B | `TwistCore`, `TwistValueEval`, `FastTwistProver` |
| `TwistShout/Applications.lean` | Section 9 | `SpeedySpartan`, `SpartanPP` |

Top-level entrypoint:

- `TwistShout.lean`: imports the four barrel modules above.

## Initial Module Map

### Preliminaries

- `EqPoly`: multilinear equality polynomial and Boolean-cube indicator facts.
- `MLE`: multilinear-extension definitions and core folding identities.
- `SumCheck`: paper-level sum-check statement, transcript shape, and verifier boundary.
- `OneHotEncoding`: plain and `d`-dimensional one-hot encodings used for memory addresses.
- `LessThanPoly`: multilinear less-than polynomial used by Twist's time-prefix reasoning.

### Shout

- `ShoutCore`: read-only memory checking identity and read-checking sum-check.
- `ShoutOneHot`: Booleanity and Hamming-weight-one enforcement for address columns.
- `FastShoutSmallMemory`: small-memory prover specialization from Section 6.
- `FastShoutStructuredMemory`: structured-memory prover specialization from Section 7.
- `ShoutLinearVariant`: Appendix C Shout variation with linear prover dependence on `d`.

### Twist

- `TwistCore`: read-write memory relation and increment-based virtual-memory semantics.
- `TwistValueEval`: reconstruction of `Val` from committed increments and the associated sum-check identities.
- `FastTwistProver`: prover-side specialization from Section 8.

### Applications

- `SpeedySpartan`: Shout-based acceleration path for non-uniform computation.
- `SpartanPP`: paper's Spartan++ application layer.

## Specs

The initial specs follow one of two shapes.

Theorem-bearing modules:

- `Purpose`
- `Target Formulas`
- `Paper Anchors`
- `Module Mapping`
- `Contract Surface`
- `Boundary Assumptions`
- `Dependency and Consumer Map`
- `Out of Scope`

Thin barrel modules:

- `Purpose`
- `Paper Anchors`
- `Modules re-exported`
- `Contract Surface`
- `Out of Scope`

## Commands

```bash
cd formal/twist-shout-lean
lake build
lake exe check
lake build TwistShoutTests
```

## Source Paper Material

- Paper transcription: [docs/twist-and-shout-paper/README.md](/Users/nicolasarqueros/.codex/worktrees/fec7/halo3/docs/twist-and-shout-paper/README.md)
- Implementer summary: [docs/twist-and-shout-ai-summary.md](/Users/nicolasarqueros/.codex/worktrees/fec7/halo3/docs/twist-and-shout-ai-summary.md)

## Closure Audit

- Coverage matrix: [COVERAGE.md](/Users/nicolasarqueros/.codex/worktrees/fec7/halo3/formal/twist-shout-lean/COVERAGE.md)
- Theorem-bearing coverage includes Sections 3 through 9, Appendix B, and Appendix C.
- The abstract, introduction, overview/baseline section, Appendix A, and references are
  audited as exposition rather than standalone formal targets for this package.

## Paper Errata

- Section 8.3's alternative-algorithm cost summary is arithmetically inconsistent with the three
  component bullets immediately above it. The published bullets sum to
  `(5 log(K) + 2d^2 + 4d + 7)T`, while the paper's closed form states
  `(5 log(K) + 2d^2 + 4d + 4)T`.
- The Lean formalization keeps both surfaces explicit in
  [FastTwistProver.lean](/Users/nicolasarqueros/.codex/worktrees/fec7/halo3/formal/twist-shout-lean/TwistShout/FastTwistProver.lean):
  `alternativeTwistLeadingCost` is the published Section 8.3 formula, while
  `alternativeTwistComponentLeadingCost` is the actual sum of the Section 8.1 and 8.2.5 bounds.
- Section 9.2.3's arithmetic-circuit cost paragraph states that the `d = 2, m = n` online
  sum-check work is `25m`, but the paper's own general formula and Figure 11 give
  `19m + 8n`, hence `27m` on the diagonal `m = n`.
- The Lean formalization follows the general formula / Figure 11 surface in
  [SpeedySpartan.lean](/Users/nicolasarqueros/.codex/worktrees/fec7/halo3/formal/twist-shout-lean/TwistShout/SpeedySpartan.lean),
  where `speedySpartanFieldMultiplications_d2_diag` proves the diagonal value `27m`.
