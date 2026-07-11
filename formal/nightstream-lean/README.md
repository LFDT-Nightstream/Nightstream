# Nightstream Lean

This is the active assurance-first formalization. The existing Lean projects
(`../superneo-lean`, `../direct-ccs-fprime-lean`, `../twist-shout-lean`) are
legacy reference material and are not dependencies. This package supersedes
them for the Nightstream assurance roadmap; no theorem is inherited merely
because it exists in a legacy package.

The normative project structure, threat model, property matrix, evidence states,
and completion gates are defined in
[`specs/formal-verification.md`](specs/formal-verification.md).

## Current status

The project currently provides:

- explicit paper-level CCS and CE membership shapes;
- a Construction-2 state matching the authority-bearing core of the Rust state;
- an executable F' envelope checker;
- a proved `check_sound` theorem for branch, counter, fixed-`pc`, immutable
  boundary, and public-trace coherence;
- the exact end-to-end verifier reduction target the project is working toward;
- an artifact-checked theorem (`CIR-U64CANON`) over the exact generated R1CS
  rows of the canonical-u64 gadget, with a Rust drift gate and twin
  honest/forged witness checks on both sides;
- an artifact-checked no-wrap increment theorem (`CIR-U64INC`) over the exact
  255 rows used by the F' counter path, with `u64::MAX -> 0` rejected at the
  final carry equation in both Rust and Lean;
- an artifact-checked no-wrap addition theorem (`CIR-U64ADD`) over the exact
  319 rows used for `step_count + rows_in_chunk`, with the same universal
  integer-sum guarantee and overflow rejection.
- an artifact-checked theorem (`CIR-FPR-COUNTER`) over the exact 660-row
  production-used recursive F' counter block, including source-image binding,
  fixed batch cardinality, both integer transition equations, and three
  cross-language adversarial witnesses;
- generic checked-program theorems for exact R1CS programs that mix
  deterministic definitions with retained verifier assertions, proving both
  soundness and completeness without solving assertions for prover inputs;
- artifact-checked soundness, output uniqueness, and completeness for the
  exact 600-row production Poseidon2 permutation and the full 6,661-row F'
  chunk-shape digest program;
- an artifact-checked complete plain base-step program: all 12,498 production
  rows, classified as 10,900 SSA definitions and 1,598 assertions, with Rust
  drift/adversarial gates and Lean soundness, x_out uniqueness, and
  completeness theorems;
- exact artifact theorems for F' base authority pins, cross-step state links,
  terminal delayed links, public encoding, and one-claim CE continuity;
- artifact-checked `CIR-SOUND` for the exact 4,076,614-row plain/stateless
  `[1,1]` full-history artifact: one recursive invocation, terminal fold,
  direct terminal CE, and the minimal-supported-bit-carrier relation yield a
  two-edge `ValidExecution` with terminal validity or a named recursive/terminal
  PiRLC root event;
- artifact-checked `CIR-COMPLETE` for that same fixed profile: independent
  successful compiler executions reconstruct satisfaction of every exact row,
  without carrying `Satisfies` or a prover-supplied verifier conclusion;
- an exact compact manifest for a 2,640,071-row production steady-recursive F' profile,
  with a Rust-generated Lean data mirror, gap-free top-level and nested NIFS
  ownership proofs, sparse-triplet/source drift gates, and an explicit semantic
  decoding interface whose facts compose to `RecursiveLocalHolds`;
- a corrected PiRLC projection boundary: satisfying the exact 714-row exported
  production helper implies the complete bounded-polynomial acceptance
  predicate for every canonical assignment; acceptance then implies coefficient
  equality or a named bad root. Honest, `E(X)=X-7` at beta 7, and row-forgery
  regressions pin all three boundaries;
- a production PiRLC projection census: one 1,892-row shared ladder/rho block
  and 31 equal-shape identities of 1,916 rows and 15 pairs each, plus a generic
  Lean theorem lifting extracted trace satisfaction to batch acceptance;
- model-proved concrete CCS and CE membership (`REL-CCS`, `REL-CE`,
  `REL-CONCRETE`) over Goldilocks, the production quadratic/cyclotomic rings,
  centered norm, Ajtai action, prefix projection, and multilinear evaluation;
- a verifier-owned Appendix-B.2 production profile (`PARAM-GLOBAL`) whose
  maximum-arity inequality is inherited by every advertised batch size and
  whose literals are checked by a Rust golden-vector regression;
- an executable SumCheck checker (`SUM-CLAIM`) proved equivalent to its logical
  verifier predicate, plus a false-acceptance reduction (`SUM-SOUND`) to a
  concrete bounded-degree sampled-challenge collision;
- model-proved Π_CCS product completeness and strong extraction shape,
  Π_RLC shared-coefficient completeness and weak ambient-extraction interface,
  and Π_DEC exact recomposition/knowledge reduction (`FOLD-PICCS`,
  `FOLD-PIRLC`, `FOLD-PIDEC`);
- a composed fold theorem (`FOLD-COMPOSE`) that starts from valid final Π_DEC
  children and returns all original source witnesses or an explicit SumCheck,
  sampling, or relaxed-binding event;
- a collision-explicit canonical `x_out` authority theorem (`FPR-HASH`) that
  distinguishes directly absorbed coordinates from verifier-derived and
  equality-pinned coordinates, including stateful/stateless and the source
  Nebula lane, with separate outer-hash and inner-lane collision events;
- a true base-step theorem (`FPR-BASE`) and a proved zero-arity interpretation
  of Rust's empty `RunningInstance` (`FPR-BASE-SPEC`);
- an executable recursive F' relation (`FPR-RECURSIVE`) covering prior-link and
  accumulator authority, typed NIFS transcript context and deterministic NIFS.V
  output, semantic-state and Nebula transition, nonempty installed batches,
  exact state advance, and the outgoing link;
- exact trace induction (`TRACE-VALID`) from retained accepted inputs/proofs to
  rich semantic reachability, nonzero schedules, exact split counters, pinned
  final state, and the top-level `ValidExecution` predicate.
- a terminal CE relation and executable checker (`TERM-CE`) covering
  verifier-derived child authority, witness cardinality, public width,
  commitment, projection, low norm, evaluation shape/content, constant terms,
  and sidecars, with soundness and completeness theorems;
- Rust-shaped F' and terminal verifier programs (`RUST-REFINE`) proved
  equivalent to the M3/terminal predicates on every input, including named
  rejection theorems, production lifecycle replay, adversarial terminal
  mutations, compact-decider fail-closed regression, and full-file drift gates.

This is not yet end-to-end production security verification of SuperNeo or
F'. The fold results are model-level over explicit algebra,
rewinding, sampling, and binding boundaries; their probability bounds and the
compact-decider reduction remain open. M3's NIFS, fresh-link, application,
digest, and Nebula functions are explicit executable semantic parameters;
their exact circuit and Rust implementations are deliberately not smuggled
into the theorem as assumptions. M0, M0.5, M1, M2, M3, M4, and the
supported-surface M5 conformance milestone are complete under the normative
definitions for their advertised profiles. M4 is bounded to the exact fixed
profile above. `FPrimeFullHistoryCircuit.fPrimeCircuit_sound_or_bad` consumes
satisfaction of all 4,076,614 sparse rows in manifest order and reconstructs
the closed two-edge execution plus direct terminal predicate, or returns a
named recursive/terminal PiRLC root event.
`FPrimeFullHistoryCircuit.fPrimeCircuit_complete` reconstructs every exact row
from independent successful compiler executions. Stateful, Nebula, other
schedules, multiple recursive invocations, alternate carriers, and
parameterized circuit families are not claimed. The Fiat-Shamir/SIS
probability bounds for both root events remain M6 obligations.
The public compact decider is intentionally fail-closed with `Unsupported`;
`DEC-SOUND` cannot advance until that verifier exists.

## Ownership and dependency direction

```text
SuperNeo model       HyperNova model
          \           /
        Protocol composition
                |
      Implementation model
                |
       Assurance theorems
```

SuperNeo and HyperNova are sibling foundations. Concrete composition owns their
integration; implementation and assurance layers consume it. There are no
per-file interface façades. A boundary is introduced only when a real consumer
exists.

## Verification

```bash
lake build
lake exe check
```

The next critical path is M6: compact-decider `DEC-SOUND`, verifier reduction,
and the recursive/terminal root-event probability bounds.
