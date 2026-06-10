# Security

> **Research software warning.** No independent audit, no formal verification of the
> Rust implementation. Do not deploy. This page describes the security *design* and
> where its current edges are.

## Assumptions

- **Module-SIS** (post-quantum): Ajtai commitment binding for openings with
  ℓ∞ norm < B. Norm discipline (Π_DEC every fold, low-norm F′ image) exists precisely
  to stay inside this binding regime.
- **Random-oracle Fiat-Shamir** over Poseidon2: SuperNeo's strong/weak interactive
  reductions (§6) compose, and HyperNova Appendix B's transform applies, when every
  challenge binds all preceding public data. The binding discipline is specified in
  `specs/direct-ccs-superneo-transcript-binding.md`.
- **Sum-check soundness over `K = F_{q²}`** with per-shape effective λ (floor 96,
  target 125) validated at preprocessing — see
  [Parameters](protocol/parameters.md).

## Enforced safeguards

- Parameter validation at construction time (`neo-params`): RLC norm bound, extension
  policy; invalid parameter bundles are unrepresentable.
- Poseidon2-only hashing and a static-label namespace in all protocol-binding paths.
- Digest authority rules (see [Transcript & digests](protocol/transcript-and-digests.md)):
  digests are compression, never authority; carried digests are recomputed or replayed,
  and verifiers re-check opened witnesses against claims rather than trusting digest
  chains.
- Verifier-owned Ajtai setup via the global PP registry — proof- or prover-supplied
  setup is rejected by construction.
- Verifiers recompute rather than trust: Π_RLC's verifier recomputes the combined
  claim; NIFS.V recomputes next-running claims; `verify_uncompressed` re-checks the
  final accumulator's openings, projections, norms, and CE relations.
- Red-team suites asserting specific rejections for tampered proofs, transcripts, and
  audit trails — see [Testing](development/testing.md).

## Current soundness edges (by design, tracked)

1. **The F′-encoding gap.** The chain proof attests instance satisfiability, correct
   folding, and state-chain binding — but not yet that each folded instance *is* the
   encoding of "F′ ran". That in-circuit binding is the PR5 decider's job. Until it
   lands, third-party-verifiable computation is not provided; self-prover use is.
   (`frontends/mod.rs` documents this boundary; see [Roadmap](roadmap.md).)
2. **Terminal-only verification is single-chunk.** `verify_uncompressed` rejects
   multi-chunk histories; they need the linear-time audit replay
   (`verify_uncompressed_audit`) until compressed verification exists.
3. **`paper/terminal_ce` is fail-closed but not accepting.** The compact terminal-CE
   statement exists; the sound direct verifier today is `paper/decider_ce_relation`.
4. **Side channels not addressed** — norm computations and big-int paths are not
   constant-time.
5. **Parameter hardening for production is open** (estimator review, concrete-security
   margins).

## Project security rules

The repo-level rules in [CLAUDE.md](../CLAUDE.md) bind all contributions: digest
authority rules, Poseidon2-only protocol paths, and the expectation that soundness
fixes come with red-team tests that fail while the gap exists. Design-freeze notes:
the `y_zcol` / Π_DEC indirect-binding semantics are accepted design — challenges to
them need a concrete accepting-forgery demonstration, not re-litigation.
