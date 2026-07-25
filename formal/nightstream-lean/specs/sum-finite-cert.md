# SUM-FINITE-CERT — finite verifier-visible SumCheck certificate

```text
property_id: SUM-FINITE-CERT
claim:
  A prover round is a raw finite constant-first coefficient list. Canonical
  shape is checked by acceptance, evaluation is executable by Horner's rule,
  degree is derived from list length rather than declared, and claimed-chain
  checking is exact. Semantic ghosts and verifier challenges are separate from
  the certificate. Accepted finite chains project conditionally into the
  symbolic truth-path model.
assumptions:
  - Verifier-owned initial value, challenge list, terminal value, and degree
    cap are inputs to acceptance, not certificate fields.
  - DecidableEq on the field carrier, for the executable checker only.
non_goals:
  - Symbolic SumCheck soundness and bad-challenge extraction; those are
    SUM-CLAIM and SUM-SOUND.
  - Root counting or Schwartz-Zippel probability; that is SUM-POLY-ENC and,
    for the production split, the alpha/gamma mixing boundary.
  - Production PiCCS/SplitNc integration; that is FOLD-PICCS-ARITH.
  - Transcript derivation, Fiat-Shamir, or Poseidon2.
paper_sources:
  - docs/superneo-paper/07-7-neo-s-folding-scheme-for-ccs.md:76-95 (SumCheck
    round structure and the verifier's terminal evaluation check)
  - docs/superneo-paper/13-d-deferred-theorems-and-proofs.md:245-256 (D.4's
    epsilon_SC, which this certificate must eventually be bounded against)
rust_surfaces:
  - none. This is the generic paper-layer certificate; it emits no rows.
circuit_or_encoding_artifacts:
  - none.
failure_class:
  The checker accepts a certificate whose claimed chain does not close, whose
  declared degree exceeds the verifier cap, whose shape is non-canonical (so
  degree accounting is ambiguous), or whose message count does not match the
  verifier's challenge count.
counterexample_or_witness:
  tests/SumCheckFiniteRejection.lean covers every checkChain rejection branch
  against a proved-accepting control:
    - honest_accepted / honest_chain     positive control, executable and logical
    - emptyMessage_rejected              empty coefficient list, claimed chain
                                         chosen to close so shape is the only
                                         failing branch
    - trailingZero_rejected              non-canonical trailing zero, cap raised
                                         so canonicality is isolated
    - degreeTwo_accepted_at_cap_two      control: the same certificate is
                                         accepted at cap 2
    - degreeAboveCap_rejected            and rejected at cap 1, isolating the cap
    - brokenInitialClaim_rejected        claimed initial not p(0) + p(1)
    - brokenTerminal_rejected            terminal not the replayed evaluation
    - missingChallenge_rejected          fewer challenges than messages
    - extraChallenge_rejected            more challenges than messages
    - honest_lockstep                    length law exercised, not assumed
lean_theorems:
  - Nightstream.SuperNeo.SumCheck.Finite.Message
  - Nightstream.SuperNeo.SumCheck.Finite.Message.canonicalCheck_eq_true_iff
  - Nightstream.SuperNeo.SumCheck.Finite.check_eq_true_iff_accepted
  - Nightstream.SuperNeo.SumCheck.Finite.complete_of_canonical_chain
  - Nightstream.SuperNeo.SumCheck.Finite.Chain.messages_length_eq_challenges_length
  - Nightstream.SuperNeo.SumCheck.Finite.accepted_implies_symbolicAccepted_and_truthPath
axiom_report:
  Guarded fail-closed in tests/Axioms/SumCheckFinite.lean, seven audits. No
  theorem in this property depends on Lean.trustCompiler; the rejection
  witnesses use kernel decide.
proof_hash:
  VerifierCertificate.lean 11bcb4f4243336f2
  Polynomial.lean          e2f0105dc287dda7
  SumCheckFiniteRejection.lean f2ef493c97077902
conformance_status:
  model-proved. Generic paper-layer only. Not artifact-checked, not
  rust-conformant. Promoted from specified on 2026-07-24 once the Section-8
  negative witness existed; before that the property had theorems and an axiom
  report but no rejection evidence, so "exact" was untested against
  over-acceptance.
retest_commands:
  - cd formal/nightstream-lean && lake build tests.SumCheckFiniteRejection
      tests.Axioms.SumCheckFinite
```

## Why the witnesses are shaped this way

A rejection test only means something if the fixture would otherwise be
accepted. Two of the branches are easy to test accidentally:

- an empty coefficient list also breaks the claimed-chain equation, so the
  fixture fixes `initial = 0` and `terminal = 0` to make that equation close
  and leave canonical shape as the only failure;
- a degree-2 message also changes `p(0) + p(1)` and `p(r)`, so the fixture uses
  the true values `5` and `16` and pins the cap by proving acceptance at cap 2
  and rejection at cap 1.

Without those adjustments both witnesses would have re-tested
`brokenInitialClaim_rejected` under a different name.
