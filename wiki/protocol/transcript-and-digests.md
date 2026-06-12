# Transcript & Digests

## Poseidon2-only policy

Every protocol-binding path — Fiat-Shamir transcripts, public digests, hash chains —
uses Poseidon2 over Goldilocks, configured once in `neo_params::poseidon2_goldilocks`.
Mixed hash families (Blake3/SHA prehashes feeding a protocol digest) are banned without
explicit approval ([CLAUDE.md](../../CLAUDE.md)). The reason is on-chain verification:
the terminal proof must stay verifiable by a circuit-friendly verifier, and one hash
family keeps that surface auditable.

## The transcript (`neo-transcript`)

A Merlin-inspired, byte-first API:

- `Transcript` trait — `append_message` / `append_fields`, `challenge_bytes` /
  `challenge_field(s)`, `fork(scope)` for domain-separated sub-transcripts, `digest32`.
- `TranscriptProtocol` — typed absorb helpers (`absorb_ccs_header`,
  `absorb_poly_sparse`, `absorb_commit_coords`, `absorb_public_fields`).
- `Poseidon2Transcript` — the only production implementation.
- `labels` module — the label namespace; every absorb and challenge carries a static
  label, which is what makes transcript audits tractable.
- Feature `fs-guard` — runtime guard against Fiat-Shamir misuse in tests;
  feature `debug-log` — transcript event logging.
- `TranscriptRng` — transcript-derived randomness for prover-side sampling.

Spec: `crates/neo-transcript/specs/Transcript.spec.md`.

## What must be bound, where

`specs/direct-ccs-superneo-transcript-binding.md` is the normative answer for the
direct-CCS path. The invariant:

> A verifier challenge must be unpredictable until all public inputs and prover
> messages that precede that challenge have been fixed.

Three layers use Fiat-Shamir differently:

| Layer | Fiat-Shamir role |
|---|---|
| SuperNeo chunk (Π_CCS → Π_RLC → Π_DEC) | Derives the folding challenges (α, γ, r′, ρ_i) from a Poseidon2 transcript that has absorbed the structure, instances, and prior prover messages. |
| F′ (Construction 2) | *Recomputes* the SuperNeo transcript to re-run NIFS.V in-circuit; separately hashes the compact Construction-2 public image (`x_out`). The image hash is linkage, not a substitute for the folding transcript. |
| Spartan compression | Invents no new SuperNeo challenges; proves the F′ transcript and terminal relation checks were satisfied. |

Red-team coverage: `crates/neo-fold-clean/tests/f_prime/transcript_redteam.rs` and
`tests/reductions/nifs_v_transcript.rs` mutate absorbed material and assert challenge
divergence.

## Digest authority rules

From the project security policy ([CLAUDE.md](../../CLAUDE.md)) — these are design
invariants the code is audited against:

1. **Digests are compression, never authority.** A matching digest is binding
   material; it does not make the underlying data true.
2. Across a trust boundary, every carried digest must be either **recomputed from
   authoritative inputs**, **replayed into a verifier-driven transcript/proof**, or
   explicitly treated as non-authoritative structure.
3. Self-consistent digest chains are not evidence of soundness: if an attacker can
   mutate data and re-digest upward, the verifier must still fail.

`crates/neo-fold-clean/src/paper/digest.rs` owns the digest taxonomy: structure and
params digests (recomputed from preprocessing, never trusted from the wire),
`state_x_out_digest` (the Construction-2 hash chain), accumulator digests, and the
chunk public digest. In-circuit mirrors live in `paper/f_prime/digest_circuit.rs` and
are byte-for-byte parity-tested against the native functions
(`tests/f_prime/digest_circuit.rs`).

A concrete consequence of rule 1: the F′ chain's `acc_digest` commits to the public CE
claims, but the terminal verifier still independently checks the opened witnesses
against those claims (the terminal CE relation) — the digest alone proves nothing
about the witness. See [Decider](../architecture/decider.md).
