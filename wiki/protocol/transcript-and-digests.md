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

The selected protocol contract and active Lean transcript model are the
authority. Rust framing tests live in `crates/neo-transcript/tests`.

## What must be bound, where

[NS-TRANSCRIPT-ORDER](../../protocol-contract/src/normative/80-nightstream-verifier.md#ns-transcript-order--fold-transcript-schedule)
defines the selected fold schedule. Its binding invariant is:

> A verifier challenge must be unpredictable until all public inputs and prover
> messages that precede that challenge have been fixed.

Three layers use Fiat-Shamir differently:

| Layer | Fiat-Shamir role |
|---|---|
| SuperNeo chunk (Π_CCS → Π_RLC → Π_DEC) | Derives the folding challenges (α, γ, r′, ρ_i) from a Poseidon2 transcript that has absorbed the structure, instances, and prior prover messages. |
| F′ (Construction 2) | *Recomputes* the SuperNeo transcript to re-run NIFS.V in-circuit; separately hashes the compact Construction-2 public image (`x_out`). The image hash is linkage, not a substitute for the folding transcript. |

Current sampler parity checks compare Rust with Lean. The staged fold checks
verify the complete continued C/R/D transcript and its returned values.

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

`nightstream-fprime/src/identity.rs` owns package and verifier-context binding.
`nightstream/src/folding/transcript.rs` and `lifecycle/inputs.rs` own the fold
transcript and state inputs. The active Lean package supplies the corresponding
protocol and circuit definitions.

A concrete consequence of rule 1: the F′ chain's `acc_digest` commits to the public CE
claims, but the terminal verifier still independently checks the opened witnesses
against those claims (the terminal CE relation) — the digest alone proves nothing
about the witness. See [Decider](../architecture/decider.md).
