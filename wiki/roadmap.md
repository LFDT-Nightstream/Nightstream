# Status & Roadmap

Snapshot as of 2026-06-10 (branch `nico/terminal-ce-is-base-fix`). For day-to-day
detail see `TODO.md` at the repo root.

## Works today

- SuperNeo folding pipeline Π_CCS → Π_RLC → Π_DEC (`neo-reductions`, optimized +
  paper-exact engines), under the Appendix B.2 Goldilocks profile.
- IVC lifecycle in `neo-fold-clean`: `prove` / `extend` chains of CCS step instances,
  `finish_uncompressed`, terminal-only verification (single-chunk) and audit-replay
  verification (any chain).
- F′ recursive-step shell: low-norm bit-image layout, mixed-gate CCS structure,
  encoder, stateful R1CS step compiler, recursive-step plan.
- Direct-CCS, R1CS-F′, and Bellpepper frontends; SHA-256 end-to-end via Bellpepper.
- Full-history decider R1CS synthesis (audit artifact) and terminal-CE relation
  checks; red-team suites across lifecycle, frontends, and transcripts.
- WASM / iOS / Android demo builds (`demos/`).

## In progress / open

### PR5 — the compressed Spartan decider

The seam is fully wired (`lifecycle::compress` finalizes, replay-verifies, builds the
`decider::Statement`), but `decider::prove` / `verify` return `Unsupported`. Landing it
closes the two biggest gaps at once:

- third-party verifiability (the in-circuit "this instance encodes F′ ran" binding —
  the frontend soundness boundary), and
- multi-chunk verification without linear-time audit replay.

The constant-size decider also requires the lifecycle to fold encoded-F′ instances on
the online path (today's full-history audit R1CS is linear in steps).

### Compact terminal-CE proof

`paper/terminal_ce` holds the backend-neutral public statement and a fail-closed
circuit entry; an accepting verifier replacing the direct
`decider_ce_relation` checks is future work.

### Other

- Chain-facing deployment wiring (on-chain verifier target) — unfinished; proofs and
  digests are kept Poseidon2-only to preserve that target.
- Independent audit and parameter hardening — not started; see
  [Security](security.md).

## Recently landed (context for readers of older docs)

- `neo-fold-prototype` (RV32IM/CHIP-8 sandbox) and the `nstream-midnight-bridge`
  crate were removed; `neo-fold-clean` is the only proving crate. Older documents
  referencing a "published RV32IM proof boundary" describe deleted code.
- Terminal-CE audit gap closed; `is_base` is now derived from the step counter rather
  than carried (commits `ba6b60ce`, `ac25ea8e`).
