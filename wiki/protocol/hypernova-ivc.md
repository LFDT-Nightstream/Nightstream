# HyperNova IVC (Construction 2 and F′)

HyperNova §6.3's Construction 2 compiles a folding scheme into IVC. Nightstream
implements it in `crates/neo-fold-clean/src/paper/construction2/` over the SuperNeo
NIFS, specialized to ℓ = 1 (a single step function; `pc` is the constant
`TRIVIAL_PC`).

## State carried between steps

`construction2::State` is the IVC carrier:

| Field | Paper symbol | Meaning |
|---|---|---|
| `chunk_count` | i | Chunk counter |
| `step_count` | — | K-aware step counter (multiple rows can fold per step) |
| `z_0`, `z_i` | z_0, z_i | Initial and current application state |
| `pc` | pc_i | Program counter — constant 1 in the ℓ=1 build |
| `proof` | — | `ProofState`, see below |

`ProofState` is a tagged enum so the base case is structurally distinct:

- `Initial` — the i = 0 base case (`U = u_⊥`): zero counters, `z_0 = z_i`.
- `Active { running, latest }` — `RunningInstance` (U_i with prover-side witness W_i)
  plus `LatestInstance` (u_i / w_i, the not-yet-folded instances from the last step).

`VerifierKey` (`vk_fs`) is derived from `(params, structure)` at preprocess time and
pins the protocol the chain runs.

## One step

`advance_state` / the F′ step entry points (`paper::f_prime::{prove, verify}`,
re-exported through `construction2`) perform:

1. **NIFS fold**: `NIFS.P(running, latest) → (next_running, NifsProof)` — the
   Π_CCS → Π_RLC → Π_DEC chain (`paper/nifs/prover.rs`). The verifier side mirrors it
   and *recomputes* the next running claims rather than trusting them
   (`paper/nifs/verifier.rs`).
2. **State advance**: counters, `z_i → z_{i+1}`, base/recursive branch checks
   (`construction2/transition.rs`).
3. **Hash-chain bind**: `x_{i+1} = H(vk_fs, i+1, z_0, z_{i+1}, acc_digest, …)` via
   `paper::digest::state_x_out_digest`. `EncInst` carries the bit-decomposition of
   x_out for the public boundary encoding.

The new `latest` becomes the encoding of this step's F′ execution, to be folded by the
*next* step — the standard Construction 2 one-step lag.

## F′ — the augmented function

F′ is "the application step plus the verifier work that makes recursion sound":
it executes the app's step function **and** re-runs `NIFS.V` on the previous
step's fold, then computes the public hash chain. Two layers implement it:

- **`paper/f_prime/`** — the app-agnostic relation: native execution (`native.rs`),
  in-circuit Poseidon2 digest mirrors parity-tested against the native digests
  (`digest_circuit.rs`), the strict F′ R1CS with base/recursive entry points
  (`r1cs.rs`), bit-valued source-image boundary encodings (`source_image.rs`), and
  Poseidon/ring-action trace primitives. This layer knows no frontend.
- **`frontends/f_prime/`** — the encoded F′ *image shell*: the low-norm bit-image
  layout, mixed-gate CCS structure, encoder, and recursive-step plan that app
  frontends (e.g. `frontends/r1cs_f_prime`) build on. Dependency direction is one-way:
  `frontends::* → paper::f_prime`.

The Fiat-Shamir discipline for what F′ must absorb before each NIFS.V challenge is
specified in `specs/direct-ccs-superneo-transcript-binding.md` — see
[Transcript & digests](transcript-and-digests.md).

## Finalization and chunks

The last `extend` leaves `latest` un-folded. `finish_uncompressed*` /
`compress` flush it with one final NIFS fold (label
`FINAL_FOLD_TRANSCRIPT_LABEL`, `construction2/finalization.rs`) so the final running
accumulator covers every batch.

A **chunk** is a run of folds whose evidence a single terminal fold authenticates.
`verify_uncompressed` (terminal-only) accepts only single-chunk chains; multi-chunk
histories carry evidence in per-step rows that the terminal-only proof drops, so they
need the audit-replay path or the compressed decider. See
[Lifecycle API](../architecture/lifecycle.md).

## Divergences from the paper worth knowing

- NIFS outputs `k` CE children, not one linearized instance (the lattice Π_DEC step) —
  the "running instance" is a fixed-width claim vector.
- ℓ = 1: there is no NIVC program-counter dispatch in the IVC layer; non-uniformity,
  where needed, is a frontend-circuit concern.
- The in-circuit proof that each folded instance *is* the encoding of "F′ ran" is the
  decider's job and is not finished — see the soundness boundary in
  [Frontends](../architecture/frontends.md) and the [Roadmap](../roadmap.md).
