# RV64IM Recursive Backend Instantiation

STATE: WORK IN PROGRESS (IMPLEMENTATION PRECONDITIONS NOT YET FROZEN)

## Scope

This file instantiates the abstract recursive/export-boundary architecture in
`riscv-recursive-proof.md` for the current repo backend.

It is the concrete implementation-contract companion to:

- `riscv-recursive-proof.md`, which owns the theorem-facing export boundary,
- `riscv-main-relation.md`, which owns the fixed RV64IM main relation
  `R_main^SN` and its bridge theorem to carried `CE(b, L)^k` semantics,
- `riscv-witness-backed-side-bridge.md`, which owns the fixed witness-backed
  side-bridge relation that the later recursive/compression backend must
  compile,
- and `riscv-kernel.md`, which owns the concrete SuperNeo backend contract,
  Goldilocks parameterization, challenge domain, and chunk-local role split.

This file owns:

- the concrete recursive proof backend,
- the concrete compression backend,
- the canonical encodings of `U_i`, `W_i`, `u_i`, and `w_i`,
- the canonical public accumulator handle `U_N^pub` if one is used,
- the chunk adapter from the current `neo-fold-next` proof path,
- and the final public Rust API that replaces the current proof-complete export.

It shall not weaken the carried semantics below the backend-grounded paper
meaning fixed by `riscv-kernel.md`.

## 1. Concrete Backend Choices

The following implementation choices shall be frozen here before protocol-
critical implementation begins:

- `RecursiveBackend := ...`
- `CompressionBackend := ...`
- `AccumulatorHandle := ...`

Where:

- `RecursiveBackend` is the concrete IVC/recursive proof system that proves the
  chunk-step verifier relation from `riscv-recursive-proof.md`,
- `CompressionBackend` is the outer compressor that proves the recursive
  verifier relation as the final theorem-facing proof,
- and `AccumulatorHandle` is either the terminal running instance `U_N` or a
  canonical backend binding digest `H_acc(U_N)`.

These names must be concrete system names, not generic phrases such as
"HyperNova-style" or "Spartan-style".

No protocol-critical implementation shall merge while any of these remain
undefined.

## 1A. Semantic Freeze

The concrete repo instantiation shall preserve the following semantic choices
from `riscv-recursive-proof.md`:

- chunk boundaries are canonical, not witness-chosen,
- `NextBoundary(Sched_SN, n, i, s_i)` is derived from the fixed schedule
  metadata and exact semantic step count `n`,
- `N := ChunkCount(Sched_SN, n)` is the canonical terminal chunk index,
- and post-halt padding is forbidden because `n` counts exact legal RV64IM
  semantic transitions.

## 2. Canonical Recursive Object Encodings

This file shall define exact encodings for:

```text
Enc_main_relation_stmt
Wit_main_relation
Enc_CE_bundle
Wit_CE_bundle
Enc_CCS_chunk_bundle
Wit_CCS_chunk_bundle
Enc_chunk_inst
H_acc
Init_SN
```

For each encoding, this file shall specify:

- field order,
- byte order and canonical serialization,
- hash domain separators,
- transcript personalization labels,
- field packing rules,
- how public machine-state fields are represented,
- and how chunk interval metadata is represented.

At this layer, the carried and fresh objects are still backend-owned encodings
of the paper relations:

- `Enc_main_relation_stmt`, `Wit_main_relation` preserve the exact fixed
  public/private theorem owned by `riscv-main-relation.md`,
- `U_i`, `W_i` represent carried `CE(b, L)^k` bundles,
- `u_i`, `w_i` represent fresh chunk-local `CCS` claim bundles.

This file may define compact repo-local encodings, but it may not replace those
paper semantics with a generic “proof object” abstraction.

This file shall also specify the knowledge-soundness or ordinary soundness
assumptions actually provided by the chosen recursive and compression backends.

## 3. Chunk Adapter From the Current Proof Path

The current public RV64IM proof path is chunked and bridge-heavy.

In particular, the current exported statement includes:

- `root_params_id`,
- `fold_schedule`,
- `chunk_count`,
- `stage_claims_digest`,
- `stage_packages_digest`,
- `kernel_opening_digest`,
- `prepared_step_bindings_digest`,
- `execution_digest`,
- `final_state_digest`,
- `transcript_final_digest`,
- `main_lane_surface_digest`,
- `root_lane_columns_digest`,
- `public_step_count`,
- `final_pc`,
- and `halted`.

The current proof object also carries projection bundles, packaged proofs,
kernel-opening bundles, derived claim bundles, and witness-side copies of those
bindings.

This file shall define the canonical adapter:

```text
AdaptChunkToFreshCCS(current_chunk_artifacts) -> (u_i, w_i)
```

and shall classify every current artifact into exactly one of:

- public fresh-instance material,
- carried `CE` material,
- private recursive witness material,
- or audit-only material.

At minimum, the adapter section shall answer:

- which current chunk metadata becomes `u_i'`,
- which current bridge digests remain public to `NIFS.V`,
- which current claim/opening bundles become private recursive witness
  material,
- how the owned `R_main^SN` statement and witness are encoded before the
  backend-specific recursive/compression wrapper sees them,
- and which current exported objects are dropped from the theorem-facing path
  entirely.

## 4. Public API Cutover

This file shall define the canonical end-state public Rust API.

At minimum it shall name:

```rust
pub struct Rv64imStatement { ... }
pub enum Rv64imPublicOutput { ... }
pub struct Rv64imCompressedProof { ... }
pub struct Rv64imAuditBundle { ... }
```

and the canonical functions:

```rust
prove_recursive(...)
verify_recursive(...)
prove_audit(...)
verify_audit(...)
```

The current `Rv64imProof` from `src/rv64im/kernel/proof/api.rs` shall be
reclassified as a proof-complete audit artifact, not the normalized exported
theorem-facing proof.

## 4A. Binding Inventory

This file shall state the exact binding location for every consensus-critical
value, including:

- `vk_NIFS`,
- `vk_IVC`,
- `root_params_id`,
- `fold_schedule`,
- `chunk_count`,
- encoding/transcript/version identifiers,
- `dig(P)`,
- `n`,
- and `U_N^pub` if one is used.

For each value, this file shall say whether it is:

- fixed by verifier configuration,
- theorem-facing public input,
- recursive-internal public input,
- or private witness material.

The exhaustive current side-bridge field classification is owned by
`riscv-witness-backed-side-bridge.md`, Section 3A. No side-bridge field may be
classified differently here. This file may only further specialize that
inventory into concrete backend encodings and recursive/compression ownership.

## 5. Minimal Shapes To Freeze

The following shapes should be frozen early to prevent interface drift:

```rust
pub struct Rv64imStatement {
    pub program_digest: ProgramDigest,
    pub public_step_count: u64,
    pub initial_state: PublicMachineState,
    pub output: Rv64imOutput,
}

pub enum Rv64imOutput {
    FinalState(PublicMachineState),
    Projection {
        output_id: OutputProjectionId,
        bytes: Vec<u8>,
    },
}

pub struct Rv64imCompressedProof {
    pub stmt: Rv64imStatement,
    pub terminal_accumulator_handle: Option<Vec<u8>>,
    pub proof_bytes: Vec<u8>,
}

pub struct Rv64imAuditBundle {
    pub legacy_proof_complete: Rv64imProof,
}

pub struct RecursiveChunkPublic {
    pub program_digest: ProgramDigest,
    pub chunk_index: u32,
    pub step_lo: u64,
    pub step_hi: u64,
    pub state_in: PublicMachineState,
    pub state_out: PublicMachineState,
    pub halted_out: bool,
}
```

These are interface targets, not final code. If the concrete backend requires a
different accumulator-handle representation than `Option<Vec<u8>>`, this file
shall replace it with a canonical concrete type.

For the current RV64IM hybrid side-bridge compiler path, the base-component
layout is fixed to exactly four digests in this order:

1. stage-claim proof bundle digest
2. stage-package proof bundle digest
3. kernel-opening proof bundle digest
4. kernel-claim proof bundle digest

That four-component layout is compiler policy, not a caller-controlled
container shape. Any change to that order or count is a protocol/compiler
boundary change and must update both this file and the owning Rust contract.

For the current RV64IM hybrid side-bridge compiler path, the chunk-transition
layout is also frozen:

- maximum chunk-transition slots: `64`
- active slots: the carried Nightstream chunk summaries
- padded tail slots: canonical zero summaries and zero handoff digests

The canonical padded chunk summary is:

```rust
FixedShapeChunkSummary {
    start_index: semantic_step_count,
    public_step_count: 0,
    public_chunk_digest: [0; 32],
    chunk_relation_digest: [0; 32],
}
```

Any carried hybrid side-bridge target/relation for fewer than `64` chunks must
pad the tail with exactly that summary and must pad the corresponding
chunk-transition witness digests with `[0; 32]`.

## 6. Implementation Sequence

Implementation should proceed in the following order:

1. Freeze `RecursiveBackend`, `CompressionBackend`, and `AccumulatorHandle`.
2. Freeze chunk-step recursion as the only recursive unit.
3. Define canonical encodings for `U_i`, `W_i`, `u_i`, `w_i`, `u_i'`, and
   `H_acc`.
4. Implement `Init_SN(dig(P), z_0) -> (U_0, W_0)` using the backend contract
   from `riscv-kernel.md`.
5. Implement `AdaptChunkToFreshCCS(current_chunk_artifacts) -> (u_i, w_i)`.
6. Implement the recursive step relation over chunk intervals.
7. Implement recursive proof generation and verification for the full
   execution.
8. Implement the outer compression proof over the recursive verifier relation.
9. Introduce the new theorem-facing public API.
10. Demote the current proof-complete export to audit/debug use only.
11. Add size and correctness gates showing that exported proof size shrinks
    substantially and no longer exports proof-complete sidecars.

## 7. Exit Criteria

This instantiation is complete only when all of the following are true:

- the recursive unit is frozen as one chunk,
- the recursive and compression backends are named concretely,
- the canonical encodings are specified exactly,
- the current chunked proof path has a defined adapter into `(u_i, w_i)`,
- the theorem-facing public API no longer exports the current bridge-owned
  digest bundle by default,
- and the current proof-complete RV64IM artifact is clearly audit-only.
