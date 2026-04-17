# RV64IM Recursive Backend Instantiation

## Scope

This file instantiates the abstract recursive/export-boundary architecture in
`riscv-recursive-proof.md` for the repo backend.

It is the concrete implementation-contract companion to:

- `riscv-recursive-proof.md`, which owns the theorem-facing export boundary,
- `riscv-main-relation.md`, which owns the fixed RV64IM main relation
  `R_main^SN` and its bridge theorem to carried `CE(b, L)^k` semantics,
- `riscv-authoritative-side-proof-bundle.md`, which owns the fixed RV64IM side
  opening theorem, optional packaged side verifier, and Spartan linkage rules
  that the later recursive/compression backend must preserve,
- and `riscv-kernel.md`, which owns the concrete SuperNeo backend contract,
  Goldilocks parameterization, challenge domain, and chunk-local role split.

This file owns:

- the concrete recursive proof backend,
- the concrete compression backend,
- the canonical encodings of `U_i`, `W_i`, `u_i`, and `w_i`,
- the canonical public accumulator handle `U_N^pub` if one is used,
- the chunk adapter from concrete repo-local artifacts,
- and the final public Rust API.

It shall not weaken the carried semantics below the backend-grounded paper
meaning fixed by `riscv-kernel.md`.

## 1. Concrete Backend Choices

This file owns the concrete names and parameter bindings of:

- `RecursiveBackend`
- `CompressionBackend`
- `AccumulatorHandle`

Where:

- `RecursiveBackend` is the concrete IVC/recursive proof system that proves the
  chunk-step verifier relation from `riscv-recursive-proof.md`,
- `CompressionBackend` is the outer compressor that proves the recursive
  verifier relation as the final theorem-facing proof,
- and `AccumulatorHandle` is either the terminal running instance `U_N` or a
  canonical backend binding digest `H_acc(U_N)`.

These names must be concrete system names, not generic phrases such as
"HyperNova-style" or "Spartan-style".

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

## 3. Chunk Adapter

If the repo instantiation adapts existing RV64IM chunk artifacts into the
recursive backend, the adapter shall classify and encode those artifacts
explicitly.

In particular, a repo-local adapter may consume a statement surface including:

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

A repo-local proof-complete artifact may also carry projection bundles,
packaged proofs, kernel-opening bundles, derived claim bundles, and witness-
side copies of those bindings.

This file shall define the canonical adapter:

```text
AdaptChunkToFreshCCS(repo_chunk_artifacts) -> (u_i, w_i)
```

and shall classify every repo-local artifact into exactly one of:

- public fresh-instance material,
- carried `CE` material,
- private recursive witness material,
- or audit-only material.

At minimum, the adapter section shall answer:

- which repo-local chunk metadata becomes `u_i'`,
- which repo-local digests remain public to `NIFS.V`,
- which repo-local claim/opening bundles become private recursive witness
  material,
- how the owned `R_main^SN` statement and witness are encoded before the
  backend-specific recursive/compression wrapper sees them,
- and which repo-local exported objects are dropped from the theorem-facing path
  entirely.

Side-lane tuple/proof/linkage details are owned by
`riscv-authoritative-side-proof-bundle.md`. If a chunk adapter carries
side-related objects, this file may classify their concrete backend encoding,
but it may not redefine their theorem meaning.

## 4. Public API

This file shall define the canonical public Rust API for the recursive backend
and any optional audit export.

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

Legacy proof-complete artifacts, including `Rv64imProof` from
`src/rv64im/kernel/proof/api.rs`, are audit-only rather than theorem-facing
proof objects in the normalized exported API.

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

The side theorem/public-instance/proof-object split is owned by
`riscv-authoritative-side-proof-bundle.md`. This file may only further
specialize that boundary into concrete backend encodings and
recursive/compression ownership.

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

If a concrete instantiation fixes any legacy digest layout, padded chunk-summary
shape, or fixed-width chunk-transition surface, that layout must be specified
here as canonical compiler policy rather than inferred from an implementation
artifact. Such policy is backend-local and does not create a second theorem
owner.

## 6. Conformance

A concrete recursive backend instantiation is conforming only if all of the
following hold:

- `RecursiveBackend`, `CompressionBackend`, and `AccumulatorHandle` are fixed as
  concrete named systems or canonical bindings;
- canonical encodings are specified exactly for the recursive statement,
  carried bundle, fresh chunk bundle, and accumulator handle;
- any adapter from repo-local chunk artifacts into `(u_i, w_i)` is specified
  exactly and classifies each carried field as theorem-facing public,
  recursive-internal public, private witness, or audit-only;
- the theorem-facing public API exports only the normalized recursive proof
  boundary from `riscv-recursive-proof.md`; and
- any audit-only artifact is explicitly non-theorem-facing.
