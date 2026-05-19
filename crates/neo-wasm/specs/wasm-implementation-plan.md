# WASM Implementation Plan For `neo-fold-prototype`

## Purpose

This document is a detailed implementation plan for rebuilding the historical
WASM prototype on top of the `neo-fold-prototype` architecture.

It is based on:

- the historical WASM proving strategy,
- the current `neo-fold-prototype` VM/frontend contracts,
- the CHIP-8 implementation as the primary frontend reference,
- and the maintained RV64 path as the main reference for soundness boundaries
  and machine-data preparation discipline.

This is an implementation plan, not a theorem spec.

## Progress Checklist

- [x] Add clean-room WASM strategy doc under `neo-fold-prototype/specs/`.
- [x] Add `neo-fold-prototype`-specific implementation plan under `neo-fold-prototype/specs/`.
- [x] Create a dedicated `src/wasm/` frontend subtree in `neo-fold-prototype`.
- [x] Wire `neo-fold-prototype` crate exports for the new WASM frontend.
- [x] Define WASM opcode taxonomy and supported phase-1 subset.
- [x] Define frontend-local normalized trace records and builder error types.
- [x] Define a fixed-width WASM row layout and a minimal `CoreCcsSpec`.
- [x] Implement `RwasmVmSpec`.
- [x] Implement `RwasmTraceBuilder` that packages rows into `StepBuild`.
- [x] Add one end-to-end prove/verify test through the generic `neo-fold-prototype` spine.
- [x] Add direct tests for tracer normalization from concrete `wasm::Tracer` rows.
- [x] Move WASM test coverage into `tests/` rather than inline impl-file tests.
- [x] Bind selectors to concrete opcode bytes in the real WASM CCS.
- [x] Enforce stack-address discipline for the phase-1 supported row shapes.
- [x] Enforce direct arithmetic semantics for `i32.add` and `i32.sub`.
- [x] Enforce basic shout metadata consistency for lookup-backed rows.
- [x] Emit frontend-owned lookup payloads for the historical lookup-routed opcode family.
- [x] Start a dedicated `wasm/stage1/` module for Shout-channel ownership and stage-1 summary extraction.
- [x] Add the first transcripted Stage-1 prover/verifier slice for one real WASM channel (`i32.eqz`).
- [x] Generalize the transcripted Stage-1 slice to the binary lookup-routed WASM channels.
- [x] Start a dedicated `wasm/stage2/` module for stack-memory ownership and summary extraction.
- [x] Add the first transcripted Stage-2 prover/verifier slice for WASM stack replay consistency.
- [x] Add a dedicated `wasm/stage3/` module for continuity / bridge ownership.
- [x] Add the first transcripted Stage-3 prover/verifier slice for WASM boundary continuity.
- [x] Strengthen Stage 2 toward a Twist-style shape with shared stack access families and value-from-inc claims.
- [x] Add a minimal `wasm/kernel/` owner surface above Stage 1 / 2 / 3.
- [x] Add a verifier-checked kernel-opening summary with selected row / step references and stage digests.
- [x] Add a folded root-run bridge from the WASM kernel into the generic `neo-fold-prototype` CCS proving path.
- [x] Replace Wasmtime adapter string-based opcode parsing with structural decode from `wasmparser::Operator`.
- [x] Add `local.get`, `local.set`, `local.tee` opcodes to the ISA, IR, layout, CCS, and adapters.
- [x] Add `COL_LOCALS_FBP`, `COL_LOCAL_INDEX`, `COL_LOCAL_VALUE` columns and row-local CCS constraints for locals.
- [ ] Add Stage-2 locals access family (`local_read` / `local_write`) alongside the stack family.
- [ ] Implement the function-info ROM for FBP updates and bounds checking (multi-function / recursion support).
- [ ] Implement the label ROM for stack-depth tracking on control-flow branches.
- [ ] Flesh out remaining direct row-local semantics beyond the initial phase-1 subset.
- [ ] Reintroduce packed auxiliary lookup semantics for the historical WASM ALU subset.
- [ ] Strengthen the current Stage-2 linkage batching beyond the present exact-row / recomputed-summary shape.
- [ ] Replace the execution-source wording with a normalized WASM trace interface and keep `wasm` as one adapter.
- [ ] Decide whether phase 2 should remain frontend-owned or continue growing the staged WASM kernel now in place.

## Module ROMs

WASM execution relies on two categories of static per-module data that cannot be
derived from the execution trace alone and must be supplied as read-only lookup
tables (ROMs) to the proving layers.

### Function-info ROM

**Purpose:** Supply the number of locals declared by each function in the module.

**Use cases:**

- **Locals FBP update on call/return:** When a function is called, the locals
  frame base pointer (FBP) advances by `num_locals[callee]`. On return it
  retreats by `num_locals[current_function]`. This is the multi-function
  generalisation of the single-function case where FBP is always 0.
- **Bounds checking:** Each `local.get/set/tee` must satisfy
  `local_index < num_locals[current_function]`. The function-info ROM supplies
  `num_locals` per function for this check.

**Current status:** Not yet implemented. In the current single-function scope,
FBP is always 0 and bounds checking is deferred. The columns `COL_LOCALS_FBP`
and `COL_LOCAL_INDEX` are present in the layout to make the future extension
straightforward. A note about this deferral appears in the adapters.

**Shape:** `(function_index: u32) → (base_addr: u64, num_locals: u32)`

### Label ROM (Control Flow)

**Purpose:** Supply the operand stack depth at each branch label in the module.

**Use cases:**

- **Stack unwinding on `br` / `br_if`:** When a branch target has a different
  stack depth than the current depth, the stack must be unwound by
  `current_depth - target_depth` elements before jumping. The label ROM gives
  the target depth without requiring the prover to trace the full control-flow
  graph.
- **`block` / `loop` / `if` exit depths:** Each structured control block carries
  an expected stack depth at its exit. The ROM canonicalises this per label index.

**Current status:** Not yet implemented. Control flow support (`br`, `br_if`,
`loop`, `block`) is out of scope for phase 1.

**Shape:** `(function_index: u32, label_index: u32) → stack_depth: u32`

---

Both ROMs are immutable for the lifetime of a module. They are natural candidates
for Twist-style read-only memory arguments rather than interactive lookup tables,
since each entry is read many times but never written.

## Known Technical Debt

These are engineering-quality issues identified in Phase 1 review. They do not block
Phase 1 correctness but must be addressed before any soundness claim is made.

### Semantic gaps

- **Lookup-routed ops have no end-to-end semantic verification** (`ccs.rs`,
  `stage1/`): For `i32.mul`, `i32.and`, `i32.or`, `i32.xor`, `i32.eqz`,
  `i32.eq`, `i32.ne`, `i32.lt_s`, `i32.lt_u`, the CCS only validates that a
  shout claim exists, not what it computes. Real arithmetic truth lives in the
  backend lookup table and is not end-to-end verified in Phase 1. The Stage-1
  prover batches claim consistency but does not connect to a ground-truth table.
  Must be closed in Phase 2 before any soundness claim.

- **Opening summary is structural, not cryptographic** (`kernel/openings.rs`):
  The kernel opening summary builds digests and counts but does not perform
  cryptographic opening verification. This is the gap between "a kernel exists"
  and "the kernel is sound." Must be closed before production use.

### Soundness assumptions that must be documented or strengthened

- **Stage 2 oracle is additive, not ordered** (`stage2/`): The value-from-increment
  claim checks `final = init + Σwrites - Σreads` per address family but does not
  verify access order. This is sound only under strict LIFO stack discipline. Any
  future opcode that touches the same address out of LIFO order would produce a
  false trace the prover would accept. Either document this assumption explicitly
  in the module header or strengthen the oracle before non-stack memory is added.

- **Stage 3 verifies endpoints only** (`stage3/`): Per-step boundary continuity
  is enforced in the CCS, not in Stage 3. Stage 3 checks only the start and end
  state. This is architecturally correct but a reader of the stage in isolation
  may assume it covers the full trace. Add a module-level comment stating the
  boundary.

### Engineering quality

- **`Select` auxiliary constraints are hardcoded** (`builder.rs`): `COL_AUX0`
  and `COL_AUX1` are filled inside a hard-coded `if matches!(opcode, Select)`
  block. Does not scale to additional opcodes needing auxiliary columns. Needs a
  metadata-driven dispatch before any new opcode with aux constraints is added.

- **Opcode↔code round-trip has no single source of truth** (`isa.rs`):
  `code_to_concrete` uses a chain of `if x == opcode_code(...)` guards. A
  static lookup table would remove the duplicated inverse-mapping logic and
  make exhaustiveness checking easier.

## Assumptions

These are the assumptions used in the plan:

1. The long-term owner should be `neo-fold-prototype`, not legacy `neo-fold`.
2. Phase 1 should reproduce the historical proving envelope, not solve full
   WASM proving.
3. The initial target is a frontend-integrated proving path that can build
   `StepBuild` records and run through the generic `neo-fold-prototype` proving
   spine.
4. A full WASM kernel comparable to the CHIP-8 or RV64 kernel is still out of
   scope for phase 1, but the phase-1 surface should be compatible with one.
5. The branch should remain lean. New code should be added only where it has a
   clear ownership boundary in the `neo-fold-prototype` structure.

These are conventions and not necessities:

1. Mirroring the CHIP-8 directory shape exactly.
2. Reusing the historical witness layout verbatim.
3. Building Stage 1 / Stage 2 / Stage 3 submodules immediately.

## Executive Recommendation

Implement WASM in two phases:

### Phase 1

Build a **frontend-owned WASM row builder** inside `neo-fold-prototype` that:

- owns opcode metadata,
- owns trace normalization from `wasm::Tracer`,
- owns a fixed-width core CCS spec,
- builds `StepBuild` records directly,
- and proves the historical narrow subset through the generic `neo-fold-prototype`
  run/prove/verify flow.

This phase should not attempt to build a full staged kernel, but it should own
enough structure to preserve the old prototype's proof split.

### Phase 2

After phase 1 is stable, decide whether WASM should:

- stay as a lightweight frontend with only generic `neo-fold-prototype` proof
  packaging,
- or grow a staged kernel analogous to CHIP-8 and RV64 with explicit auxiliary
  commitment/opening stages.

This split keeps the initial rewrite tractable.

## Migration Status

The migration from the historical branch is not complete yet.

What is already in place:

- normalized per-step trace records,
- fixed-width main-lane CCS,
- lookup payload extraction,
- Stage 1 ownership for the current lookup-routed subset,
- a stronger Stage 2 ownership boundary with access-family and value-from-inc summaries,
- a first Stage 3 continuity / bridge slice,
- a minimal staged `wasm/kernel/` owner boundary,
- a verifier-checked kernel-opening summary,
- and a folded root-run bridge into the generic `neo-fold-prototype` proving spine.

What is still missing before the old prototype should be considered migrated:

- the remaining direct row-local opcode semantics,
- the packed auxiliary lookup route used by the historical ALU subset,
- a stronger Stage-2 linkage/oracle story than the current exact-row replay summaries,
- fuller opening/package artifacts if CHIP-8-level packaging parity is desired,
- and a cleaner execution-source abstraction so `wasm` is only one adapter.

So the current branch should be read as:

- substantial migration of the proving architecture,
- but not full semantic parity with the old branch yet.

## Proof Strategy Split

The long-term WASM proof surface should mirror the CHIP-8 split even if the
execution source eventually changes.

### Main-Lane CCS

Owner:

- `crates/neo-fold-prototype/src/wasm/ccs.rs`

Purpose:

- prove row-local structure cheaply,
- keep the generic folded CCS pipeline responsible for exported per-step rows.

Current scope:

- selector booleanness and one-hotness,
- selector-to-opcode-byte binding,
- stack-pointer update,
- direct stack-address formulas for the supported row shapes,
- simple PC update rules for the currently supported non-branch rows,
- direct arithmetic for `i32.add` and `i32.sub`,
- the simplified boolean-guarded `select` relation,
- and shout metadata consistency for lookup-backed rows.

This layer should remain local. It should not attempt to prove table/memory
consistency by itself.

### Stage 1 / Shout

Owner:

- `crates/neo-fold-prototype/src/wasm/stage1/`

Purpose:

- prove lookup-routed read-only opcode semantics.

Current scope:

- `i32.eqz`
- `i32.eq`
- `i32.ne`
- `i32.lt_s`
- `i32.lt_u`
- `i32.and`
- `i32.or`
- `i32.xor`
- `i32.mul`

Current proof style:

- transcript-bound batched semantic checks per shout channel over exact exported
  lookup rows.

Future direction:

- replace the current exact-row batch with a proper shout/table argument once
  the kernel story is chosen.

### Stage 2 / Twist

Owner:

- `crates/neo-fold-prototype/src/wasm/stage2/`

Purpose:

- prove mutable shared stack-memory consistency.

Current scope:

- exact replay of stack reads/writes against one shared stack map,
- explicit access-family summaries for `read0`, `read1`, `read2`, and `write1`,
- a value-from-inc surface over the shared stack state,
- final stack snapshot export,
- transcript-bound batched read consistency,
- and a first transcripted linkage batch over the exported Stage-2 claims.

Future direction:

- move toward a real Twist-style argument with access families, batched
  read/write claims, and value-from-inc semantics.

### Stage 3 / Continuity And Bridge

Owner:

- `crates/neo-fold-prototype/src/wasm/stage3/`

Purpose:

- prove adjacent-row boundary continuity,
- own row bindings for future root export.

Current scope:

- boundary continuity across `pc`, `sp`, and `halted`,
- start/end boundary summaries,
- row binding export for future root packaging.

### Folded Proofs And Openings

Owner:

- generic `neo-fold-prototype` root proving and opening layers.

Purpose:

- prove/export the folded CCS session over prepared WASM steps,
- compress opening obligations.

Current scope:

- a staged kernel proof bundle (`Stage 1`, `Stage 2`, `Stage 3`),
- a verifier-checked kernel-opening summary over stage rows and prepared steps,
- and a `prove_kernel_run` / `verify_kernel_run` bridge into the generic folded CCS session.

Important non-goal:

- these folded/opening layers are not where WASM opcode or memory semantics
  should live.

## Execution Boundary

The stable proving boundary should be a **normalized WASM execution trace**,
not a hard dependency on `wasm::Tracer`.

Current adapter:

- `wasm::Tracer` via `execute.rs` / `lower.rs`

Future adapters:

- Wasmtime debug tracing
- any other concrete runtime trace that can be normalized into the same
  frontend-owned step shape

The key architectural rule is:

- execution-source differences should terminate at the normalization boundary;
  the CCS and Stage 1 / 2 / 3 layers should consume the normalized trace shape.

## Mapping The Historical Design To The New Architecture

## Old responsibility -> New owner

### Opcode metadata

Old role:

- `neo-memory::wasm::opcode`

New owner:

- `crates/neo-fold-prototype/src/wasm/isa.rs`
- `crates/neo-fold-prototype/src/wasm/tables.rs`

Reason:

- CHIP-8 keeps opcode taxonomy and decode/table metadata inside the VM frontend.
- `neo-fold-prototype::vm` only owns generic contracts, not VM-specific opcode sets.

### Trace normalization

Old role:

- `neo-memory::wasm::tracer_adapter`

New owner:

- `crates/neo-fold-prototype/src/wasm/execute.rs`
- `crates/neo-fold-prototype/src/wasm/lower.rs`

Reason:

- in CHIP-8, execution and lowering are frontend-owned,
- the builder consumes already normalized per-step trace data,
- keeping normalization frontend-owned prevents generic crate pollution.

### Step witness layout and core CCS

Old role:

- `neo-memory::wasm::arith`

New owner:

- `crates/neo-fold-prototype/src/wasm/layout.rs`
- `crates/neo-fold-prototype/src/wasm/ccs.rs`

Reason:

- CHIP-8 keeps fixed layout constants and core CCS in `layout.rs` and `ccs.rs`,
- `VmSpec::core_ccs_spec()` expects the frontend to own its core CCS.

### Step packaging into proof inputs

Old role:

- `WasmTraceArith`,
- direct legacy `R1csCpu` / shared-bus witness path.

New owner:

- `crates/neo-fold-prototype/src/wasm/builder.rs`

Reason:

- CHIP-8’s `Chip8TraceBuilder` is the direct model.
- The frontend should produce `StepBuild` records with `StepInput` and
  `extension_data`.

### VM contract boundary

Old role:

- implicit through legacy shared-bus/session APIs.

New owner:

- `crates/neo-fold-prototype/src/wasm/spec.rs`
- `crates/neo-fold-prototype/src/wasm/mod.rs`

Reason:

- CHIP-8 exposes `Chip8VmSpec` and a thin compatibility surface.
- WASM should have an explicit `RwasmVmSpec` implementing `VmSpec`.

## Proposed File Structure

Add a new frontend subtree:

```text
crates/neo-fold-prototype/src/wasm/
├── mod.rs
├── spec.rs
├── isa.rs
├── layout.rs
├── ccs.rs
├── tables.rs
├── execute.rs
├── lower.rs
├── builder.rs
└── trace.rs
```

### Ownership rules

`mod.rs`

- owns the frontend barrel,
- exports only the curated public WASM frontend surface.

`spec.rs`

- thin compatibility / curated re-export layer,
- should stay small.

`isa.rs`

- owns opcode ids,
- opcode classification,
- stack arity metadata,
- stable lookup ids,
- trace-visible semantic categories.

`layout.rs`

- owns all witness column and width constants,
- owns public-prefix definition.

`ccs.rs`

- owns `RwasmVmSpec`,
- owns `CoreCcsSpec`,
- owns core row-local CCS construction.

`tables.rs`

- owns lookup-family metadata,
- owns packed relation family declarations,
- owns mapping from opcode family to frontend lookup channels.

`execute.rs`

- owns direct translation from `wasm::Tracer` rows into frontend-local trace
  records.

`lower.rs`

- owns any trace normalization or row expansion required before proof building,
- should initially remain shallow because phase 1 is one-row-per-step.

`builder.rs`

- owns `RwasmTraceBuilder`,
- converts normalized rows into `StepBuild`,
- creates `StepInput` values by packing row-major witness vectors and committing
  them.

`trace.rs`

- thin compatibility barrel over `execute` + `lower` + `builder`.

## Detailed Phase Plan

## Phase 0: Documentation And Surface Definition

Goal:

- land the strategy and implementation plan before code.

Tasks:

1. Add this plan and the clean-room strategy doc to `neo-fold-prototype/specs/`.
2. Keep the docs frontend-oriented, not legacy-API-oriented.
3. State explicitly that phase 1 proves stack semantics and selected ALU ops
   only.

Exit condition:

- reviewers can point to one obvious intended ownership structure for WASM.

## Phase 1: Frontend Skeleton

Goal:

- create a compilable WASM frontend skeleton inside `neo-fold-prototype`.

Tasks:

1. Add `src/wasm/mod.rs`.
2. Add `src/wasm/spec.rs`.
3. Wire `pub mod wasm;` into `crates/neo-fold-prototype/src/lib.rs`.
4. Re-export:
   - `RwasmVmSpec`,
   - `RwasmTraceBuilder`,
   - frontend-local trace/build error types.

Design constraint:

- keep the barrel thin like CHIP-8.

Exit condition:

- `neo-fold-prototype` exposes a placeholder WASM frontend namespace with no proving
  logic yet.

## Phase 2: Opcode And Trace Taxonomy

Goal:

- rebuild the historical metadata layer under frontend ownership.

Tasks:

1. In `isa.rs`, define:
   - `RwasmOpcodeId` or reuse the concrete rWASM opcode code type,
   - `WasmOpcodeClass`,
   - `WasmShoutOpcode`,
   - `WasmOpcodeInfo`.
2. Preserve stable `WasmShoutOpcode -> ShoutId` numbering from the old branch.
3. Expose:
   - stack read count,
   - stack write count,
   - whether the row is direct arithmetic or auxiliary-routed,
   - whether the row touches observed linear memory or tables.
4. Define the explicit **phase-1 supported subset** in one obvious function or
   constant set.

Trade-off:

- The metadata table may cover more opcodes than phase 1 proves.
- That is acceptable as long as the supported subset is explicit.

Exit condition:

- frontend code can classify any tracer row and reject unsupported rows
  deterministically.

## Phase 3: Frontend-Local Trace Model

Goal:

- define the normalized row representation that the builder consumes.

Tasks:

1. In `execute.rs`, define `RwasmStepTrace` with at least:
   - cycle,
   - `pc_before`,
   - `pc_after`,
   - opcode id,
   - opcode metadata snapshot or recoverable key,
   - `sp_before`,
   - `sp_after`,
   - stack lane values,
   - optional observed linear-memory changes,
   - optional observed table changes,
   - optional lookup payload for auxiliary-routed ops,
   - halted flag.
2. Keep this frontend-local rather than forcing direct use of
   `neo-vm-trace::StepTrace`.
3. Implement translation from `wasm::Tracer` to `Vec<RwasmStepTrace>`.

Why not use `neo-vm-trace` directly here:

- `neo-fold-prototype` does not require it at the frontend seam,
- the builder only needs enough structured data to produce `StepBuild`,
- frontend-local trace records keep the new design independent from legacy
  shared-bus API assumptions.

Exit condition:

- given a tracer, the frontend can produce deterministic normalized rows with
  the historical stack-pointer and lane semantics.

## Phase 4: Core Layout

Goal:

- define the fixed-width row shape for the WASM main lane.

Tasks:

1. In `layout.rs`, define:
   - public prefix length,
   - witness width,
   - fixed `ONE` column,
   - opcode / PC / SP columns,
   - stack lane selector/address/value columns,
   - lookup metadata columns,
   - helper columns for direct ops.
2. Preserve the historical three-lane stack convention explicitly in comments.
3. Do **not** blindly copy the historical numeric column indices. Reassign them
   cleanly if needed, but preserve the semantic fields.
4. Keep witness width narrow enough that phase 1 remains small.

Recommended policy:

- prefer a fresh contiguous layout rather than carrying historical sparse
  numbering.

Exit condition:

- one file owns the canonical row shape for the WASM frontend.

## Phase 5: Core CCS Spec

Goal:

- make WASM a real `VmSpec`.

Tasks:

1. In `ccs.rs`, define `RwasmVmSpec { core: CoreCcsSpec, ... }`.
2. Implement `VmSpec`:
   - `name() -> "wasm"`,
   - `state_spec()`,
   - `shout_tables()`,
   - `twist_tables()`,
   - `opcode_classes()`,
   - `decode_spec()`,
   - `core_ccs_spec()`.
3. Build the core CCS using `vm::r1cs_builder::R1csBuilder`, like CHIP-8.
4. Phase 1 direct obligations should include:
   - selector booleanness,
   - stack-pointer update,
   - non-branch PC update,
   - stack lane occupancy consistency,
   - direct arithmetic relations for supported direct ops.
5. For auxiliary-routed ops, the row-local lane should only enforce the local
   metadata consistency needed for phase 1.

Important design decision:

- Phase 1 should **not** attempt to encode all packed lookup soundness in the
  main CCS.
- Keep the main lane row-local and narrow, following the CHIP-8 and RV64 kernel
  design principle that non-local obligations belong outside the main lane.

Exit condition:

- `RwasmVmSpec::new()` produces a stable `CoreCcsSpec`.

## Phase 6: Step Builder

Goal:

- package normalized rows into `StepBuild`.

Tasks:

1. In `builder.rs`, define `RwasmTraceBuilder<'a, L> { log: &'a L }`.
2. Follow the `Chip8TraceBuilder` pattern:
   - build the frontend-local row witness vector,
   - pack row-major into `Mat<F>`,
   - split public prefix vs witness suffix,
   - commit the row matrix,
   - emit `StepInput`,
   - wrap as `StepBuild`.
3. Define a label format that is stable and trace-friendly, for example:
   - `wasm@pc:<pc>:op:<opcode>`
4. Populate `extension_data` conservatively.

Recommended phase-1 extension mapping:

- `bytecode_fetch`:
  - use when the row has a meaningful fetch address / opcode code pair.
- `register_reads` / `register_writes`:
  - leave empty in phase 1 unless a concrete register-bank story exists.
- `ram_reads` / `ram_writes`:
  - use only if representing stack memory through this generic shape is more
    helpful than harmful.

Important point:

- CHIP-8’s `StepExtensionData` is optimized for CHIP-8 audit data.
- WASM should not contort itself to overuse these fields if the semantics do
  not fit. Keep them shallow in phase 1.

Exit condition:

- given `Vec<RwasmStepTrace>`, the frontend can produce `Vec<StepBuild>`.

## Phase 7: Public API And Trace Barrel

Goal:

- expose a frontend surface similar in quality to CHIP-8.

Tasks:

1. In `trace.rs`, re-export:
   - the builder,
   - the execution / normalization entrypoints,
   - the frontend-local build error.
2. In `spec.rs`, re-export:
   - `RwasmVmSpec`,
   - selected ISA/taxonomy items.
3. Keep both files thin.

Exit condition:

- external callers can discover WASM through one curated frontend surface.

## Phase 8: Phase-1 Tests

Goal:

- reproduce the historical proving envelope on the new spine.

Tasks:

1. Add tests under `crates/neo-fold-prototype/tests/`.
2. Create direct frontend tests for:
   - tracer normalization,
   - stack-pointer updates,
   - lane mapping,
   - supported-subset rejection.
3. Create prove/verify tests using:
   - `RwasmVmSpec`,
   - `RwasmTraceBuilder`,
   - `run::prove_run` / `run::verify_run`,
   - or packaged proof APIs where appropriate.
4. Cover historical examples:
   - `i32.const + i32.add`,
   - `i32.sub`,
   - `i32.popcnt`,
   - `i32.mul`,
   - `i32.and`,
   - `i32.or`,
   - `i32.xor`,
   - `i32.eq`,
   - `i32.ne`,
   - `i32.lt_s`,
   - `i32.lt_u`,
   - `i32.eqz`.

Testing constraint:

- phase 1 tests should validate the frontend and generic proof spine.
- they should not claim a full staged kernel exists.

Exit condition:

- the historical supported subset proves end to end on `neo-fold-prototype`.

## Phase 9: Auxiliary Lookup Design Decision

Goal:

- decide how to represent the historical packed lookup families in the new
  architecture without prematurely building a full staged kernel.

There are two viable phase-1 options.

### Option A: Frontend-local direct encoding only

Description:

- implement the historical supported subset entirely with row-local direct
  constraints in the core CCS.

Pros:

- simplest phase-1 integration with `neo-fold-prototype`,
- no new auxiliary protocol owner needed immediately.

Cons:

- loses the historical “reuse packed Route-A semantics” design,
- scales worse as more WASM ops are added,
- diverges from the intended long-term staged-kernel direction.

### Option B: Frontend-local placeholder auxiliary family

Description:

- keep row-local CCS narrow,
- define frontend-owned packed relation payload columns and proof-side metadata,
- but postpone full kernelization.

Pros:

- preserves the direct-vs-auxiliary split,
- aligns better with the CHIP-8 / RV64 philosophy,
- eases later migration into a staged kernel.

Cons:

- more design work in phase 1,
- some scaffolding may later move again.

Recommendation:

- choose Option B if the team is committed to a future WASM kernel,
- choose Option A only if the near-term goal is proving the historical subset as
  quickly as possible with minimal architecture work.

Given the stated goal of upstream maintenance on Nico’s converging architecture,
Option B is the better fit.

## Proposed Phase-1 Scope Boundary

Phase 1 should prove:

- stack pointer continuity inside each row,
- direct semantics for:
  - `i32.const`,
  - `i32.add`,
  - `i32.sub`,
  - `i32.popcnt`,
  - `select`,
  - `br_if_eqz`,
  - `return`,
- auxiliary-routed or placeholder-aux families for:
  - `i32.mul`,
  - `i32.and`,
  - `i32.or`,
  - `i32.xor`,
  - `i32.eqz`,
  - `i32.eq`,
  - `i32.ne`,
  - `i32.lt_s`,
  - `i32.lt_u`.

Phase 1 should **not** prove:

- full linear-memory writes,
- table writes,
- arbitrary unsupported control-flow forms,
- multi-step continuity,
- full staged opening/kernel artifacts.

## Integration With The Generic Proof Spine

The target proving flow should look like:

1. Build `RwasmVmSpec`.
2. Normalize tracer rows into `Vec<RwasmStepTrace>`.
3. Use `RwasmTraceBuilder` to produce `Vec<StepBuild>`.
4. Extract `prepared` fields into `Vec<StepInput>`.
5. Call:
   - `run::prove_run` / `run::verify_run`, or
   - `run::prove_and_package` / `run::verify_packaged`.

This matches the current generic `neo-fold-prototype` API and avoids routing phase 1
through the older `neo-fold` session/shared-bus entrypoints.

## Suggested Milestones

### Milestone 1: Skeleton

- `wasm/` subtree exists,
- `RwasmVmSpec` compiles,
- tracer normalization compiles,
- no proof tests yet.

### Milestone 2: Direct subset

- direct row-local ops prove end to end,
- no auxiliary lookup families yet.

### Milestone 3: Historical supported subset

- all historically covered ops prove end to end,
- labels, extension data, and public-step packaging are stable.

### Milestone 4: Kernel decision

- decide whether to:
  - stay frontend-only for now,
  - or begin a true staged `wasm/kernel/` subtree.

## Future Kernelization Path

If phase 1 succeeds and the project decides to grow WASM into a first-class
kernel, the natural next structure is:

```text
crates/neo-fold-prototype/src/wasm/
├── mod.rs
├── spec.rs
├── isa.rs
├── layout.rs
├── ccs.rs
├── tables.rs
├── execute.rs
├── lower.rs
├── builder.rs
├── trace.rs
├── stage1/
├── stage2/
├── stage3/
└── kernel/
```

But phase 1 should not create those directories prematurely.

## Concrete Initial Work Queue

A practical first coding sequence is:

1. Add `src/wasm/mod.rs`, `spec.rs`, `isa.rs`.
2. Add `layout.rs` with a fresh contiguous witness layout.
3. Add `ccs.rs` with `RwasmVmSpec` and only direct-op row-local constraints.
4. Add `execute.rs` with tracer normalization and stack-pointer replay.
5. Add `builder.rs` producing `StepBuild`.
6. Add one direct prove/verify test for `i32.const + i32.add`.
7. Add direct tests for `i32.sub`, `i32.popcnt`.
8. Add placeholder auxiliary-family interfaces in `tables.rs`.
9. Extend to the historical lookup-routed subset.
10. Only then evaluate whether a proper kernel subtree is justified.

## Summary

The right first implementation on `neo-fold-prototype` is not “port the old files”.

It is:

- build a new WASM frontend in the CHIP-8 style,
- keep the main lane row-local and narrow,
- preserve the historical trace and stack semantics,
- preserve the direct-vs-auxiliary split,
- and defer full kernelization until the frontend proves the historical subset
  cleanly through the new proof spine.
