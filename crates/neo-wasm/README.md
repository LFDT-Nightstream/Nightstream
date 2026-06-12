# neo-wasm

WASM VM frontend for the Neo/Nightstream folding stack: lowers wasmtime
execution traces into per-step witnesses, proves per-step semantics as
R1CS-derived CCS folded through `neo-fold-clean`'s `r1cs_f_prime` chain, and
exposes ROM/memory/lookup access tuples as metadata
(`WasmLookupBindingLayout`) for the proving system's memory/lookup arguments.

## Status and terminology

- The memory-consistency check (MCC) is owned by the underlying proving
  system (Nebula-style); this crate's job is to expose well-formed access
  tuples, not to prove memory consistency itself.
- The ALU **op tables** (`WasmOpTable`, `op_table_*` columns) are
  backend-agnostic metadata: each describes an opcode-table relation that
  the proving system must enforce with some lookup argument — most likely
  Nebula-based ROM lookups, since Shout is descoped from the
  proving-system implementation. The binding layer deliberately does not
  assume a backend, so it could be swapped (e.g., to Shout) without
  reshaping the frontend.

## TODO

### Soundness gaps (proof does not yet mean "this program ran")

- [ ] **Wire the lookup/ROM layer.** `lookup_semantics.rs` is a debug
  checker only. All ALU op-table families, the `linear_memory_bounds`
  family, and every `WasmLookupBindingSpec` binding must be proven by the
  proving system's ROM/lookup argument.
- [ ] **Bind the program to the proof.** `verify()` pins the canonical VM
  structure and initial-state digest, but the program ROM tables
  (`program_opcodes`, `pc_rom`, `call_targets`, …) are never committed or
  made public input. Today the proof means "*some* valid execution under
  this preprocessing". Needs a program commitment on the verifier surface,
  presumably via the ROM side of the memory/lookup argument.
- [ ] **Trap coverage.** Traps are modeled as a provable terminal state: a
  carried `trapped` flag enters the semantic-state digest, nothing executes
  after it, and it is mutually exclusive with a captured output (see
  `tests/wasm_trap.rs`). Proven causes so far: `unreachable`, div/rem by
  zero, signed division overflow (`min_value / -1`), and `call_indirect`
  null entry / callee type mismatch — zero-test gates on the faulting row
  feed `div_trap` / `ci_trap`, which de-gate the row's op-table lookup or
  callee-metadata reads. Other trapping executions fail loudly at trace
  time. Remaining causes:
  - [ ] OOB linear-memory access — land together with the
    `linear_memory_bounds` argument (same comparison, complementary sides).
  - [ ] `call_indirect` OOB entry (`index >= table size`) — needs the same
    comparison primitive as the bounds argument; land them together.
- [ ] **Linear-memory bounds proof.** The `linear_memory_bounds` binding is
  explicitly "unproven" (see the TODO in `lookup_binding_builder/mod.rs`);
  nothing constrains accesses against `memory_pages` yet. Revisit the
  lookup shape when wiring the real argument.
- [x] **Authoritative function-table init.** Active element segments and
  declared table sizes parse into the preload (`tables_init` /
  `table_sizes_init`); `FirstReadDefines` is removed (`tables` is
  zero-default for null entries, `table_sizes` strict). Imported tables,
  `table64`, non-`RefNull` table init exprs, and non-`i32.const` element
  offsets are rejected at parse time.
- [ ] **Host/imported function calls.** Imported callees produce no guest
  rows, so a host function's return value enters the operand stack
  unconstrained. Needs a host-I/O commitment (Nebula-style) or an enforced
  "no imports" restriction at parse time. Host calls also pop their args
  on-row (guest calls pop via param-init aux rows), so a host callee with
  more than 3 popped operands (args + indirect index) is unprovable under
  the 3-read-lane budget; the future component-ABI host model should move
  host arg pops to pop-only aux rows.
- [ ] **Production parameters.** `wasm_tiny_params()` uses λ = 40,
  `Params::test_only_from_neo_params`, and a fixed demo Ajtai seed.

### ISA / wasm-feature completeness

Floats are unsupported by policy: float ops decode to `Unsupported` and
`f32/f64.const` global initializers are rejected at parse time.


- [ ] Bulk memory and references: `memory.copy/fill/init`,
  `table.grow/copy/fill/init`, `ref.null/is_null`, typed `select`. Passive
  data/element segments are skipped, which is only correct while
  `memory.init` / `table.init` are unsupported.
- [ ] Module-level limits: single memory only, no memory64, data-segment
  offsets must be `i32.const`, components use only the first embedded core
  module, single-value outputs (no multi-value returns).
- [ ] Verify `memory.grow` failure semantics (the -1 path) and `start`
  function handling.

### Proof pipeline

- [ ] **Compression path.** `WasmProof` is `UncompressedAudit` only — no
  Spartan compression analog of the RV64IM flow, hence no succinct
  artifact.
- [ ] **Move `WasmtimeTraceRun::results` out of the public API.** It is the
  reference interpreter's output, used only as a test oracle (see
  `assert_output_matches_reference` in `tests/common`); leaving it public
  invites someone to trust it as the proven output. The proof-bound output
  is the final-state claim checked by `verify`.

### MCC (Nebula-style) integration readiness

`WasmMemorySpec` already exposes per-row access tuples in the right shape
(multi-limb address columns, value column, Read/Write kind, boolean
activation gates, `is_rom`, `value_before_column` on RMW writes), and
`preload_from_program_artifacts` provides the authoritative init multiset.
Remaining work:

- [ ] **IVC-state slots for MCC accumulators.** Cross-step links carry only
  scalar VM state into the semantic digest; an incremental MCC needs its
  running multiset accumulators carried in the folded state across
  steps/batches. The F' image plan needs room for those. Design this
  before the layout ossifies.
- [ ] **Intra-row access ordering.** No timestamp columns; row index gives
  inter-row order, but rows make several accesses to one memory (stack
  read0/1/2 + write0, three linear-memory lanes, reused stack columns under
  output-capture/param-init gates). Define the canonical intra-row order
  (spec column order is the natural choice) and have the MCC consume it.
- [ ] **Address-space packing.** Addresses are multi-limb and memories are
  distinguished by name only. Define the canonical packing of
  (memory id, limbs) into the MCC address domain, with disjointness
  between named memories.
- [ ] **Padding rows emit no accesses.** Padding must provably gate off all
  memory access tuples (it crosses batch boundaries); add a test/assert
  before the MCC lands.
- [ ] `value_before_column` contract: `Some(c)` requires the MCC to equate
  its read value with column `c`; `None` writes still need the MCC-internal
  read tuple (documented on `WasmMemoryColumnKind`).
