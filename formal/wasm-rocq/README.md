# Wasm circuit proofs in Rocq

`run_refinement` (`ExecutionRefinement.v`) proves circuit traces against SpecTec's Wasm 2.0 semantics for
`i32.add/sub`, `local.get/set`, `nop`, `unreachable`, direct guest calls, `return` from any depth of
blocks, and blocks, loops and `if`s with no or one integer result with `br`, `br_if` and `br_table` to any enclosing
label: a branch to a block or an `if` leaves it, a branch to a loop runs it again. A `br` may also target the
function's own label, which is a `return`; a `br_if` or `br_table` to it is rejected by the Rust parser. Blocks, loops and `if`s nest to any
depth, calls nest to any depth, and any supported instruction, calls included, may appear inside them.
Conditions may be computed, with multiple branches per block; a taken branch
keeps its target block's results and discards every other operand above the
block's base, as the circuit's frame slide does; a `return` likewise keeps only
its function's results. Code after a
`return`, `br`, `br_table` or `unreachable` is dead: nothing falls through to it, so it needs no stack height.
`Fragment.v` lists the supported instructions, one constructor each with its
restrictions, and states the branch-site rule (`codeBranches`). Parameters, results, locals and stack values may mix i32 and i64. Arithmetic remains i32.add/sub. The theorem
takes plain rows, refines them one at a time, and relates circuit and SpecTec states through
`CircuitConfig` (`ControlDecode.v`): a decoder reads the configuration from memory
at a static control shape, a code path per frame through its enclosing blocks.
The operand height below each label follows from the code's static stack effects;
every function in the store must pass SpecTec validation. A run may stop anywhere
in its base frame, including inside blocks, inside calls, or during a callee's
frame initialization; the final shape resumes the next run. The store is an input:
the circuit holds only its tables (`TableState.v`: null as 0, function address a as a + 1),
which the fragment never writes. An activation may remain running, return a captured i32 or i64 result (or no result), or trap.
`ActivationConfig` wraps the running decoder in the base function's SpecTec frame;
`ExecutionConfig` binds the outcome to the last row. A terminating run stops on that row.
Padding before termination is supported. An explicitly empty `else` is rejected by the Rust parser (SpecTec's `if` has no such form; strip it before
proving). `call_indirect` is supported when every function in the program decodes as a guest of
the caller's module, with i32/i64 parameters, results and locals: its resolve row checks the table entry and leaves SpecTec at `CALL_ADDR`
(`siteResolved`), its invocation row enters the callee (`IndirectRows.v`), and a failed check traps
(`TrapRows.v`). Host rows, and blocks and loops with parameters or typed by index are excluded.
Selected zero-test steps are also proved. `TrapFacts.v` connects the supported opcode set to
trap accounting: only `unreachable` and an indirect call's resolve step trap in this fragment.
The decoder uses the declared types to read low limbs for i32 and both limbs for i64, including
pending moves and captured results. `tests/MixedWitness.v` checks wide values through branches,
nested calls, local initialization, indirect calls and terminal capture.
Field primality and memory-event consistency are assumed; the offline memory
argument is not proved. ROMs are dense: keys the encoder does not list read 0, so
proofs never rely on a key being absent. An instruction with k control choices (`if`,
`br_if`, `br_table`) takes k addresses: the pc ROM and the slide ROMs have one address column,
and hold the edge of choice c at the instruction's address plus c (`opEdge` slots in
`CodeLayout.v`). A call's return address is not in a ROM: the circuit pins it to the next address. The program ROM encoder covers this guest-only fragment;
its correspondence with Rust preprocessing is not yet proved. The `tests/*Witness.v`
files check exported Rust traces against the refinement proofs. The traces
(`tests/*Trace.v`) are not in git; `make` generates them with
`cargo test -p neo-wasm --release --test wasm_rocq_witness_export -- --ignored`.

## Build

`RomSharing.shared_rom_replay` is a separate bridge from shared ROM tables to
logical memory replay. It permits ROM-read reordering while preserving the
complete RAM event sequence, including prior values on read-modify-write events.
Its premises require read-only ROMs, corresponding shared reads, equal initial
RAM, and agreement of the two initial images at every active logical ROM read.
Its conclusion supplies a logical replay with unchanged logical ROM and the same
final RAM. Unused logical ROM cells need not equal the shared table's cells.

This theorem does not prove the Rust table mapping or its ownership and opcode
conditions. An application must establish `SharedTrace` and `ImagesAgree` for
that mapping; physical slot compaction remains outside this event-level bridge.
The instruction refinement theorem is unchanged. The bridge is included in
`make audit`; `tests/RomSharing.v` checks disjoint sharing, reordered ROM reads,
and rejection of a changed owned cell, reordered RAM, or an incorrect prior value.

`StackPorts.full_stack_port_gates` derives the separate-access and read-modify-write
gates from the exported circuit. The arithmetic and indirect-resolution stack
proofs obtain operand zero from the combined write's prior value. Frame moves,
initialization, and indirect invocation retain separate accesses. The gate theorem
is included in the audit; the supported instruction fragment is unchanged.

`ProgramBinding.full_host_context_refs` derives the saved export/import packing
equation and both 16-bit bounds from the exported circuit. It proves that the
packed word fits 32 bits and that remainder and division recover the two
references. The theorem is included in the audit; it does not add host execution
to the proved instruction fragment.

Requires the repository Rust toolchain (see `rust-toolchain.toml`), opam, and the
checked versions of Coq/Rocq 9.0.1, MathComp ssreflect 2.4.0, and RecordUpdate 0.3.6:

```sh
opam switch create nightstream-rocq 5.3.0
eval "$(opam env --switch=nightstream-rocq --set-switch)"
opam repository add rocq-released https://rocq-prover.org/opam/released
opam install coq.9.0.1 coq-mathcomp-ssreflect.2.4.0 coq-record-update.0.3.6
cd formal/wasm-rocq
make
```

For an existing switch, install those packages there and run `make` with that
switch active. `make` and `make audit` first check all four Rust-generated exports
without rewriting them; `make check-exports` runs that check alone. Regenerate stale
files with the command below, then rebuild. `make` also generates the witness traces
and rewrites only those that changed; `make traces` runs that step alone. SpecTec is not needed to build the
checked-in `Wasm2.v`.

Run `make audit` (requires Python 3) to build and print `run_refinement`
and its assumptions. It fails if axiom names differ from `scripts/audit.expected`
or the report cannot be parsed. Review changes before editing that file; changes
to existing axioms' types are not detected.

## Update inputs

From the Nightstream repository root, regenerate the circuit export with:

```sh
cargo run --release -p neo-wasm --bin export_wasm_rocq
```

`Wasm2.v` is copied verbatim from
[`spectec/test-rocq/theories/wasm.v`](https://github.com/Wasm-DSL/spectec/blob/95c256c2c/spectec/test-rocq/theories/wasm.v)
at commit `95c256c2c` (`rocq-backend-proof-final`). It includes upstream proofs
in addition to generated definitions. Upstream lists `spectec/` under
[Apache-2.0](https://github.com/Wasm-DSL/spectec/blob/95c256c2c/LICENSE).
To update it from a local SpecTec clone, run from the Nightstream root:

```sh
git -C /path/to/spectec show 95c256c2c:spectec/test-rocq/theories/wasm.v > formal/wasm-rocq/Wasm2.v
```
