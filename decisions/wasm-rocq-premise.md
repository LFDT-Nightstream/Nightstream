# WASM Premise of the Memory Claims

**Status:** Proposed

## Problem

The memory claims `Claims.Nebula.memory_sound`, `Claims.Nebula.memory_complete`
and `Claims.Nebula.memory_light_complete` take a memory application
(`Lifecycle.Nebula.PortApp`). Their premise row `app.*` in
`formal/nightstream-fprime/SECURITY_MODEL.md` is `open`. No Lean instance
exists for an application that Rust builds, and the WASM relation is such an
application.

Rocq proves `run_refinement` (`formal/wasm-rocq/ExecutionRefinement.v`). Rows
that satisfy `validCircuitRun` refine the Wasm 2.0 steps of SpecTec for the
fragment of `Fragment.v`. `validCircuitRun` has four parts: each row satisfies
the relation, the rows chain, the events replay, and each event has 32-bit
values. Lean gives the last two parts from a Nebula execution of the WASM
application of an accepted binding export
(`Export.WasmBinding.runMemory_logical_of_stepBinds`, `SPEC.md` section 7.6,
WASM binding export). The owner must select the class of the WASM case of
`app.*`.

## SuperNeo

SuperNeo v1.2 does not fix the application. It folds the relation that the
application gives. The WASM relation, its memory catalog and its Rocq proof
are project choices. Nebula fixes the memory argument and leaves the
application open.

## Decision

This record is a proposal. It does not change the premise table.

1. **Statement.** The WASM case of `app.*` is
   `Export.WasmBinding.WasmModel`: each execution of the model of the WASM
   application from the program images has step witnesses whose trace rows
   satisfy the first two parts of `validCircuitRun`, and whose slots obey
   rules 1 to 4 of `SPEC.md` section 7.6 at the bound ports of the export of
   its size (`Export.WasmBinding.StepBinds`).
   `Export.WasmBinding.circuitRun_of_wasmModel` proves that this statement and
   an accepted export give `validCircuitRun`, with the first two parts as a
   parameter. `Export.WasmBinding.circuitRun_of_attests` gives the same from a
   run that attests a statement, which `Claims.Nebula.memory_sound` gives
   outside the event that it bounds. The Rocq proof then applies to the trace
   rows. Lean does not
   tie the export to `app`: the pinned digests name the export of each
   size, and the premise names that export.
2. **Class.** Give the WASM case of `app.*` the class `assumption`, with one
   new assumption in section Assumptions of `SECURITY_MODEL.md`. The
   assumption has three parts:
   - Rocq `run_refinement` holds for the relation that Rust exports, with the
     axioms of `formal/wasm-rocq/scripts/audit.expected`, for the fragment of
     `Fragment.v`.
   - The rows of the Rust WASM application give
     `Export.WasmBinding.WasmModel`: the relation rows give the first two
     parts of `validCircuitRun`, and the rows of the catalog binding give
     `Export.WasmBinding.StepBinds` over the field.
   - The program images of a preload give the Rocq `programLoaded` of the
     program. No proof covers each program. Rocq `tests/ProgramRom.v` checks
     it on a corpus of modules inside the fragment.
3. **Why not `link`.** A `link` row needs data that Lean emits and a Rust
   test that compares it. Here the binding data goes from Rust to Lean, and
   `Export.WasmBinding.check_sound` checks it. No Lean or Rocq result
   connects the Rust relation rows, or the Rust rows of rules 1 to 4, to the
   statement. A `link` class would claim more than the evidence.
4. **Why not `lean`.** A `lean` row needs a Lean proof at the production
   parameters. The owner closes Lean–Rust gaps with emitted data,
   cross-checks and named assumptions, not with a Lean semantics of the
   Rust rows.
5. **Split, optional.** If the owner later wants the checked part apart, the
   binding part of `Export.WasmBinding.StepBinds` (catalog, regions, slot
   plan and columns) can become its own row with the evidence of
   `Export.WasmBinding.check_sound`, the exports of the four sizes and their
   pinned digests.
