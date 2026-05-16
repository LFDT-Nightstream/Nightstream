# Rv64IMExactNarrowMemoryHelperResultSemantics Spec

## Purpose

- **What it is**: The exact execution-level owner for the Stage-1 helper-result
  bridge used by RV64IM narrow loads and narrow stores.
- **What it is not**: It is not the owner of Stage-1 helper arithmetic itself,
  and it does not replace Stage-2 aligned-word RAM authentication.
- **Protocol role**: It packages the theorem-facing bridge from authenticated
  aligned-word inputs to `executionRow.results.aluResult` through
  `extractExtend` and `blend`.

## Central Object

`ExactNarrowMemoryHelperResultSemantics(pkg, families)` packages:

- the exact load-side binding from the helper package to authenticated state:
  `addr = MEM_ADDR`,
  `word = rvRamWord`,
  `out = ALU_RESULT`,
  `unsigned = memUnsigned_dec`,
- the exact store-side binding from the helper package to authenticated state:
  `addr = MEM_ADDR`,
  `word = rvRamWord`,
  `src = rvRs2`,
  `out = ALU_RESULT`.

## Required Constructors

- `exactNarrowMemoryHelperResultSemantics_of_exactOpcodeFamilySemantics`
- `exactNarrowMemoryHelperResultSemantics_of_stepComposition`

## Required Derived Theorems

- `loadExtractHelperResult_of_exactNarrowMemoryHelperResultSemantics`
- `storeBlendHelperResult_of_exactNarrowMemoryHelperResultSemantics`

These theorems must expose:

- `wordToNat(ALU_RESULT) = extractExtend(rawAlignedWord, off, width, memUnsigned)`
  on narrow loads,
- `wordToNat(ALU_RESULT) = blend(rawAlignedWord, rs2Word, off, width)`
  on narrow stores.

## Proof Obligations

- The helper-result owner is the first theorem-facing layer where the exact
  Stage-1 helper formulas are tied to authenticated row state.
- The owner must not silently replace `rvRamWord` or `rvRs2` with unrelated
  witnesses.
- The owner may consume helper proof packages from `Rv64IMStepComposition`,
  but it may not weaken them into boundary-only existence claims.

## Paper Anchors

- **Source**: `./crates/neo-fold-prototype/specs/riscv-kernel.md`
- Anchors:
  - `VExtractLoad`
  - `VBlendStore`
  - narrow-memory helper arithmetic
  - `LB / LBU / LH / LHU / LW / LWU`
  - `SB / SH / SW`

## Module Mapping

| Lean file | Local owner |
|---|---|
| `Nightstream/Rv64IM/Execution/ExactNarrowMemoryHelperResultSemantics.lean` | Exact narrow-memory helper-result bridge |
| `Nightstream/Rv64IM/Execution/ExactNarrowMemoryHelperResultSemanticsInterface.lean` | Theorem-facing re-export surface |
