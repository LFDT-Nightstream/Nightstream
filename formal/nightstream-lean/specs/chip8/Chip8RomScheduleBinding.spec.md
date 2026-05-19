# Chip8KernelInputBinding Spec

## Purpose

- **What it is**: The theorem-facing authenticated public-input binding
  contract for the final CHIP-8 kernel.
- **Key property**: `kernelPublicInputsBound_of_authenticatedInputs`: the
  public metadata fixed in `meta_pub` determines the exact ROM table,
  program-shape metadata, padding row metadata, root-encoding parameters, and
  authenticated initial machine state used by the kernel.
- **Protocol role**: This is the layer that makes the kernel's public-input
  bundle theorem-facing instead of leaving it only inside Rust structs or
  transcript prose.

## Target Formulas

### Public kernel input bundle

The final kernel public boundary includes at least:

- `vm_spec`
- `public_program_image`
- `initial_state`
- `transcript_seed`
- the `meta_pub` fields absorbed into `root0`

The theorem-facing owner packages those into one authenticated public-input
object.

### Public metadata bindings

Let the public metadata package include at least:

- program image digest
- table digests
- field / extension identifiers
- initialization-mode identifier
- lowering / visibility-order identifier
- padding-convention identifier
- public-table-authentication identifier
- opening-reduction-mode identifier
- trace length `N`
- `program_word_count`
- `program_base_addr`
- `pad_pc_word`
- initial-state digest(s)
- root-encoding / root-protocol version identifiers

Define:

$$
\mathrm{ProgramDigestBound}(meta, romTable)
$$

$$
\mathrm{ProgramShapeBound}(meta, romTable)
$$

$$
\mathrm{PadRowMetadataBound}(meta)
$$

$$
\mathrm{InitialStateDigestBound}(meta, init)
$$

$$
\mathrm{RootParamsBound}(meta, vmSpec)
$$

These express, respectively, that:

- the public program digest matches the exact committed absolute ROM table
- `program_word_count` and `program_base_addr` match the loaded absolute ROM
  interval
- `pad_pc_word` is the exact public self-loop padding address
- the public initial-state digest(s) match the exact authenticated initial
  register/RAM state
- `vm_spec` fixes the canonical root witness-encoding parameters used by
  `RootEncode`
- the relation-shaping metadata ids are fixed exactly before `root0` and are
  not hidden prover-chosen conventions

### Authenticated public inputs

Define:

$$
\mathrm{AuthenticatedProgramImage}(publicInput, romTable)
$$

$$
\mathrm{AuthenticatedInitialState}(publicInput, init)
$$

$$
\mathrm{AuthenticatedKernelMeta}(publicInput, meta)
$$

to package the exact authenticated witnesses used to derive the bound
relations.

### Bundled public-input theorem

Define:

$$
\mathrm{KernelPublicInputsBound}(publicInput, meta, romTable, init)
$$

to mean the conjunction of:

- `ProgramDigestBound(meta, romTable)`
- `ProgramShapeBound(meta, romTable)`
- `PadRowMetadataBound(meta)`
- `InitialStateDigestBound(meta, init)`
- `RootParamsBound(meta, publicInput.vmSpec)`

The target theorem is:

$$
\mathrm{AuthenticatedProgramImage}
\land
\mathrm{AuthenticatedInitialState}
\land
\mathrm{AuthenticatedKernelMeta}
\Longrightarrow
\mathrm{KernelPublicInputsBound}.
$$

### Transport theorems

If the relevant digest/equality functions are injective, then shared public
metadata determines a unique ROM table or initial state.

These transport theorems let later Stage-1, Stage-2, and Stage-3 modules move
facts across a fixed public input bundle without re-proving the public bindings.

## Paper Anchors

- **Source**: `./crates/neo-fold-prototype/specs/chip8-kernel.md`
- Anchors:
  - commitment bundle / `meta_pub`
  - fixed hypercube domains and absolute ROM addressing
  - public pad-row rule
  - Rust-facing kernel boundary
  - prepared-step construction / canonical root parameters

## Module Mapping

| Lean file | Local owner |
|---|---|
| `Nightstream/Chip8/RomScheduleBinding.lean` | Authenticated public-input binding theorems for the final CHIP-8 kernel |
| `Nightstream/Chip8/RomScheduleBindingInterface.lean` | Theorem-facing re-export surface |

## Contract Surface

| Group | Lean surface | Kind | Role | Guarantee |
|---|---|---|---|---|
| Public inputs | `KernelPublicInput` | def | Definitional | The theorem-facing public input bundle of the final kernel |
| Bindings | `ProgramDigestBound` | def | Definitional | Public metadata matches the exact committed ROM table |
| Bindings | `ProgramShapeBound` | def | Definitional | `program_word_count` and `program_base_addr` match the actual absolute ROM layout |
| Bindings | `PadRowMetadataBound` | def | Definitional | `pad_pc_word` and the public pad-row metadata are fixed exactly |
| Bindings | `InitialStateDigestBound` | def | Definitional | Public initial-state digest(s) match the exact initial register/RAM state |
| Bindings | `RootParamsBound` | def | Definitional | `vm_spec` fixes the canonical root encoding parameters |
| Evidence | `AuthenticatedProgramImage` | def | Definitional | Explicit authenticated ROM-table witness |
| Evidence | `AuthenticatedInitialState` | def | Definitional | Explicit authenticated initial-state witness |
| Evidence | `AuthenticatedKernelMeta` | def | Definitional | Explicit authenticated `meta_pub` witness |
| Bundle | `KernelPublicInputsBound` | def | Definitional | Exact bundled theorem surface for the public kernel inputs |
| Theorem | `kernelPublicInputsBound_of_authenticatedInputs` | theorem | Theorem-Target | Authenticated public inputs imply the bundled public-input bounds |
| Theorem | `romTable_eq_of_sharedMeta` | theorem | Theorem-Target | Shared authenticated public metadata determines one exact ROM table |
| Theorem | `initialState_eq_of_sharedMeta` | theorem | Theorem-Target | Shared authenticated public metadata determines one exact initial state |

## Proof Obligations

- The public-input layer must make `meta_pub` theorem-facing.
- The exact pad-row metadata and initial-state digest bindings must be explicit.
- The root witness-encoding parameters must be fixed by the public boundary, not
  left as hidden prover-chosen values.
- The exact `meta_pub` field order absorbed into `root0` must be fixed by the
  public boundary, not left as an implementation-defined struct serialization.
- The exact relation-shaping metadata ids fixed by `meta_pub` must be explicit
  at the public boundary, not left implicit in implementation convention.
- Shared-public-input transport theorems must require only the exact needed
  injectivity hypotheses.

## Assumption Ledger

- Poseidon2 security and collision resistance are external to this module.
- This module does not prove Stage-1, Stage-2, or Stage-3 semantic facts.
- This module binds public objects to semantic ones; it does not re-prove the
  PCS commitments themselves.

## Dependency and Consumer Map

- **Upstream dependencies**:
  - final kernel public boundary from `chip8-kernel.md`
- **Downstream consumers**:
  - `Nightstream/Chip8/FetchDecodeBinding.lean`
  - `Nightstream/Chip8/WitnessMemoryBinding.lean`
  - `Nightstream/Chip8/ContinuityBridge.lean`
  - `Nightstream/Chip8/EvidenceCoverage.lean`

## Implementation Plan

1. Define the theorem-facing public input and metadata objects.
2. Define the exact bound relations for ROM shape, padding, initialization, and
   root parameters.
3. Define the authenticated witness structures.
4. Prove the bundled public-input theorem and the transport lemmas.

## Quality Expectations

- Keep the module small and ownership-specific.
- Make `meta_pub` explicit.
- Separate authenticated public inputs from later semantic extraction.

## Acceptance Criteria

1. `lake build Nightstream.Chip8.RomScheduleBinding` succeeds.
2. The theorem surface makes `meta_pub`, pad-row metadata, and initial-state
   authentication explicit.
3. No `sorry`.

## Out of Scope

- Stage-1 proofs
- Stage-2 proofs
- Stage-3 proofs
- Poseidon2 security proofs
