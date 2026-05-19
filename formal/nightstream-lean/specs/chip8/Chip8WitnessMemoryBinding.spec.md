# Chip8WitnessMemoryBinding Spec

## Purpose

- **What it is**: The theorem-facing Stage-2 binding contract from semantic
  CHIP-8 state to the final kernel's lane values, register/RAM Twist ports,
  memory-transfer values, RAF support equations, and initial-state anchors.
- **Key property**: `memoryBound_memValue_total`: when the local Stage-2 memory
  binding holds, the lane value `MEM_VALUE` is total and equals exactly the
  semantic value required by the authenticated row kind.
- **Protocol role**: This is the local owner for the CHIP-8-specific semantic
  meaning of Stage-2 register and RAM claims. It does not re-prove generic
  Twist soundness.

## Target Formulas

### Semantic state

The local Stage-2 layer reasons about semantic state objects:

- `pre`: machine state before the current row
- `post`: machine state after the current row
- `init`: authenticated initial machine state for the chunk

For the current kernel, the state carrier must expose at least:

- `pc_word`
- `i`
- `V[0..15]`
- `RAM[0..4095]`

### Lane row binding

This layer reasons over the final 24-coordinate semantic row, not the older
reduced routing row.

Define:

$$
\mathrm{WitnessBinds}(pre, post, dec, z)
$$

to mean that the lane row `z` carries:

- `PC = pre.pc_word`
- `PC_NEXT = post.pc_word`
- `REG_X = primaryValue(pre, dec)`
- `REG_Y = secondaryValue(pre, dec)`
- `REG_X_NEXT = primaryValue(post, dec)`
- `I_REG = pre.i`
- `I_NEXT = post.i`
- `KK = dec.kk`
- `NNN_ADDR = dec.nnnAddr`
- `NNN_WORD = dec.nnnWord`
- `MEM_VALUE = memValueOf(pre, post, dec)`
- `X_IDX = xIndexOf(dec)`
- `Y_IDX = yIndexOf(dec)`
- `BURST_LAST = burstLastOf(dec)`
- `RAM_ADDR = ramAddrOf(pre, dec)`

The control columns and `LOOKUP_OUTPUT` are imported from Stage 1; this module
owns the memory-derived lane values.

### Memory-transfer value

Define the exact semantic transfer value:

$$
\mathrm{memValueOf}(pre, post, dec)
:=
\begin{cases}
REG\_X(pre,dec) & \text{if } writesRam(dec) = 1, \\
REG\_X(post,dec) & \text{if } readsRam(dec) = 1, \\
0 & \text{otherwise.}
\end{cases}
$$

The Stage-2 owner must prove the lane totality rule:

- if `writesRam = 1`, then `MEM_VALUE = REG_X`
- if `readsRam = 1`, then `MEM_VALUE` is the authenticated RAM-read value
- otherwise `MEM_VALUE = 0`

### Register subsystem

The register-file domain is:

- slots `0..15` for `V[0]..V[15]`
- slot `16` for `I`
- slot `17` for `⊥_reg`

Define the Stage-2 register objects:

- `RegInc`
- `RegRaX`
- `RegRaY`
- `RegRaI`
- `RegWa`
- virtual `RegVal`

The CHIP-8-local register-port semantics are:

- `RegRaX` reads slot `X_IDX`
- `RegRaY` reads slot `Y_IDX` when `usesY = 1`, else `⊥_reg`
- `RegRaI` reads slot `16`
- `RegWa` writes:
  - slot `X_IDX` when `WritesLookupToX + WritesMemToX = 1`
  - slot `16` when `WritesNnnToI = 1`
  - slot `⊥_reg` otherwise

Sink semantics:

$$
\mathrm{RegVal}(\bot_{reg},0) = 0
$$

$$
\mathrm{RegVal}(\bot_{reg},j+1) = \mathrm{RegVal}(\bot_{reg},j)
$$

$$
\mathrm{RegInc}(j) = 0 \text{ whenever } RegWa \text{ points to } \bot_{reg}
$$

### RAM subsystem

The RAM domain is:

- slots `0..4095` for RAM
- slot `4096` for `⊥_ram`

Define the Stage-2 RAM objects:

- `RamInc`
- `RamRa`
- `RamWa`
- virtual `RamVal`

The CHIP-8-local RAM-port semantics are:

- `RamRa` points to `RAM_ADDR` when `readsRam = 1`, else `⊥_ram`
- `RamWa` points to `RAM_ADDR` when `writesRam = 1`, else `⊥_ram`

Sink semantics:

$$
\mathrm{RamVal}(\bot_{ram},0) = 0
$$

$$
\mathrm{RamVal}(\bot_{ram},j+1) = \mathrm{RamVal}(\bot_{ram},j)
$$

$$
\mathrm{RamInc}(j) = 0 \text{ whenever } RamWa \text{ points to } \bot_{ram}
$$

### Stage-2 lane linkage

Define:

$$
\mathrm{Stage2LaneLinkBound}(pre, post, dec, z)
$$

to mean the exact scalar equalities consumed by the register/RAM subclaims:

- `rv_x = REG_X`
- `rv_y = REG_Y`
- `rv_i = I_REG`
- `wv_reg = (WritesLookupToX + WritesMemToX) * REG_X_NEXT + WritesNnnToI * I_NEXT`
- `rv_ram = readsRam * MEM_VALUE`
- `wv_ram = writesRam * MEM_VALUE`

This is the theorem-facing content of the Stage-2 linkage batch; it is not a
free consequence of generic Twist theory.

### RAM RAF support

Define:

$$
\mathrm{RamRafBound}(dec, z)
$$

to mean the exact support equations:

$$
\Sigma_a ra\_read(a)\cdot unmap\_{chip8}(a) = readsRam \cdot RAM\_ADDR
$$

$$
\Sigma_a ra\_write(a)\cdot unmap\_{chip8}(a) = writesRam \cdot RAM\_ADDR
$$

where the active RAM domain is exactly `0..4095` and `⊥_ram = 4096`.

### Initial-state authentication

Define:

$$
\mathrm{InitialStateBound}(init)
$$

to mean:

- `RegVal(a,0) = init_reg[a]` for `a in {0..16}`
- `RegVal(17,0) = 0`
- `RamVal(a,0) = init_ram[a]` for `a in {0..4095}`
- `RamVal(4096,0) = 0`

This is the final kernel's chosen non-zero-initialization route. It is not a
synthetic preload-write encoding. The modified `init + Inc -> Val` identity for
that route is owned by `Nightstream/NonZeroInitTwist.lean`; this module imports
that bridge and specializes it to the CHIP-8 register and RAM surfaces.

The theorem-facing initialization mode is therefore fixed positively:

- `initialization_mode = authenticated_nonzero_init`
- the public / transcript metadata must carry the matching `init_mode_id`
- any proof surface that appeals to Stage-2 `Val` consequences is appealing to
  the authenticated non-zero-init identity, not to a zero-init theorem plus an
  implicit preload-write reduction.

### Memory bound

Define:

$$
\mathrm{MemoryBound}(pre, post, init, dec, z)
$$

to mean the conjunction of:

- `WitnessBinds(pre, post, dec, z)` for the memory-derived lane values
- register-port semantics
- RAM-port semantics
- `Stage2LaneLinkBound(pre, post, dec, z)`
- `RamRafBound(dec, z)`
- `InitialStateBound(init)`

This is the local theorem surface that later composition layers import as the
CHIP-8 meaning of the Stage-2 subsystem.

## Paper Anchors

- **Source**: `./crates/neo-fold-prototype/specs/chip8-kernel.md`
- Anchors:
  - register-file domain and ports
  - register-file lane linkage
  - RAM domain and ports
  - RAM lane linkage
  - RAM RAF support relation
  - initial-state authentication

## Module Mapping

| Lean file | Local owner |
|---|---|
| `Nightstream/Chip8/WitnessMemoryBinding.lean` | CHIP-8-local semantic meaning of Stage-2 lane values, Twist ports, RAF, and initialization |
| `Nightstream/Chip8/WitnessMemoryBindingInterface.lean` | Theorem-facing re-export surface |

## Contract Surface

| Group | Lean surface | Kind | Role | Guarantee |
|---|---|---|---|---|
| Semantic objects | `MachineState` | def | Definitional | CHIP-8 semantic state carrier for Stage 2 |
| Semantic objects | `InitialState` | def | Definitional | Authenticated chunk-initial state carrier |
| Binding | `WitnessBinds` | def | Definitional | Binds memory-derived lane values to semantic state and decoded row |
| Binding | `memValueOf` | def | Definitional | Exact semantic source of `MEM_VALUE` |
| Register | `RegisterPortBound` | def | Definitional | Exact CHIP-8-local meaning of `RegRaX`, `RegRaY`, `RegRaI`, and `RegWa` |
| RAM | `RamPortBound` | def | Definitional | Exact CHIP-8-local meaning of `RamRa` and `RamWa` |
| Linkage | `Stage2LaneLinkBound` | def | Definitional | Exact scalar linkage between lane columns and Stage-2 read/write claims |
| RAF | `RamRafBound` | def | Definitional | Exact CHIP-8 RAM RAF support equations |
| Init | `InitialStateBound` | def | Definitional | Exact initial register/RAM base case for the virtual `Val` surfaces |
| Binding | `MemoryBound` | def | Definitional | Complete CHIP-8-local Stage-2 semantic binding bundle |
| Theorem | `memoryBound_memValue_total` | theorem | Theorem-Target | `MEM_VALUE` obeys the exact totality rule |
| Theorem | `registerPorts_exact` | theorem | Theorem-Target | Register ports realize exactly the intended CHIP-8 reads/writes/sink rows |
| Theorem | `ramPorts_exact` | theorem | Theorem-Target | RAM ports realize exactly the intended CHIP-8 reads/writes/sink rows |
| Theorem | `ramRaf_tracks_laneAddress` | theorem | Theorem-Target | RAM RAF support ties the committed address family back to `RAM_ADDR` |
| Theorem | `initialStateBound_exact` | theorem | Theorem-Target | The Stage-2 base case is the authenticated public initialization surface |

## Proof Obligations

- This module must target the final 24-coordinate lane row, not the older
  reduced routing witness.
- Sink semantics for `⊥_reg` and `⊥_ram` must remain explicit.
- `MemoryBound` must expose the lane-linkage equalities and the RAF equations as
  theorem-facing objects.
- Initial-state authentication must follow the final kernel's non-zero-init
  route directly, not a preload-write reduction.

## Assumption Ledger

- This module does not re-prove generic Twist read/write soundness.
- The modified non-zero-init `Val` identity is imported from the Nightstream
  bridge owner rather than reproved here.
- This module does not re-prove generic address-correctness for the one-hot
  families.
- This module does not prove Stage-1 lookup correctness.
- The virtual `RegVal` / `RamVal` surfaces exported here must remain available
  as theorem-facing inputs to the downstream temporal-consistency owner; this
  module itself does not prove the cross-row temporal consequences.
- This module does not prove Stage-3 continuity or bridge binding.

## Dependency and Consumer Map

- **Upstream dependencies**:
  - `Nightstream/Chip8/FetchDecodeBinding.lean`
  - `Nightstream/Chip8/DecodeAddressBinding.lean`
  - `Nightstream/Chip8/Routing.lean`
- **Downstream consumers**:
  - `Nightstream/Chip8/EvidenceCoverage.lean`
  - `Nightstream/Chip8/StepComposition.lean`
  - later Rust-refinement theorems for Stage-2 proof objects

## Implementation Plan

1. Define the semantic lane-value and port objects.
2. Define `memValueOf`, register/RAM port semantics, RAF, and initialization.
3. Define the bundled `MemoryBound`.
4. Prove the exact totality, sink-routing, RAF, and initialization lemmas.

## Quality Expectations

- Keep Stage-2 ownership explicit and separate from generic Twist theorems.
- Use the exact sink-routing and initialization rules from the final kernel.
- Do not collapse Stage 2 into an opaque “memory passed” predicate.

## Acceptance Criteria

1. `lake build Nightstream.Chip8.WitnessMemoryBinding` succeeds.
2. The theorem surface matches the final Stage-2 register/RAM design.
3. `MEM_VALUE` totality, RAF support, and initialization are explicit.
4. No `sorry`.

## Out of Scope

- generic Twist theorem proofs
- Stage-1 Shout proofs
- Stage-3 continuity and bridge binding
