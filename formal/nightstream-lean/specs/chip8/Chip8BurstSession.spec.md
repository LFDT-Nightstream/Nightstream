# Chip8BurstSession Spec

## Purpose

- **What it is**: The theorem-facing session contract for the exact decomposed
  CHIP-8 families in the final kernel, namely `StoreRegs` (`Fx55`) and
  `LoadRegs` (`Fx65`).
- **Key property**: `instructionCorrect_of_burstSession`: if a burst session is
  correctly scheduled, anchored, authenticated, and continuity-aware, then the
  whole macro instruction is semantically correct.
- **Protocol role**: This is the layer that closes decomposed-instruction
  soundness using authenticated execution frames rather than a weak parallel
  list of descriptors and states.

## Target Formulas

### Burst session objects

For a macro decoded instruction `dec`, define a burst session as a finite
sequence of authenticated execution frames:

$$
\mathrm{BurstSession}(frames).
$$

Each frame carries:

- the microstep-local decoded row
- the semantic pre-state
- the semantic post-state
- the authenticated 24-coordinate lane row

The session models the exact decomposition of one macro CHIP-8 instruction into
its chunk-local memory-prefix microsteps.

### Anchoring and chaining

The burst session must be anchored to the macro pre/post states:

$$
\mathrm{BurstAnchored}(dec, pre, post, frames)
$$

meaning:

- the first frame pre-state is the macro pre-state
- the frame at local cursor `x` has post-state equal to the macro post-state

The session must also be chained:

$$
\mathrm{BurstChained}(frames)
\iff
\forall i < n-1,\; frames_i.post = frames_{i+1}.pre.
$$

### Exact schedule derivation

For decomposed families the frame descriptors must be derived from the macro
decoded instruction and the exact final-kernel mem-op handoff:

$$
\mathrm{BurstDerivedFrom}(dec, frames)
$$

meaning:

- every frame descriptor `frames_i.dec` has the same authenticated decoded core
  as `dec`
- `dec.family ∈ \{StoreRegs, LoadRegs\}`
- the microstep cursor is exactly the local list position `i`
- the microstep-local RAM address is exactly `frames_i.pre.I + i`
- the microstep-local mem-op handoff is exact (`IsMemOp = 1`,
  `X_BOUND = dec.x`)
- the final covered frame therefore has `BURST_LAST = 1`

### Coverage and cursor progression

For `Fx55` / `Fx65`-style prefix instructions:

$$
\mathrm{BurstCoversPrefix}(dec, frames)
$$

means the session covers exactly the intended cursor range `0..x`, neither
omitting nor duplicating microsteps.

$$
\mathrm{BurstCursorMonotone}(frames)
$$

means the cursor progresses exactly one step at a time.

### Frame conditions

The session must make explicit which parts of machine state are touched and
which are preserved:

$$
\mathrm{BurstFrameCorrect}(dec, pre, post)
$$

meaning:

- for `StoreRegs`, the RAM prefix `I..I+x` is written from the matching
  register prefix and the register file is otherwise preserved
- for `LoadRegs`, the register prefix `0..x` is written from the matching RAM
  prefix and RAM is otherwise preserved
- `I` and all unsupported machine components are preserved exactly as required
  by the current final kernel

### Authenticated frames and chunk-local continuity

The final kernel is chunk-local and exports row continuity through Stage 3.
The burst layer therefore assumes an imported authenticated frame-trace
boundary rather than proving cross-chunk linkage internally:

$$
\mathrm{BurstFramesBound}(rom,\sigma,frames)
$$

and

$$
\mathrm{BurstContinuityBound}(frames).
$$

This means:

- every frame in the session is authenticated and semantically row-correct
- adjacent microsteps in the same chunk satisfy the internal linking law
- chunk boundaries are linked only through authenticated Stage-3 continuity and
  row-binding claims
- the burst theorem never infers hidden future rows from sparse/random openings

### Whole-instruction theorem

Define:

$$
\mathrm{BurstScheduleCorrect}(rom,\sigma,dec,pre,post,frames)
$$

to mean:

- `BurstSession(frames)`
- `BurstAnchored(dec, pre, post, frames)`
- `BurstChained(frames)`
- `BurstDerivedFrom(dec, frames)`
- `BurstCoversPrefix(dec, frames)`
- `BurstCursorMonotone(frames)`
- `BurstFrameCorrect(dec, pre, post)`
- `BurstFramesBound(rom,\sigma,frames)`
- `BurstContinuityBound(frames)`

Then the target theorem is:

$$
\mathrm{BurstScheduleCorrect}(rom,\sigma,dec,pre,post,frames)
\Longrightarrow
\mathrm{InstructionCorrect}(rom,\sigma,dec,pre,post).
$$

## Paper Anchors

- **Sources**:
  - `./crates/neo-fold-prototype/specs/chip8-kernel.md`
  - `./docs/soundness-specs/twist-and-shout-requirements.md`
- Anchors:
  - decomposed instruction schedule for memory-prefix families
  - exact address/cursor binding for CHIP-8 burst instructions
  - chunk-local continuity exported through Stage 3

## Module Mapping

| Lean file | Local owner |
|---|---|
| `Nightstream/Chip8/BurstSession.lean` | Burst-session correctness theorems for decomposed CHIP-8 instructions |
| `Nightstream/Chip8/BurstSessionInterface.lean` | Theorem-facing re-export surface |

## Contract Surface

| Group | Lean surface | Kind | Role | Guarantee |
|---|---|---|---|---|
| Sessions | `BurstSession` | def | Definitional | Packages one authenticated burst-trace session |
| Sessions | `BurstAnchored` | def | Definitional | Connects macro pre/post state to the authenticated session endpoints |
| Sessions | `BurstChained` | def | Definitional | Enforces post-to-next-pre chaining |
| Sessions | `BurstDerivedFrom` | def | Definitional | Derives each frame descriptor from the macro decoded instruction |
| Sessions | `BurstCoversPrefix` | def | Definitional | Enforces exact microstep coverage |
| Sessions | `BurstCursorMonotone` | def | Definitional | Enforces exact cursor progression |
| Sessions | `BurstFrameCorrect` | def | Definitional | Enforces touched-address correctness and untouched-state preservation |
| Sessions | `BurstFramesBound` | def | Definitional | Every frame carries authenticated row-local semantic correctness |
| Sessions | `BurstContinuityBound` | def | Definitional | Imports the exact chunk-local continuity boundary used by the final kernel |
| Sessions | `BurstScheduleCorrect` | def | Definitional | Complete authenticated burst-session correctness condition |
| Theorem | `instructionCorrect_of_burstSession` | theorem | Theorem-Target | Correct authenticated burst session implies whole-instruction correctness |

## Proof Obligations

- `BurstScheduleCorrect` must be stronger than a bare list-equality or
  membership predicate.
- The theorem surface must make the state chaining explicit.
- The theorem surface must make the cursor/address derivation explicit.
- The theorem surface must make the authenticated frame evidence explicit.
- The theorem surface must respect the final kernel's chunk-local continuity
  boundary instead of implicitly assuming one global contiguous trace witness.

## Assumption Ledger

- `MicrostepCorrect`, `InstructionCorrect`, and `ExecutionFrameBound` are
  imported from `Chip8ExecutionSemantics`.
- The exact register/address cursor semantics are imported from the decode and
  address-binding layers.
- Continuity across chunk boundaries is imported from the Stage-3
  continuity/bridge layer.

## Dependency and Consumer Map

- **Upstream dependencies**:
  - `Nightstream/Chip8/ExecutionSemantics.lean`
  - `Nightstream/Chip8/DecodeAddressBinding.lean`
  - `Nightstream/Chip8/ContinuityBridge.lean`
- **Downstream consumers**:
  - later ROM-specific whole-instruction and execution theorems

## Implementation Plan

1. Define the authenticated burst session predicates over execution frames.
2. State the exact strengthened `BurstScheduleCorrect`.
3. Prove `instructionCorrect_of_burstSession`.

## Quality Expectations

- Keep the burst layer explicit about chaining, authentication, and framing.
- Avoid weak schedule predicates that could overclaim instruction correctness.
- Keep the theorem exact for the supported `StoreRegs` / `LoadRegs` families.

## Acceptance Criteria

1. `lake build Nightstream.Chip8.BurstSession` succeeds.
2. The theorem surface explicitly closes the decomposed-instruction schedule
   hole for the selected CHIP-8 families.
3. No `sorry`.

## Out of Scope

- re-proving local microstep correctness
- proving authenticated claim coverage
- proving PCS binding or Fiat-Shamir security
- extending the theorem to unsupported CHIP-8 families such as `DRW`, stack
  operations, timers, or keypad-dependent rows
