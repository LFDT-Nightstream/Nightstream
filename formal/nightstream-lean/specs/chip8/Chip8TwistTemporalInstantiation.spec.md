# Chip8TwistTemporalInstantiation Spec

## Purpose

- **What it is**: The theorem-facing Stage-2 temporal-support owner for one
  CHIP-8 execution trace.
- **Key property**: It packages the concrete register / `I` and RAM timeline
  witnesses required by the strengthened kernel proof and shows how they arise
  from one chunk-global authenticated Stage-2 temporal context. Once the
  explicit Stage-3 `pc` bridge is supplied, that context produces the generic
  temporal instantiation consumed by `Chip8TemporalConsistency`.
- **Protocol role**: This is the Stage-2 half of strong adjacent-state linking.
  It owns the concrete timeline bundle, not cryptographic authentication of the
  underlying Twist claims.

## Target Formulas

### Stage-2 temporal bundle

Define:

$$
\mathrm{Stage2TemporalContext}(trace)
$$

to package:

- one shared register / `I` timeline `RegVal` over the whole semantic prefix
- one shared RAM timeline `RamVal` over the whole semantic prefix
- one row-indexed provenance map showing that each row-local Stage-2 temporal
  seed reads from and writes back to those same shared timelines

Its proposition form is:

$$
\mathrm{Stage2TemporalContextBound}(trace).
$$

This is the exact Stage-2 closure object that must be recoverable from
authenticated Stage-2 evidence before any whole-trace link theorem can be
claimed.

Classification:

- `Stage2TemporalContext` / `Stage2TemporalContextBound` are theorem-level
  kernel closure objects;
- they are not direct opening claims, not new commitment families, and not
  audit/provenance summaries.

Define:

$$
\mathrm{Stage2TemporalInstantiation}(trace)
$$

to package:

- one register / `I` timeline witness `RegVal`,
- one RAM timeline witness `RamVal`,
- `RegisterTemporalBound(RegVal, trace)`,
- `RamTemporalBound(RamVal, trace)`.

Its proposition form is:

$$
\mathrm{Stage2TemporalInstantiationBound}(trace).
$$

The Stage-2 temporal instantiation is derived from the chunk-global context:

$$
\mathrm{Stage2TemporalContextBound}(trace)
\Longrightarrow
\mathrm{Stage2TemporalInstantiationBound}(trace).
$$

### Bridge to full temporal instantiation

The Stage-2 bundle is not by itself a whole-trace link theorem because it does
not carry the `pc` component.

The first theorem target is:

$$
\mathrm{Stage2TemporalInstantiationBound}(trace)
\land
\mathrm{PcTemporalBound}(trace)
\Longrightarrow
\mathrm{TemporalInstantiationBound}(trace).
$$

The protocol-shaped version uses the explicit Stage-3 support witness:

$$
\mathrm{ExecutionFrameBound}(trace)
\land
\mathrm{StateWellFormed}(trace)
\land
\mathrm{Stage2TemporalInstantiationBound}(trace)
\land
\mathrm{PcAdjacentBridge}(trace)
\Longrightarrow
\mathrm{TemporalInstantiationBound}(trace).
$$

This theorem is intentionally the exact handoff point from:

- Stage-2 chunk-global temporal support,
- Stage-3 semantic `pc` bridge,

to the generic temporal-instantiation witness used by `Chip8TemporalConsistency`
and `Chip8AuthenticatedTrace`.

## Paper Anchors

- **Sources**:
  - `./crates/neo-fold-prototype/specs/chip8-kernel.md`
  - `./docs/soundness-specs/twist-and-shout-requirements.md`
- Anchors:
  - strong adjacent-state linking is component-wise
  - Stage 2 owns the register / `I` and RAM temporal components
  - Stage 3 contributes the `pc` component only

## Module Mapping

| Lean file | Local owner |
|---|---|
| `Nightstream/Chip8/TwistTemporalInstantiation.lean` | Stage-2 temporal bundle and bridge to generic temporal instantiation |
| `Nightstream/Chip8/TwistTemporalInstantiationInterface.lean` | Theorem-facing re-export surface |

## Contract Surface

| Group | Lean surface | Kind | Role | Guarantee |
|---|---|---|---|---|
| Temporal | `Stage2TemporalContext` | structure | Definitional | Packages the chunk-global authenticated Stage-2 register / RAM timeline context |
| Temporal | `Stage2TemporalContextBound` | def | Definitional | Named proposition that the chunk-global Stage-2 temporal context exists |
| Temporal | `Stage2TemporalInstantiation` | structure | Definitional | Packages the concrete Stage-2 register / RAM timeline witnesses |
| Temporal | `Stage2TemporalInstantiationBound` | def | Definitional | Named proposition that the Stage-2 temporal bundle exists |
| Theorem | `stage2TemporalInstantiationBound_of_context` | theorem | Theorem-Target | The chunk-global Stage-2 temporal context yields the concrete Stage-2 temporal-instantiation bundle |
| Constructor | `temporalInstantiation_of_stage2_and_pc` | def | Definitional | Stage-2 temporal support plus `pc` continuity yields one full temporal instantiation witness |
| Theorem | `temporalInstantiationBound_of_stage2_and_pc` | theorem | Theorem-Target | Proposition-level version of the same bridge |
| Theorem | `temporalInstantiationBound_of_stage2_and_bridge` | theorem | Theorem-Target | Stage-2 temporal support plus the explicit Stage-3 `pc` bridge yields the generic temporal-instantiation contract |

## Proof Obligations

- This owner must remain Stage-2 focused: it owns register / `I` and RAM
  timelines, not the `pc` component.
- It must make the chunk-global Stage-2 temporal context explicit; the
  row-local temporal seeds exported by `Chip8EvidenceCoverage` are not by
  themselves enough.
- It must not silently collapse the explicit Stage-3 `pc` bridge back into a
  generic temporal assumption.
- It must not claim cryptographic authentication of the Stage-2 surfaces; it
  only packages their theorem-facing meaning.

## Assumption Ledger

- This owner assumes the chunk-global Stage-2 temporal context it states
  explicitly, and from that context derives the concrete register / RAM
  timeline witnesses.
- The bridge theorem additionally assumes the theorem-facing Stage-3 `pc`
  bridge together with the row well-formedness / frame-bound hypotheses needed
  to interpret it semantically.

## Dependency and Consumer Map

- **Upstream dependencies**:
  - `Chip8RegisterTimeline`
  - `Chip8RamTimeline`
  - `Chip8PcContinuityBridge`
  - `Chip8TemporalConsistency`
- **Downstream consumers**:
  - `Chip8AuthenticatedTrace`
  - `Chip8KernelSoundness`
  - `Chip8KernelArtifactAudit`
