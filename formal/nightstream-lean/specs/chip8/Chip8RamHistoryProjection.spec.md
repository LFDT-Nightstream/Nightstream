# Chip8RamHistoryProjection Spec

## Purpose

- **What it is**: The concrete Nightstream bridge owner for the CHIP-8
  `RamHistory` extension family.
- **What it is not**: It is not a generic Twist proof and it does not restate
  the whole Stage-2 semantic kernel.
- **Protocol role**: It packages the exact authenticated RAM-side Stage-2 trace
  consequences already proved for the CHIP-8 kernel into one concrete
  release-path family id and proves how that family is routed by the
  Nightstream bridge.

## Target Formulas

Define the concrete family id:

$$
\mathrm{RamHistoryFamily} := \mathrm{RamHistory}.
$$

For one exact authenticated frame list `frames`, define the concrete
RAM-history bundle:

$$
\mathrm{RamHistoryBundle}(frames),
$$

whose fields are:

- the exact non-zero-init Stage-2 initial-state surface
  `InitialStateBound(init)`,
- the exact tracewise RAM-side role-session extraction bundle
  `ExactTraceRoleSessionsBundle(frames)`,
- the canonical RAM timeline witness
  `RamTemporalBound(RamTimeline(traceOf(frames)), traceOf(frames))`.

The theorem-facing constructor target is:

$$
\mathrm{ExactTraceEvidence}(frames)
\Longrightarrow
\mathrm{RamHistoryBundleBound}(frames).
$$

This bundle must preserve the exact non-zero-init RAM boundary:

$$
\forall addr < \mathrm{RamSinkAddr},\;
\mathrm{initialRamValue}(init, addr) = init.RAM[addr]
$$

and

$$
\mathrm{initialRamValue}(init, \mathrm{RamSinkAddr}) = 0.
$$

It must also preserve the exact authenticated RAM role-session shape on the
rows that actually perform RAM accesses:

$$
\forall frame \in frames,\;
frame.\mathrm{opcodeId} = \mathrm{loadRegs}
\Longrightarrow
\mathrm{ramReads}(frame)
=
[(ramRa, rv_{\mathrm{mem}})]
$$

$$
\forall frame \in frames,\;
frame.\mathrm{opcodeId} = \mathrm{storeRegs}
\Longrightarrow
\mathrm{ramWrites}(frame)
=
[(ramWa, wv_{\mathrm{mem}})]
$$

and the extracted role-session values must agree with the kernel's exact RAM
read/write semantics:

$$
\forall frame \in frames,\;
frame.\mathrm{opcodeId} = \mathrm{loadRegs}
\Longrightarrow
rv_{\mathrm{mem}} = \mathrm{ramReadValue}(frame.pre, frame.dec)
$$

$$
\forall frame \in frames,\;
frame.\mathrm{opcodeId} = \mathrm{storeRegs}
\Longrightarrow
wv_{\mathrm{mem}} = \mathrm{ramWriteValue}(frame.post, frame.dec).
$$

Define the emitted obligation family:

$$
\mathrm{RamHistoryProjection}(p)
:=
\mathrm{TwistValProjection}(\mathrm{RamHistoryFamily}, p).
$$

This family is a concrete `TwistValEval` family, so it never enters the
Nightstream main lane directly:

$$
\neg
\mathrm{MainLaneAdmissible}
(\Pi.f_{\mathrm{main}}, \Pi.p_{\mathrm{main}}, \mathrm{RamHistoryProjection}(p)).
$$

Its classification therefore follows the generic separate-fold support rule:

$$
\Pi.S(\mathrm{RamHistoryFamily}, \mathrm{TwistValEval}, p)
\Longrightarrow
\mathrm{decideFamily}(\Pi, \mathrm{RamHistoryProjection}(p)) = \mathrm{foldSeparate}.
$$

$$
\neg \Pi.S(\mathrm{RamHistoryFamily}, \mathrm{TwistValEval}, p)
\Longrightarrow
\mathrm{decideFamily}(\Pi, \mathrm{RamHistoryProjection}(p)) = \mathrm{exportFinal}.
$$

## Contract Surface

| Group | Lean surface | Kind | Role | Guarantee |
|---|---|---|---|---|
| Family id | `ramHistoryFamily` | def | Definitional | Fixes the concrete family id to `ExtensionFamily.ramHistory` |
| Bundle | `RamHistoryBundle` | structure | Definitional | Packages exact non-zero-init state, exact RAM-side role-session extraction, and the canonical RAM timeline witness |
| Bundle | `RamHistoryBundleBound` | def | Definitional | Named proposition that the exact authenticated trace admits that concrete bundle |
| Constructor | `ramHistoryBundle_of_exactTrace` | def | Theorem-Target | Exact authenticated trace evidence yields the concrete RAM-history bundle |
| Theorem | `ramHistoryBundleBound_of_exactTrace` | theorem | Theorem-Target | Proposition-level version of the same construction |
| Theorem | `ramHistoryBundle_initialRamValue` | theorem | Theorem-Target | The bundle preserves the exact non-zero-init RAM surface at time `0` below the sink address |
| Theorem | `ramHistoryBundle_initialRamSinkValue` | theorem | Theorem-Target | The bundle preserves the exact zero-valued RAM sink surface at time `0` |
| Theorem | `loadRamReads_eq_roleValues_tracewise` | theorem | Theorem-Target | Authenticated `loadRegs` rows rewrite RAM reads directly in terms of the extracted authenticated role-session value |
| Theorem | `storeRamWrites_eq_roleValues_tracewise` | theorem | Theorem-Target | Authenticated `storeRegs` rows rewrite RAM writes directly in terms of the extracted authenticated role-session value |
| Theorem | `loadRamReadMemValue_eq_preRam_tracewise` | theorem | Theorem-Target | Authenticated `loadRegs` RAM read values agree with the exact pre-state RAM semantics |
| Theorem | `storeRamWriteMemValue_eq_postRam_tracewise` | theorem | Theorem-Target | Authenticated `storeRegs` RAM write values agree with the exact post-state RAM semantics |
| Projection | `ramHistoryProjection` | def | Definitional | Concrete `TwistValProjection` for the RAM-history family |
| Theorem | `ramHistoryProjection_not_mainLane` | theorem | Theorem-Target | Concrete RAM-history family stays out of the main lane |
| Theorem | `ramHistoryProjection_decide_eq_foldSeparate_of_supported` | theorem | Theorem-Target | Explicit family support routes RAM history to separate folding |
| Theorem | `ramHistoryProjection_decide_eq_exportFinal_of_unsupported` | theorem | Theorem-Target | Without explicit support the family remains final |

## Dependency and Consumer Map

- **Upstream dependencies**:
  - `Nightstream/Chip8/ExtensionFamily.lean`
  - `Nightstream/Chip8/Stage2/WitnessMemoryBinding.lean`
  - `Nightstream/Chip8/Stage2/TwistTraceRoleSessions.lean`
  - `Nightstream/Chip8/Trace/RamTimeline.lean`
  - `Nightstream/Chip8/Trace/AuthenticatedTrace.lean`
  - `Nightstream/ShardComposition.lean`
- **Downstream consumers**:
  - later release-path CHIP-8 family routing owners
  - later Rust refinement for `families/ram_history.rs`

## Proof Obligations

- The concrete family id must stay aligned with the Rust `ExtensionFamily`
  boundary.
- The RAM-history family must be routed as a `TwistValEval` family, not as a
  main-lane `CE` family.
- This owner must keep the exact non-zero-init route explicit through
  `InitialStateBound(init)`.
- This owner may assume only the exact Stage-2/trace theorem surfaces already
  owned by `Chip8WitnessMemoryBinding`, `Chip8TwistTraceRoleSessions`, and
  `Chip8AuthenticatedTrace`.

## Paper Anchors

- **Source**:
  - `./docs/assurance-strategy.md`
  - `./docs/soundness-specs/twist-and-shout-requirements.md`
  - `./crates/neo-fold-prototype/specs/chip8-kernel.md`
  - `./crates/neo-fold-prototype/src/families/ram_history.rs`
  - `./formal/nightstream-lean/specs/chip8/Chip8TwistTraceRoleSessions.spec.md`
  - `./formal/nightstream-lean/specs/chip8/Chip8RamTimeline.spec.md`

## Out of Scope

- generic Twist soundness proofs
- register-history routing
- main-lane admissibility for non-`CE` families
