# Chip8RegisterHistoryProjection Spec

## Purpose

- **What it is**: The concrete Nightstream bridge owner for the CHIP-8
  `RegisterHistory` extension family.
- **What it is not**: It is not a generic Twist proof and it does not restate
  the whole Stage-2 semantic kernel.
- **Protocol role**: It packages the exact authenticated register-side
  Stage-2 trace consequences already proved for the CHIP-8 kernel into one
  concrete release-path family id and proves how that family is routed by the
  Nightstream bridge.

## Target Formulas

Define the concrete family id:

$$
\mathrm{RegisterHistoryFamily} := \mathrm{RegisterHistory}.
$$

For one exact authenticated frame list `frames`, define the concrete
register-history bundle:

$$
\mathrm{RegisterHistoryBundle}(frames),
$$

whose fields are:

- the exact non-zero-init Stage-2 initial-state surface
  `InitialStateBound(init)`,
- the exact tracewise register-side role-session extraction bundle
  `ExactTraceRoleSessionsBundle(frames)`,
- the canonical register timeline witness
  `RegisterTemporalBound(RegisterTimeline(traceOf(frames)), traceOf(frames))`.

The theorem-facing constructor target is:

$$
\mathrm{ExactTraceEvidence}(frames)
\Longrightarrow
\mathrm{RegisterHistoryBundleBound}(frames).
$$

This bundle must preserve the exact tracewise authenticated register access
shape:

$$
\forall frame \in frames,\;
\mathrm{registerReads}(frame)
=
[(regRaX, rv_X), (regRaY, rv_Y), (regRaI, rv_I)]
$$

and

$$
\forall frame \in frames,\;
\mathrm{registerWrites}(frame)
=
[(regWa, wv_{\mathrm{reg}})].
$$

It must also preserve the exact non-zero-init register boundary:

$$
\forall idx \in \{0,\dots,15\},\;
\mathrm{initialRegisterValue}(init, idx) = init.V[idx]
$$

and

$$
\mathrm{initialRegisterValue}(init, 16) = init.I.
$$

Define the emitted obligation family:

$$
\mathrm{RegisterHistoryProjection}(p)
:=
\mathrm{TwistValProjection}(\mathrm{RegisterHistoryFamily}, p).
$$

This family is a concrete `TwistValEval` family, so it never enters the
Nightstream main lane directly:

$$
\neg
\mathrm{MainLaneAdmissible}
(\Pi.f_{\mathrm{main}}, \Pi.p_{\mathrm{main}}, \mathrm{RegisterHistoryProjection}(p)).
$$

Its classification therefore follows the generic separate-fold support rule:

$$
\Pi.S(\mathrm{RegisterHistoryFamily}, \mathrm{TwistValEval}, p)
\Longrightarrow
\mathrm{decideFamily}(\Pi, \mathrm{RegisterHistoryProjection}(p)) = \mathrm{foldSeparate}.
$$

$$
\neg \Pi.S(\mathrm{RegisterHistoryFamily}, \mathrm{TwistValEval}, p)
\Longrightarrow
\mathrm{decideFamily}(\Pi, \mathrm{RegisterHistoryProjection}(p)) = \mathrm{exportFinal}.
$$

## Contract Surface

| Group | Lean surface | Kind | Role | Guarantee |
|---|---|---|---|---|
| Family id | `registerHistoryFamily` | def | Definitional | Fixes the concrete family id to `ExtensionFamily.registerHistory` |
| Bundle | `RegisterHistoryBundle` | structure | Definitional | Packages exact non-zero-init state, exact register-side role-session extraction, and the canonical register timeline witness |
| Bundle | `RegisterHistoryBundleBound` | def | Definitional | Named proposition that the exact authenticated trace admits that concrete bundle |
| Constructor | `registerHistoryBundle_of_exactTrace` | def | Theorem-Target | Exact authenticated trace evidence yields the concrete register-history bundle |
| Theorem | `registerHistoryBundleBound_of_exactTrace` | theorem | Theorem-Target | Proposition-level version of the same construction |
| Theorem | `registerHistoryBundle_initialRegisterValue` | theorem | Theorem-Target | The bundle preserves the exact non-zero-init `V[idx]` surface at time `0` |
| Theorem | `registerHistoryBundle_initialIValue` | theorem | Theorem-Target | The bundle preserves the exact non-zero-init `I` surface at time `0` |
| Theorem | `registerHistoryReads_eq_roleValues_tracewise` | theorem | Theorem-Target | The exact authenticated register-read trace is rewritten directly in terms of the extracted authenticated role-session values |
| Theorem | `registerHistoryWrites_eq_roleValues_tracewise` | theorem | Theorem-Target | The exact authenticated register-write trace is rewritten directly in terms of the extracted authenticated role-session value |
| Projection | `registerHistoryProjection` | def | Definitional | Concrete `TwistValProjection` for the register-history family |
| Theorem | `registerHistoryProjection_not_mainLane` | theorem | Theorem-Target | Concrete register-history family stays out of the main lane |
| Theorem | `registerHistoryProjection_decide_eq_foldSeparate_of_supported` | theorem | Theorem-Target | Explicit family support routes register history to separate folding |
| Theorem | `registerHistoryProjection_decide_eq_exportFinal_of_unsupported` | theorem | Theorem-Target | Without explicit support the family remains final |

## Dependency and Consumer Map

- **Upstream dependencies**:
  - `Nightstream/Chip8/ExtensionFamily.lean`
  - `Nightstream/Chip8/Stage2/WitnessMemoryBinding.lean`
  - `Nightstream/Chip8/Stage2/TwistTraceRoleSessions.lean`
  - `Nightstream/Chip8/Trace/RegisterTimeline.lean`
  - `Nightstream/Chip8/Trace/AuthenticatedTrace.lean`
  - `Nightstream/ShardComposition.lean`
- **Downstream consumers**:
  - later release-path CHIP-8 family routing owners
  - later Rust refinement for `families/register_history.rs`

## Proof Obligations

- The concrete family id must stay aligned with the Rust `ExtensionFamily`
  boundary.
- The register-history family must be routed as a `TwistValEval` family, not
  as a main-lane `CE` family.
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
  - `./crates/neo-fold-prototype/src/families/register_history.rs`
  - `./formal/nightstream-lean/specs/chip8/Chip8TwistTraceRoleSessions.spec.md`
  - `./formal/nightstream-lean/specs/chip8/Chip8RegisterTimeline.spec.md`

## Out of Scope

- generic Twist soundness proofs
- RAM-history routing
- main-lane admissibility for non-`CE` families
