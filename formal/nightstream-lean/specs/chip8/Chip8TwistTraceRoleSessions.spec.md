# Chip8TwistTraceRoleSessions Spec

## Purpose

- **What it is**: the trace-level Stage-2 role-session extraction owner for
  authenticated CHIP-8 rows.
- **Key property**: from one exact authenticated frame it recovers the explicit
  authenticated register-side Twist sessions for `regRaX`, `regRaY`, `regRaI`,
  and `regWa`, together with the conditional RAM-side load/store sessions when
  the opcode uses them.
- **Protocol role**: this owner sits strictly below chunk-global Stage-2
  temporal reconstruction. It exposes the concrete per-row authenticated
  objects that a later owner must compose into one shared register timeline and
  one shared RAM timeline.
- **Additional extraction surface**: it also exposes the exact per-row
  Stage-2 register/RAM access traces carried by the seed bundle, rewritten
  directly in terms of the extracted authenticated role-session values, and
  lifts those equalities to one explicit tracewise `List.Forall` surface over
  the full chunk.

## Target Formulas

Define:

$$
\mathrm{ExactFrameRoleSessions}(frame)
$$

to package:

- one authenticated row-local Stage-2 seed bundle `seed`,
- the exact `\Gamma_3` claim list carried by that seed,
- the explicit authenticated register-side sessions extracted from `seed`,
- the conditional authenticated RAM-side load session when
  `frame.dec.opcodeId = \mathrm{loadRegs}`,
- the conditional authenticated RAM-side store session when
  `frame.dec.opcodeId = \mathrm{storeRegs}`.

Its proposition form is:

$$
\mathrm{ExactFrameRoleSessionsBound}(frame).
$$

The theorem target is:

$$
\mathrm{ExactFrameEvidence}(frame)
\Longrightarrow
\mathrm{ExactFrameRoleSessionsBound}(frame).
$$

At the trace level, this owner exposes:

$$
\mathrm{ExactTraceEvidence}(frames)
\Longrightarrow
\forall frame \in frames,\ \mathrm{ExactFrameRoleSessionsBound}(frame).
$$

It also exposes one explicit chunk-level extraction object:

$$
\mathrm{ExactTraceRoleSessionsBundle}(frames),
$$

which returns the extracted `\mathrm{ExactFrameRoleSessions}(frame)` package for
any `frame \in frames`. This remains row-indexed Stage-2 evidence, not a
shared temporal context.

This is still not the chunk-global Stage-2 temporal context. It is the exact
row-indexed extraction layer that removes nested existential noise from the
authenticated Stage-2 seed summary before the whole-trace reconstruction
theorem is attempted.

## Paper Anchors

- **Sources**:
  - `/Users/nicolasarqueros/starstream/develop/nightstream-clean-up/crates/neo-fold-prototype/specs/chip8-kernel.md`
  - `/Users/nicolasarqueros/starstream/develop/nightstream-clean-up/docs/soundness-specs/twist-and-shout-requirements.md`
- Anchors:
  - Stage 2 must preserve the exact `Inc -> Val -> rv / wv` dependency chain
  - strong trace soundness requires a chunk-global register/RAM temporal
    consequence, not just row-local sessions
  - row-local session extraction should remain separate from whole-trace
    temporal reconstruction

## Module Mapping

| Lean file | Local owner |
|---|---|
| `Nightstream/Chip8/TwistTraceRoleSessions.lean` | Exact-frame and exact-trace extraction of explicit Stage-2 role sessions |
| `Nightstream/Chip8/TwistTraceRoleSessionsInterface.lean` | Theorem-facing re-export surface |

## Contract Surface

| Group | Lean surface | Kind | Role | Guarantee |
|---|---|---|---|---|
| Extraction | `ExactFrameRoleSessions` | structure | Definitional | Packages the exact authenticated Stage-2 seed together with the explicit row-local register/RAM role sessions |
| Extraction | `ExactFrameRoleSessionsBound` | def | Definitional | Named proposition that one exact frame admits that explicit role-session package |
| Extraction | `ExactTraceRoleSessionsBundle` | structure | Definitional | Chunk-level bundle that returns the extracted row-local role-session package for any frame in the trace |
| Extraction | `ExactTraceRoleSessionsBundleBound` | def | Definitional | Named proposition that the trace admits that explicit chunk-level extraction bundle |
| Constructor | `exactFrameRoleSessions_of_exactFrameEvidence` | def | Definitional | Builds the explicit role-session package from exact authenticated frame evidence |
| Theorem | `exactFrameRoleSessionsBound_of_exactFrameEvidence` | theorem | Theorem-Target | Proposition-level version of the same extraction |
| Theorem | `exactTraceRoleSessionsBound_of_frames` | theorem | Theorem-Target | Every exact authenticated frame in the trace admits the explicit role-session package |
| Constructor | `exactTraceRoleSessionsBundle_of_frames` | def | Definitional | Builds the chunk-level extraction bundle across the full trace |
| Theorem | `exactTraceRoleSessionsBundleBound_of_frames` | theorem | Theorem-Target | Proposition-level version of the chunk-level extraction bundle |
| Theorem | `registerReadXValue_eq_primaryValue_of_traceBundle` | theorem | Theorem-Target | The bundle exposes the authenticated row-local `regRaX` value directly as the concrete `pre`-state primary value |
| Theorem | `registerReadYValue_eq_secondaryValue_of_traceBundle` | theorem | Theorem-Target | The bundle exposes the authenticated row-local `regRaY` value directly as the concrete `pre`-state secondary value |
| Theorem | `registerReadIValue_eq_preI_of_traceBundle` | theorem | Theorem-Target | The bundle exposes the authenticated row-local `regRaI` value directly as the concrete `pre.i` value |
| Theorem | `registerWriteRegValue_eq_postValue_of_traceBundle` | theorem | Theorem-Target | The bundle exposes the authenticated row-local `regWa` value directly as the concrete `post`-state write value |
| Theorem | `registerReadsExpected_of_traceBundle` | theorem | Theorem-Target | The bundle exposes the exact expected per-row register-read trace carried by the Stage-2 seed |
| Theorem | `registerWritesExpected_of_traceBundle` | theorem | Theorem-Target | The bundle exposes the exact expected per-row register-write trace carried by the Stage-2 seed |
| Theorem | `ramReadsExpected_of_traceBundle` | theorem | Theorem-Target | The bundle exposes the exact expected per-row RAM-read trace carried by the Stage-2 seed |
| Theorem | `ramWritesExpected_of_traceBundle` | theorem | Theorem-Target | The bundle exposes the exact expected per-row RAM-write trace carried by the Stage-2 seed |
| Theorem | `registerReads_eq_roleValues_of_traceBundle` | theorem | Theorem-Target | The per-row register-read trace can be rewritten directly in terms of the extracted `regRaX/regRaY/regRaI` authenticated values |
| Theorem | `registerWrites_eq_roleValues_of_traceBundle` | theorem | Theorem-Target | The per-row register-write trace can be rewritten directly in terms of the extracted `regWa` authenticated value |
| Theorem | `registerReads_eq_roleValues_tracewise` | theorem | Theorem-Target | The full trace exposes one `List.Forall` register-read access shape written directly in terms of extracted authenticated role-session values |
| Theorem | `registerWrites_eq_roleValues_tracewise` | theorem | Theorem-Target | The full trace exposes one `List.Forall` register-write access shape written directly in terms of extracted authenticated role-session values |
| Theorem | `loadRamReads_eq_roleValues_of_traceBundle` | theorem | Theorem-Target | On `loadRegs` rows, the per-row RAM-read trace can be rewritten directly in terms of the extracted authenticated load session value |
| Theorem | `storeRamWrites_eq_roleValues_of_traceBundle` | theorem | Theorem-Target | On `storeRegs` rows, the per-row RAM-write trace can be rewritten directly in terms of the extracted authenticated store session value |
| Theorem | `loadRamReads_eq_roleValues_tracewise` | theorem | Theorem-Target | On `loadRegs` rows, the full trace exposes one `List.Forall` RAM-read access shape written directly in terms of extracted authenticated load-session values |
| Theorem | `storeRamWrites_eq_roleValues_tracewise` | theorem | Theorem-Target | On `storeRegs` rows, the full trace exposes one `List.Forall` RAM-write access shape written directly in terms of extracted authenticated store-session values |
| Theorem | `loadRamReadMemValue_eq_preRam_of_traceBundle` | theorem | Theorem-Target | On `loadRegs` rows, the bundle exposes the authenticated RAM read value directly as the concrete pre-state RAM slot value |
| Theorem | `storeRamWriteMemValue_eq_postRam_of_traceBundle` | theorem | Theorem-Target | On `storeRegs` rows, the bundle exposes the authenticated RAM write value directly as the concrete post-state RAM slot value |
| Theorem | `loadRamReadMemValue_eq_preRam_tracewise` | theorem | Theorem-Target | On `loadRegs` rows, the full trace exposes one `List.Forall` authenticated RAM-read-value surface equal to the concrete pre-state RAM slot value |
| Theorem | `storeRamWriteMemValue_eq_postRam_tracewise` | theorem | Theorem-Target | On `storeRegs` rows, the full trace exposes one `List.Forall` authenticated RAM-write-value surface equal to the concrete post-state RAM slot value |

## Proof Obligations

- This owner must stay row-local and extraction-only. It does not own the
  chunk-global register/RAM timeline theorem.
- It must preserve the exact Stage-2 ownership split:
  register and RAM role sessions come from the Stage-2 seed bundle, not from
  Stage 1 or Stage 3.
- It must not replace the authenticated seed bundle with a weaker summary.
- It must keep conditional RAM load/store extraction explicit rather than
  pretending every row has both.

## Assumption Ledger

- This owner assumes only the exact authenticated frame evidence exported by
  `Chip8AuthenticatedTrace` plus the row-local Stage-2 seed extraction already
  owned by `Chip8EvidenceCoverage`.
- It introduces no new temporal or cross-row assumptions.

## Dependency and Consumer Map

- **Upstream dependencies**:
  - `Chip8AuthenticatedTrace`
  - `Chip8EvidenceCoverage`
  - `Chip8TwistRoleSessions`
- **Downstream consumers**:
  - the future chunk-global Stage-2 temporal reconstruction owner
  - any audit surface that needs explicit row-local authenticated Stage-2
    session objects rather than nested existentials
