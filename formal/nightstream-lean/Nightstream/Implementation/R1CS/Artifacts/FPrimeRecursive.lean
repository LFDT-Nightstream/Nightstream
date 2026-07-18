import Nightstream.Implementation.R1CS.Artifacts.FPrimeRecursive.Generated.FPrimeRecursiveOutputAuthoritySboxManifestData
import Nightstream.Implementation.R1CS.Artifacts.FPrimeRecursive.OrdinaryPlacement
import Nightstream.Implementation.R1CS.Artifacts.FPrimeRecursive.PiRlcChallenge
import Nightstream.Implementation.R1CS.Artifacts.FPrimeRecursive.PiRlcProjection
import Nightstream.Implementation.R1CS.Artifacts.FPrimeRecursive.SourceRoleCensus
import Nightstream.Implementation.R1CS.Artifacts.FPrimeRecursive.YZcolProjection

/-!
Stable facade for fixed-F-prime recursive generated evidence.

Owns: the public import boundary for the reviewed output-authority Poseidon2
call geometry, checked base/recursive source-role censuses, derived
ordinary-private source-loop placements, and the two active parent `y_zcol`
projection identities with their shared beta/rho and output-evaluation owners.

Does not own: encoded/CE coordinate layouts, global source-row identity,
whole-matrix no-escape, semantic soundness, centered substitution, or
permission to remove rows or slots.

Emits constraints: no.

Authority boundary: generated Rust evidence flags and source-census drift
checks report external conformance. The source census is source-only and says
nothing about encoded/CE coordinates. The output-authority metadata does not
prove `SourceCallRowsMatch` or `WholeMatrixNoEscape` in Lean.

| Child | Evidence | Semantic owner |
|---|---|---|
| `FPrimeRecursiveOutputAuthoritySboxManifestData` | 422 call geometries, 86 offsets, and exact censuses | `Correspondence.Poseidon2.OutputAuthoritySboxManifestProofs` |
| `SourceRoleCensus` | exact base/recursive source partitions and role counts | `Correspondence.FieldEncoding.PackedSourceCensus` |
| `OrdinaryPlacement` | exact ordinary 41-word starts derived from source roles plus compact source-phase/final-width metadata | `Correspondence.FieldEncoding.OrdinaryPlacement` |
| `PiRlcChallenge.ChallengeWiring` | exact 15 x 54 physical alias from selection outputs to projection rho consumers | sampler-row semantics and transcript authority remain open |
| `PiRlcChallenge.SamplerLayout` | typed affine rows/columns, block-major lane predecessors, and fixed sampler profile counts | row satisfaction, sampler semantics, and transcript authority remain open |
| `PiRlcChallenge.TranscriptLayout` | exact 291-pin / 78-call source partition, ordered emissions, state continuity, 240 field-output aliases, and four external bind-input columns | row satisfaction, transcript replay, and transcript authority remain open |
| `YZcolProjection` | two 54-coefficient evaluator leaves bound to returned parent wires, with 216 exact row equations | semantic and transcript correspondence remain open |
| `PiRlcProjection.BetaLadder` | one 55-power ladder bound to both `y_zcol` leaves, with 272 exact row equations | transcript derivation remains open |
| `PiRlcProjection.RhoEvaluations` | 15 shared 54-coefficient rho evaluators, with 1,620 exact row equations | transcript derivation and semantic rho ownership remain open |
| `PiRlcProjection.YZcolIdentities` | both 1,916-row degree-106 identity schedules and 3,616 newly owned local source rows | conditional exact-or-bad-root correspondence; transcript/semantic authority remains open |
-/
