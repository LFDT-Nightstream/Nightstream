import Nightstream.Implementation.R1CS.Artifacts.FPrimeRecursive.Generated.FPrimeRecursiveOutputAuthoritySboxManifestData
import Nightstream.Implementation.R1CS.Artifacts.FPrimeRecursive.OrdinaryPlacement
import Nightstream.Implementation.R1CS.Artifacts.FPrimeRecursive.SourceRoleCensus

/-!
Stable facade for fixed-F-prime recursive generated evidence.

Owns: the public import boundary for the reviewed output-authority Poseidon2
call geometry, checked base/recursive source-role censuses, and derived
ordinary-private source-loop placements.

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
-/
