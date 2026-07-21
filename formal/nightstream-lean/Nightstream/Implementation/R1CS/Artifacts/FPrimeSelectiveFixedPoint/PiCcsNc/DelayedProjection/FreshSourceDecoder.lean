import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.FreshSourceDecoder.Exact

/-!
Public artifact facade for the fixed-active fresh public-`X` decoder.

Owns: the stable import surface for the exact generated certificate.

Does not own: coordinate values, semantic public-input binding, full witness
data, commitment authority, or rows.

Emits constraints: none.

| Child path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `FreshSourceDecoder.Exact` | exact bounded column and disposition provenance | artifact-checked | `Exact` |
-/
