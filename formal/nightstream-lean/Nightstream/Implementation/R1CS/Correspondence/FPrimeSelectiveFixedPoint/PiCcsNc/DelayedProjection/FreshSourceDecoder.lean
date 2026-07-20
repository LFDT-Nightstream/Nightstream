import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.FreshSourceDecoder.Refinement

/-!
Correspondence facade for the fixed-active fresh public-`X` decoder.

Owns: the stable import surface for exact artifact provenance and the
conditional public-input-field refinement.

Does not own: coordinate binding rows, complete source-product authority,
full witness data, commitment authority, or rows.

Emits constraints: none.

| Child path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `FreshSourceDecoder.Refinement` | bound source columns and direct dataflow imply the fresh public-input field | conditional correspondence | `Refinement` |
-/
