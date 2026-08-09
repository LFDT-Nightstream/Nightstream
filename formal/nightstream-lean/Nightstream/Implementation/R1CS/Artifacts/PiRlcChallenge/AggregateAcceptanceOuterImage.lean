import Nightstream.Implementation.R1CS.Artifacts.PiRlcChallenge.Generated.AggregateAcceptanceOuterImageData

/-!
Stable facade for the generated fixed-recursive aggregate-acceptance image.

Owns: the public artifact import boundary for exact dimensions, fifteen
challenge shards, and 960 ordered direct-decoder chunk records.

Does not own: semantic interpretation, Rust refinement, constraint
soundness, cost authority, or permission to remove rows.

Emits constraints: no.

Authority boundary: generated data remains non-authoritative evidence. Import
this facade only from handwritten correspondence that assigns mathematical
meaning to the records.

| Child branch | Records | Semantic consumer |
|---|---:|---|
| shape | one fixed direct-decoder profile | outer-image artifact refinement |
| challenges | 15 × 64 chunks | physical placement refinement |
-/
