import Nightstream.Implementation.R1CS.Correspondence.PiCcsMatrix.Phi81BarMatrixRefinement
import Nightstream.Implementation.R1CS.Correspondence.PiCcsMatrix.Phi81MatrixSourceRefinement

/-!
Public ownership root for the concrete `Pi_CCS` matrix correspondence.

Owns: the shallow map from Rust matrix evidence to independently specified
coefficient-embedding semantics.

Does not own: matrix packing loops, CCS row evaluation, R1CS lowering, row
removal, or constraint accounting.

Emits constraints: no.

Authority boundary: generated implementation data can enter this layer only
through a handwritten theorem against an independent mathematical definition.

| Child path | Mathematical obligation | Emits constraints? | Owner |
|---|---|---|---|
| `Phi81BarMatrixRefinement` | all runtime bar-matrix entries equal the independent Phi81 transform | no | artifact-to-semantics correspondence |
| `Phi81MatrixSourceRefinement` | every runtime-bar-derived coefficient matrix leaf equals the independent complete-carrier source | no | artifact-to-source correspondence |
-/
