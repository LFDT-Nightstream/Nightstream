import Nightstream.SuperNeo.Folding.PiCCS.PaperRectangular
import Nightstream.Implementation.Rust.CanonicalConformance.PiCcsPaperRectangular.Generated.Layout

/-!
Bounded Rust-to-Lean conformance for the canonical PiCCS coefficient layout.

Owns: exact agreement of all 324 Rust-exported carried gamma slots, the fresh
and norm slots, and the two tested rectangular domain directions with the
independent PaperJoint coordinate model.

Does not own: Rust matrix evaluation, transcript bytes, proof serialization,
SumCheck rounds, general dimensions, R1CS lowering, or security reduction.
-/

namespace Nightstream.Implementation.Rust.CanonicalConformance.PiCcsPaperRectangular

open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

namespace Artifact

def shape : Shape where
  cubeVariables := Generated.columnVariablesWhenNltM
  freshCount := Generated.freshCount
  runningCount := Generated.runningCount
  matrixCount := Generated.matrixCount
  coefficientCount := Generated.coefficientCount

def expectedFreshGammaExponents : List Nat :=
  (canonicalFinIndices shape.freshCount).map Fin.val

def expectedNormGammaExponents : List Nat :=
  (canonicalFinIndices shape.sourceCount).map fun source =>
    shape.normOffset + source.val

def expectedCarriedGammaExponents : List Nat :=
  (canonicalCarriedCoordinates shape).map CarriedCoordinate.gammaExponent

theorem carried_input_length :
    Generated.carriedGammaExponents.length = Generated.carriedCount := by
  set_option maxRecDepth 2000 in
    decide

theorem carried_count_eq_shape :
    Generated.carriedCount = shape.carriedEvaluationCount := by
  decide

theorem fresh_gamma_slots_match :
    Generated.freshGammaExponents = expectedFreshGammaExponents := by
  decide

theorem norm_gamma_slots_match :
    Generated.normGammaExponents = expectedNormGammaExponents := by
  decide

theorem carried_gamma_slots_match :
    Generated.carriedGammaExponents = expectedCarriedGammaExponents := by
  set_option maxRecDepth 2000 in
    decide

theorem covers_both_rectangular_directions :
    Generated.rowVariablesWhenNltM < Generated.columnVariablesWhenNltM ∧
    Generated.columnVariablesWhenNgtM < Generated.rowVariablesWhenNgtM := by
  decide

end Artifact

end Nightstream.Implementation.Rust.CanonicalConformance.PiCcsPaperRectangular
