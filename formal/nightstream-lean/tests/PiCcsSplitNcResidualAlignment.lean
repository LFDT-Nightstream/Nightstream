import Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionRefinement.ResidualAlignment

/-!
Focused regressions for the Split-NC residual-slot obstruction and semantic
replacement.
-/

set_option autoImplicit false

namespace NightstreamTests.PiCcsSplitNcResidualAlignment

open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial
open Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionRefinement.ResidualAlignment

example : ¬ LiteralResidualSlotAlignment :=
  not_literalResidualSlotAlignment

example :
    productionCarriedGammaExponent Witness.shape Witness.running
      Witness.matrix = 5 :=
  Witness.productionCarriedGammaExponent_eq_five

example :
    paperCarriedGammaExponent Witness.shape Witness.running Witness.matrix
      Witness.coefficientZero = 4 :=
  Witness.paperCoefficientZeroGammaExponent_eq_four

example :
    paperCarriedGammaExponent Witness.shape Witness.running Witness.matrix
      Witness.coefficientOne = 6 :=
  Witness.paperCoefficientOneGammaExponent_eq_six

example :
    Witness.shape.rowVariables + productionLaneVariables = 7 ∧
      Witness.shape.paperShape.cubeVariables = 1 :=
  ⟨Witness.feRoundArity_eq_seven, Witness.paperRoundArity_eq_one⟩

example :
    Nc.Mixing.sourceExponent Witness.shape .paperNc Witness.firstSource = 0 ∧
      Nc.Mixing.sourceExponent Witness.shape .paperJointQ
        Witness.firstSource = 1 :=
  ⟨Witness.ncRelativeExponent_eq_zero, Witness.ncAbsoluteExponent_eq_one⟩

example
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    {shape : SemanticShape}
    (data : Data shape) :
    Semantics.ResidualsZero data ↔ Semantics.Paper.Holds data :=
  semanticResidualsZero_iff_paperHolds noZeroDivisors data

end NightstreamTests.PiCcsSplitNcResidualAlignment
