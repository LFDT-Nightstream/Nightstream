import Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionRefinement
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Fe
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.Mixing.Gamma

/-!
Residual-coordinate classification for the production Split-NC prefix.

Assurance tier: model-level registered-deviation refinement.

Owns: the exact obstruction to literal paper-slot identity, the surviving
relative/absolute NC exponent equations, and the semantic residual reduction
used by the production verifier.

Does not own: probability bounds, Fiat--Shamir, field certificates,
commitment security, extraction, Rust, R1CS, costs, or rows.

Emits constraints: no.

| Stage path | Owned statement | Authority |
|---|---|---|
| `fprime.piccs.production.residual_slots` | literal slot identity is false | kernel-checked obstruction |
| `fprime.piccs.production.residual_semantics` | split residual truth is the paper relation | derived |
-/

set_option autoImplicit false

namespace Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionRefinement.ResidualAlignment

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial

/-- The actual FE carried exponent after applying both the inner production
group exponent and the outer carried-block shift. -/
def productionCarriedGammaExponent
    (shape : SemanticShape)
    (running : Fin shape.runningCount)
    (matrix : Fin shape.matrixCount) : Nat :=
  shape.sourceCount + Fe.carriedGammaExponent shape running matrix

/-- The paper coordinate corresponding to one production running source,
matrix, and Phi81 coefficient lane. -/
def paperCarriedCoordinate
    (shape : SemanticShape)
    (running : Fin shape.runningCount)
    (matrix : Fin shape.matrixCount)
    (coefficient : Fin ringDegree) :
    CarriedCoordinate shape.paperShape where
  running := running
  matrix := matrix
  coefficient := coefficient

/-- The absolute Section-7.3 gamma exponent of one carried coefficient. -/
def paperCarriedGammaExponent
    (shape : SemanticShape)
    (running : Fin shape.runningCount)
    (matrix : Fin shape.matrixCount)
    (coefficient : Fin ringDegree) : Nat :=
  (paperCarriedCoordinate shape running matrix coefficient).gammaExponent

/-- The production Phi81 coefficient lane uses six Boolean variables. -/
def productionLaneVariables : Nat := 6

/-- Necessary conditions for literal production/paper slot identity.

The condition deliberately says identity, not semantic equivalence: it asks
the FE product domain to have the paper row arity, relative NC weights to
already occupy the joint polynomial's absolute norm block, and every carried
coefficient to use its paper gamma exponent. -/
def LiteralResidualSlotAlignment : Prop :=
  (∀ (shape : SemanticShape),
      shape.rowVariables + productionLaneVariables =
        shape.paperShape.cubeVariables) ∧
  (∀ (shape : SemanticShape) (source : Fin shape.sourceCount),
      Nc.Mixing.sourceExponent shape .paperNc source =
        shape.paperShape.normOffset + source.val) ∧
  (∀ (shape : SemanticShape)
      (running : Fin shape.runningCount)
      (matrix : Fin shape.matrixCount)
      (coefficient : Fin ringDegree),
      productionCarriedGammaExponent shape running matrix =
        paperCarriedGammaExponent shape running matrix coefficient)

/-- Relative production NC weights are exactly the paper's local NC weights. -/
theorem ncRelativeExponent_eq_paperLocal
    (shape : SemanticShape)
    (source : Fin shape.sourceCount) :
    Nc.Mixing.sourceExponent shape .paperNc source = source.val := by
  rfl

/-- Applying the paper joint-Q convention places a norm residual at the
absolute Section-7.3 norm-block exponent. -/
theorem ncJointExponent_eq_paperNormSlot
    (shape : SemanticShape)
    (source : Fin shape.sourceCount) :
    Nc.Mixing.sourceExponent shape .paperJointQ source =
      shape.paperShape.normOffset + source.val := by
  rfl

namespace Witness

/-- Smallest useful shape exposing both the matrix stride and a nontrivial
Phi81 coefficient axis. -/
def shape : SemanticShape where
  rowVariables := 1
  logicalWidth := 1
  freshCount := 1
  runningCount := 1
  matrixCount := 2

def running : Fin shape.runningCount := ⟨0, by decide⟩
def matrix : Fin shape.matrixCount := ⟨1, by decide⟩
def coefficientZero : Fin ringDegree := ⟨0, by decide⟩
def coefficientOne : Fin ringDegree := ⟨1, by decide⟩
def firstSource : Fin shape.sourceCount := ⟨0, by decide⟩

theorem productionCarriedGammaExponent_eq_five :
    productionCarriedGammaExponent shape running matrix = 5 := by
  decide

theorem paperCoefficientZeroGammaExponent_eq_four :
    paperCarriedGammaExponent shape running matrix coefficientZero = 4 := by
  decide

theorem paperCoefficientOneGammaExponent_eq_six :
    paperCarriedGammaExponent shape running matrix coefficientOne = 6 := by
  decide

theorem feRoundArity_eq_seven :
    shape.rowVariables + productionLaneVariables = 7 := by
  decide

theorem paperRoundArity_eq_one :
    shape.paperShape.cubeVariables = 1 := by
  decide

theorem ncRelativeExponent_eq_zero :
    Nc.Mixing.sourceExponent shape .paperNc firstSource = 0 := by
  decide

theorem ncAbsoluteExponent_eq_one :
    Nc.Mixing.sourceExponent shape .paperJointQ firstSource = 1 := by
  decide

end Witness

/-- A single production carried gamma group owns multiple paper coefficient
slots. The coefficient coordinate is represented by the lane SumCheck axis,
not by distinct gamma powers. -/
theorem carriedCoefficientAxis_is_not_gammaAxis :
    productionCarriedGammaExponent Witness.shape Witness.running
        Witness.matrix =
      productionCarriedGammaExponent Witness.shape Witness.running
        Witness.matrix ∧
    paperCarriedGammaExponent Witness.shape Witness.running Witness.matrix
        Witness.coefficientZero ≠
      paperCarriedGammaExponent Witness.shape Witness.running Witness.matrix
        Witness.coefficientOne := by
  constructor
  · rfl
  · rw [Witness.paperCoefficientZeroGammaExponent_eq_four,
      Witness.paperCoefficientOneGammaExponent_eq_six]
    decide

/-- Kernel-checked obstruction to the literal alignment target.

This does not obstruct a semantic refinement: it proves only that the
production product-domain split cannot be described as slot-identical to the
paper's one row-domain gamma polynomial. -/
theorem not_literalResidualSlotAlignment :
    ¬ LiteralResidualSlotAlignment := by
  intro alignment
  have equal := alignment.2.2 Witness.shape Witness.running Witness.matrix
    Witness.coefficientZero
  rw [Witness.productionCarriedGammaExponent_eq_five,
    Witness.paperCoefficientZeroGammaExponent_eq_four] at equal
  omega

/-- The soundness-relevant replacement for false slot identity: the complete
uncompressed split residual family is sound and complete for the independently
stated Section-7.3 relation. -/
theorem semanticResidualsZero_iff_paperHolds
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    {shape : SemanticShape}
    (data : Data shape) :
    Semantics.ResidualsZero data ↔ Semantics.Paper.Holds data :=
  Semantics.residualsZero_iff_paperHolds noZeroDivisors data

universe uState

variable
  {shape : SemanticShape}
  {State : Type uState}
  {publicRingColumns verifierRows : Nat}
  {publicFits : ringDegree * publicRingColumns <= shape.carrierWidth}

/-- Production acceptance uses the semantic refinement rather than a false
message- or slot-identity claim. All algebraic disagreement remains in the
existing exact FE/NC event families. -/
theorem accepted_implies_paper_or_residual_failure
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (input : AuthoritativeInput (shape := shape) (State := State)
      (publicRingColumns := publicRingColumns) (verifierRows := verifierRows)
      (publicFits := publicFits))
    (certificate : Certificate input)
    (accepted : ProductionVerifierAccepts input certificate) :
    Semantics.Paper.Holds input.data ∨
      FeFailure input certificate ∨ NcFailure input certificate :=
  accepted_implies_paper_or_algebraic_failure noZeroDivisors input certificate
    accepted

end Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionRefinement.ResidualAlignment
