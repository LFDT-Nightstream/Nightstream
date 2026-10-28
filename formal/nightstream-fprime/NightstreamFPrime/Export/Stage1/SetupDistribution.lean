import NightstreamFPrime.Spec.AjtaiSetupV1.ReductionBias
import NightstreamFPrime.Spec.AjtaiSetupV1.Programming
import NightstreamFPrime.Export.Stage1.Poseidon2HashChainV1Setup

/-! The ideal modular-reduction error budget at the exact selected key size,
and the ideal-model binding bound of `AjtaiSetupV1.Programming` at the
production key. Both require independent uniform 256-bit chunks, which is
premise P1 (SHAKE128 as a random oracle). This file makes no claim about
SHAKE128 itself. -/

namespace NightstreamFPrime.Export.Stage1.Poseidon2HashChainV1Setup

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

def setupCoefficientCount : Nat := verifierRows * messageColumns * ringDegree

theorem setupCoefficientCount_eq : setupCoefficientCount = 967203072 := by
  rw [setupCoefficientCount, verifierRows_eq, messageColumns_eq]
  rfl

def idealReductionErrorBudget : ℚ :=
  setupCoefficientCount * ((2 ^ 256 % goldilocksModulus : Nat) : ℚ) / 2 ^ 256

theorem idealReductionErrorBudget_eq :
    idealReductionErrorBudget = (967203072 : ℚ) * 4294967295 / 2 ^ 256 := by
  rw [idealReductionErrorBudget, setupCoefficientCount_eq,
    AjtaiSetupV1.ReductionBias.wide_remainder_eq]
  rfl

theorem idealReductionErrorBudget_lt : idealReductionErrorBudget < 1 / 2 ^ 191 := by
  rw [idealReductionErrorBudget_eq]
  norm_num

open AjtaiSetupV1.Programming in
theorem programmingError_eq :
    (Fintype.card (KeyIndex verifierRows messageColumns) : ℚ) * (2 * 4294967295 / 2 ^ 256) =
      2 * idealReductionErrorBudget := by
  rw [idealReductionErrorBudget_eq, Fintype.card_prod, Fintype.card_prod, Fintype.card_fin,
    Fintype.card_fin, Fintype.card_fin, verifierRows_eq, messageColumns_eq]
  norm_num [ringDegree]

open AjtaiSetupV1.Programming in
theorem programmingError_lt :
    (Fintype.card (KeyIndex verifierRows messageColumns) : ℚ) * (2 * 4294967295 / 2 ^ 256) <
      1 / 2 ^ 190 := by
  rw [programmingError_eq, idealReductionErrorBudget_eq]
  norm_num

theorem productionKey_eq_chunks :
    productionAjtaiKey =
      AjtaiSetupV1.Programming.keyOf (AjtaiSetupV1.Programming.residues
        (AjtaiSetupV1.Programming.setupChunks productionSeedBytes)) :=
  AjtaiSetupV1.Programming.verifierKey_eq productionSetup

abbrev commitmentShape : Phi81Relation.Shape :=
  Lifecycle.PaperAlgebra.FullShape
    (PerApplicationFixedPoint.logicalWidth Poseidon2HashChainV1Package.application)
    (PerApplicationFixedPoint.publicFits Poseidon2HashChainV1Package.application)

open Classical AjtaiSetupV1.Programming in
theorem production_binding_lt_solver {Extra : Type} [Fintype Extra] [Nonempty Extra]
    (bound : Nat)
    (attack : (KeyIndex verifierRows (Phi81ColumnLayout.blockCount commitmentShape.carrierWidth) →
        Chunk) × Extra →
      Phi81Relation.Assignment commitmentShape × Phi81Relation.Assignment commitmentShape) :
    real (fun view => IsCollision bound (keyOf (residues view.1)) (attack view)) <
      solverSuccess (shape := commitmentShape) (rows := verifierRows) bound attack +
        1 / 2 ^ 190 := by
  have ideal := binding_le_solver (shape := commitmentShape) (rows := verifierRows) bound attack
  have error : (Fintype.card
      (KeyIndex verifierRows (Phi81ColumnLayout.blockCount commitmentShape.carrierWidth)) : ℚ) *
        (2 * 4294967295 / 2 ^ 256) < 1 / 2 ^ 190 :=
    programmingError_lt
  linarith

end NightstreamFPrime.Export.Stage1.Poseidon2HashChainV1Setup
