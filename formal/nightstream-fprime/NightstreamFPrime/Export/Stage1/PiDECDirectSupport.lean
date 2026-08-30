import NightstreamFPrime.Export.Stage1.Data
import NightstreamFPrime.Export.Stage1.PiDECArithmetic
import NightstreamFPrime.Layout.R1CS.Support
import NightstreamFPrime.Layout.Stage1.PiDECSourceSupportData
import NightstreamFPrime.Lifecycle.PiDEC.v1_1.Support.VariableSupport

/-!
Owns exact retained-source support for the canonical PiDEC ordinary rows.
The source predicate is selected by the Stage 1 layout; the generic PiDEC
support lemmas prove that every child reads only that predicate.
-/

namespace NightstreamFPrime.Export.Stage1.PiDECDirectSupport

open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Layout.Stage1.PiDECSourceSupport
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiDEC.v1_1
open NightstreamFPrime.Lifecycle.PiRLC.v1_1
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

private theorem combinationOutput_varsSatisfy
    {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : PiRLC.v1_1.CombinationFamily.Interface blockCount cellCount)
    (offset : Nat) (block : Fin blockCount) (lane : Fin ringDegree)
    (cell : Fin cellCount)
    (includes : ∀ column,
      InRange
        (PiRLC.v1_1.CombinationFamily.stepOffset offset
          PiRLC.v1_1.CombinationFamily.finalSource.val blockCount cellCount)
        (PiRLC.v1_1.CombinationStep.privateCount blockCount cellCount) column →
      Source column) :
    (PiRLC.v1_1.CombinationFamily.output interface offset block lane cell
      ).VarsSatisfy Source := by
  simp only [PiRLC.v1_1.CombinationFamily.output,
    PiRLC.v1_1.CombinationStep.output, Expr.VarsSatisfy]
  apply includes
  constructor
  · omega
  · have bound :=
      (PiRLC.v1_1.CombinationStep.indexOf block lane cell).isLt
    omega

theorem parentCommitment_supported
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (row : Fin productionProfile.commitmentWidth) (lane : Fin ringDegree) :
    (((PiDECInputs.interface logicalWidth publicFits).parent
      PiDECInputs.phaseOffset).commitment row lane).VarsSatisfy Source := by
  change (PiRLC.v1_1.CommitmentCombination.output
    (PiRLC.v1_1.Formal.commitmentInterface
      (PiDECInputs.piRlcSharedInterface logicalWidth publicFits))
    PiRLCStarts.commitmentLogicalStart row lane).VarsSatisfy Source
  unfold PiRLC.v1_1.CommitmentCombination.output
  apply combinationOutput_varsSatisfy
  intro column inside
  exact parent_source column (parentCommitment column inside)

theorem parentPublicInput_supported
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (column : Fin (FullShape logicalWidth publicFits).publicWidth) :
    (((PiDECInputs.interface logicalWidth publicFits).parent
      PiDECInputs.phaseOffset).publicInput column).VarsSatisfy Source := by
  change (PiRLC.v1_1.PublicInputCombination.output
    (PiRLC.v1_1.Formal.publicInputInterface
      (PiDECInputs.piRlcSharedInterface logicalWidth publicFits))
    PiRLCStarts.publicInputLogicalStart column).VarsSatisfy Source
  unfold PiRLC.v1_1.PublicInputCombination.output
  apply combinationOutput_varsSatisfy
  intro source inside
  exact parent_source source (parentPublicInput source inside)

theorem parentEvalK_supported
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (coefficient : Fin productionShape.coefficientCount) :
    RingKRecomposition.KSupported Source
      (((PiDECInputs.interface logicalWidth publicFits).parent
        PiDECInputs.phaseOffset).evaluation.eval_K coefficient) := by
  constructor
  · change (PiRLC.v1_1.CombinationFamily.output
      (PiRLC.v1_1.RingKCombination.familyInterface
        (PiRLC.v1_1.EvalKCombination.ringInterface
          (PiRLC.v1_1.Formal.evalKInterface
            (PiDECInputs.piRlcSharedInterface logicalWidth publicFits))))
      PiRLCStarts.evalKLogicalStart PiRLC.v1_1.EvalKCombination.block
      (Fin.cast PiRLC.v1_1.EvalKCombination.coefficientCount_eq coefficient)
      PiRLC.v1_1.RingKCombination.c0Cell).VarsSatisfy Source
    apply combinationOutput_varsSatisfy
    intro source inside
    exact parent_source source (parentEvalK source inside)
  · change (PiRLC.v1_1.CombinationFamily.output
      (PiRLC.v1_1.RingKCombination.familyInterface
        (PiRLC.v1_1.EvalKCombination.ringInterface
          (PiRLC.v1_1.Formal.evalKInterface
            (PiDECInputs.piRlcSharedInterface logicalWidth publicFits))))
      PiRLCStarts.evalKLogicalStart PiRLC.v1_1.EvalKCombination.block
      (Fin.cast PiRLC.v1_1.EvalKCombination.coefficientCount_eq coefficient)
      PiRLC.v1_1.RingKCombination.c1Cell).VarsSatisfy Source
    apply combinationOutput_varsSatisfy
    intro source inside
    exact parent_source source (parentEvalK source inside)

theorem parentEvalA_supported
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (matrix : Fin productionShape.matrixCount)
    (coefficient : Fin productionShape.coefficientCount) :
    RingKRecomposition.KSupported Source
      (((PiDECInputs.interface logicalWidth publicFits).parent
        PiDECInputs.phaseOffset).evaluation.eval_A matrix coefficient) := by
  constructor
  · change (PiRLC.v1_1.CombinationFamily.output
      (PiRLC.v1_1.RingKCombination.familyInterface
        (PiRLC.v1_1.EvalACombination.ringInterface
          (PiRLC.v1_1.Formal.evalAInterface
            (PiDECInputs.piRlcSharedInterface logicalWidth publicFits))))
      PiRLCStarts.evalALogicalStart matrix
      (Fin.cast PiRLC.v1_1.EvalKCombination.coefficientCount_eq coefficient)
      PiRLC.v1_1.RingKCombination.c0Cell).VarsSatisfy Source
    apply combinationOutput_varsSatisfy
    intro source inside
    exact parent_source source (parentEvalA source inside)
  · change (PiRLC.v1_1.CombinationFamily.output
      (PiRLC.v1_1.RingKCombination.familyInterface
        (PiRLC.v1_1.EvalACombination.ringInterface
          (PiRLC.v1_1.Formal.evalAInterface
            (PiDECInputs.piRlcSharedInterface logicalWidth publicFits))))
      PiRLCStarts.evalALogicalStart matrix
      (Fin.cast PiRLC.v1_1.EvalKCombination.coefficientCount_eq coefficient)
      PiRLC.v1_1.RingKCombination.c1Cell).VarsSatisfy Source
    apply combinationOutput_varsSatisfy
    intro source inside
    exact parent_source source (parentEvalA source inside)

theorem messageCommitment_supported
    (child : Spec.Phi81Relation.PiDECAlgebra.Radix.ChildIndex)
    (row : Fin productionProfile.commitmentWidth) (lane : Fin ringDegree) :
    ((PiDECInputs.message child).commitment row lane).VarsSatisfy Source := by
  simp only [PiDECInputs.message, PiDECInputs.childCommitment,
    Expr.VarsSatisfy]
  apply proof_source
  unfold InRange
  have childBound := child.isLt
  have rowBound := row.isLt
  have laneBound := lane.isLt
  norm_num [productionGlobalParams, productionProfile] at childBound rowBound
  norm_num [ringDegree] at laneBound
  norm_num [PiDECInputs.childCommitmentStart,
    PiDECInputs.commitmentInputStart, PiDECInputs.proofInputStart,
    PiDECInputs.proofInputColumnCount, PiDECInputs.childCount,
    PiDECInputs.commitmentWordsPerChild, PiDECInputs.evalKWordsPerChild,
    PiDECInputs.evalAWordsPerChild, PiDECInputs.publicInputWordsPerChild,
    ringDegree]
  omega

theorem messageEvalK_supported
    (child : Spec.Phi81Relation.PiDECAlgebra.Radix.ChildIndex)
    (coefficient : Fin productionShape.coefficientCount) :
    RingKRecomposition.KSupported Source
      ((PiDECInputs.message child).evaluation.eval_K coefficient) := by
  simp only [PiDECInputs.message, PiDECInputs.childEvalK,
    RingKRecomposition.KSupported, Expr.VarsSatisfy]
  constructor <;> apply proof_source <;> unfold InRange
  all_goals
    have childBound := child.isLt
    have coefficientBound := coefficient.isLt
    change child.val < 16 at childBound
    change coefficient.val < 54 at coefficientBound
    norm_num [PiDECInputs.childEvalKStart, PiDECInputs.evalKInputStart,
      PiDECInputs.commitmentInputStart, PiDECInputs.proofInputStart,
      PiDECInputs.proofInputColumnCount, PiDECInputs.childCount,
      PiDECInputs.commitmentWordsPerChild, PiDECInputs.evalKWordsPerChild,
      PiDECInputs.evalAWordsPerChild, PiDECInputs.publicInputWordsPerChild]
    omega

theorem messageEvalA_supported
    (child : Spec.Phi81Relation.PiDECAlgebra.Radix.ChildIndex)
    (matrix : Fin productionShape.matrixCount)
    (coefficient : Fin productionShape.coefficientCount) :
    RingKRecomposition.KSupported Source
      ((PiDECInputs.message child).evaluation.eval_A matrix coefficient) := by
  simp only [PiDECInputs.message, PiDECInputs.childEvalA,
    RingKRecomposition.KSupported, Expr.VarsSatisfy]
  constructor <;> apply proof_source <;> unfold InRange
  all_goals
    have childBound := child.isLt
    have matrixBound := matrix.isLt
    have coefficientBound := coefficient.isLt
    change child.val < 16 at childBound
    change matrix.val < 14 at matrixBound
    change coefficient.val < 54 at coefficientBound
    norm_num [PiDECInputs.childEvalAStart, PiDECInputs.evalAInputStart,
      PiDECInputs.evalKInputStart, PiDECInputs.commitmentInputStart,
      PiDECInputs.proofInputStart, PiDECInputs.proofInputColumnCount,
      PiDECInputs.childCount, PiDECInputs.commitmentWordsPerChild,
      PiDECInputs.evalKWordsPerChild, PiDECInputs.evalAWordsPerChild,
      PiDECInputs.publicInputWordsPerChild]
    omega

theorem digit_supported
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (child : Spec.Phi81Relation.PiDECAlgebra.Radix.ChildIndex)
    (coordinate : Fin
      (PiDEC.v1_1.PublicInputSplit.coordinateCount logicalWidth publicFits)) :
    ((PiDECInputs.interface logicalWidth publicFits).digit PiDECInputs.phaseOffset
      child coordinate).VarsSatisfy Source := by
  simp only [PiDECInputs.interface, PiDECInputs.childPublicInput,
    Expr.VarsSatisfy]
  apply proof_source
  unfold InRange
  have childBound := child.isLt
  have coordinateBound := coordinate.isLt
  norm_num [productionGlobalParams] at childBound
  norm_num [PiDEC.v1_1.PublicInputSplit.coordinateCount_eq] at coordinateBound
  norm_num [PiDECInputs.childPublicInputStart, PiDECInputs.publicInputStart,
    PiDECInputs.evalAInputStart, PiDECInputs.evalKInputStart,
    PiDECInputs.commitmentInputStart, PiDECInputs.proofInputStart,
    PiDECInputs.proofInputColumnCount, PiDECInputs.childCount,
    PiDECInputs.commitmentWordsPerChild, PiDECInputs.evalKWordsPerChild,
    PiDECInputs.evalAWordsPerChild, PiDECInputs.publicInputWordsPerChild]
  omega

structure ConstraintsSupported (constraints : List Expr) : Prop where
  get : ∀ expression ∈ constraints, expression.VarsSatisfy Source

theorem publicConstraints_varsSatisfy
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth} :
    ConstraintsSupported
      (PiDECArithmetic.publicInputConstraints logicalWidth publicFits) := by
  constructor
  change ∀ expression ∈ flatConstraints (Circuit.ops
    (PiDEC.v1_1.PublicInputSplit.circuit
      (PiDEC.v1_1.Formal.publicInputInterface
        (PiDEC.v1_1.Formal.atOffset
          (PiDECInputs.interface logicalWidth publicFits)
          PiDECInputs.phaseOffset))).main PiDECInputs.phaseOffset), _
  apply PiDEC.v1_1.PublicInputSplit.flatConstraints_varsSatisfy Source
  · intro coordinate
    exact parentPublicInput_supported coordinate
  · intro child coordinate
    exact digit_supported child coordinate
  · intro index lower upper
    apply logical_source
    unfold InRange
    constructor
    · simpa [PiDECStarts.phaseLogicalStart] using lower
    · rw [PiDEC.v1_1.PublicInputSplit.logicalPrivateCount_eq] at upper
      simpa [PiDECStarts.phaseLogicalStart] using upper

theorem commitmentConstraints_varsSatisfy
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth} :
    ConstraintsSupported
      (PiDECArithmetic.commitmentConstraints logicalWidth publicFits) := by
  constructor
  change ∀ expression ∈ flatConstraints (Circuit.ops
    (PiDEC.v1_1.CommitmentRecomposition.circuit
      (PiDEC.v1_1.Formal.commitmentInterface
        (PiDEC.v1_1.Formal.atOffset
          (PiDECInputs.interface logicalWidth publicFits)
          PiDECInputs.phaseOffset))).main
      (PiDEC.v1_1.Formal.commitmentOffset PiDECInputs.phaseOffset)), _
  rw [PiDEC.v1_1.RadixRecomposition.circuit_ops]
  apply PiDEC.v1_1.RadixRecomposition.flatConstraints_varsSatisfy Source
  · intro coordinate
    exact parentCommitment_supported
      (logicalWidth := logicalWidth) (publicFits := publicFits)
      (PiDEC.v1_1.CommitmentRecomposition.coordinates coordinate).1
      (PiDEC.v1_1.CommitmentRecomposition.coordinates coordinate).2
  · intro child coordinate
    exact messageCommitment_supported child
      (PiDEC.v1_1.CommitmentRecomposition.coordinates coordinate).1
      (PiDEC.v1_1.CommitmentRecomposition.coordinates coordinate).2

theorem evalKConstraints_varsSatisfy
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth} :
    ConstraintsSupported
      (PiDECArithmetic.evalKConstraints logicalWidth publicFits) := by
  constructor
  change ∀ expression ∈ flatConstraints (Circuit.ops
    (PiDEC.v1_1.RingKRecomposition.circuit
      (PiDEC.v1_1.EvalKRecomposition.ringInterface
        (PiDEC.v1_1.Formal.evalKInterface
          (PiDEC.v1_1.Formal.atOffset
            (PiDECInputs.interface logicalWidth publicFits)
            PiDECInputs.phaseOffset)))).main
      (PiDEC.v1_1.Formal.evalKOffset PiDECInputs.phaseOffset)), _
  apply PiDEC.v1_1.RingKRecomposition.flatConstraints_varsSatisfy Source
  · intro _block lane
    exact parentEvalK_supported
      (PiDEC.v1_1.EvalKRecomposition.coefficient lane)
  · intro child _block lane
    exact messageEvalK_supported child
      (PiDEC.v1_1.EvalKRecomposition.coefficient lane)

theorem evalAConstraints_varsSatisfy
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth} :
    ConstraintsSupported
      (PiDECArithmetic.evalAConstraints logicalWidth publicFits) := by
  constructor
  change ∀ expression ∈ flatConstraints (Circuit.ops
    (PiDEC.v1_1.RingKRecomposition.circuit
      (PiDEC.v1_1.EvalARecomposition.ringInterface
        (PiDEC.v1_1.Formal.evalAInterface
          (PiDEC.v1_1.Formal.atOffset
            (PiDECInputs.interface logicalWidth publicFits)
            PiDECInputs.phaseOffset)))).main
      (PiDEC.v1_1.Formal.evalAOffset PiDECInputs.phaseOffset)), _
  apply PiDEC.v1_1.RingKRecomposition.flatConstraints_varsSatisfy Source
  · intro matrix lane
    exact parentEvalA_supported matrix
      (PiDEC.v1_1.EvalKRecomposition.coefficient lane)
  · intro child matrix lane
    exact messageEvalA_supported child matrix
      (PiDEC.v1_1.EvalKRecomposition.coefficient lane)

theorem productionPublicConstraints_supported
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth} :
    ConstraintsSupported
      (PiDECArithmetic.publicInputConstraints logicalWidth publicFits) :=
  publicConstraints_varsSatisfy
    (logicalWidth := logicalWidth) (publicFits := publicFits)

theorem productionCommitmentConstraints_supported
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth} :
    ConstraintsSupported
      (PiDECArithmetic.commitmentConstraints logicalWidth publicFits) :=
  commitmentConstraints_varsSatisfy
    (logicalWidth := logicalWidth) (publicFits := publicFits)

theorem productionEvalKConstraints_supported
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth} :
    ConstraintsSupported
      (PiDECArithmetic.evalKConstraints logicalWidth publicFits) :=
  evalKConstraints_varsSatisfy
    (logicalWidth := logicalWidth) (publicFits := publicFits)

theorem productionEvalAConstraints_supported
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth} :
    ConstraintsSupported
      (PiDECArithmetic.evalAConstraints logicalWidth publicFits) :=
  evalAConstraints_varsSatisfy
    (logicalWidth := logicalWidth) (publicFits := publicFits)

end NightstreamFPrime.Export.Stage1.PiDECDirectSupport
