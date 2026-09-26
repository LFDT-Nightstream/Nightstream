import NightstreamFPrime.Export.Stage1.PiDECCommitmentHonestMessages
import NightstreamFPrime.Export.Stage1.PiDECEvaluationHonestMessages
import NightstreamFPrime.Export.Stage1.HyperNovaInput

/-! Assemble the existing D kernel fields and identify their exact child
openings. All statement algebra is proved with symbolic relation, key, parent,
and fields before the selected block/evaluation kernels are substituted. -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiDECComputedChildren

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.Nifs.StoredAssignmentArithmetic (StoredAssignment view)
open NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Layout.Stage1
open Poseidon2HashChainV1Package (application fits)
open Poseidon2HashChainV1Setup (productionSetup productionAjtaiKey)

private abbrev shape := FullShape
  (PerApplicationFixedPoint.logicalWidth application)
  (PerApplicationFixedPoint.publicFits application)

abbrev Children := Vector (StoredAssignment shape.carrierWidth) productionGlobalParams.k

private def fieldMessages
    (commitments : Fin productionGlobalParams.k → PaperAlgebra.Commitment)
    (evaluations : Fin productionGlobalParams.k → PaperAlgebra.Evaluation) :
    Fin productionGlobalParams.k →
      PiDEC.PaperVerifier.ChildMessage PaperAlgebra.Evaluation PaperAlgebra.Commitment :=
  fun child => ⟨commitments child, #[evaluations child]⟩

private def assemble
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (parent : CE.Instance (PaperAlgebra.Structure logicalWidth)
      (PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
      PaperAlgebra.Point PaperAlgebra.Evaluation PaperAlgebra.Commitment)
    (commitments : Fin productionGlobalParams.k → PaperAlgebra.Commitment)
    (evaluations : Fin productionGlobalParams.k → PaperAlgebra.Evaluation) :
    Running (logicalWidth := logicalWidth) (publicFits := publicFits) where
  point := parent.point
  commitments := commitments
  publicInputs := (publicInputSplit ajtai).split parent.publicInput
  evaluations := evaluations

private theorem assembled_statement
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (parent : CE.Instance (PaperAlgebra.Structure logicalWidth)
      (PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
      PaperAlgebra.Point PaperAlgebra.Evaluation PaperAlgebra.Commitment)
    (commitments : Fin productionGlobalParams.k → PaperAlgebra.Commitment)
    (evaluations : Fin productionGlobalParams.k → PaperAlgebra.Evaluation)
    (system : parent.constraintSystem = Lifecycle.PiRLC.v1_1.InputBinding.relationSource relation)
    (child : Fin productionGlobalParams.k) :
    Lifecycle.runningStatement relation (assemble ajtai parent commitments evaluations) child =
      PiDEC.PaperVerifier.children (publicInputSplit ajtai)
        ⟨parent, fieldMessages commitments evaluations⟩ child := by
  simp only [Lifecycle.runningStatement, assemble, fieldMessages,
    PiDEC.PaperVerifier.children, system, PaperAlgebra.relationSource,
    Lifecycle.PiRLC.v1_1.InputBinding.relationSource]

private theorem assembled_openings
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (parent : CE.Instance (PaperAlgebra.Structure logicalWidth)
      (PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
      PaperAlgebra.Point PaperAlgebra.Evaluation PaperAlgebra.Commitment)
    (parentWitness : PaperAlgebra.Assignment (logicalWidth := logicalWidth) (publicFits := publicFits))
    (commitments : Fin productionGlobalParams.k → PaperAlgebra.Commitment)
    (evaluations : Fin productionGlobalParams.k → PaperAlgebra.Evaluation)
    (system : parent.constraintSystem = Lifecycle.PiRLC.v1_1.InputBinding.relationSource relation)
    (combined : parent.stage = .combined)
    (valid : CE.Holds (semantics ajtai) productionGlobalParams parent parentWitness)
    (messagesEq : fieldMessages commitments evaluations =
      PiDEC.PaperVerifier.honestMessages (piDecAlgebra ajtai) parent parentWitness) :
    ∀ child, CE.Holds (semantics ajtai) productionGlobalParams
      (Lifecycle.runningStatement relation (assemble ajtai parent commitments evaluations) child)
      ((piDecAlgebra ajtai).splitAssignment parentWitness child) := by
  have honest := (PiDEC.PaperVerifier.complete (semantics ajtai) productionGlobalParams
    (piDecAlgebra ajtai) (publicInputSplit ajtai) (evaluationArity ajtai)
    parent parentWitness combined valid).2
  have attemptEq :
      (⟨parent, fieldMessages commitments evaluations⟩ :
        PiDEC.PaperVerifier.Attempt _ _ _ _ _ productionGlobalParams) =
      PiDEC.PaperVerifier.honestAttempt (piDecAlgebra ajtai) parent parentWitness := by
    rw [messagesEq]
    rfl
  intro child
  rw [assembled_statement relation ajtai parent commitments evaluations system child, attemptEq]
  exact honest child

private def commitments (children : Children) :
    Fin productionGlobalParams.k → PaperAlgebra.Commitment :=
  fun child row =>
    (PiDECCommitmentFold.sum fun block =>
      (PiDECCommitmentBlock.contributions productionSetup row block
        (PiDECCommitmentFold.childBlocks (shape := shape) children block)).get child).get

/-- Exact existing block commitment and selected evaluation kernels. -/
def result (parent : PiDECInputCheck.ParentValues) (children : Children) :
    Running (logicalWidth := shape.logicalWidth) (publicFits := shape.publicFits) :=
  assemble productionAjtaiKey (PiDECInputCheck.parent parent)
    (commitments children) (PiDECEvaluationHonestMessages.family children parent.point)

def messages (parent : PiDECInputCheck.ParentValues) (children : Children) :
    PiDECInputCheck.Messages :=
  HyperNovaInput.runningInput (result parent children)

def witnesses (children : Children) : Stage1.Terminal.RunningWitness
    (logicalWidth := shape.logicalWidth) (publicFits := shape.publicFits) :=
  fun child => view (children.get child)

private theorem message_ext
    (left right : PiDEC.PaperVerifier.ChildMessage PaperAlgebra.Evaluation PaperAlgebra.Commitment)
    (commitment : left.commitment = right.commitment)
    (evaluations : left.evaluations = right.evaluations) : left = right := by
  cases left
  cases right
  cases commitment
  cases evaluations
  rfl

private theorem computed_messages
    (parent : PiDECInputCheck.ParentValues)
    (parentWitness : StoredAssignment shape.carrierWidth) (children : Children)
    (success : StoredSplit.splitChecked parentWitness = some children) :
    fieldMessages (commitments children) (PiDECEvaluationHonestMessages.family children parent.point) =
      PiDEC.PaperVerifier.honestMessages (piDecAlgebra productionAjtaiKey)
        (PiDECInputCheck.parent parent) (view parentWitness) := by
  funext child
  apply message_ext
  · change commitments children child = _
    funext row
    unfold commitments
    exact PiDECCommitmentHonestMessages.sum_contributions_honestMessages
      (PiDECInputCheck.parent parent) parentWitness children success child row
  · exact PiDECEvaluationHonestMessages.family_honestMessages
      parent parentWitness children success child

private theorem child_openings_checkedRelation
    (parent : PiDECInputCheck.ParentValues)
    (parentWitness : StoredAssignment shape.carrierWidth) (children : Children)
    (success : StoredSplit.splitChecked parentWitness = some children)
    (parentValid : CE.Holds (semantics productionAjtaiKey) productionGlobalParams
      (PiDECInputCheck.parent parent) (view parentWitness)) :
    ∀ child, CE.Holds (semantics productionAjtaiKey) productionGlobalParams
      (Lifecycle.runningStatement PiDECInputCheck.relation
        (PiCCSInputCheck.runningFromInput (messages parent children)) child)
      (witnesses children child) := by
  have system : (PiDECInputCheck.parent parent).constraintSystem =
      Lifecycle.PiRLC.v1_1.InputBinding.relationSource PiDECInputCheck.relation := by
    simp only [PiDECInputCheck.parent]
  have opening := assembled_openings PiDECInputCheck.relation productionAjtaiKey
    (PiDECInputCheck.parent parent) (view parentWitness) (commitments children)
    (PiDECEvaluationHonestMessages.family children parent.point) system rfl parentValid
    (computed_messages parent parentWitness children success)
  intro child
  rw [messages]
  erw [HyperNovaInput.runningFromInput_runningInput]
  have witness := (PiDECStoredSplitHonestWitness.splitChecked_honestWitness
    (PiDECInputCheck.parent parent) parentWitness children success child).1
  change CE.Holds _ _ _ (view (children.get child))
  rw [witness]
  exact opening child

/-- All child statements and openings use the actual computed kernel fields.
The only opening premise is validity of the exact combined parent. -/
theorem child_openings
    (parent : PiDECInputCheck.ParentValues)
    (parentWitness : StoredAssignment shape.carrierWidth) (children : Children)
    (success : StoredSplit.splitChecked parentWitness = some children)
    (parentValid : CE.Holds (semantics productionAjtaiKey) productionGlobalParams
      (PiDECInputCheck.parent parent) (view parentWitness)) :
    ∀ child, CE.Holds (semantics productionAjtaiKey) productionGlobalParams
      (Lifecycle.runningStatement (PerApplicationFixedPoint.relation application fits)
        (PiCCSInputCheck.runningFromInput (messages parent children)) child)
      (witnesses children child) := by
  have checked := child_openings_checkedRelation parent parentWitness children success parentValid
  rw [PiDECInputCheck.relation_eq_selected] at checked
  exact checked

end NightstreamFPrime.Export.Stage1.PiDECComputedChildren
