import NightstreamFPrime.Export.Stage1.PiCCSStoredWitnessCheck
import NightstreamFPrime.Lifecycle.Relation

/-!
The selected NIFS source return supplies the exact source CCS and CE
openings. Fresh public fields come from the verifier-owned input; the returned
value supplies its private tail and all sixteen running assignments.
This is a value and relation bridge, with no probability or work assumption.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.HyperNovaSource

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open StrongReduction ConcreteCarrier UnifiedSources
open NightstreamFPrime.Lifecycle

open PiCCSStoredWitnessCheck (carrier)
open Poseidon2HashChainV1Setup (productionAjtaiKey)

/-- The returned running vectors keep their existing source order. -/
def runningWitness
    (values : WitnessProjection.SourceWitness productionShape carrier) :
    Fin productionShape.runningCount → Phi81Relation.Assignment carrier :=
  fun index => (values.running.get index).get

/-- Reattach the exact public prefix to the returned fresh private tail. -/
def freshWitness (input : PiCCSInputCheck.Input)
    (values : WitnessProjection.SourceWitness productionShape carrier) :
    Phi81Relation.Assignment carrier :=
  WitnessProjection.joinFresh
    ((PiCCSInputCheck.fresh input).publicInputs ⟨0, by decide⟩)
    (values.fresh.get ⟨0, by decide⟩)

section SemanticAgreement

variable {logicalWidth : Nat}
    {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}

private abbrev paperSemantics
    (key : PaperAlgebra.AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits)) :=
  paperRelationSemantics (shape := productionShape)
    (blockCount := Phi81ColumnLayout.blockCount
      (Phi81CarrierLayout.carrierWidth logicalWidth))
    baseOps extensionOps K.embed (PaperAlgebra.openingMaps key)

private theorem openingMaps_projection
    (key : PaperAlgebra.AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    CheckedWitnessExtraction.openingMaps
        (carrier := PaperAlgebra.FullShape logicalWidth publicFits)
        (PaperAlgebra.openingMaps key).commit = PaperAlgebra.openingMaps key := by
  rfl

private theorem ccsAgreement
    (key : PaperAlgebra.AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (statement : CCS.Instance (PaperAlgebra.Structure logicalWidth)
      (PaperAlgebra.PublicInput
        (logicalWidth := logicalWidth) (publicFits := publicFits))
      PaperAlgebra.Commitment)
    (assignment : PaperAlgebra.Assignment
      (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    CCS.Holds (paperSemantics key) productionGlobalParams statement assignment ↔
      CCS.Holds (PaperAlgebra.semantics key) productionGlobalParams statement assignment := by
  exact and_congr (PaperAlgebra.openingAgreement key
    (statement.stage.bound productionGlobalParams) statement.commitment
    statement.publicInput assignment) Iff.rfl

private theorem ceAgreement
    (key : PaperAlgebra.AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (layout : UnifiedSources.ColumnLayout productionShape.cubeVariables
      (Phi81CarrierLayout.carrierWidth logicalWidth))
    (system : Phi81Relation.Structure (PaperAlgebra.FullShape logicalWidth publicFits))
    (statement : CE.Instance (PaperAlgebra.Structure logicalWidth)
      (PaperAlgebra.PublicInput
        (logicalWidth := logicalWidth) (publicFits := publicFits))
      PaperAlgebra.Point PaperAlgebra.Evaluation PaperAlgebra.Commitment)
    (assignment : PaperAlgebra.Assignment
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (source : statement.constraintSystem = PaperAlgebra.relationSource layout system) :
    CE.Holds (paperSemantics key) productionGlobalParams statement assignment ↔
      CE.Holds (PaperAlgebra.semantics key) productionGlobalParams statement assignment := by
  unfold CE.Holds
  rw [PaperAlgebra.openingAgreement]
  change (_ ∧ True ∧ _ = _) ↔ (_ ∧ True ∧ _ = _)
  rw [source, PaperAlgebra.evaluations_eq_paper key layout system]
  rfl

private theorem runningAgreement
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (key : PaperAlgebra.AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (running : Lifecycle.Running
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (index : Fin productionShape.runningCount)
    (assignment : PaperAlgebra.Assignment
      (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    CE.Holds (paperSemantics key) productionGlobalParams
        (Lifecycle.runningStatement relation running index) assignment ↔
      CE.Holds (PaperAlgebra.semantics key) productionGlobalParams
        (Lifecycle.runningStatement relation running index) assignment :=
  ceAgreement key
    (NightstreamFPrime.Spec.Folding.PiCCS.CanonicalRowLayout.layout cubeVariables
      (Phi81CarrierLayout.carrierWidth logicalWidth) relation.cubeFits)
    relation.system (Lifecycle.runningStatement relation running index) assignment rfl

end SemanticAgreement

private theorem selectedOpeningMaps :
    CheckedWitnessExtraction.openingMaps (carrier := carrier) PiCCSStoredWitnessCheck.commit =
      PaperAlgebra.openingMaps
        (logicalWidth := PiDECInputCheck.logicalWidth)
        (publicFits := PiDECInputCheck.publicFits) productionAjtaiKey :=
  openingMaps_projection
    (logicalWidth := PiDECInputCheck.logicalWidth)
    (publicFits := PiDECInputCheck.publicFits) productionAjtaiKey

private theorem addCases_running {shape : Shape} {Value : Type*}
    (fresh : Fin shape.freshCount → Value) (running : Fin shape.runningCount → Value)
    (index : Fin shape.runningCount) :
    Fin.addCases fresh running (runningSourceIndex index) = running index :=
  Fin.addCases_right (m := shape.freshCount) (n := shape.runningCount)
    (motive := fun _ => Value) (left := fresh) (right := running) index

private theorem freshInstance_eq (input : PiCCSInputCheck.Input)
    (index : Fin productionShape.freshCount) :
    SourceMembership.freshInstance (PiCCSStoredWitnessCheck.statement input) index =
      Lifecycle.freshStatement PiDECInputCheck.relation (PiCCSInputCheck.fresh input) := by
  have same : index = ⟨0, by decide⟩ := by
    apply Fin.ext
    have bound := index.isLt
    change index.val < 1 at bound
    change index.val = 0
    omega
  subst index
  simp only [SourceMembership.freshInstance, PiCCSStoredWitnessCheck.statement,
    PiCCSInputCheck.outputCommitments, PiCCSInputCheck.outputPublicInputs]
  rfl

private theorem ceInstance_ext {S P R E C : Type*}
    (left right : CE.Instance S P R E C)
    (source : left.constraintSystem = right.constraintSystem)
    (commitment : left.commitment = right.commitment)
    (publicInput : left.publicInput = right.publicInput)
    (point : left.point = right.point)
    (evaluations : left.evaluations = right.evaluations)
    (stage : left.stage = right.stage) : left = right := by
  cases left
  cases right
  cases source
  cases commitment
  cases publicInput
  cases point
  cases evaluations
  cases stage
  rfl

private theorem runningInstance_eq (input : PiCCSInputCheck.Input)
    (index : Fin productionShape.runningCount) :
    SourceMembership.runningInstance (PiCCSStoredWitnessCheck.statement input) index =
      Lifecycle.runningStatement PiDECInputCheck.relation (PiCCSInputCheck.running input) index := by
  refine ceInstance_ext _ _ ?_ ?_ ?_ ?_ ?_ ?_
  · rfl
  · change Fin.addCases (PiCCSInputCheck.fresh input).commitments
        (PiCCSInputCheck.running input).commitments (runningSourceIndex index) =
      (PiCCSInputCheck.running input).commitments index
    exact addCases_running (shape := productionShape) (Value := PaperAlgebra.Commitment)
      (PiCCSInputCheck.fresh input).commitments (PiCCSInputCheck.running input).commitments index
  · change Fin.addCases (PiCCSInputCheck.fresh input).publicInputs
        (PiCCSInputCheck.running input).publicInputs (runningSourceIndex index) =
      (PiCCSInputCheck.running input).publicInputs index
    exact addCases_running (shape := productionShape) (Value := PiCCSInputCheck.PublicInput)
      (PiCCSInputCheck.fresh input).publicInputs (PiCCSInputCheck.running input).publicInputs index
  · rfl
  · rfl
  · rfl

private theorem reconstruct_fresh (input : PiCCSInputCheck.Input)
    (values : WitnessProjection.SourceWitness productionShape carrier)
    (index : Fin productionShape.freshCount) :
    (WitnessProjection.reconstruct
      (PiCCSStoredWitnessCheck.statement input).publicInputs values).assignments
        (freshSourceIndex index) = freshWitness input values := by
  rw [WitnessProjection.reconstruct_fresh]
  have same : index = ⟨0, by decide⟩ := by
    apply Fin.ext
    have bound := index.isLt
    change index.val < 1 at bound
    change index.val = 0
    omega
  subst index
  simp only [PiCCSStoredWitnessCheck.statement, PiCCSInputCheck.outputPublicInputs]
  rfl

/-- Every successful source result has exactly the source CCS and CE
memberships, with its returned running vectors and reconstructed fresh prefix.
The result remains explicit; no witness is selected from an existential. -/
theorem sourceReturned_iff_terminalHolds (input : PiCCSInputCheck.Input)
    (result : Option (WitnessProjection.SourceWitness productionShape carrier)) :
    CheckedWitnessExtraction.SourceReturned PiCCSStoredWitnessCheck.commit
        productionGlobalParams (PiCCSStoredWitnessCheck.statement input) result ↔
      ∃ values, result = some values ∧
        Lifecycle.TerminalHolds PiDECInputCheck.relation productionAjtaiKey
          (PiCCSInputCheck.running input) (runningWitness values)
          (PiCCSInputCheck.fresh input) (freshWitness input values) := by
  rw [CheckedWitnessExtraction.sourceReturned_iff_memberships
    (freshBound := (rfl : productionGlobalParams.b = 2)), selectedOpeningMaps]
  constructor
  · rintro ⟨values, returned, fresh, running⟩
    refine ⟨values, returned, ?_, ?_⟩
    · intro index
      have member := running index
      rw [runningInstance_eq, WitnessProjection.reconstruct_running] at member
      exact (runningAgreement
        (logicalWidth := PiDECInputCheck.logicalWidth) (publicFits := PiDECInputCheck.publicFits)
        PiDECInputCheck.relation productionAjtaiKey
        (PiCCSInputCheck.running input) index (runningWitness values index)).mp member
    · have member := fresh ⟨0, by decide⟩
      rw [freshInstance_eq, reconstruct_fresh] at member
      exact (ccsAgreement
        (logicalWidth := PiDECInputCheck.logicalWidth) (publicFits := PiDECInputCheck.publicFits)
        productionAjtaiKey
        (Lifecycle.freshStatement PiDECInputCheck.relation (PiCCSInputCheck.fresh input))
        (freshWitness input values)).mp member
  · rintro ⟨values, returned, running, fresh⟩
    refine ⟨values, returned, ?_, ?_⟩
    · intro index
      rw [freshInstance_eq, reconstruct_fresh]
      exact (ccsAgreement
        (logicalWidth := PiDECInputCheck.logicalWidth) (publicFits := PiDECInputCheck.publicFits)
        productionAjtaiKey
        (Lifecycle.freshStatement PiDECInputCheck.relation (PiCCSInputCheck.fresh input))
        (freshWitness input values)).mpr fresh
    · intro index
      rw [runningInstance_eq, WitnessProjection.reconstruct_running]
      exact (runningAgreement
        (logicalWidth := PiDECInputCheck.logicalWidth) (publicFits := PiDECInputCheck.publicFits)
        PiDECInputCheck.relation productionAjtaiKey
        (PiCCSInputCheck.running input) index (runningWitness values index)).mpr (running index)

/-- The actual checked NIFS source return identifies the complete source
CCS and CE memberships consumed by terminal verification. -/
theorem finishValue_source_iff_terminalHolds (input : PiCCSInputCheck.Input)
    (outcome : CheckedWitnessExtraction.StoredOutcome productionShape carrier) :
    CheckedWitnessExtraction.SourceReturned PiCCSStoredWitnessCheck.commit
        productionGlobalParams (PiCCSStoredWitnessCheck.statement input)
        (PiCCSStoredWitnessCheck.finishValue input outcome) ↔
      ∃ values, PiCCSStoredWitnessCheck.finishValue input outcome = some values ∧
        Lifecycle.TerminalHolds PiDECInputCheck.relation productionAjtaiKey
          (PiCCSInputCheck.running input) (runningWitness values)
          (PiCCSInputCheck.fresh input) (freshWitness input values) :=
  sourceReturned_iff_terminalHolds input (PiCCSStoredWitnessCheck.finishValue input outcome)

end NightstreamFPrime.Export.Stage1.HyperNovaSource
