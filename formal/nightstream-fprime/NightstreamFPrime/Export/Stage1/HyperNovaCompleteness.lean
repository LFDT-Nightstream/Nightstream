import NightstreamFPrime.Lifecycle.Stage1.Terminal
import NightstreamFPrime.Lifecycle.PiDEC.v1_1.OutputWitnessConsumer
import NightstreamFPrime.Spec.Folding.Nifs.PaperNonInteractive.Completeness

/-!
Owns the honest production NIFS call from an accepted recursive terminal.
Its fresh and running openings supply source membership. The total verifier
sampler supplies the actual response after the causal PiCCS messages.
Verifier acceptance and every returned child opening are conclusions.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.HyperNovaCompleteness

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.StrongReduction
open ConcreteCarrier UnifiedSources
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open _root_.NightstreamFPrime.Spec.SumCheck.Finite
open NightstreamFPrime.Spec.HyperNova.Construction2.Paper

private theorem addCases_fresh {shape : Shape} {Value : Type*}
    (fresh : Fin shape.freshCount → Value) (running : Fin shape.runningCount → Value)
    (index : Fin shape.freshCount) :
    Fin.addCases fresh running (freshSourceIndex index) = fresh index :=
  Fin.addCases_left (m := shape.freshCount) (n := shape.runningCount)
    (motive := fun _ => Value) (left := fresh) (right := running) index

private theorem addCases_running {shape : Shape} {Value : Type*}
    (fresh : Fin shape.freshCount → Value) (running : Fin shape.runningCount → Value)
    (index : Fin shape.runningCount) :
    Fin.addCases fresh running (runningSourceIndex index) = running index :=
  Fin.addCases_right (m := shape.freshCount) (n := shape.runningCount)
    (motive := fun _ => Value) (left := fresh) (right := running) index

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

section SourceMembership

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}
  (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
  (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
  (running : Running (logicalWidth := logicalWidth) (publicFits := publicFits))
  (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))

private def sourceWitness {columns : Nat}
    (runningWitness : Fin productionShape.runningCount → PaperLinearAlgebra.Assignment F columns)
    (freshWitness : PaperLinearAlgebra.Assignment F columns) :
    OutputWitness productionShape columns where
  assignments := Fin.addCases (fun _ => freshWitness) runningWitness

private theorem source_fresh_eq (index : Fin productionShape.freshCount) :
    SourceMembership.freshInstance
        ((ProductionKey.key relation ajtai).statement running fresh) index =
      Lifecycle.freshStatement relation fresh := by
  have zero : index = ⟨0, by decide⟩ := by
    apply Fin.ext
    have bound := index.isLt
    change index.val < 1 at bound
    change index.val = 0
    omega
  subst index
  rfl

private theorem source_running_eq (index : Fin productionShape.runningCount) :
    SourceMembership.runningInstance
        ((ProductionKey.key relation ajtai).statement running fresh) index =
      Lifecycle.runningStatement relation running index := by
  refine ceInstance_ext _ _ ?_ ?_ ?_ ?_ ?_ ?_
  · rfl
  · change Fin.addCases fresh.commitments running.commitments (runningSourceIndex index) =
      running.commitments index
    exact addCases_running (shape := productionShape) fresh.commitments running.commitments index
  · change Fin.addCases fresh.publicInputs running.publicInputs (runningSourceIndex index) =
      running.publicInputs index
    exact addCases_running (shape := productionShape) fresh.publicInputs running.publicInputs index
  · rfl
  · rfl
  · rfl

private theorem sourceHolds_of_terminalHolds
    (runningWitness : Stage1.Terminal.RunningWitness
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (freshWitness : Stage1.Terminal.FreshWitness
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (valid : Lifecycle.TerminalHolds relation ajtai running runningWitness fresh freshWitness) :
    SourceHolds extensionOps K.embed (PaperAlgebra.openingMaps ajtai) productionGlobalParams
      ((ProductionKey.key relation ajtai).statement running fresh)
      (sourceWitness runningWitness freshWitness) := by
  apply (SourceMembership.sourceHolds_iff_memberships extensionOps extensionLaws K.embed
    (PaperAlgebra.openingMaps ajtai) productionGlobalParams rfl
    ((ProductionKey.key relation ajtai).statement running fresh)
    (sourceWitness runningWitness freshWitness)).mpr
  let paper := paperRelationSemantics (shape := productionShape)
    (blockCount := Phi81ColumnLayout.blockCount (Phi81CarrierLayout.carrierWidth logicalWidth))
    baseOps extensionOps K.embed (PaperAlgebra.openingMaps ajtai)
  constructor
  · intro index
    have witnessEq : (sourceWitness runningWitness freshWitness).assignments
        (freshSourceIndex index) = freshWitness :=
      addCases_fresh (shape := productionShape) (fun _ => freshWitness) runningWitness index
    apply Eq.mpr (congrArg₂ (CCS.Holds paper productionGlobalParams)
      (source_fresh_eq relation ajtai running fresh index) witnessEq)
    refine ⟨?_, valid.2.2⟩
    exact (PaperAlgebra.openingAgreement ajtai 2 _ _ _).mpr valid.2.1
  · intro index
    have witnessEq : (sourceWitness runningWitness freshWitness).assignments
        (runningSourceIndex index) = runningWitness index :=
      addCases_running (shape := productionShape) (fun _ => freshWitness) runningWitness index
    apply Eq.mpr (congrArg₂ (CE.Holds paper productionGlobalParams)
      (source_running_eq relation ajtai running fresh index) witnessEq)
    have member := valid.1 index
    refine ⟨?_, trivial, ?_⟩
    · exact (PaperAlgebra.openingAgreement ajtai 2 _ _ _).mpr member.1
    · exact (PaperAlgebra.evaluations_eq_paper ajtai
        (PiCCS.CanonicalRowLayout.layout cubeVariables
          (Phi81CarrierLayout.carrierWidth logicalWidth) relation.cubeFits)
        relation.system (runningWitness index) running.point).symm.trans member.2.2

end SourceMembership

/-- Accepted terminal openings construct an accepted production fold and its exact children.
No sampler-success, accepted-local-proof or new-opening premise is required. -/
theorem recursive_nifs
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (context : KeyDigest) (application : Stage1.Application.Program)
    (statement : TerminalStatement AppState)
    (payload : TerminalProof
      (Running (logicalWidth := logicalWidth) (publicFits := publicFits))
      (Stage1.Terminal.RunningWitness (logicalWidth := logicalWidth) (publicFits := publicFits))
      (Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
      (Stage1.Terminal.FreshWitness (logicalWidth := logicalWidth) (publicFits := publicFits)) slotCount)
    (accepted : Stage1.Terminal.HoldsFor relation ajtai context application statement (.recursive payload)) :
    ∃ (proof : Lifecycle.Proof 9)
      (result : Running (logicalWidth := logicalWidth) (publicFits := publicFits))
      (children : Stage1.Terminal.RunningWitness (logicalWidth := logicalWidth) (publicFits := publicFits)),
      Nifs.PaperNonInteractive.verify (ProductionKey.key relation ajtai)
        (payload.running functionIndex) payload.fresh proof = some result ∧
      ∀ child, CE.Holds (semantics ajtai) productionGlobalParams
        (Lifecycle.runningStatement relation result child) (children child) := by
  let key := ProductionKey.key relation ajtai
  obtain ⟨_valid, _pc, _positive, _public, runningMember, freshMember⟩ :=
    (Stage1.Terminal.holdsFor_recursive_iff relation ajtai context application statement payload).mp accepted
  have memberships : Lifecycle.TerminalHolds relation ajtai
      (payload.running functionIndex) (payload.runningWitness functionIndex)
      payload.fresh payload.freshWitness :=
    ⟨runningMember functionIndex, freshMember⟩
  have valid := sourceHolds_of_terminalHolds relation ajtai
    (payload.running functionIndex) payload.fresh
    (payload.runningWitness functionIndex) payload.freshWitness memberships
  obtain ⟨_messages, _fullOutput, _cAccepted, _cOpenings, continuation⟩ :=
    Nifs.PaperNonInteractive.Completeness.exists_honest_proof_of_sampler_success key
      (payload.running functionIndex) payload.fresh
      (sourceWitness (payload.runningWitness functionIndex) payload.freshWitness) valid
  obtain ⟨proof, result, children, _roundsEq, _outputEq, _sampleEq, verified, childValid⟩ :=
    continuation _ (ProductionKey.key_response relation ajtai _)
  refine ⟨proof, result, children, verified, ?_⟩
  intro child
  have member := childValid child
  rw [Lifecycle.PiDEC.v1_1.OutputWitnessConsumer.runningStatement_eq relation ajtai result child] at member
  exact member

end NightstreamFPrime.Export.Stage1.HyperNovaCompleteness
