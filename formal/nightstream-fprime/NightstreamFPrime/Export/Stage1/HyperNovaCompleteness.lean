import NightstreamFPrime.Export.Stage1.HyperNovaHistory
import NightstreamFPrime.Lifecycle.PiDEC.v1_1.OutputWitnessConsumer
import NightstreamFPrime.Spec.Folding.Nifs.PaperNonInteractive.Completeness

/-!
Owns the honest selected NIFS call from an accepted recursive terminal
payload. Its existing fresh and running witnesses supply source membership.
C messages are fixed before actual sampler success; the normal verifier and
all new running openings are conclusions. No security or work law is assumed.
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
open Poseidon2HashChainV1Package (application fits)
open Poseidon2HashChainV1Setup (productionSetup productionAjtaiKey)

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

/-- An accepted selected recursive payload supplies the actual old witnesses.
They construct one causal C prefix and, when its actual PiRLC sampler returns,
a normal production NIFS proof with valid openings for every returned child.
No accepted local proof, intermediate output, or new witness validity is an
input. This does not construct the next fresh application assignment. -/
theorem recursive_nifs_of_sampler_success
    (statement : HyperNovaHistory.Statement) (payload : HyperNovaHistory.Payload)
    (accepted : PerApplicationTerminal.Holds application fits productionSetup
      statement (.recursive payload)) :
    let relation := PerApplicationFixedPoint.relation application fits
    let key := ProductionKey.key relation productionAjtaiKey
    ∃ (messages : Fin productionShape.cubeVariables → FixedPolynomial K 9)
      (fullOutput : FullOutputCoordinates.FullOutput K productionShape),
      let coins := FiatShamir.derive key.oracle.transcript
        ({ priorState := key.publicInputState (payload.running functionIndex) payload.fresh
           input := (key.statement (payload.running functionIndex) payload.fresh).verifierInput key.lift } :
          PiCCS.TranscriptReplay.Statement K Transcript.State productionShape)
        { rounds := fun round => (messages round).toMessage }
      ∀ rho : Fin key.arity.total → RingF,
        key.piRlcResponse (key.absorbPiCcsOutput coins.finalState fullOutput) = some rho →
        ∃ (proof : Lifecycle.Proof 9)
          (result : Running
            (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
            (publicFits := PerApplicationFixedPoint.publicFits application))
          (children : Stage1.Terminal.RunningWitness
            (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
            (publicFits := PerApplicationFixedPoint.publicFits application)),
          proof.piCcsRounds = messages ∧ proof.piCcsOutput = fullOutput ∧
          key.piRlcChallenges (payload.running functionIndex) payload.fresh proof = some rho ∧
          Nifs.PaperNonInteractive.verify key (payload.running functionIndex) payload.fresh proof = some result ∧
          ∀ child, CE.Holds (semantics productionAjtaiKey) productionGlobalParams
            (Lifecycle.runningStatement relation result child) (children child) := by
  let relation := PerApplicationFixedPoint.relation application fits
  let key := ProductionKey.key relation productionAjtaiKey
  obtain ⟨_statementValid, _pcValid, _positive, _public, runningMember, freshMember⟩ :=
    (PerApplicationTerminal.holds_recursive_iff application fits productionSetup statement payload).mp accepted
  have memberships : Lifecycle.TerminalHolds relation productionAjtaiKey
      (payload.running functionIndex) (payload.runningWitness functionIndex)
      payload.fresh payload.freshWitness :=
    ⟨runningMember functionIndex, freshMember⟩
  have valid := sourceHolds_of_terminalHolds relation productionAjtaiKey
    (payload.running functionIndex) payload.fresh
    (payload.runningWitness functionIndex) payload.freshWitness memberships
  obtain ⟨messages, fullOutput, _cAccepted, _cOpenings, continuation⟩ :=
    Nifs.PaperNonInteractive.Completeness.exists_honest_proof_of_sampler_success key
      (payload.running functionIndex) payload.fresh
      (sourceWitness (payload.runningWitness functionIndex) payload.freshWitness) valid
  refine ⟨messages, fullOutput, ?_⟩
  intro coins rho sampled
  obtain ⟨proof, result, children, roundsEq, outputEq, sampleEq, verified, childValid⟩ :=
    continuation rho sampled
  refine ⟨proof, result, children, roundsEq, outputEq, sampleEq, verified, ?_⟩
  intro child
  have member := childValid child
  rw [Lifecycle.PiDEC.v1_1.OutputWitnessConsumer.runningStatement_eq
    relation productionAjtaiKey result child] at member
  exact member

end NightstreamFPrime.Export.Stage1.HyperNovaCompleteness
