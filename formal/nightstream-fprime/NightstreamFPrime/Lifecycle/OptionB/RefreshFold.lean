import NightstreamFPrime.Lifecycle.OptionB.RefreshPiCCSPhase
import NightstreamFPrime.Lifecycle.OptionB.Step
import NightstreamFPrime.Lifecycle.PiDEC.v1_2.Semantics
import NightstreamFPrime.Lifecycle.PiRLC.v1_2.Semantics
import NightstreamFPrime.Lifecycle.PiCCS.v1_2.OutputBinding

/-! The refresh fold (CW-3.2). Today's Π_DEC on the slot's claim, the refresh Π_CCS phase on the
16 children and the fresh instance, and today's Π_RLC make the option-B refresh fold accept the
slot's claim, with the output slot claim that the circuit's words carry. This is
`Refresh.verify`: SuperNeo v1.2 Sections 7.5, 7.3 and 7.4, with Π_DEC moved to the front. It is
the fold term of the refresh case of `stepHolds_iff`.

The Π_RLC inputs of the 16 running sources take their public inputs from the Π_DEC digit words,
which the Π_DEC rows bind to the split of the slot's public input
(`OutputAccepted.childPublicInput_eq`). -/

namespace NightstreamFPrime.Lifecycle.OptionB.RefreshFold

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Lifecycle.PiCCS.v1_2

variable {logicalWidth : Nat}
  {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

/-- The Π_DEC child that gives running source `child`. -/
def outputIndex (child : Fin productionShape.runningCount) :
    Spec.Phi81Relation.PiDECAlgebra.Radix.ChildIndex :=
  Fin.cast rfl child

/-- The refresh Π_CCS running inputs: the slot's point and public-input split, with the proof's
commitments and evaluation families (`Refresh.children`). -/
noncomputable abbrev children (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (slot : SlotClaim logicalWidth publicFits)
    (split : Spec.Folding.Refresh.Split PaperAlgebra.Commitment productionShape) :
    Children logicalWidth publicFits :=
  Spec.Folding.Refresh.children (refreshKey relation ajtai)
    (toClaim (refreshKey relation ajtai).relationSource slot) split

/-- The wires of Π_DEC: its parent is the slot's claim and its messages are the proof's
children. -/
structure DecWired (dec : Lifecycle.PiDEC.v1_2.Formal.Interface logicalWidth publicFits)
    (decOffset : Nat) (slot : SlotClaim logicalWidth publicFits)
    (split : Spec.Folding.Refresh.Split PaperAlgebra.Commitment productionShape) (env : Env) :
    Prop where
  parentCommitment : ∀ row lane,
    ((dec.parent decOffset).commitment row lane).eval env = slot.commitment row lane
  parentPublicInput : ∀ column,
    ((dec.parent decOffset).publicInput column).eval env = slot.publicInput column
  parentPoint : Lifecycle.PiDEC.v1_2.InputBinding.evalPoint (dec.point decOffset) env = slot.point
  parentEvaluation : StatementAbsorption.evalEvaluation (dec.parent decOffset).evaluation env =
    slot.evaluations
  messageCommitment : ∀ child row lane,
    ((dec.message decOffset (outputIndex child)).commitment row lane).eval env =
      split.commitments child row lane
  messageEvaluation : ∀ child,
    StatementAbsorption.evalEvaluation (dec.message decOffset (outputIndex child)).evaluation
      env = split.evaluations child

/-- The wires of Π_RLC: the sampler starts from the state after the Π_CCS output binding, the
point is the SumCheck point, the fresh source is the fresh instance, the running sources are the
children with the Π_DEC digit words as public inputs, and the evaluation families are the Π_CCS
output. -/
structure RlcWired {degreeBound : Nat}
    (dec : Lifecycle.PiDEC.v1_2.Formal.Interface logicalWidth publicFits) (decOffset : Nat)
    (piccs : RefreshPiCCSPhase.Leaves logicalWidth degreeBound publicFits)
    (rlc : Lifecycle.PiRLC.v1_2.Formal.Interface logicalWidth publicFits) (rlcOffset : Nat)
    (fresh : StepFresh logicalWidth publicFits)
    (split : Spec.Folding.Refresh.Split PaperAlgebra.Commitment productionShape)
    (output : FullOutputCoordinates.FullOutput K productionShape) (env : Env) : Prop where
  samplerStart : Lifecycle.PiRLC.v1_2.Sampler.evalState env (rlc.initialState rlcOffset) =
    List.ofFn (Layer.evalState env (OutputBinding.finalState piccs.output piccs.outputOffset))
  point : (Lifecycle.PiRLC.v1_2.InputBinding.evalPoint (rlc.point rlcOffset) env).coordinates =
    (RoundTranscript.evalRoundPoint piccs.rounds piccs.roundsOffset env).coordinates
  commitment : ∀ source row lane,
    ((rlc.input rlcOffset source).commitment row lane).eval env =
      (Fin.addCases fresh.commitments split.commitments source : PaperAlgebra.Commitment) row
        lane
  publicInput : ∀ source column,
    ((rlc.input rlcOffset source).publicInput column).eval env =
      (Fin.addCases fresh.publicInputs
        (fun child column => (dec.digit decOffset (outputIndex child) column).eval env) source :
          PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
        column
  evaluation : ∀ source,
    StatementAbsorption.evalEvaluation (rlc.input rlcOffset source).evaluation env =
      ⟨output.padCoordinate source, output.matrixCoordinate source⟩

/-- The slot claim that the circuit's words carry after a refresh fold: the Π_RLC output. -/
def nextSlot (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (rlc : Lifecycle.PiRLC.v1_2.Formal.Interface logicalWidth publicFits) (rlcOffset : Nat)
    (env : Env) : SlotClaim logicalWidth publicFits where
  commitment := (Lifecycle.PiRLC.v1_2.Semantics.evalOutput relation rlc rlcOffset env).commitment
  publicInput :=
    (Lifecycle.PiRLC.v1_2.Semantics.evalOutput relation rlc rlcOffset env).publicInput
  point := (Lifecycle.PiRLC.v1_2.Semantics.evalOutput relation rlc rlcOffset env).point
  evaluations := Lifecycle.PiRLC.v1_2.OutputBinding.evalEvaluation
    (Lifecycle.PiRLC.v1_2.Formal.outputBindingInterface
      (Lifecycle.PiRLC.v1_2.Formal.atOffset rlc rlcOffset) rlcOffset)
    (Lifecycle.PiRLC.v1_2.Formal.outputBindingOffset rlcOffset) env

theorem toClaim_nextSlot (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (rlc : Lifecycle.PiRLC.v1_2.Formal.Interface logicalWidth publicFits) (rlcOffset : Nat)
    (env : Env) :
    toClaim (Lifecycle.PiRLC.v1_2.InputBinding.relationSource relation)
        (nextSlot relation rlc rlcOffset env) =
      Lifecycle.PiRLC.v1_2.Semantics.evalOutput relation rlc rlcOffset env :=
  rfl

private theorem instance_ext
    {left right : Lifecycle.PiRLC.v1_2.InputBinding.InputInstance logicalWidth publicFits}
    (constraintSystem : left.constraintSystem = right.constraintSystem)
    (commitment : left.commitment = right.commitment)
    (publicInput : left.publicInput = right.publicInput)
    (point : left.point = right.point)
    (evaluations : left.evaluations = right.evaluations)
    (stage : left.stage = right.stage) : left = right := by
  cases left
  cases right
  simp_all

private theorem message_ext
    {left right : Spec.Folding.PiDEC.PaperVerifier.ChildMessage PaperAlgebra.Evaluation
      PaperAlgebra.Commitment}
    (commitment : left.commitment = right.commitment)
    (evaluations : left.evaluations = right.evaluations) : left = right := by
  cases left
  cases right
  simp_all

private theorem attempt_ext
    {left right : Lifecycle.PiDEC.v1_2.InputBinding.Attempt logicalWidth publicFits}
    (parent : left.parent = right.parent) (messages : left.messages = right.messages) :
    left = right := by
  cases left
  cases right
  simp_all

private theorem point_eq {count : Nat} {left right : CubePoint K count}
    (coordinates : left.coordinates = right.coordinates) : left = right := by
  cases left
  cases right
  simp_all

section Fold

variable (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
  (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
  (slot : SlotClaim logicalWidth publicFits) (fresh : StepFresh logicalWidth publicFits)
  (split : Spec.Folding.Refresh.Split PaperAlgebra.Commitment productionShape)
  (dec : Lifecycle.PiDEC.v1_2.Formal.Interface logicalWidth publicFits) (decOffset : Nat)
  (env : Env)

/-- The Π_DEC attempt that the wires carry is the refresh verifier's attempt on the slot's
claim. -/
theorem decAttempt_eq (wired : DecWired dec decOffset slot split env) :
    Lifecycle.PiDEC.v1_2.Semantics.inputAttempt relation dec decOffset env =
      Spec.Folding.Refresh.inputAttempt (refreshKey relation ajtai)
        (toClaim (refreshKey relation ajtai).relationSource slot) split := by
  apply attempt_ext
  · apply instance_ext
    · rfl
    · exact funext fun row => funext fun lane => wired.parentCommitment row lane
    · exact funext wired.parentPublicInput
    · exact wired.parentPoint
    · exact congrArg (fun value => #[value]) wired.parentEvaluation
    · rfl
  · funext child
    apply message_ext
    · exact funext fun row => funext fun lane =>
        wired.messageCommitment (Fin.cast rfl child) row lane
    · exact congrArg (fun value => #[value]) (wired.messageEvaluation (Fin.cast rfl child))

/-- The Π_DEC digit words are the children's public inputs: the split of the slot's public
input. -/
theorem digits_eq (decSpec : Lifecycle.PiDEC.v1_2.Formal.SpecHolds relation dec decOffset env)
    (wired : DecWired dec decOffset slot split env) :
    (fun child column => (dec.digit decOffset (outputIndex child) column).eval env) =
      (children relation ajtai slot split).publicInputs := by
  have phase := Lifecycle.PiDEC.v1_2.Semantics.spec_implies_phaseHolds relation ajtai dec
    decOffset env decSpec
  funext child column
  have splitEq := congrFun (Spec.Folding.PiDEC.PaperVerifier.OutputAccepted.childPublicInput_eq
    phase (outputIndex child)) column
  rw [decAttempt_eq relation ajtai slot split dec decOffset env wired] at splitEq
  exact splitEq

end Fold

/-- The refresh fold term of `stepHolds_iff`: the Π_DEC, Π_CCS and Π_RLC rows and their wires
make the option-B refresh fold accept the slot's claim with the next slot's claim. -/
theorem spec_implies_fold (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (slot : SlotClaim logicalWidth publicFits) (fresh : StepFresh logicalWidth publicFits)
    (proof : Spec.Folding.Refresh.Proof PaperAlgebra.Commitment productionShape
      (ProductionKey.degreeBound relation))
    (dec : Lifecycle.PiDEC.v1_2.Formal.Interface logicalWidth publicFits) (decOffset : Nat)
    (piccs : RefreshPiCCSPhase.Leaves logicalWidth (ProductionKey.degreeBound relation) publicFits)
    (rlc : Lifecycle.PiRLC.v1_2.Formal.Interface logicalWidth publicFits) (rlcOffset : Nat)
    (env : Env)
    (decSpec : Lifecycle.PiDEC.v1_2.Formal.SpecHolds relation dec decOffset env)
    (decWired : DecWired dec decOffset slot proof.split env)
    (piccsSpec : RefreshPiCCSPhase.SpecHolds relation piccs env)
    (piccsWired : RefreshPiCCSPhase.Wired relation ajtai
      (children relation ajtai slot proof.split) fresh proof.nifs piccs env)
    (rlcSpec : Lifecycle.PiRLC.v1_2.Formal.SpecHolds relation rlc rlcOffset env)
    (rlcWired : RlcWired dec decOffset piccs rlc rlcOffset fresh proof.split
      proof.nifs.piCcsOutput env) :
    HyperNova.Construction2.OptionBFolds.folds .refresh (keys relation ajtai)
        (toClaim (refreshKey relation ajtai).relationSource slot) fresh (.refresh proof) =
      some (toClaim (refreshKey relation ajtai).relationSource
        (nextSlot relation rlc rlcOffset env)) := by
  have phase := RefreshPiCCSPhase.spec_implies_phaseHolds relation ajtai
    (children relation ajtai slot proof.split) fresh proof.nifs piccs env piccsSpec piccsWired
  have rlcPhase := Lifecycle.PiRLC.v1_2.Semantics.spec_implies_phaseHolds relation ajtai rlc
    rlcOffset env rlcSpec
  have decAccepted := Lifecycle.PiDEC.v1_2.Semantics.accepted relation ajtai dec decOffset env
    decSpec
  rw [decAttempt_eq relation ajtai slot proof.split dec decOffset env decWired] at decAccepted
  have digits := digits_eq relation ajtai slot proof.split dec decOffset env decSpec decWired
  have inputsEq : Lifecycle.PiRLC.v1_2.Semantics.evalInputs relation rlc rlcOffset env =
      (refreshKey relation ajtai).piCcsOutputs (children relation ajtai slot proof.split) fresh
        proof.nifs := by
    funext source
    let joint : Fin productionShape.sourceCount :=
      Fin.cast (refreshKey relation ajtai).total_eq_sourceCount source
    have sourceEq : Lifecycle.PiRLC.v1_2.Semantics.sourceIndex source = joint := Fin.ext rfl
    apply instance_ext
    · rfl
    · change (fun row lane => ((rlc.input rlcOffset
          (Lifecycle.PiRLC.v1_2.Semantics.sourceIndex source)).commitment row lane).eval env) =
        Fin.addCases fresh.commitments proof.split.commitments joint
      rw [sourceEq]
      exact funext fun row => funext fun lane => rlcWired.commitment joint row lane
    · change (fun column => ((rlc.input rlcOffset
          (Lifecycle.PiRLC.v1_2.Semantics.sourceIndex source)).publicInput column).eval env) =
        Fin.addCases fresh.publicInputs (children relation ajtai slot proof.split).publicInputs
          joint
      rw [sourceEq, ← digits]
      exact funext fun column => rlcWired.publicInput joint column
    · exact point_eq (rlcWired.point.trans phase.roundPoint)
    · change #[StatementAbsorption.evalEvaluation (rlc.input rlcOffset
          (Lifecycle.PiRLC.v1_2.Semantics.sourceIndex source)).evaluation env] =
        #[⟨proof.nifs.piCcsOutput.padCoordinate joint,
          proof.nifs.piCcsOutput.matrixCoordinate joint⟩]
      rw [sourceEq, rlcWired.evaluation joint]
    · rfl
  have challengesEq : (refreshKey relation ajtai).piRlcChallenges
      (children relation ajtai slot proof.split) fresh proof.nifs =
      some (Lifecycle.PiRLC.v1_2.Semantics.evalChallenges rlc rlcOffset env) := by
    have stateEq : Lifecycle.PiRLC.v1_2.Sampler.evalState env (rlc.initialState rlcOffset) =
        ((refreshKey relation ajtai).piCcsExecution (children relation ajtai slot proof.split)
          fresh proof.nifs).outgoingState :=
      rlcWired.samplerStart.trans phase.outgoingState
    change (refreshKey relation ajtai).piRlcResponse
      ((refreshKey relation ajtai).piCcsExecution (children relation ajtai slot proof.split)
        fresh proof.nifs).outgoingState = _
    rw [← stateEq]
    exact rlcPhase.response
  have outputEq : Lifecycle.PiRLC.v1_2.Semantics.evalOutput relation rlc rlcOffset env =
      (refreshKey relation ajtai).parentForChallenges (children relation ajtai slot proof.split)
        fresh proof.nifs (Lifecycle.PiRLC.v1_2.Semantics.evalChallenges rlc rlcOffset env) := by
    let first : Fin Spec.Folding.Nifs.PaperProfile.arity.total := ⟨0, by decide⟩
    have structureEq : (Lifecycle.PiRLC.v1_2.Semantics.evalOutput relation rlc rlcOffset
        env).constraintSystem = (refreshKey relation ajtai).relationSource :=
      (rlcPhase.accepted.sameStructure first).symm.trans
        (congrArg (fun inputs => (inputs first).constraintSystem) inputsEq)
    have pointEq : (Lifecycle.PiRLC.v1_2.Semantics.evalOutput relation rlc rlcOffset env).point =
        ((refreshKey relation ajtai).piCcsExecution (children relation ajtai slot proof.split)
          fresh proof.nifs).coins.roundPoint :=
      (rlcPhase.accepted.samePoint first).symm.trans
        (congrArg (fun inputs => (inputs first).point) inputsEq)
    rw [Lifecycle.PiRLC.v1_2.Semantics.output_eq_combinedOutput relation ajtai rlc rlcOffset env
      rlcPhase]
    unfold Spec.Folding.Nifs.PaperNonInteractive.Key.parentForChallenges
    rw [inputsEq, structureEq, pointEq]
    rfl
  have parentEq : (refreshKey relation ajtai).parent (children relation ajtai slot proof.split)
      fresh proof.nifs =
      some (Lifecycle.PiRLC.v1_2.Semantics.evalOutput relation rlc rlcOffset env) := by
    unfold Spec.Folding.Nifs.PaperNonInteractive.Key.parent
    rw [challengesEq, outputEq]
    rfl
  rw [HyperNova.Construction2.OptionBFolds.refresh_eq, Spec.Folding.Refresh.verify_eq_some_iff]
  exact ⟨decAccepted, phase.accepted, parentEq⟩

end NightstreamFPrime.Lifecycle.OptionB.RefreshFold
