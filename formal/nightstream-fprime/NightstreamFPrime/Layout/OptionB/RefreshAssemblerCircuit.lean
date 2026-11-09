import NightstreamFPrime.Layout.OptionB.RefreshAssemblerCompleteness

/-! The refresh step circuit as one `FormalCircuit` at the source layout (CW-3.9). Its spec is
that every child's spec holds, as for the light step circuit (`LightAssemblerCircuit`). Soundness
is the rows themselves; `RefreshAssembler.rows_implies_stepHolds` then gives the step relation
from the words. Completeness (`completeOfChildren`) fills the children from their specs: the step
gadgets read only shared words (`completeGadgetsFrom`); today's Π_DEC phase holds on the shared
words (`PiDEC.v1_2.Semantics.phaseHolds_of_agree`); the refresh Π_CCS check accepts the inputs
that the shared words carry (`RefreshPiCCSPhase.spec_implies_phaseHolds`, then
`RefreshPiCCSCompleteness.completeFrom`); today's Π_RLC fills from its inputs; the new slot's
hash fills because the folded slot is the same before and after the fill; and the digest-select
rows read only shared words (`DigestSelect.rows_congr`). -/

namespace NightstreamFPrime.Layout.OptionB.RefreshAssemblerCircuit

open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.OptionB

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Gadgets.SumCheck
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.HyperNova.Construction2
open NightstreamFPrime.Lifecycle.PiCCS.v1_2
open NightstreamFPrime.Lifecycle.OptionB.SlotWords
open NightstreamFPrime.Lifecycle.OptionB.RefreshStepLayout
open NightstreamFPrime.Layout.OptionB.LightAssembler (slotAt slotAt_below slotWordCount slotsStart
  witnessStart)
open NightstreamFPrime.Layout.OptionB.RefreshAssembler

variable {logicalWidth : Nat}
  {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}
  (program : Stage1.Application.Program) (base : Nat) (source : Env)

/-- The active slot's claim that the shared words carry. -/
def decodedSlot : SlotClaim logicalWidth publicFits :=
  ((interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).active.value source).1

/-- The fresh instance that the shared words carry. -/
def decodedFresh : StepFresh logicalWidth publicFits :=
  StatementAbsorption.evalFresh (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).fresh source

/-- The refresh proof that the shared words carry. The Π_DEC fields of the inner proof are not
read by the refresh fold; they repeat the children. -/
def decodedProof : Spec.Folding.Refresh.Proof PaperAlgebra.Commitment productionShape 8 where
  split :=
    ⟨fun child row lane => ((interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).childCommitment child row lane).eval source,
      fun child => StatementAbsorption.evalEvaluation ((interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).childEvaluation child) source⟩
  nifs :=
    { piCcsRounds := fun roundIndex => ((interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).round roundIndex).semanticPolynomial source
      piCcsOutput :=
        ⟨fun origin coefficient => ((interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).output.padCoordinate origin coefficient).eval source,
          fun origin matrix coefficient =>
            ((interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).output.matrixCoordinate origin matrix coefficient).eval source⟩
      piDecCommitments := fun child row lane =>
        ((interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).childCommitment child row lane).eval source
      piDecEvaluations := fun child =>
        StatementAbsorption.evalEvaluation ((interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).childEvaluation child) source }

section Shared

variable (target : Env) (agree : ∀ column, column < (sharedEnd base program.witnessWordCount) → target column = source column)

include agree

private theorem shared_eval (expression : Expr) (below : expression.VarsBelow (sharedEnd base program.witnessWordCount)) :
    expression.eval target = expression.eval source :=
  expression.eval_eq_of_agree_below _ target source below agree

private theorem shared_kEval (value : KExpr) (below : value.VarsBelow (sharedEnd base program.witnessWordCount)) :
    value.eval target = value.eval source :=
  value.eval_eq_of_agree_below _ target source below agree

private theorem shared_words (words : List Expr)
    (below : ∀ word ∈ words, word.VarsBelow (sharedEnd base program.witnessWordCount)) :
    Hash.evalList target words = Hash.evalList source words := by
  unfold Hash.evalList
  apply List.map_congr_left
  intro word member
  exact shared_eval program base source target agree word (below word member)

/-- On values that agree below `sharedEnd`, the words carry the decoded refresh Π_CCS inputs. -/
theorem piccsRepresents (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (offset : Nat) :
    RefreshPiCCSLayout.Represents
      (RefreshFold.children relation ajtai (decodedSlot program base source)
        (decodedProof (logicalWidth := logicalWidth) (publicFits := publicFits) program base
          source).split)
      (decodedFresh program base source)
      (decodedProof (logicalWidth := logicalWidth) (publicFits := publicFits) program base
        source).nifs (piccsInterface program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount)) offset target := by
  have below := inputsBelow (logicalWidth := logicalWidth) (publicFits := publicFits) base
    program.witnessWordCount
  refine {
    fresh := ?_
    point := ?_
    commitments := fun child row lane => shared_eval program base source target agree _
      (below.childCommitment child row lane)
    evaluations := fun child => ?_
    rounds := fun roundIndex => by
      unfold RoundTranscript.Message.semanticPolynomial
      exact FixedChain.Round.semanticPolynomial_eq_of_agree_below _ _ target source
        ⟨fun coefficient => below.round roundIndex coefficient, ⟨trivial, trivial⟩⟩ agree
    outputPad := fun origin coefficient => (shared_kEval program base source target agree _
      (below.outputPad origin coefficient)).trans rfl
    outputMatrix := fun origin matrix coefficient => (shared_kEval program base source target agree
      _ (below.outputMatrix origin matrix coefficient)).trans rfl }
  · unfold decodedFresh StatementAbsorption.evalFresh
    congr 1
    · funext origin row lane
      exact shared_eval program base source target agree _ (below.freshCommitment origin row lane)
    · funext origin column
      exact shared_eval program base source target agree _ (below.freshPublicInput origin column)
  · have pointEq : (fun coordinate => ((interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).active.point coordinate).eval target) =
        fun coordinate => ((interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).active.point coordinate).eval source :=
      funext fun coordinate => shared_kEval program base source target agree _
        (below.activePoint coordinate)
    change (canonicalFinIndices productionShape.cubeVariables).map
        (fun coordinate => ((interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).active.point coordinate).eval target) = _
    rw [pointEq]
    simp [canonicalFinIndices, RefreshFold.children, Spec.Folding.Refresh.children, toClaim,
      decodedSlot, SlotExpr.value, StatementAbsorption.evalPoint]
  · unfold StatementAbsorption.evalEvaluation
    congr 1
    · funext coefficient
      exact shared_kEval program base source target agree _ (below.childPad child coefficient)
    · funext matrix coefficient
      exact shared_kEval program base source target agree _
        (below.childMatrix child matrix coefficient)

/-- On values that agree below `sharedEnd`, today's Π_DEC reads the decoded slot and children. -/
theorem decWired (circuit : Fin 2) (offset : Nat) :
    RefreshFold.DecWired (decInterface program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount)) (decOffset program circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) offset)
      (decodedSlot program base source)
      (decodedProof (logicalWidth := logicalWidth) (publicFits := publicFits) program base
        source).split target := by
  have below := inputsBelow (logicalWidth := logicalWidth) (publicFits := publicFits) base
    program.witnessWordCount
  refine {
    parentCommitment := fun row lane => shared_eval program base source target agree _
      (below.activeCommitment row lane)
    parentPublicInput := fun column => shared_eval program base source target agree _
      (below.activePublicInput column)
    parentPoint := ?_
    parentEvaluation := ?_
    messageCommitment := fun child row lane => shared_eval program base source target agree _
      (below.childCommitment child row lane)
    messageEvaluation := fun child => ?_ }
  · change StatementAbsorption.evalPoint (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).active.point target =
      StatementAbsorption.evalPoint (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).active.point source
    unfold StatementAbsorption.evalPoint
    congr 1
    exact congrArg List.ofFn (funext fun coordinate => shared_kEval program base source target
      agree _ (below.activePoint coordinate))
  · change StatementAbsorption.evalEvaluation (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).active.evaluation target =
      StatementAbsorption.evalEvaluation (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).active.evaluation source
    unfold StatementAbsorption.evalEvaluation
    congr 1
    · funext coefficient
      exact shared_kEval program base source target agree _ (below.activePad coefficient)
    · funext matrix coefficient
      exact shared_kEval program base source target agree _ (below.activeMatrix matrix coefficient)
  · change StatementAbsorption.evalEvaluation ((interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).childEvaluation (childOf
        (RefreshFold.outputIndex child))) target =
      StatementAbsorption.evalEvaluation ((interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).childEvaluation child) source
    unfold StatementAbsorption.evalEvaluation
    congr 1
    · funext coefficient
      exact shared_kEval program base source target agree _ (below.childPad child coefficient)
    · funext matrix coefficient
      exact shared_kEval program base source target agree _
        (below.childMatrix child matrix coefficient)

/-- On values that agree below `sharedEnd`, today's Π_RLC reads the decoded fresh instance,
children and Π_CCS output. -/
theorem rlcWired (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (circuit : Fin 2) (offset : Nat) :
    RefreshFold.RlcWired (decInterface program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount)) (decOffset program circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) offset)
      (piccsLeaves program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) offset)
      (rlcInterface program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) offset)
      (rlcOffset program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) offset) (decodedFresh program base source)
      (decodedProof (logicalWidth := logicalWidth) (publicFits := publicFits) program base
        source).split
      (decodedProof (logicalWidth := logicalWidth) (publicFits := publicFits) program base
        source).nifs.piCcsOutput target where
  samplerStart := by
    simp only [rlcInterface, Lifecycle.PiRLC.v1_2.Sampler.evalState]
  point := by
    simp only [rlcInterface, Lifecycle.PiRLC.v1_2.InputBinding.evalPoint,
      StatementAbsorption.evalPoint, RoundTranscript.evalRoundPoint, canonicalFinIndices,
      List.map_ofFn]
    rfl
  commitment := fun source' row lane => by
    have below := inputsBelow (logicalWidth := logicalWidth) (publicFits := publicFits) base
      program.witnessWordCount
    change ((Fin.addCases (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).fresh.commitment (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).childCommitment source' :
        Fin productionProfile.commitmentWidth → Fin ringDegree → Expr) row lane).eval target =
      (Fin.addCases (decodedFresh program base source).commitments
        (decodedProof (logicalWidth := logicalWidth) (publicFits := publicFits) program base
          source).split.commitments source' : PaperAlgebra.Commitment) row lane
    refine Fin.addCases (fun index => ?_) (fun child => ?_) source'
    · simp only [Fin.addCases_left]
      exact shared_eval program base source target agree _ (below.freshCommitment index row lane)
    · simp only [Fin.addCases_right]
      exact shared_eval program base source target agree _ (below.childCommitment child row lane)
  publicInput := fun source' column => by
    have below := inputsBelow (logicalWidth := logicalWidth) (publicFits := publicFits) base
      program.witnessWordCount
    change ((Fin.addCases (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).fresh.publicInput (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).digit source' :
        Fin (PaperAlgebra.FullShape logicalWidth publicFits).publicWidth → Expr) column).eval
          target =
      (Fin.addCases (decodedFresh program base source).publicInputs
        (fun child column =>
          ((decInterface program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount)).digit (decOffset program circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) offset)
            (RefreshFold.outputIndex child) column).eval target) source' :
          PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
        column
    refine Fin.addCases (fun index => ?_) (fun child => ?_) source'
    · simp only [Fin.addCases_left]
      exact shared_eval program base source target agree _ (below.freshPublicInput index column)
    · simp only [Fin.addCases_right]
      rfl
  evaluation := fun source' => by
    have below := inputsBelow (logicalWidth := logicalWidth) (publicFits := publicFits) base
      program.witnessWordCount
    change StatementAbsorption.evalEvaluation
        ⟨(interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).output.padCoordinate source', (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).output.matrixCoordinate source'⟩ target = _
    unfold StatementAbsorption.evalEvaluation
    congr 1
    · funext coefficient
      exact shared_kEval program base source target agree _ (below.outputPad source' coefficient)
    · funext matrix coefficient
      exact shared_kEval program base source target agree _
        (below.outputMatrix source' matrix coefficient)

end Shared

private theorem slotSpec_at {words : List Expr} {digest : Fin 4 → Expr} {left right : Nat}
    {env : Env}
    (spec : OutputHash.SpecHolds (LightStepLayout.slotHashInterface words digest) left env) :
    OutputHash.SpecHolds (LightStepLayout.slotHashInterface words digest) right env :=
  spec

private theorem priorStateSpec_at
    (interface : RefreshStepLayout.Interface logicalWidth publicFits program.witnessWordCount)
    {left right : Nat} {env : Env}
    (spec : PriorStateHash.SpecHolds (priorStateInterface program interface) left env) :
    PriorStateHash.SpecHolds (priorStateInterface program interface) right env :=
  spec

private theorem nextStateSpec_at (circuit : Fin 2)
    (interface : RefreshStepLayout.Interface logicalWidth publicFits program.witnessWordCount)
    {left right : Nat} {env : Env}
    (spec : OutputHash.SpecHolds (nextStateInterface program circuit interface) left env) :
    OutputHash.SpecHolds (nextStateInterface program circuit interface) right env :=
  spec

/-- Today's Π_DEC phase on the step's words does not read its offset: the Π_DEC interface reads
only shared words. The offsets are variables here, so Lean never evaluates a layout offset. -/
private theorem decPhase_at (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (interface : RefreshStepLayout.Interface logicalWidth publicFits program.witnessWordCount)
    {left right : Nat} {env : Env}
    (phase : Lifecycle.PiDEC.v1_2.Semantics.PhaseHolds relation ajtai
      (decInterface program interface) left env) :
    Lifecycle.PiDEC.v1_2.Semantics.PhaseHolds relation ajtai (decInterface program interface)
      right env :=
  phase

private theorem assertions_localLength (expressions : List Expr) :
    localLength (expressions.map Op.assertZero) = 0 := by
  induction expressions with
  | nil => rfl
  | cons _ rest ih =>
      simp only [List.map_cons, localLength, Op.localLength, List.sum_cons, Nat.zero_add]
      simpa [localLength] using ih

/-- Completeness of the refresh step circuit from its children's specs: when every child's spec
holds on `source`, the children fill into flat rows that agree with `source` outside the step's
private words. -/
theorem completeOfChildren (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (circuit : Fin 2)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (children : holds source (RefreshStepLayout.opsAt program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount))) :
    ∃ completed, AgreesOutside source completed (sharedEnd base program.witnessWordCount)
        (localLength (Circuit.ops (RefreshStepLayout.main program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount)) (sharedEnd base program.witnessWordCount))) ∧
      holdsFlat completed (Circuit.ops (RefreshStepLayout.main program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount))
        (sharedEnd base program.witnessWordCount)) := by
  have below := inputsBelow (logicalWidth := logicalWidth) (publicFits := publicFits) base
    program.witnessWordCount
  have endEq := sharedEnd_eq base program.witnessWordCount
  have assumptionsAt := RefreshStepBounds.assumptions program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount) below
  obtain ⟨stepSpec, selectRows0⟩ := RefreshStepLayout.soundness program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount)
    (sharedEnd base program.witnessWordCount) program.step (fun _ _ => circuit) rfl (fun _ _ _ => rfl) source (assumptionsAt source)
    children relation
  have applicationSpec : (applicationCircuit program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount)).spec (sharedEnd base program.witnessWordCount) source := by
    have callHolds := children (Formal.childOp "refresh.application"
      (applicationCircuit program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount)) (sharedEnd base program.witnessWordCount)) (by
        simp only [RefreshStepLayout.opsAt, List.mem_append]
        exact Or.inl (Or.inl (Or.inl (Or.inl (by simp [gadgetOps])))))
    change (applicationCircuit program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount)).assumptions (sharedEnd base program.witnessWordCount) source →
      (applicationCircuit program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount)).spec (sharedEnd base program.witnessWordCount) source at callHolds
    exact callHolds (assumptionsAt source).application
  have agreeOf : ∀ completed : Sequence.Prefix source (sharedEnd base program.witnessWordCount), ∀ column, column < (sharedEnd base program.witnessWordCount) →
      completed.current column = source column :=
    fun completed column bound => completed.agrees column (Or.inl bound)
  -- The step gadgets.
  obtain ⟨p1, ops1, end1, preserves1⟩ := RefreshStepCompleteness.completeGadgetsFrom program
    circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount) relation (Sequence.empty source (sharedEnd base program.witnessWordCount)) rfl below applicationSpec
    (fun slot => slotSpec_at (stepSpec.priorSlots slot))
    (priorStateSpec_at program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) stepSpec.prior)
    (nextStateSpec_at program circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) stepSpec.next)
  -- Today's Π_DEC: its phase reads only shared words.
  have decShared : Lifecycle.PiDEC.v1_2.Formal.Assumptions relation (decInterface program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount))
      (sharedEnd base program.witnessWordCount) source := ⟨{
    point := below.activePoint
    parentCommitment := below.activeCommitment
    parentPublicInput := below.activePublicInput
    parentEval_K := below.activePad
    parentEval_A := below.activeMatrix
    messageCommitment := fun child => below.childCommitment (childOf child)
    messageEval_K := fun child => below.childPad (childOf child)
    messageEval_A := fun child => below.childMatrix (childOf child)
    digit := fun child => below.digit (childOf child) }⟩
  have decSource : Lifecycle.PiDEC.v1_2.Semantics.PhaseHolds relation ajtai
      (decInterface program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount)) (sharedEnd base program.witnessWordCount) source :=
    decPhase_at program relation ajtai (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount)
      (Lifecycle.PiDEC.v1_2.Semantics.spec_implies_phaseHolds relation ajtai
        (decInterface program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount)) (decOffset program circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)) source stepSpec.dec)
  have decPhase : Lifecycle.PiDEC.v1_2.Semantics.PhaseHolds relation ajtai
      (decInterface program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount)) (decOffset program circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)) p1.current :=
    decPhase_at program relation ajtai (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount)
      (Lifecycle.PiDEC.v1_2.Semantics.phaseHolds_of_agree relation ajtai
        (decInterface program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount)) (sharedEnd base program.witnessWordCount) source p1.current decShared
        (fun index bound => (agreeOf p1 index bound).symm) decSource)
  obtain ⟨p2, ops2, end2, preserves2⟩ := RefreshStepCompleteness.completeDecFrom program circuit
    (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount) relation ajtai p1 end1 below decPhase
  -- The refresh Π_CCS: the check accepts the inputs that the shared words carry.
  have piccsBelow := RefreshStepBounds.piccsBelow program circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount) below
  have wiredAt := fun (env : Env) (agreeEnv : ∀ column, column < (sharedEnd base program.witnessWordCount) → env column = source column) =>
    RefreshPiCCSLayout.wired relation (piccsOffset program circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)) relation ajtai _ _ _
      (piccsInterface program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount)) env
      (piccsRepresents program base source env agreeEnv relation ajtai _)
  obtain ⟨p3, ops3, end3, preserves3⟩ := RefreshPiCCSCompleteness.completeFrom relation
    (piccsInterface program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount)) (piccsOffset program circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)) ajtai
    (RefreshFold.children relation ajtai (decodedSlot program base source)
      (decodedProof (logicalWidth := logicalWidth) (publicFits := publicFits) program base
        source).split)
    (decodedFresh program base source)
    (decodedProof (logicalWidth := logicalWidth) (publicFits := publicFits) program base
      source).nifs
    p2 end2 piccsBelow relation (piccsRepresents program base source p2.current (agreeOf p2)
      relation ajtai _)
    (RefreshPiCCSPhase.spec_implies_phaseHolds relation ajtai _ _ _ _ source stepSpec.piccs
      (wiredAt source (fun _ _ => rfl))).accepted
  -- Today's Π_RLC.
  obtain ⟨p4, ops4, end4, preserves4, -⟩ := RefreshStepCompleteness.completeRlcPhaseFrom relation
    ajtai (rlcInterface program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount))
    (rlcOffset program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)) p3 end3
    (RefreshStepBounds.rlcAssumptions program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount) below p3.current)
  have filledRows : holdsFlat p4.current
      (gadgetOps program circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount) ++
        Circuit.ops (Lifecycle.PiDEC.v1_2.Formal.main relation (decInterface program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount)))
          (decOffset program circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)) ++
        RefreshPiCCSLayout.opsAt relation (piccsInterface program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount))
          (piccsOffset program circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)) ++
        Circuit.ops (Lifecycle.PiRLC.v1_2.Formal.main relation
          (rlcInterface program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)))
          (rlcOffset program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount))) := by
    have rows := p4.rows
    rw [ops4, ops3, ops2, ops1] at rows
    simpa only [Sequence.empty, List.nil_append] using rows
  have rows4 := holdsFlat_implies_holds _ _ filledRows
  simp only [holds_append] at rows4
  obtain ⟨⟨⟨_, decRows⟩, piccsRows⟩, rlcRows⟩ := rows4
  -- The folded slot is the same before and after the fill.
  have foldAt := fun (env : Env) (agreeEnv : ∀ column, column < (sharedEnd base program.witnessWordCount) → env column = source column)
      (decSpec : Lifecycle.PiDEC.v1_2.Formal.SpecHolds relation (decInterface program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount))
        (decOffset program circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)) env)
      (piccsSpec : RefreshPiCCSPhase.SpecHolds relation
        (piccsLeaves program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)) env)
      (rlcSpec : Lifecycle.PiRLC.v1_2.Formal.SpecHolds relation
        (rlcInterface program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount))
        (rlcOffset program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)) env) =>
    RefreshFold.spec_implies_fold relation ajtai (decodedSlot program base source)
      (decodedFresh program base source)
      (decodedProof (logicalWidth := logicalWidth) (publicFits := publicFits) program base source)
      (decInterface program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount)) (decOffset program circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount))
      (piccsLeaves program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount))
      (rlcInterface program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount))
      (rlcOffset program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)) env decSpec
      (decWired program base source env agreeEnv circuit (sharedEnd base program.witnessWordCount)) piccsSpec (wiredAt env agreeEnv)
      rlcSpec (rlcWired program base source env agreeEnv relation circuit (sharedEnd base program.witnessWordCount))
  have foldSource := foldAt source (fun _ _ => rfl) stepSpec.dec stepSpec.piccs stepSpec.rlc
  have foldFilled := foldAt p4.current (agreeOf p4)
    (RefreshStepLayout.decSpec_any (Lifecycle.PiDEC.v1_2.Formal.soundness relation
      (decInterface program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount)) (decOffset program circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)) p4.current
      (RefreshStepBounds.decAssumptions program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount) below p4.current)
      decRows))
    (RefreshPiCCSLayout.soundness relation (piccsOffset program circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount))
      (piccsInterface program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount)) p4.current
      (RefreshPiCCSBounds.assumptions relation _ _ piccsBelow p4.current) piccsRows relation)
    (RefreshStepLayout.rlcSpec_any (Lifecycle.PiRLC.v1_2.Formal.soundness relation
      (rlcInterface program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount))
      (rlcOffset program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)) p4.current
      (RefreshStepBounds.rlcAssumptions program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount) below p4.current)
      rlcRows))
  have nextEq : RefreshFold.nextSlot relation (rlcInterface program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount))
      (rlcOffset program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)) p4.current =
      RefreshFold.nextSlot relation (rlcInterface program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount))
        (rlcOffset program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)) source :=
    LightFold.toClaim_injective _ (Option.some.inj (foldFilled.symm.trans foldSource))
  have newEq : Hash.evalList p4.current (newSlot program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)).words =
      Hash.evalList source (newSlot program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)).words := by
    rw [SlotExpr.words_eval, SlotExpr.words_eval, RefreshStepSoundness.newSlot_value,
      RefreshStepSoundness.newSlot_value, nextEq]
  -- The new slot's hash.
  have digestEq : ∀ (digest : Fin 4 → Expr) (target : Env)
      (agreeTarget : ∀ column, column < (sharedEnd base program.witnessWordCount) → target column = source column),
      (∀ lane, (digest lane).VarsBelow (sharedEnd base program.witnessWordCount)) → digestValue digest target = digestValue digest source :=
    fun digest target agreeTarget below => congrArg List.ofFn (funext fun lane =>
      shared_eval program base source target agreeTarget _ (below lane))
  have newSpec : OutputHash.SpecHolds (newSlotInterface program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount))
      (newSlotOffset program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)) p4.current := by
    have sourceSpec := stepSpec.newSlot
    change digestValue (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).newDigest source =
      Spec.Poseidon2.hash (Hash.evalList source (slotDomainChunk.map Expr.const ++
        (newSlot program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)).words)) at sourceSpec
    change digestValue (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).newDigest p4.current =
      Spec.Poseidon2.hash (Hash.evalList p4.current (slotDomainChunk.map Expr.const ++
        (newSlot program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)).words))
    have preimageEq : Hash.evalList p4.current (slotDomainChunk.map Expr.const ++
        (newSlot program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)).words) =
        Hash.evalList source (slotDomainChunk.map Expr.const ++
          (newSlot program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)).words) := by
      simp only [Hash.evalList, List.map_append] at newEq ⊢
      rw [newEq]
      simp [Function.comp_def]
    rw [preimageEq, digestEq _ p4.current (agreeOf p4) below.newDigest]
    exact sourceSpec
  obtain ⟨built5, agrees5, rows5⟩ := OutputHash.completeness
    (newSlotInterface program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)) p4.current
    (newSlotOffset program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)) (assumptionsAt p4.current).newSlot newSpec
  obtain ⟨p5, ops5, end5, preserves5, -⟩ :=
    Formal.CompletenessSupport.appendBuiltAt p4 "refresh.new_slot"
      (newSlotCircuit program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount))
      (newSlotOffset program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)) end4
      (OutputHash.flatConstraints_varsBelow _ _ (env := p4.current)
        (assumptionsAt p4.current).newSlot)
      built5 agrees5 rows5
  -- The digest-select rows read only shared words.
  have selectRows5 := DigestSelect.rows_congr (selectInterface program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount))
    (selectOffset program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)) source p5.current
    (agreeOf p5 base (by omega)) (agreeOf p5 (base + 1) (by omega))
    (agreeOf p5 (base + 2) (by omega)) (agreeOf p5 (base + 3) (by omega))
    (agreeOf p5 (base + 4) (by omega))
    (fun index => shared_eval program base source p5.current (agreeOf p5) _
      (below.initialState index))
    (fun index => shared_eval program base source p5.current (agreeOf p5) _
      (below.currentState index))
    (fun slot => shared_words program base source p5.current (agreeOf p5) _ (below.prior slot))
    (shared_words program base source p5.current (agreeOf p5) _ (slotAt_below
      (logicalWidth := logicalWidth) (publicFits := publicFits)
      (slotsStart base program.witnessWordCount + 2 * slotWordCount) (sharedEnd base program.witnessWordCount) (by
        simp only [slotsStart, witnessStart, slotWordCount] at endEq ⊢
        rw [endEq]
        omega)))
    (fun slot => digestEq _ p5.current (agreeOf p5) (below.priorDigest slot))
    (digestEq _ p5.current (agreeOf p5) below.newDigest)
    (fun slot => digestEq _ p5.current (agreeOf p5) (below.innerDigest slot))
    (fun slot => digestEq _ p5.current (agreeOf p5) (below.nextDigest slot))
    selectRows0
  have filledRows5 : holdsFlat p5.current
      (gadgetOps program circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount) ++
        Circuit.ops (Lifecycle.PiDEC.v1_2.Formal.main relation (decInterface program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount)))
          (decOffset program circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)) ++
        RefreshPiCCSLayout.opsAt relation (piccsInterface program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount))
          (piccsOffset program circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)) ++
        Circuit.ops (Lifecycle.PiRLC.v1_2.Formal.main relation
          (rlcInterface program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)))
          (rlcOffset program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)) ++
        [Formal.childOp "refresh.new_slot" (newSlotCircuit program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount))
          (newSlotOffset program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount))]) := by
    have rows := p5.rows
    rw [ops5, ops4, ops3, ops2, ops1] at rows
    simpa only [Sequence.empty, List.nil_append] using rows
  have selectLength : localLength (Circuit.ops (selectCircuit program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount)).main
      (selectOffset program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount))) = 0 :=
    assertions_localLength _
  have split : ∀ (front : List Op) (first second : Op),
      front ++ [first, second] = (front ++ [first]) ++ [second] := by
    intro front first second
    simp
  refine ⟨p5.current, ?_, ?_⟩
  · have lengthEq : localLength (RefreshStepLayout.opsAt program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)) =
        localLength p5.operations := by
      rw [ops5, ops4, ops3, ops2, ops1]
      unfold RefreshStepLayout.opsAt
      rw [split, Sequence.localLength_append, Sequence.localLength_singleton,
        Formal.CompletenessSupport.childOp_localLength]
      unfold Formal.childLength
      rw [selectLength, Nat.add_zero]
      simp only [Sequence.empty, List.nil_append]
      rfl
    change AgreesOutside source p5.current (sharedEnd base program.witnessWordCount)
      (localLength (RefreshStepLayout.opsAt program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)))
    rw [lengthEq]
    exact p5.agrees
  · change holdsFlat p5.current (RefreshStepLayout.opsAt program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount))
    unfold RefreshStepLayout.opsAt
    rw [split]
    unfold holdsFlat
    rw [show ∀ left right : List Op, flatConstraints (left ++ right) =
        flatConstraints left ++ flatConstraints right from
      fun left right => by simp [flatConstraints, List.flatMap_append]]
    refine (constraintsHold_append _ _ _).mpr ⟨filledRows5, ?_⟩
    intro row member
    simp only [flatConstraints, List.flatMap_cons, List.flatMap_nil, List.append_nil] at member
    change row ∈ flatConstraints ((DigestSelect.rows (selectInterface program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount))
      (selectOffset program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount))).map Op.assertZero) at member
    simp only [flatConstraints, List.flatMap_map, List.mem_flatMap, Op.flatConstraints,
      List.mem_singleton] at member
    obtain ⟨original, inside, rfl⟩ := member
    exact selectRows5 _ inside

/-- The refresh step circuit at the source layout as one `FormalCircuit`. Its spec is that every
child's spec holds; it is complete by `completeOfChildren`.
`RefreshAssembler.rows_implies_stepHolds` gives the step relation from its rows and words, and
`RefreshAssemblerCompleteness.complete` gives its rows from an honest refresh step. -/
def stepCircuit (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (circuit : Fin 2)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    FormalCircuit where
  main := RefreshStepLayout.main program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount)
  assumptions := fun offset _ => offset = (sharedEnd base program.witnessWordCount)
  spec := fun offset env =>
    holds env (Circuit.ops (RefreshStepLayout.main program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount)) offset)
  soundness := fun _ _ _ rows => rows
  completeness := by
    intro env offset atEnd children
    subst atEnd
    exact completeOfChildren program base env relation circuit ajtai children

end NightstreamFPrime.Layout.OptionB.RefreshAssemblerCircuit
