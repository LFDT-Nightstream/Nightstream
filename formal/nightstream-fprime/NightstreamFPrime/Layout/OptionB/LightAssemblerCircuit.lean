import NightstreamFPrime.Layout.OptionB.LightAssemblerCompleteness

/-! The light step circuit as one `FormalCircuit` at the source layout (assembler A4). Its spec is
that every child's spec holds, as for today's step circuit (`Lifecycle.Stage1.circuit`). Soundness
is the rows themselves; `LightAssembler.rows_implies_stepHolds` then gives the step relation from
the words. Completeness (`completeOfChildren`) fills the children from their specs: the step
gadgets read only shared words (`completeGadgetsFrom`), the light Π_CCS check accepts the inputs
that the shared words carry (`LightPiCCSPhase.spec_implies_phaseHolds`, then
`LightPiCCSCompleteness.completeFrom`), the Π_RLC fills from its inputs, the new slot's hash fills
because the folded slot is the same before and after the fill, and the digest-select rows read
only shared words (`LightSelect.rows_congr`). -/

namespace NightstreamFPrime.Layout.OptionB.LightAssemblerCircuit

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
open NightstreamFPrime.Lifecycle.OptionB.LightStepLayout
open NightstreamFPrime.Layout.OptionB.LightAssembler

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

/-- The light proof that the shared words carry. -/
def decodedProof : Spec.Folding.Light.Proof LightShape 8 where
  piCcsRounds := fun roundIndex => ((interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).round roundIndex).semanticPolynomial source
  piCcsOutput := ⟨fun origin coefficient => ((interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).output.padCoordinate origin coefficient).eval source,
    fun origin matrix coefficient =>
      ((interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).output.matrixCoordinate origin matrix coefficient).eval source⟩

section Shared

variable (target : Env)
  (agree : ∀ column, column < (sharedEnd base program.witnessWordCount) → target column = source column)

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

private theorem fresh_eq :
    StatementAbsorption.evalFresh (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).fresh target = decodedFresh program base source := by
  have below := inputsBelow (logicalWidth := logicalWidth) (publicFits := publicFits) base
    program.witnessWordCount
  unfold decodedFresh StatementAbsorption.evalFresh
  congr 1
  · funext origin row lane
    exact shared_eval program base source target agree _ (below.freshCommitment origin row lane)
  · funext origin column
    exact shared_eval program base source target agree _ (below.freshPublicInput origin column)

/-- On values that agree below `sharedEnd`, the words carry the decoded light Π_CCS inputs. -/
theorem piccsRepresents (offset : Nat) :
    LightPiCCSLayout.Represents (LightFold.lightRunning (decodedSlot program base source))
      (OptionBFolds.freshToLight (decodedFresh program base source))
      (decodedProof (logicalWidth := logicalWidth) (publicFits := publicFits) program base source) (piccsInterface program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount)) offset target := by
  have below := inputsBelow (logicalWidth := logicalWidth) (publicFits := publicFits) base
    program.witnessWordCount
  refine {
    fresh := fresh_eq program base source target agree
    runningPoint := ?_
    runningPad := fun coefficient => (shared_kEval program base source target agree _
      (below.activePad coefficient)).trans rfl
    runningMatrix := fun matrix coefficient => (shared_kEval program base source target agree _
      (below.activeMatrix matrix coefficient)).trans rfl
    rounds := fun roundIndex => by
      unfold RoundTranscript.Message.semanticPolynomial
      exact FixedChain.Round.semanticPolynomial_eq_of_agree_below _ _ target source
        ⟨fun coefficient => below.round roundIndex coefficient, ⟨trivial, trivial⟩⟩ agree
    outputPad := fun origin coefficient => (shared_kEval program base source target agree _
      (below.outputPad origin coefficient)).trans rfl
    outputMatrix := fun origin matrix coefficient => (shared_kEval program base source target agree
      _ (below.outputMatrix origin matrix coefficient)).trans rfl }
  have pointEq : (fun coordinate => ((interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).active.point coordinate).eval target) =
      fun coordinate => ((interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).active.point coordinate).eval source :=
    funext fun coordinate => shared_kEval program base source target agree _
      (below.activePoint coordinate)
  change (canonicalFinIndices LightShape.cubeVariables).map
      (fun coordinate => ((interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).active.point coordinate).eval target) = _
  rw [pointEq]
  simp [canonicalFinIndices, LightFold.lightRunning, decodedSlot, SlotExpr.value,
    StatementAbsorption.evalPoint]
  rfl

/-- On values that agree below `sharedEnd`, the words carry the decoded light Π_RLC inputs. -/
theorem rlcRepresents (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (circuit : Fin 2) :
    LightRlcLayout.Represents (piccsLeaves program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount))
      (rlcInterface program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount)) (rlcOffset program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount))
      (decodedSlot program base source) (decodedFresh program base source)
      (decodedProof (logicalWidth := logicalWidth) (publicFits := publicFits) program base source).piCcsOutput target := by
  have below := inputsBelow (logicalWidth := logicalWidth) (publicFits := publicFits) base
    program.witnessWordCount
  exact {
    runningCommitment := funext fun row => funext fun lane =>
      shared_eval program base source target agree _ (below.activeCommitment row lane)
    runningPublicInput := funext fun column =>
      shared_eval program base source target agree _ (below.activePublicInput column)
    freshWords := fresh_eq program base source target agree
    outputPad := fun origin coefficient => (shared_kEval program base source target agree _
      (below.outputPad origin coefficient)).trans rfl
    outputMatrix := fun origin matrix coefficient => (shared_kEval program base source target agree
      _ (below.outputMatrix origin matrix coefficient)).trans rfl }

end Shared

private theorem slotSpec_at {words : List Expr} {digest : Fin 4 → Expr} {left right : Nat}
    {env : Env} (spec : OutputHash.SpecHolds (slotHashInterface words digest) left env) :
    OutputHash.SpecHolds (slotHashInterface words digest) right env :=
  spec

private theorem priorStateSpec_at
    (interface : LightStepLayout.Interface logicalWidth publicFits program.witnessWordCount)
    {left right : Nat} {env : Env}
    (spec : PriorStateHash.SpecHolds (priorStateInterface program interface) left env) :
    PriorStateHash.SpecHolds (priorStateInterface program interface) right env :=
  spec

private theorem nextStateSpec_at (circuit : Fin 2)
    (interface : LightStepLayout.Interface logicalWidth publicFits program.witnessWordCount)
    {left right : Nat} {env : Env}
    (spec : OutputHash.SpecHolds (nextStateInterface program circuit interface) left env) :
    OutputHash.SpecHolds (nextStateInterface program circuit interface) right env :=
  spec

private theorem assertions_localLength (expressions : List Expr) :
    localLength (expressions.map Op.assertZero) = 0 := by
  induction expressions with
  | nil => rfl
  | cons _ rest ih =>
      simp only [List.map_cons, localLength, Op.localLength, List.sum_cons, Nat.zero_add]
      simpa [localLength] using ih

/-- Completeness of the light step circuit from its children's specs: when every child's spec
holds on `source`, the children fill into flat rows that agree with `source` outside the step's
private words. -/
theorem completeOfChildren (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (circuit : Fin 2)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (children : holds source (LightStepLayout.opsAt program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount))) :
    ∃ completed, AgreesOutside source completed (sharedEnd base program.witnessWordCount)
        (localLength (Circuit.ops (LightStepLayout.main program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount)) (sharedEnd base program.witnessWordCount))) ∧
      holdsFlat completed
        (Circuit.ops (LightStepLayout.main program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount)) (sharedEnd base program.witnessWordCount)) := by
  have below := inputsBelow (logicalWidth := logicalWidth) (publicFits := publicFits) base
    program.witnessWordCount
  have endEq := sharedEnd_eq base program.witnessWordCount
  have assumptionsAt := LightStepBounds.assumptions program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount) below
  obtain ⟨stepSpec, activeSpec, selectRows0⟩ := LightStepLayout.soundness program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)
    program.step (fun _ _ => circuit) rfl (fun _ _ _ => rfl) source (assumptionsAt source)
    children relation
  have applicationSpec : (applicationCircuit program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount)).spec (sharedEnd base program.witnessWordCount) source := by
    have callHolds := children (Formal.childOp "light.application"
      (applicationCircuit program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount)) (sharedEnd base program.witnessWordCount)) (by
        simp only [LightStepLayout.opsAt, List.mem_append]
        exact Or.inl (Or.inl (Or.inl (by simp [gadgetOps]))))
    change (applicationCircuit program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount)).assumptions (sharedEnd base program.witnessWordCount) source →
      (applicationCircuit program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount)).spec (sharedEnd base program.witnessWordCount) source at callHolds
    exact callHolds (assumptionsAt source).application
  have depthSmall : ((interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).active.depth.eval source).val <
      Folding.Schedule.productionLightFoldCap :=
    DepthGuard.word_lt_cap (rangeInterface program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount)) source
      (rangeOffset program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)) stepSpec.depth stepSpec.depthHigh
  have agreeOf : ∀ completed : Sequence.Prefix source (sharedEnd base program.witnessWordCount), ∀ column, column < (sharedEnd base program.witnessWordCount) →
      completed.current column = source column :=
    fun completed column bound => completed.agrees column (Or.inl bound)
  -- The step gadgets, the light Π_CCS and the light Π_RLC.
  obtain ⟨p1, ops1, end1, preserves1⟩ := LightStepCompleteness.completeGadgetsFrom program
    circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount) relation (Sequence.empty source (sharedEnd base program.witnessWordCount)) rfl below applicationSpec
    depthSmall (slotSpec_at activeSpec)
    (priorStateSpec_at program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) stepSpec.prior)
    (nextStateSpec_at program circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) stepSpec.next)
  have piccsBelow := LightStepBounds.piccsBelow program circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount) below
  have rlcBelow := LightStepBounds.rlcBelow program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount) below
  have wiredAt := fun (env : Env) (agreeEnv : ∀ column, column < (sharedEnd base program.witnessWordCount) → env column = source column) =>
    LightPiCCSLayout.wired relation (piccsOffset program circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)) relation ajtai _ _ _
      (piccsInterface program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount)) env (piccsRepresents program base source env agreeEnv _)
  obtain ⟨p2, ops2, end2, preserves2⟩ := LightPiCCSCompleteness.completeFrom relation
    (piccsInterface program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount)) (piccsOffset program circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)) ajtai
    (LightFold.lightRunning (decodedSlot program base source))
    (OptionBFolds.freshToLight (decodedFresh program base source))
    (decodedProof (logicalWidth := logicalWidth) (publicFits := publicFits) program base source)
    p1 end1 piccsBelow relation (piccsRepresents program base source p1.current (agreeOf p1) _)
    (LightPiCCSPhase.spec_implies_phaseHolds relation ajtai _ _ _ _ source stepSpec.piccs
      (wiredAt source (fun _ _ => rfl))).accepted
  obtain ⟨p3, ops3, end3, preserves3⟩ := LightRlcCompleteness.completeFrom
    (piccsLeaves program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)) (rlcInterface program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount))
    (rlcOffset program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)) p2 end2 rlcBelow
  have filledRows : holdsFlat p3.current (gadgetOps program circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount) ++
      LightPiCCSLayout.opsAt relation (piccsInterface program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount))
        (piccsOffset program circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)) ++
      LightRlcLayout.opsAt (piccsLeaves program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount))
        (rlcInterface program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount)) (rlcOffset program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount))) := by
    have rows := p3.rows
    rw [ops3, ops2, ops1] at rows
    simpa only [Sequence.empty, List.nil_append] using rows
  have rows3 := holdsFlat_implies_holds _ _ filledRows
  simp only [holds_append] at rows3
  obtain ⟨⟨_, piccsRows⟩, rlcRows⟩ := rows3
  -- The folded slot is the same before and after the fill.
  have foldAt := fun (env : Env) (agreeEnv : ∀ column, column < (sharedEnd base program.witnessWordCount) → env column = source column)
      (piccsSpec : LightPiCCSPhase.SpecHolds relation
        (piccsLeaves program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)) env)
      (rlcSpec : LightFold.RlcSpecHolds (rlcLeaves program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)) env) =>
    LightFold.spec_implies_fold relation ajtai (decodedSlot program base source)
      (decodedFresh program base source)
      (decodedProof (logicalWidth := logicalWidth) (publicFits := publicFits) program base source)
      (piccsLeaves program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount))
      (rlcLeaves program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)) env piccsSpec (wiredAt env agreeEnv) rlcSpec
      (LightRlcLayout.wired (piccsLeaves program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount))
        (rlcInterface program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount)) (rlcOffset program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)) _ _ _ env
        (rlcRepresents program base source env agreeEnv relation circuit))
  have foldSource := foldAt source (fun _ _ => rfl) stepSpec.piccs stepSpec.rlc
  have foldFilled := foldAt p3.current (agreeOf p3)
    (LightPiCCSLayout.soundness relation (piccsOffset program circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount))
      (piccsInterface program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount)) p3.current
      (LightPiCCSBounds.assumptions relation _ _ piccsBelow p3.current) piccsRows relation)
    (LightRlcLayout.soundness (piccsLeaves program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount))
      (rlcInterface program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount)) (rlcOffset program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)) p3.current
      (LightRlcBounds.assumptions _ _ _ rlcBelow p3.current) rlcRows)
  have nextEq : LightFold.nextSlot (piccsLeaves program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount))
      (rlcLeaves program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)) p3.current =
      LightFold.nextSlot (piccsLeaves program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount))
        (rlcLeaves program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)) source :=
    LightFold.toClaim_injective _ (Option.some.inj (foldFilled.symm.trans foldSource))
  have depthEq : ((interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).active.depth + 1).eval p3.current = ((interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).active.depth + 1).eval source :=
    shared_eval program base source p3.current (agreeOf p3) _
      (Expr.VarsBelow.add _ _ _ below.activeDepth trivial)
  have newEq : Hash.evalList p3.current (newSlot program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)).words =
      Hash.evalList source (newSlot program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)).words := by
    rw [SlotExpr.words_eval, SlotExpr.words_eval]
    apply congrArg serializeSlot
    apply Prod.ext
    · exact (LightStepSoundness.newSlot_value program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) relation circuit (sharedEnd base program.witnessWordCount) p3.current).trans
        (nextEq.trans
          (LightStepSoundness.newSlot_value program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) relation circuit (sharedEnd base program.witnessWordCount) source).symm)
    · exact congrArg Fin.val depthEq
  -- The new slot's hash.
  have digestEq : ∀ (digest : Fin 4 → Expr) (target : Env)
      (agreeTarget : ∀ column, column < (sharedEnd base program.witnessWordCount) → target column = source column),
      (∀ lane, (digest lane).VarsBelow (sharedEnd base program.witnessWordCount)) → digestValue digest target = digestValue digest source :=
    fun digest target agreeTarget below => congrArg List.ofFn (funext fun lane =>
      shared_eval program base source target agreeTarget _ (below lane))
  have assumptionsAt := LightStepBounds.assumptions program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount) below
  have newSpec : OutputHash.SpecHolds (newSlotInterface program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount))
      (newSlotOffset program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)) p3.current := by
    have sourceSpec := stepSpec.newSlot
    change digestValue (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).newDigest source =
      Spec.Poseidon2.hash (Hash.evalList source (slotDomainChunk.map Expr.const ++
        (newSlot program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)).words)) at sourceSpec
    change digestValue (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).newDigest p3.current =
      Spec.Poseidon2.hash (Hash.evalList p3.current (slotDomainChunk.map Expr.const ++
        (newSlot program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)).words))
    have preimageEq : Hash.evalList p3.current (slotDomainChunk.map Expr.const ++
        (newSlot program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)).words) =
        Hash.evalList source (slotDomainChunk.map Expr.const ++
          (newSlot program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)).words) := by
      simp only [Hash.evalList, List.map_append] at newEq ⊢
      rw [newEq]
      simp [Function.comp_def]
    rw [preimageEq, digestEq _ p3.current (agreeOf p3) below.newDigest]
    exact sourceSpec
  obtain ⟨built4, agrees4, rows4⟩ := OutputHash.completeness
    (newSlotInterface program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)) p3.current
    (newSlotOffset program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)) (assumptionsAt p3.current).newSlot newSpec
  obtain ⟨p4, ops4, end4, preserves4, -⟩ :=
    Formal.CompletenessSupport.appendBuiltAt p3 "light.new_slot"
      (newSlotCircuit program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount))
      (newSlotOffset program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)) end3
      (OutputHash.flatConstraints_varsBelow _ _ (env := p3.current)
        (assumptionsAt p3.current).newSlot)
      built4 agrees4 rows4
  -- The digest-select rows read only shared words.
  have selectRows4 := LightSelect.rows_congr (selectInterface program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount))
    (selectOffset program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)) source p4.current
    (agreeOf p4 base (by omega)) (agreeOf p4 (base + 1) (by omega))
    (agreeOf p4 (base + 2) (by omega)) (agreeOf p4 (base + 3) (by omega))
    (agreeOf p4 (base + 4) (by omega))
    (fun index => shared_eval program base source p4.current (agreeOf p4) _
      (below.initialState index))
    (fun index => shared_eval program base source p4.current (agreeOf p4) _
      (below.currentState index))
    (digestEq _ p4.current (agreeOf p4) below.activeDigest)
    (fun slot => digestEq _ p4.current (agreeOf p4) (below.priorDigest slot))
    (digestEq _ p4.current (agreeOf p4) below.newDigest)
    (fun slot => digestEq _ p4.current (agreeOf p4) (below.innerDigest slot))
    (fun slot => digestEq _ p4.current (agreeOf p4) (below.nextDigest slot))
    selectRows0
  have filledRows4 : holdsFlat p4.current
      (gadgetOps program circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount) ++
        LightPiCCSLayout.opsAt relation (piccsInterface program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount))
          (piccsOffset program circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)) ++
        LightRlcLayout.opsAt (piccsLeaves program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount))
          (rlcInterface program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount)) (rlcOffset program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)) ++
        [Formal.childOp "light.new_slot" (newSlotCircuit program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount))
          (newSlotOffset program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount))]) := by
    have rows := p4.rows
    rw [ops4, ops3, ops2, ops1] at rows
    simpa only [Sequence.empty, List.nil_append] using rows
  have selectLength : localLength (Circuit.ops (selectCircuit program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount)).main
      (selectOffset program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount))) = 0 :=
    assertions_localLength _
  have split : ∀ (front : List Op) (first second : Op),
      front ++ [first, second] = (front ++ [first]) ++ [second] := by
    intro front first second
    simp
  refine ⟨p4.current, ?_, ?_⟩
  · have lengthEq : localLength (LightStepLayout.opsAt program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)) =
        localLength p4.operations := by
      rw [ops4, ops3, ops2, ops1]
      unfold LightStepLayout.opsAt
      rw [split, Sequence.localLength_append, Sequence.localLength_singleton,
        Formal.CompletenessSupport.childOp_localLength]
      unfold Formal.childLength
      rw [selectLength, Nat.add_zero]
      simp only [Sequence.empty, List.nil_append]
    change AgreesOutside source p4.current (sharedEnd base program.witnessWordCount)
      (localLength (LightStepLayout.opsAt program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)))
    rw [lengthEq]
    exact p4.agrees
  · change holdsFlat p4.current (LightStepLayout.opsAt program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount))
    unfold LightStepLayout.opsAt
    rw [split]
    unfold holdsFlat
    rw [show ∀ left right : List Op, flatConstraints (left ++ right) =
        flatConstraints left ++ flatConstraints right from
      fun left right => by simp [flatConstraints, List.flatMap_append]]
    refine (constraintsHold_append _ _ _).mpr ⟨filledRows4, ?_⟩
    intro row member
    simp only [flatConstraints, List.flatMap_cons, List.flatMap_nil, List.append_nil] at member
    change row ∈ flatConstraints ((LightSelect.rows (selectInterface program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount))
      (selectOffset program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount))).map Op.assertZero) at member
    simp only [flatConstraints, List.flatMap_map, List.mem_flatMap, Op.flatConstraints,
      List.mem_singleton] at member
    obtain ⟨original, inside, rfl⟩ := member
    exact selectRows4 _ inside

/-- The light step circuit at the source layout as one `FormalCircuit`. Its spec is that every
child's spec holds; it is complete by `completeOfChildren`. `LightAssembler.rows_implies_stepHolds`
gives the step relation from its rows and words, and
`LightAssemblerCompleteness.complete` gives its rows from an honest light step. -/
def stepCircuit (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (circuit : Fin 2)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    FormalCircuit where
  main := LightStepLayout.main program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount)
  assumptions := fun offset _ => offset = (sharedEnd base program.witnessWordCount)
  spec := fun offset env =>
    holds env (Circuit.ops (LightStepLayout.main program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount)) offset)
  soundness := fun _ _ _ rows => rows
  completeness := by
    intro env offset atEnd children
    subst atEnd
    exact completeOfChildren program base env relation circuit ajtai children

end NightstreamFPrime.Layout.OptionB.LightAssemblerCircuit
