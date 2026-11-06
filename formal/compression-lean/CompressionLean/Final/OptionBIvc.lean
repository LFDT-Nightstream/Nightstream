import CompressionLean.Final.OptionBParent
import CompressionLean.Spec.OptionBIvc
import CompressionLean.Game.Law

/-! The join of the final verifier's bound with option B's IVC bound (CW-10.1).

`lastFold_of_holds`: the claim of `option_b_bound` on the emitted finish package gives a last fold
in the walk's terms (`Spec.OptionB.LastFold`). The proof is that of `parentHolds_of_holds`, with two
more facts of the final program's checks: the iteration is nonzero, and the key digests in the
finish columns are the state keys. When the state keys are the setup's digests
(`Spec.OptionB.KeysAre`), the finish checks hold under the setup's keys (`decodedKeys_of_keys`).

`lastFold_covered`: a last fold whose statement has iteration at most `depth` and no valid advice
reaches it gives, for each answer, a failed finish stage or an IVC false acceptance of the chain's
output (F′'s `FinishWalk.finish_covered`). `chain_event` lifts this to the outcomes of an
experiment. `chain_le` is the union bound on one law: the outcome's law and the finish extractor's
coins. -/

set_option autoImplicit false

namespace CompressionLean.Final.OptionB

open CompressionLean.Spec CompressionLean.Spec.OptionB CompressionLean.OptionB
open CompressionLean.Game
open NightstreamFPrime NightstreamFPrime.Spec NightstreamFPrime.Circuit NightstreamFPrime.Lifecycle
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.HyperNova.ReverseStages (NifsAdversary NifsExtractor ExtractionFails)
open NightstreamFPrime.Spec.HyperNova.Construction2.Paper (TerminalStatement)
open NightstreamFPrime.Layout.OptionB
open NightstreamFPrime.Export.OptionB.StepRelation
open NightstreamFPrime.Export.OptionB.TerminalBridge (Exports decodedKeys_eq)
open NightstreamFPrime.Export.OptionB.FinishWalk
open NightstreamFPrime.Export.OptionB.IvcWalk (WalkSucceeds WalkExtracted Answer Reaches)
open NightstreamFPrime.Export.OptionB.IvcSecurity (FalseAcceptance)
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.RingFPolynomial (modulus)

variable {lw : ℕ} {pf : ringDegree * PaperAlgebra.publicRingColumns ≤ Phi81CarrierLayout.carrierWidth lw}

/-- The key digests that the finish columns carry are the setup's when the state keys are the
setup's digests. -/
theorem decodedKeys_of_keys (program : Stage1.Application.Program)
    {circuitKeys : Fin 2 → Lifecycle.OptionB.CircuitKey lw pf} {stateKeys : List Replay.Word}
    (keys : KeysAre stateKeys circuitKeys) (keysShaped : ∀ key, (circuitKeys key).digest.length = 4)
    {env completed : Env}
    (words : ∀ k (inside : k < stateKeys.length), env (keysWord + k) = Replay.Word.cast stateKeys[k])
    (agree : ∀ k, k < RefreshAssembler.sharedEnd base program.witnessWordCount → completed k = env k) :
    decodedKeys circuitKeys (refreshInterface (logicalWidth := lw) (publicFits := pf) program).verifierKeys
      completed = circuitKeys := by
  have mapped : (stateKeys.map (Replay.Word.cast (F := NightstreamFPrime.Spec.F))).length = 8 := by
    rw [keys, List.length_append, keysShaped, keysShaped]
  have length : stateKeys.length = 8 := by simpa using mapped
  have baseEq : base = 270 := rfl
  refine decodedKeys_eq fun key => List.ext_getElem (by simp [decodedKeys, keysShaped]) fun i inside _ => ?_
  have lane : i < 4 := by simpa [decodedKeys] using inside
  have keyBound := key.isLt
  simp only [decodedKeys, List.getElem_ofFn]
  change completed (LightAssembler.keysStart base + 4 * key.val + i) = (circuitKeys key).digest[i]
  rw [agree _ (by rw [RefreshAssembler.sharedEnd_eq, baseEq]; simp only [LightAssembler.keysStart]; omega),
    show LightAssembler.keysStart base + 4 * key.val + i = keysWord + (4 * key.val + i) by
      simp only [LightAssembler.keysStart, keysWord, publicWords, baseEq]; omega,
    words _ (by omega)]
  have entry := List.getElem_of_eq keys (show 4 * key.val + i < (stateKeys.map
    (Replay.Word.cast (F := NightstreamFPrime.Spec.F))).length by rw [mapped]; omega)
  rw [List.getElem_map] at entry
  rw [entry, List.getElem_append]
  fin_cases key
  · simp [keysShaped, lane]
  · simp [keysShaped]

/-- The final terminal on the emitted finish package, with both slot statements in their layer-1
relations and the state keys the setup's digests, gives a last fold. -/
theorem lastFold_of_holds (program : Stage1.Application.Program)
    (relation : ProductionKey.LogicalRelation lw pf)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := lw) (publicFits := pf))
    (circuitKeys : Fin 2 → Lifecycle.OptionB.CircuitKey lw pf) (next width : ℕ)
    (batches : List NightstreamFPrime.Circuit.WitnessBatch) (rest : ℕ × ℕ × ℕ × ℕ)
    (packed : Export.Generic.CircuitRelation.packable Export.Generic.CircuitRelation.publicColumns
      (Export.OptionB.FinishRelation.pieces program relation) = true)
    (starts : Export.Generic.CircuitRelation.startsFrom (RefreshAssembler.sharedEnd base program.witnessWordCount)
      (Export.OptionB.FinishRelation.pieces program relation) = true)
    (s : Fin 2 → Layer1.Shape) (S : Fin 2 → Layer1.Setup Gl) (bar : ℕ → ℕ → Gl)
    (barEq : ∀ p l : Fin ringDegree, bar p l = gl (Phi81CoefficientKernel.nativeBarEntry p l))
    (shape : ∀ slot, ParentShape (s slot) lw) (blocks_le : ∀ slot, (s slot).blocks ≤ 2 ^ (s slot).blockVars)
    (honest : ∀ slot, HonestSetup (s slot) (S slot) (circuitKeys slot).relation ajtai)
    {stateKeys : List Replay.Word} (keys : KeysAre stateKeys circuitKeys)
    (keysShaped : ∀ key, (circuitKeys key).digest.length = 4) {x : StateWords}
    (holds : Holds (s 0) (s 1) modulus bar (S 0) (S 1)
      (FinishTerminal (finishPackage program relation next width batches rest) stateKeys)
      (slotClaim (finishPackage program relation next width batches rest) 0)
      (slotClaim (finishPackage program relation next width batches rest) 1) x) :
    LastFold program ajtai circuitKeys x := by
  obtain ⟨env, terminal, ⟨z0, rel0⟩, z1, rel1⟩ := holds
  obtain ⟨completed, agree, iteration, start, current, finish⟩ :=
    finishTerminal_holds program relation ajtai circuitKeys next width batches (program.witnessWordCount, rest)
      packed starts terminal
  obtain ⟨positive, -, -, -, words, -, -⟩ := terminal
  rw [decodedKeys_of_keys program keys keysShaped words agree] at finish
  have slotHolds : ∀ (slot : Fin 2) (z : ℕ → Gl),
      Layer1.Relation (s slot) modulus bar (S slot)
        (slotClaim (finishPackage program relation next width batches rest) slot env) z →
      ∃ witness, Spec.CE.Holds (PaperAlgebra.semantics ajtai) productionGlobalParams
        (Lifecycle.OptionB.toClaim ((circuitKeys slot).system ajtai)
          ((refreshOutput (logicalWidth := lw) (publicFits := pf) program 1 completed).runningNext slot).1)
        witness := by
    intro slot z relation_holds
    rw [slotClaim_eq (lw := lw) (pf := pf) program _ rfl
      (PiRLC.v1_2.InputBinding.relationSource (circuitKeys slot).relation) env completed agree slot]
      at relation_holds
    rw [system_eq]
    exact ⟨_, slot_holds (s slot) bar (S slot) (circuitKeys slot).relation ajtai (shape slot) (blocks_le slot)
      barEq (honest slot) _ relation_holds⟩
  have both : ∀ slot : Fin 2, ∃ witness, Spec.CE.Holds (PaperAlgebra.semantics ajtai) productionGlobalParams
      (Lifecycle.OptionB.toClaim ((circuitKeys slot).system ajtai)
        ((refreshOutput (logicalWidth := lw) (publicFits := pf) program 1 completed).runningNext slot).1)
      witness := by
    intro slot
    fin_cases slot
    · exact slotHolds 0 z0 rel0
    · exact slotHolds 1 z1 rel1
  choose witnesses witnessHolds using both
  refine ⟨completed, witnesses, positive, ?_, finish, witnessHolds⟩
  unfold finishStatement terminalOf
  rw [iteration, start, current]

/-- The deterministic link of CW-10.1: a last fold whose statement has iteration at most `depth`
and that no valid advice reaches gives, for each answer, a failed finish stage or an IVC false
acceptance of the chain's output. -/
theorem lastFold_covered {relation : ProductionKey.LogicalRelation lw pf}
    {ajtai : PaperAlgebra.AjtaiKey (logicalWidth := lw) (publicFits := pf)}
    {circuitKeys : Fin 2 → Lifecycle.OptionB.CircuitKey lw pf} {step : AppState → AppWitness → AppState}
    {cubeFits : Phi81CarrierLayout.carrierWidth lw ≤ 2 ^ cubeVariables}
    (exports : Exports relation ajtai circuitKeys step cubeFits)
    (keysShaped : ∀ key, (circuitKeys key).digest.length = 4)
    (control : AppState → AppWitness → Fin 2) (cap depth : ℕ) {x : StateWords}
    (fold : LastFold exports.program ajtai circuitKeys x) (shallow : (x 0).val ≤ depth)
    (unreached : ¬ Reaches exports.program (terminalOf x)) (answer : Answer lw pf) :
    (WalkSucceeds ajtai (lastAttempt exports.program ajtai circuitKeys x) ∧
        ¬ WalkExtracted ajtai (lastAttempt exports.program ajtai circuitKeys x) answer) ∨
      FalseAcceptance ajtai circuitKeys exports control cap
        (chainOutput exports.program ajtai circuitKeys depth x answer) := by
  obtain ⟨env, witnesses, holds⟩ := fold
  have chosen : LastFoldAt exports.program ajtai circuitKeys x
      (lastChoice exports.program ajtai circuitKeys x).1 (lastChoice exports.program ajtai circuitKeys x).2 :=
    Classical.epsilon_spec (p := fun choice => LastFoldAt exports.program ajtai circuitKeys x choice.1 choice.2)
      ⟨(env, witnesses), holds⟩
  obtain ⟨positive, statementEq, finish, slots⟩ := chosen
  have outputEq : chainOutput exports.program ajtai circuitKeys depth x answer =
      finishOutput (refreshInput exports.program (lastChoice exports.program ajtai circuitKeys x).1)
        (lastChoice exports.program ajtai circuitKeys x).2 answer := by
    rw [chainOutput, ite_eq_left shallow, finishOutput, statementEq]
  have iterationEq := congrArg (fun statement : TerminalStatement AppState => statement.iteration) statementEq
  rw [outputEq]
  refine finish_covered ajtai circuitKeys exports control cap _
    (refresh_wellFormed _ ajtai circuitKeys keysShaped _) (refresh_valid _ _) ?_ finish slots
    (statementEq ▸ unreached) answer
  change (refreshInput exports.program (lastChoice exports.program ajtai circuitKeys x).1).iteration =
    (x 0).val at iterationEq
  rw [iterationEq]
  exact Nat.pos_of_ne_zero fun zero => positive (Fin.ext zero)

/-- A statement's iteration bounds the iteration of the chain's output. -/
theorem chainOutput_iteration_le (program : Stage1.Application.Program)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := lw) (publicFits := pf))
    (circuitKeys : Fin 2 → Lifecycle.OptionB.CircuitKey lw pf) (depth : ℕ) (x : StateWords)
    (answer : Answer lw pf) :
    (chainOutput program ajtai circuitKeys depth x answer).1.iteration ≤ depth := by
  unfold chainOutput
  split_ifs with shallow
  · exact shallow
  · exact Nat.zero_le _

section Break

variable {F K : Type} [Field F] [Field K] [Algebra F K] [Fintype K] [DecidableEq K] {t0 L0 t1 L1 : ℕ}

omit [Field K] [Algebra F K] [Fintype K] [DecidableEq K] in
/-- A break of a claim names an accepted statement that fails the claim, and it is a break of a
second claim that the statement fails. -/
theorem restoreBreak_accepted {Claim Claim' : StateWords → Prop}
    (T : Shrink.Template F) (s : Shrink.Counts) (prog : StateWords → Shrink.Run F)
    (g0 : Game (Option (Layer1.Transcript F K t0 L0))) (g1 : Game (Option (Layer1.Transcript F K t1 L1)))
    (Reads0 : StateWords → (ℕ → F) → g0.Pos → Prop) (Reads1 : StateWords → (ℕ → F) → g1.Pos → Prop)
    (r : Outcome F K g0 g1) (broken : RestoreBreak Claim T s prog g0 g1 Reads0 Reads1 r) :
    ∃ x, acceptedOf r.1 = some x ∧ ¬ Claim x ∧
      (¬ Claim' x → RestoreBreak Claim' T s prog g0 g1 Reads0 Reads1 r) := by
  obtain ⟨tr, accepted, reads, failed⟩ := broken
  refine ⟨tr.inst, ?_, failed, fun failed' => ⟨tr, accepted, reads, failed'⟩⟩
  rw [acceptedOf, accepted]
  rfl

end Break

/-- The finish extractor's draws give the outcome's law on the outcome. -/
theorem law_map_fst {Attempt Answer : Type} (adversary : NifsAdversary Attempt)
    (extractor : NifsExtractor Answer adversary) : extractor.law.map Prod.fst = adversary.tape := by
  change (adversary.tape.bind fun tape => extractor.coins.map (tape, ·)).map Prod.fst = _
  rw [PMF.map_bind]
  calc _ = adversary.tape.bind PMF.pure := by
        congr 1
        funext tape
        rw [PMF.map_comp]
        exact PMF.map_const extractor.coins tape
    _ = adversary.tape := PMF.bind_pure _

/-- The union bound on one law: an event of the outcome that each draw covers with a break of the
outcome, a failed extraction or a win of the output has at most the sum of their probabilities. -/
theorem chain_le {Attempt Answer Output : Type} {Succeeds : Attempt → Prop}
    {Extracted : Attempt → Answer → Prop} (adversary : NifsAdversary Attempt)
    (extractor : NifsExtractor Answer adversary) (output : adversary.Tape × extractor.Coins → Output)
    (Target Break : adversary.Tape → Prop) (Wins : Output → Prop)
    (covered : ∀ draw : adversary.Tape × extractor.Coins, Target draw.1 →
      Break draw.1 ∨ ExtractionFails Succeeds Extracted adversary extractor draw ∨ Wins (output draw)) :
    (adversary.tape.toOuterMeasure {tape | Target tape}).toReal ≤
      (adversary.tape.toOuterMeasure {tape | Break tape}).toReal +
        (extractor.law.toOuterMeasure
          {draw | ExtractionFails Succeeds Extracted adversary extractor draw}).toReal +
        (extractor.law.toOuterMeasure {draw | Wins (output draw)}).toReal := by
  have marginal : ∀ P : adversary.Tape → Prop, adversary.tape.toOuterMeasure {tape | P tape} =
      extractor.law.toOuterMeasure {draw | P draw.1} := by
    intro P
    have mapped := congrArg (fun law => PMF.toOuterMeasure law {tape | P tape})
      (law_map_fst adversary extractor)
    simp only [PMF.toOuterMeasure_map_apply] at mapped
    exact mapped.symm
  rw [marginal Target, marginal Break]
  have bound : extractor.law.toOuterMeasure {draw | Target draw.1} ≤
      extractor.law.toOuterMeasure {draw | Break draw.1} +
        extractor.law.toOuterMeasure {draw | ExtractionFails Succeeds Extracted adversary extractor draw} +
        extractor.law.toOuterMeasure {draw | Wins (output draw)} :=
    calc _ ≤ extractor.law.toOuterMeasure ({draw | Break draw.1} ∪
          {draw | ExtractionFails Succeeds Extracted adversary extractor draw} ∪
          {draw | Wins (output draw)}) :=
          MeasureTheory.measure_mono fun draw target => by
            rcases covered draw target with broken | failed | wins
            · exact Or.inl (Or.inl broken)
            · exact Or.inl (Or.inr failed)
            · exact Or.inr wins
      _ ≤ _ := (MeasureTheory.measure_union_le _ _).trans
          (add_le_add (MeasureTheory.measure_union_le _ _) le_rfl)
  have finite : ∀ event : Set (adversary.Tape × extractor.Coins), extractor.law.toOuterMeasure event ≠ ⊤ :=
    event_ne_top _
  refine (ENNReal.toReal_mono (ENNReal.add_ne_top.mpr ⟨ENNReal.add_ne_top.mpr ⟨finite _, finite _⟩, finite _⟩)
    bound).trans (le_of_eq ?_)
  rw [ENNReal.toReal_add (ENNReal.add_ne_top.mpr ⟨finite _, finite _⟩) (finite _),
    ENNReal.toReal_add (finite _) (finite _)]

/-- The deterministic event inclusion of CW-10.1 on the outcomes of an experiment: when each target
outcome accepts a statement of iteration at most `depth` that no valid advice reaches and is a
break unless the statement has a last fold, each draw of the outcome and the finish extractor's
coins is a break, a failed finish extraction, or an IVC false acceptance of the chain's output. -/
theorem chain_event {relation : ProductionKey.LogicalRelation lw pf}
    {ajtai : PaperAlgebra.AjtaiKey (logicalWidth := lw) (publicFits := pf)}
    {circuitKeys : Fin 2 → Lifecycle.OptionB.CircuitKey lw pf} {step : AppState → AppWitness → AppState}
    {cubeFits : Phi81CarrierLayout.carrierWidth lw ≤ 2 ^ cubeVariables}
    (exports : Exports relation ajtai circuitKeys step cubeFits)
    (keysShaped : ∀ key, (circuitKeys key).digest.length = 4)
    (control : AppState → AppWitness → Fin 2) (cap depth : ℕ) {Ω : Type} (law : PMF Ω)
    (accepted : Ω → Option StateWords)
    (extractor : NifsExtractor (Answer lw pf) (finishAdversary exports.program ajtai circuitKeys law accepted))
    (Target Break : Ω → Prop)
    (target : ∀ ω, Target ω → ∃ x, accepted ω = some x ∧ (x 0).val ≤ depth ∧
      ¬ Reaches exports.program (terminalOf x) ∧ (¬ LastFold exports.program ajtai circuitKeys x → Break ω))
    (draw : Ω × extractor.Coins) (hit : Target draw.1) :
    Break draw.1 ∨
      ExtractionFails (WalkSucceeds ajtai) (WalkExtracted ajtai) _ extractor draw ∨
      FalseAcceptance ajtai circuitKeys exports control cap
        ((chainAdversary exports.program ajtai circuitKeys law accepted extractor depth).output draw) := by
  obtain ⟨x, acceptedEq, shallow, unreached, broken⟩ := target draw.1 hit
  by_cases fold : LastFold exports.program ajtai circuitKeys x
  · have outputEq : (chainAdversary exports.program ajtai circuitKeys law accepted extractor depth).output draw =
        chainOutput exports.program ajtai circuitKeys depth x (extractor.run draw.1 draw.2) := by
      change chainOutput exports.program ajtai circuitKeys depth ((accepted draw.1).getD 0) _ = _
      rw [acceptedEq]
      rfl
    rw [outputEq]
    rcases lastFold_covered exports keysShaped control cap depth fold shallow unreached
      (extractor.run draw.1 draw.2) with ⟨wins, fails⟩ | falseAcceptance
    · refine Or.inr (Or.inl ⟨lastAttempt exports.program ajtai circuitKeys x, ?_, wins, fails⟩)
      change (accepted draw.1).map (lastAttempt exports.program ajtai circuitKeys) = _
      rw [acceptedEq]
      rfl
    · exact Or.inr (Or.inr falseAcceptance)
  · exact Or.inl (broken fold)

end CompressionLean.Final.OptionB
