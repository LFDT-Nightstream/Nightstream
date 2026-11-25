import NightstreamFPrime.Export.BoundedDepth.FinalKey20All
import NightstreamFPrime.Export.BoundedDepth.FinalRows
import NightstreamFPrime.Export.BoundedDepth.IvcSecurity
import NightstreamFPrime.Spec.Compression.Game.Law

/-! The final verifier over the Akita openings of `key20.aks` against a false history (SPEC.md,
Algorithm 10; `SECURITY_MODEL.md`, adversaries 4 and 6): `FinalHistory.soundness` and
`FinalHistory.knowledge` in the game with Akita openings.

A false history is a table of the oracle for which the final verifier `finalAccepts` accepts
`(i, z0, zi)` and no `i` valid steps of the application lead from `z0` to `zi` (`FalseHistory`).

A prover of the final verifier with the extractor of Akita Theorem 10.17 is an IVC adversary
(`adversary`). Its tape is the table of the oracle and the tape of the extractor, with the
uniform law. Its output is the accepted statement with the proof `finalProof` of the extractor. A
false history is a tape for which `Verifies` rejects this proof, or a false acceptance of this
adversary. So its probability is at most `shapesBound` (`rejected_le`, from `knowledge_all`) plus
the bound of HyperNova Lemma 18 for this adversary, `IvcSecurity.walkBound` (`history_sound`).
The same bound holds for the witnesses that the reverse extractor returns
(`history_knowledge`). -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.BoundedDepth.FinalKey20

open NightstreamFPrime.Spec.Compression.Akita.Replay (CodeSlot)

open scoped BigOperators
open NightstreamFPrime.Spec NightstreamFPrime.Spec.FieldExtensionRing
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.HyperNova.Construction2
open NightstreamFPrime.Spec.HyperNova.Construction2.Paper
open NightstreamFPrime.Spec.Compression.Production
open NightstreamFPrime.Spec.Compression.Akita NightstreamFPrime.Spec.Compression.Akita.Extract
open NightstreamFPrime.Spec.Checkpoint
open NightstreamFPrime.Spec.RandomOracle (OracleComp)
open NightstreamFPrime.Lifecycle NightstreamFPrime.Lifecycle.BoundedDepth
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Export.BoundedDepth.RunCompleteness
open NightstreamFPrime.Export.BoundedDepth.FinalVerifier (Mode Claims Premises)
open NightstreamFPrime.Export.BoundedDepth.FinalAkita

attribute [local instance low] Classical.propDecidable

attribute [local instance] NightstreamFPrime.Export.BoundedDepth.FinalKey20.instAlgebraZqK
  NightstreamFPrime.Export.BoundedDepth.FinalKey20.instDegree2K

/-- The measure of an event under the uniform law is the mean of its indicator
(`Spec.Compression.Game.prob_eq_law` for one draw). -/
theorem uniform_event {Sample : Type} [Fintype Sample] [Nonempty Sample]
    (event : Sample → Prop) [DecidablePred event] :
    ((Compression.Game.uniform Sample).toOuterMeasure {sample | event sample}).toReal =
      𝔼 sample, if event sample then (1 : ℝ) else 0 := by
  rw [← PMF.bind_pure (Compression.Game.uniform Sample)]
  refine (Compression.Game.prob_eq_law (fun M _ _ => Classical.arbitrary M)
    (.draw Sample ∅ .pure) event).symm.trans (Finset.expect_congr rfl fun sample _ => ?_)
  simp only [Compression.Game.prob]
  congr

variable {logicalWidth : Nat}
  {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
    RingCarrierLayout.carrierWidth logicalWidth}
  {mode : Mode} (setup : FinalVerifier.Setup mode logicalWidth publicFits)
  (packageWidths : Fin 2 → ℕ) (step : AppState → AppWitness → AppState)
  (control : AppState → AppWitness → Fin 2) {Wire Coord : Type} {total : ℕ} (read : Reader Wire)
  (head : Wire → First logicalWidth publicFits)
  [Fintype Coord] [DecidableEq Coord] [Nonempty Coord] {ι : Type} [Fintype ι]
  (kind : Wire → Option ι) (family : ι → Opening logicalWidth publicFits Wire Coord total)
  (prover : OracleComp (Address Wire total) (Option Coord → CodeSlot K) (ℕ → Wire))
    [Finite Wire]
  (budget : ℕ)

/-- The prover of the final verifier over the Akita openings of `key20.aks`, with the extractor
of at most `budget` attempts in each retry loop, as an IVC adversary. Its tape is the table of the
oracle and the tape of the extractor, with the uniform law. Its output is the accepted statement
with the proof `finalProof` of the extractor, and otherwise the bottom proof of `(0, [], [])`. -/
noncomputable def adversary :
    HyperNova.ReverseStages.IvcAdversary (IvcSecurity.Output logicalWidth publicFits) where
  Tape := (Address Wire total → Option Coord → CodeSlot K) ×
    Tape (Address Wire total) (Option Coord) (CodeSlot K) budget total
  tape := Compression.Game.uniform _
  output sample :=
    if finalAccepts setup packageWidths step control read head kind family
        (played prover sample.1) then
      ((head (played prover sample.1).first).statement,
        finalProof setup packageWidths step control read head kind family prover budget
          sample.1 sample.2)
    else (⟨0, [], []⟩, .bottom)

/-- The final verifier accepts `(i, z0, zi)` for the table of the oracle, and no `i` valid steps
of `program`, each with a witness of the word count of the program, lead from `z0` to `zi`. -/
def FalseHistory (program : Stage1.Application.Program)
    (table : Address Wire total → Option Coord → CodeSlot K) : Prop :=
  finalAccepts setup packageWidths step control read head kind family (played prover table) ∧
    ¬ IvcWalk.Reaches program (head (played prover table).first).statement

section Uniform

local notation "ivc" =>
  adversary setup packageWidths step control read head kind family prover budget

/-- The final verifier accepts for the table of the oracle, and `Verifies` rejects the proof
`finalProof` of the extractor for the tape: the event of `knowledge_all`. -/
def Rejected (sample : (Address Wire total → Option Coord → CodeSlot K) ×
    Tape (Address Wire total) (Option Coord) (CodeSlot K) budget total) : Prop :=
  finalAccepts setup packageWidths step control read head kind family
      (played prover sample.1) ∧
    ¬ Verifies setup.ajtai setup.circuitKeys step packageWidths control
      (head (played prover sample.1).first).statement
      (finalProof setup packageWidths step control read head kind family prover budget
        sample.1 sample.2)

local notation "rejects" =>
  Rejected setup packageWidths step control read head kind family prover budget

/-- The probability of `Rejected` under the law of the tape of `adversary` is at most
`shapesBound` (`knowledge_all` under the uniform law). -/
theorem rejected_le {p : ℝ} (premises : Premises setup packageWidths)
    (blocks : setup.keyBlocks = 512) (coords : 30 ≤ Fintype.card Coord)
    (sized : ∀ index,
      (family index).Sized setup packageWidths step control read head p) {queries : ℕ}
    (bounded : prover.QueryBound queries) (positive : 0 < budget) :
    ((ivc).tape.toOuterMeasure {sample | rejects sample}).toReal ≤
      shapesBound setup packageWidths step control read head family prover p queries
        budget := by
  refine (uniform_event (Sample := (Address Wire total → Option Coord → CodeSlot K) ×
    Tape (Address Wire total) (Option Coord) (CodeSlot K) budget total) _).trans_le
    (le_trans (le_of_eq ?_) (knowledge_all setup packageWidths step control read head kind
      family prover premises blocks coords sized bounded positive))
  rw [← Finset.univ_product_univ, Finset.expect_product]
  exact Finset.expect_congr rfl fun table _ => Finset.expect_congr rfl fun tape _ =>
    if_congr Iff.rfl rfl rfl

omit [Nonempty Coord] [Fintype ι] in
/-- The statement of each output of `adversary` has at most `depth` steps, when each accepted
statement has at most `depth` steps. -/
theorem adversary_depth (depth : ℕ)
    (depthBound : ∀ table,
      finalAccepts setup packageWidths step control read head kind family
        (played prover table) → (head (played prover table).first).statement.iteration ≤ depth)
    (sample : (ivc).Tape) : ((ivc).output sample).1.iteration ≤ depth := by
  by_cases accepted : finalAccepts setup packageWidths step control read head kind family
    (played prover sample.1)
  · rw [show (ivc).output sample = _ from ite_eq_left accepted]
    exact depthBound _ accepted
  · rw [show (ivc).output sample = _ from ite_eq_right accepted]
    exact Nat.zero_le _

end Uniform

section History

variable {relation : ProductionKey.LogicalRelation logicalWidth publicFits}
  {cubeFits : RingCarrierLayout.carrierWidth logicalWidth ≤ 2 ^ cubeVariables}
  (exports : TerminalBridge.Exports relation setup.ajtai setup.circuitKeys step cubeFits)
  {Admitted : HyperNova.ReverseStages.NifsAdversary (IvcWalk.WalkAttempt logicalWidth publicFits) →
    Prop}
  {StageAdmitted : HyperNova.ReverseStages.Stage (IvcWalk.Visit logicalWidth publicFits)
    (IvcWalk.Answer logicalWidth publicFits) → Prop}
  {Efficient : (adversary :
      HyperNova.ReverseStages.NifsAdversary (IvcWalk.WalkAttempt logicalWidth publicFits)) →
    HyperNova.ReverseStages.NifsExtractor (IvcWalk.Answer logicalWidth publicFits) adversary →
      Prop}
  {error : HyperNova.ReverseStages.NifsAdversary (IvcWalk.WalkAttempt logicalWidth publicFits) → ℝ}
  (assumption : HyperNova.ReverseStages.Assumption1 (IvcWalk.WalkSucceeds setup.ajtai)
    (IvcWalk.WalkExtracted setup.ajtai) Admitted Efficient error)
  (closed : HyperNova.ReverseStages.Closed (IvcWalk.attemptOf setup.ajtai setup.circuitKeys exports)
    (IvcWalk.back setup.ajtai setup.circuitKeys exports) Admitted StageAdmitted Efficient)
  (complete : StepExtend.Complete exports)

local notation "ivc" =>
  adversary setup complete.packageWidth step control read head kind family prover budget

local notation "rejects" =>
  Rejected setup complete.packageWidth step control read head kind family prover budget

/-- The final verifier with Akita openings against a false history, in the form of
`Claims.Final.history_sound`. For each prover with at most `Q` queries to the oracle, the
probability of a false history is at most `shapesBound`, the bound of `verifier_sound`, plus
`IvcSecurity.walkBound` of the IVC adversary `adversary`: for each of the first `depth` stages
of the reverse extractor, the success chance of the Poseidon2 collision finder of the stage
(`IvcSecurity.finderChance`) and the error of the stage. The probability is over the table of
the oracle and the tape of the extractor. `rows`: the rows of the setup are the rows of the
packages (`FinalRows.premises_of_rows`). `depthBound`: each accepted statement has at most `depth`
steps. -/
theorem history_sound {p : ℝ}
    (rows : ∀ circuit : Fin 2, (setup.rows circuit).matrices =
      (FinalRows.packageRows exports complete circuit).matrices)
    (keysShaped : ∀ key, (setup.circuitKeys key).digest.length = 4)
    (blocks : setup.keyBlocks = 512) (coords : 30 ≤ Fintype.card Coord)
    (sized : ∀ index,
      (family index).Sized setup complete.packageWidth step control read head p) {queries : ℕ}
    (bounded : prover.QueryBound queries) (positive : 0 < budget)
    (admitted : StageAdmitted
      (HyperNova.ReverseStages.Stage.start (IvcSecurity.walkAdversary ivc))) (depth : ℕ)
    (depthBound : ∀ table,
      finalAccepts setup complete.packageWidth step control read head kind family
        (played prover table) → (head (played prover table).first).statement.iteration ≤ depth) :
    ((ivc).tape.toOuterMeasure {sample |
        FalseHistory setup complete.packageWidth step control read head kind family prover
          exports.program sample.1}).toReal ≤
      shapesBound setup complete.packageWidth step control read head family prover p queries
          budget +
        IvcSecurity.walkBound setup.ajtai setup.circuitKeys exports assumption closed ivc admitted
          depth := by
  have rejected := rejected_le setup complete.packageWidth step control read head kind family
    prover budget (FinalRows.premises_of_rows setup exports complete rows) blocks coords sized
    bounded positive
  have accepted := IvcSecurity.probability_bound setup.ajtai setup.circuitKeys exports assumption
    closed ivc admitted keysShaped control depth fun sample _ =>
      adversary_depth setup complete.packageWidth step control read head kind family prover
        budget depth depthBound sample
  have cover : {sample : (ivc).Tape |
      FalseHistory setup complete.packageWidth step control read head kind family prover
        exports.program sample.1} ⊆
      {sample | rejects sample} ∪
        {sample | IvcSecurity.FalseAcceptance setup.ajtai setup.circuitKeys exports control
          ((ivc).output sample)} := by
    rintro sample ⟨accepts, unreached⟩
    by_cases verifies : Verifies setup.ajtai setup.circuitKeys step complete.packageWidth
      control
      (head (played prover sample.1).first).statement
      (finalProof setup complete.packageWidth step control read head kind family prover budget
        sample.1 sample.2)
    · refine Or.inr ?_
      show IvcSecurity.FalseAcceptance _ _ _ _ ((ivc).output sample)
      rw [show (ivc).output sample = _ from ite_eq_left accepts]
      exact ⟨holds_of_verifies verifies, unreached⟩
    · exact Or.inl ⟨accepts, verifies⟩
  calc ((ivc).tape.toOuterMeasure _).toReal
      ≤ ((ivc).tape.toOuterMeasure _ + (ivc).tape.toOuterMeasure _).toReal :=
        ENNReal.toReal_mono
          (ENNReal.add_ne_top.mpr ⟨HyperNova.ReverseStages.event_ne_top _ _,
            HyperNova.ReverseStages.event_ne_top _ _⟩)
          ((MeasureTheory.measure_mono cover).trans (MeasureTheory.measure_union_le _ _))
    _ = ((ivc).tape.toOuterMeasure _).toReal + ((ivc).tape.toOuterMeasure _).toReal :=
        ENNReal.toReal_add (HyperNova.ReverseStages.event_ne_top _ _)
          (HyperNova.ReverseStages.event_ne_top _ _)
    _ ≤ _ := add_le_add rejected accepted

local notation:max "finalStages" admitted:max =>
  HyperNova.ReverseStages.reverseStages assumption closed (IvcSecurity.walkAdversary ivc) admitted

local notation:max "start" admitted:max =>
  HyperNova.ReverseStages.origin assumption closed (IvcSecurity.walkAdversary ivc) admitted

/-- The final verifier accepted the statement of the table that the stage `depth` of the reverse
extractor reads from its tape, and the witnesses that the extractor decodes from the answers of
the stage are not `iteration` valid steps of the application that lead from `z0` to `zi`
(HyperNova Definition 11 (ii)). -/
def AdviceFailure
    (admitted : StageAdmitted
      (HyperNova.ReverseStages.Stage.start (IvcSecurity.walkAdversary ivc))) (depth : ℕ)
    (tape : (finalStages admitted depth).1.Tape) : Prop :=
  finalAccepts setup complete.packageWidth step control read head kind family
      (played prover ((start admitted depth tape : (ivc).Tape).1)) ∧
    ¬ IvcWalk.ReachedBy exports.program
      (head (played prover ((start admitted depth tape : (ivc).Tape).1)).first).statement
      (IvcSecurity.extracted setup.ajtai setup.circuitKeys exports
        ((finalStages admitted depth).1.input tape) ((finalStages admitted depth).1.results tape))

/-- The knowledge form of `history_sound`, in the form of `Claims.Final.history_knowledge`: the
probability that the final verifier accepts and the witnesses that the reverse extractor returns
are not a history of the accepted statement is at most `shapesBound` plus `IvcSecurity.walkBound`.
A tape with `Rejected` at the start of the walk has the bound of `rejected_le`. For each other
tape the extraction fails at the start of the walk (`IvcSecurity.extraction_bound`). -/
theorem history_knowledge {p : ℝ}
    (rows : ∀ circuit : Fin 2, (setup.rows circuit).matrices =
      (FinalRows.packageRows exports complete circuit).matrices)
    (keysShaped : ∀ key, (setup.circuitKeys key).digest.length = 4)
    (blocks : setup.keyBlocks = 512) (coords : 30 ≤ Fintype.card Coord)
    (sized : ∀ index,
      (family index).Sized setup complete.packageWidth step control read head p) {queries : ℕ}
    (bounded : prover.QueryBound queries) (positive : 0 < budget)
    (admitted : StageAdmitted
      (HyperNova.ReverseStages.Stage.start (IvcSecurity.walkAdversary ivc))) (depth : ℕ)
    (depthBound : ∀ table,
      finalAccepts setup complete.packageWidth step control read head kind family
        (played prover table) → (head (played prover table).first).statement.iteration ≤ depth) :
    ((finalStages admitted depth).1.tape.toOuterMeasure {tape |
        AdviceFailure setup step control read head kind family prover budget exports
          assumption closed complete admitted depth tape}).toReal ≤
      shapesBound setup complete.packageWidth step control read head family prover p queries
          budget +
        IvcSecurity.walkBound setup.ajtai setup.circuitKeys exports assumption closed ivc admitted
          depth := by
  have rejected := rejected_le setup complete.packageWidth step control read head kind family
    prover budget (FinalRows.premises_of_rows setup exports complete rows) blocks coords sized
    bounded positive
  have failed := IvcSecurity.extraction_bound setup.ajtai setup.circuitKeys exports assumption
    closed ivc admitted keysShaped control depth fun sample _ =>
      adversary_depth setup complete.packageWidth step control read head kind family prover
        budget depth depthBound sample
  have cover : {tape | AdviceFailure setup step control read head kind family prover budget
        exports assumption closed complete admitted depth tape} ⊆
      start admitted depth ⁻¹' {sample : (ivc).Tape | rejects sample} ∪
        {tape | IvcSecurity.ExtractionFailure setup.ajtai setup.circuitKeys exports control
          ((finalStages admitted depth).1.input tape)
          ((finalStages admitted depth).1.results tape)} := by
    rintro tape ⟨accepts, unreached⟩
    by_cases verifies : Verifies setup.ajtai setup.circuitKeys step complete.packageWidth
      control
      (head (played prover ((start admitted depth tape : (ivc).Tape).1)).first).statement
      (finalProof setup complete.packageWidth step control read head kind family prover budget
        (start admitted depth tape : (ivc).Tape).1 (start admitted depth tape : (ivc).Tape).2)
    · have input : (finalStages admitted depth).1.input tape = IvcWalk.initial
          ((head (played prover ((start admitted depth tape : (ivc).Tape).1)).first).statement,
            finalProof setup complete.packageWidth step control read head kind family prover
              budget (start admitted depth tape : (ivc).Tape).1
              (start admitted depth tape : (ivc).Tape).2) := by
        rw [HyperNova.ReverseStages.input_origin]
        show IvcWalk.initial ((ivc).output (start admitted depth tape)) = _
        rw [show (ivc).output (start admitted depth tape) = _ from ite_eq_left accepts]
      refine Or.inr ⟨?_, ?_⟩
      · rw [input]
        exact holds_of_verifies verifies
      · rw [show ((finalStages admitted depth).1.input tape).statement = _ from
          congrArg IvcWalk.Visit.statement input]
        exact unreached
    · exact Or.inl ⟨accepts, verifies⟩
  have pulled : (finalStages admitted depth).1.tape.toOuterMeasure
      (start admitted depth ⁻¹' {sample : (ivc).Tape | rejects sample}) =
      (ivc).tape.toOuterMeasure {sample | rejects sample} := by
    rw [← PMF.toOuterMeasure_map_apply, HyperNova.ReverseStages.tape_map_origin]
    rfl
  calc ((finalStages admitted depth).1.tape.toOuterMeasure _).toReal
      ≤ ((finalStages admitted depth).1.tape.toOuterMeasure _ +
          (finalStages admitted depth).1.tape.toOuterMeasure _).toReal :=
        ENNReal.toReal_mono
          (ENNReal.add_ne_top.mpr ⟨HyperNova.ReverseStages.event_ne_top _ _,
            HyperNova.ReverseStages.event_ne_top _ _⟩)
          ((MeasureTheory.measure_mono cover).trans (MeasureTheory.measure_union_le _ _))
    _ = ((finalStages admitted depth).1.tape.toOuterMeasure _).toReal +
          ((finalStages admitted depth).1.tape.toOuterMeasure _).toReal :=
        ENNReal.toReal_add (HyperNova.ReverseStages.event_ne_top _ _)
          (HyperNova.ReverseStages.event_ne_top _ _)
    _ ≤ _ := add_le_add (by rw [pulled]; exact rejected) failed

end History

end NightstreamFPrime.Export.BoundedDepth.FinalKey20
