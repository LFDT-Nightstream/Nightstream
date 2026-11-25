import NightstreamFPrime.Export.BoundedDepth.FinalKey20History

/-! The final verifier for each claim shape and for the proof of zero steps, over a family of
verifiers of the proofs with steps (SPEC.md, Algorithm 10; `SECURITY_MODEL.md`, adversaries 4
and 6; Akita, ePrint 2026/1983, revision of 2026-11-23, Theorem 10.17).

This file has the part of `FinalKey20All.lean` and of `FinalKey20History.lean` that does not
read the schedule. A family has, for each claim shape `index`, a verifier `accepts index` of the
proofs with steps, a proof `proofs index` of an extractor, and a bound `bounds index` of the
event "the verifier accepts and `Verifies` rejects the proof of the extractor" (`each`).
`FinalKey20All.lean` and `FinalKey20History.lean` keep their statements for `key20.aks`.

- `finalAccepts` is the final verifier: the check of the proof of zero steps, or `accepts` for
  the shape that the first message names. `finalProof` is the proof of the extractor for it.
- `knowledge_all` and `sound_all` have the sum of the bounds of the shapes. The proof of zero
  steps has no draw, and `Verifies` accepts its proof (`FinalVerifier.verifies_of_bottom`).
- `adversary` is a prover of the final verifier with the extractor as an IVC adversary.
  `history_sound` and `history_knowledge` are `FinalHistory.soundness` and
  `FinalHistory.knowledge` for it: the sum of the bounds plus `IvcSecurity.walkBound`. -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.BoundedDepth.FinalShapes

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
open FinalKey20 (played played_first uniform_event)

attribute [local instance low] Classical.propDecidable

attribute [local instance] NightstreamFPrime.Export.BoundedDepth.FinalKey20.instAlgebraZqK
  NightstreamFPrime.Export.BoundedDepth.FinalKey20.instDegree2K

variable {logicalWidth : Nat}
  {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
    RingCarrierLayout.carrierWidth logicalWidth}
  {mode : Mode} (setup : FinalVerifier.Setup mode logicalWidth publicFits)
  (packageWidths : Fin 2 → ℕ) (step : AppState → AppWitness → AppState) (control : AppState → AppWitness → Fin 2) {Wire Coord : Type} {total : ℕ}
  (head : Wire → First logicalWidth publicFits)
  [Fintype Coord] [DecidableEq Coord] {ι : Type} [Fintype ι]
  (kind : Wire → Option ι) (accepts : ι → Prefix Wire (Option Coord → CodeSlot K) total → Prop)
  (proofs : ι → (budget : ℕ) → (Address Wire total → Option Coord → CodeSlot K) →
    Tape (Address Wire total) (Option Coord) (CodeSlot K) budget total →
      Terminal.ProofEnvelope (logicalWidth := logicalWidth) (publicFits := publicFits))
  (prover : OracleComp (Address Wire total) (Option Coord → CodeSlot K) (ℕ → Wire))

/-- The final verifier over a family of verifiers of the proofs with steps (SPEC.md
Algorithm 10). It accepts a transcript when the statement of the first message is a statement of
zero steps (step 1), or when the first message names the shape `kind` of the family and the
verifier `accepts` of this shape accepts. A first message with a shape outside the family has
`kind` equal to `none`. -/
def finalAccepts (point : Prefix Wire (Option Coord → CodeSlot K) total) : Prop :=
  (Terminal.StatementValid (head point.first).statement ∧
      (head point.first).statement.iteration = 0 ∧
      (head point.first).statement.zi = (head point.first).statement.z0) ∨
    ∃ index, kind point.first = some index ∧ accepts index point

/-- The proof with witnesses of the extractor for the final verifier: the bottom proof for a
statement of zero steps, and the proof `proofs` of the shape that the first message names. -/
noncomputable def finalProof (budget : ℕ) (table : Address Wire total → Option
  Coord → CodeSlot K)
    (tape : Tape (Address Wire total) (Option Coord) (CodeSlot K) budget total) :
    Terminal.ProofEnvelope (logicalWidth := logicalWidth) (publicFits := publicFits) :=
  if Terminal.StatementValid (head (prover.run table 0)).statement ∧
      (head (prover.run table 0)).statement.iteration = 0 ∧
      (head (prover.run table 0)).statement.zi = (head (prover.run table 0)).statement.z0 then
    .bottom
  else
    match kind (prover.run table 0) with
    | none => .bottom
    | some index => proofs index budget table tape

variable [Finite Wire] (budget : ℕ)

/-- Knowledge soundness of the final verifier for all shapes of the family and for the proof of
zero steps. The probability that `finalAccepts` accepts and `Verifies` rejects the proof
`finalProof` of the extractor is at most the sum of the bounds of the shapes. `each`: the bound
of each shape for its verifier and its proof. A statement of zero steps adds no term: it has no
draw, and `Verifies` accepts its proof (`FinalVerifier.verifies_of_bottom`). -/
theorem knowledge_all {bounds : ι → ℝ}
    (each : ∀ index,
      𝔼 table, 𝔼 tape : Tape (Address Wire total) (Option Coord) (CodeSlot K) budget total,
        (if accepts index (played prover table) ∧
            ¬ Verifies setup.ajtai setup.circuitKeys step packageWidths control
              (head (played prover table).first).statement (proofs index budget table tape)
          then (1 : ℝ) else 0) ≤ bounds index) :
    𝔼 table, 𝔼 tape : Tape (Address Wire total) (Option Coord) (CodeSlot K) budget total,
        (if finalAccepts head kind accepts (played prover table) ∧
            ¬ Verifies setup.ajtai setup.circuitKeys step packageWidths control
              (head (played prover table).first).statement
              (finalProof head kind proofs prover budget table tape)
          then (1 : ℝ) else 0) ≤
      ∑ index, bounds index := by
  refine le_trans ?_ (Finset.sum_le_sum fun index _ => each index)
  simp only [← Finset.expect_sum_comm]
  refine Finset.expect_le_expect fun table _ => Finset.expect_le_expect fun tape _ => ?_
  split_ifs with event
  · obtain ⟨accepted, refuted⟩ := event
    unfold finalAccepts at accepted
    rw [played_first] at accepted refuted
    by_cases bottom : Terminal.StatementValid (head (prover.run table 0)).statement ∧
        (head (prover.run table 0)).statement.iteration = 0 ∧
        (head (prover.run table 0)).statement.zi = (head (prover.run table 0)).statement.z0
    · refine absurd ?_ refuted
      unfold finalProof
      rw [ite_eq_left bottom]
      exact FinalVerifier.verifies_of_bottom setup step control bottom
    · rcases accepted with zero | ⟨index, named, accepted⟩
      · exact absurd zero bottom
      · have same : finalProof head kind proofs prover budget table tape =
            proofs index budget table tape := by
          unfold finalProof
          rw [ite_eq_right bottom, named]
        rw [same, ← played_first] at refuted
        refine le_trans (le_of_eq (ite_eq_left (show _ ∧ _ from ⟨accepted, refuted⟩)).symm)
          (Finset.single_le_sum (f := fun index =>
            if accepts index (played prover table) ∧
                ¬ Verifies setup.ajtai setup.circuitKeys step packageWidths control
                  (head (played prover table).first).statement (proofs index budget table tape)
              then (1 : ℝ) else 0)
            (fun _ _ => by split_ifs <;> norm_num) (Finset.mem_univ index))
  · exact Finset.sum_nonneg fun _ _ => by split_ifs <;> norm_num

/-- Soundness of the final verifier for all shapes of the family and for the proof of zero
steps. The probability over the table of the oracle that `finalAccepts` accepts and no proof
with witnesses satisfies `Verifies` for the statement of the first message is at most the sum of
the bounds of the shapes. `each`: the bound of each shape for its verifier and its proof. -/
theorem sound_all {bounds : ι → ℝ}
    (each : ∀ index,
      𝔼 table, 𝔼 tape : Tape (Address Wire total) (Option Coord) (CodeSlot K) budget total,
        (if accepts index (played prover table) ∧
            ¬ Verifies setup.ajtai setup.circuitKeys step packageWidths control
              (head (played prover table).first).statement (proofs index budget table tape)
          then (1 : ℝ) else 0) ≤ bounds index) :
    𝔼 table,
        (if finalAccepts head kind accepts (played prover table) ∧
            ¬ ∃ proof, Verifies setup.ajtai setup.circuitKeys step packageWidths control
              (head (played prover table).first).statement proof
          then (1 : ℝ) else 0) ≤
      ∑ index, bounds index := by
  have flat (value : ℝ) : value =
      𝔼 _tape : Tape (Address Wire total) (Option Coord) (CodeSlot K) budget total, value :=
    (Finset.expect_const Finset.univ_nonempty value).symm
  refine le_trans (Finset.expect_le_expect fun table _ => ?_)
    (knowledge_all setup packageWidths step control head kind accepts proofs prover budget
      each)
  refine (flat _).le.trans (Finset.expect_le_expect fun tape _ => ?_)
  split_ifs with lost found
  · exact le_rfl
  · exact absurd ⟨lost.1, fun verifies => lost.2 ⟨_, verifies⟩⟩ found
  · exact zero_le_one
  · exact le_rfl

/-- A prover of the final verifier, with the extractor of at most `budget` attempts in each
retry loop, as an IVC adversary. Its tape is the table of the oracle and the tape of the
extractor, with the uniform law. Its output is the accepted statement with the proof `finalProof`
of the extractor, and otherwise the bottom proof of `(0, [], [])`. -/
noncomputable def adversary :
    HyperNova.ReverseStages.IvcAdversary (IvcSecurity.Output logicalWidth publicFits) where
  Tape := (Address Wire total → Option Coord → CodeSlot K) ×
    Tape (Address Wire total) (Option Coord) (CodeSlot K) budget total
  tape := Compression.Game.uniform _
  output sample :=
    if finalAccepts head kind accepts (played prover sample.1) then
      ((head (played prover sample.1).first).statement,
        finalProof head kind proofs prover budget sample.1 sample.2)
    else (⟨0, [], []⟩, .bottom)

/-- The final verifier accepts `(i, z0, zi)` for the table of the oracle, and no `i` valid steps
of `program`, each with a witness of the word count of the program, lead from `z0` to `zi`. -/
def FalseHistory (program : Stage1.Application.Program)
    (table : Address Wire total → Option Coord → CodeSlot K) : Prop :=
  finalAccepts head kind accepts (played prover table) ∧
    ¬ IvcWalk.Reaches program (head (played prover table).first).statement

/-- The final verifier accepts for the table of the oracle, and `Verifies` rejects the proof
`finalProof` of the extractor for the tape: the event of `knowledge_all`. -/
def Rejected (sample : (Address Wire total → Option Coord → CodeSlot K) ×
    Tape (Address Wire total) (Option Coord) (CodeSlot K) budget total) : Prop :=
  finalAccepts head kind accepts (played prover sample.1) ∧
    ¬ Verifies setup.ajtai setup.circuitKeys step packageWidths control
      (head (played prover sample.1).first).statement
      (finalProof head kind proofs prover budget sample.1 sample.2)

local notation "ivc" => adversary head kind accepts proofs prover budget

local notation "rejects" =>
  Rejected setup packageWidths step control head kind accepts proofs prover budget

/-- The probability of `Rejected` under the law of the tape of `adversary` is at most the sum of
the bounds of the shapes (`knowledge_all` under the uniform law). -/
theorem rejected_le {bounds : ι → ℝ}
    (each : ∀ index,
      𝔼 table, 𝔼 tape : Tape (Address Wire total) (Option Coord) (CodeSlot K) budget total,
        (if accepts index (played prover table) ∧
            ¬ Verifies setup.ajtai setup.circuitKeys step packageWidths control
              (head (played prover table).first).statement (proofs index budget table tape)
          then (1 : ℝ) else 0) ≤ bounds index) :
    ((ivc).tape.toOuterMeasure {sample | rejects sample}).toReal ≤ ∑ index, bounds index := by
  refine (uniform_event (Sample := (Address Wire total → Option Coord → CodeSlot K) ×
    Tape (Address Wire total) (Option Coord) (CodeSlot K) budget total) _).trans_le
    (le_trans (le_of_eq ?_) (knowledge_all setup packageWidths step control head kind accepts
      proofs prover budget each))
  rw [← Finset.univ_product_univ, Finset.expect_product]
  exact Finset.expect_congr rfl fun table _ => Finset.expect_congr rfl fun tape _ =>
    if_congr Iff.rfl rfl rfl

omit [Fintype ι] in
/-- The statement of each output of `adversary` has at most `depth` steps, when each accepted
statement has at most `depth` steps. -/
theorem adversary_depth (depth : ℕ)
    (depthBound : ∀ table, finalAccepts head kind accepts (played prover table) →
      (head (played prover table).first).statement.iteration ≤ depth)
    (sample : (ivc).Tape) : ((ivc).output sample).1.iteration ≤ depth := by
  by_cases accepted : finalAccepts head kind accepts (played prover sample.1)
  · rw [show (ivc).output sample = _ from ite_eq_left accepted]
    exact depthBound _ accepted
  · rw [show (ivc).output sample = _ from ite_eq_right accepted]
    exact Nat.zero_le _

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

local notation "rejects" =>
  Rejected setup complete.packageWidth step control head kind accepts proofs prover budget

/-- The final verifier against a false history, in the form of `Claims.Final.history_sound`. The
probability of a false history is at most the sum of the bounds of the shapes plus
`IvcSecurity.walkBound` of the IVC adversary `adversary`: for each of the first `depth` stages of
the reverse extractor, the success chance of the Poseidon2 collision finder of the stage
(`IvcSecurity.finderChance`) and the error of the stage. The probability is over the table of the
oracle and the tape of the extractor. `each`: the bound of each shape for its verifier and its
proof, at the widths of the packages. `depthBound`: each accepted statement has at most `depth`
steps. -/
theorem history_sound {bounds : ι → ℝ}
    (each : ∀ index,
      𝔼 table, 𝔼 tape : Tape (Address Wire total) (Option Coord) (CodeSlot K) budget total,
        (if accepts index (played prover table) ∧
            ¬ Verifies setup.ajtai setup.circuitKeys step complete.packageWidth control
              (head (played prover table).first).statement (proofs index budget table tape)
          then (1 : ℝ) else 0) ≤ bounds index)
    (keysShaped : ∀ key, (setup.circuitKeys key).digest.length = 4)
    (admitted : StageAdmitted
      (HyperNova.ReverseStages.Stage.start (IvcSecurity.walkAdversary ivc))) (depth : ℕ)
    (depthBound : ∀ table, finalAccepts head kind accepts (played prover table) →
      (head (played prover table).first).statement.iteration ≤ depth) :
    ((ivc).tape.toOuterMeasure {sample |
        FalseHistory head kind accepts prover exports.program sample.1}).toReal ≤
      ∑ index, bounds index +
        IvcSecurity.walkBound setup.ajtai setup.circuitKeys exports assumption closed ivc admitted
          depth := by
  have rejected := rejected_le setup complete.packageWidth step control head kind accepts
    proofs prover budget each
  have accepted := IvcSecurity.probability_bound setup.ajtai setup.circuitKeys exports assumption
    closed ivc admitted keysShaped control depth fun sample _ =>
      adversary_depth head kind accepts proofs prover budget depth depthBound sample
  have cover : {sample : (ivc).Tape |
      FalseHistory head kind accepts prover exports.program sample.1} ⊆
      {sample | rejects sample} ∪
        {sample | IvcSecurity.FalseAcceptance setup.ajtai setup.circuitKeys exports control
          ((ivc).output sample)} := by
    rintro sample ⟨accepted, unreached⟩
    by_cases verifies : Verifies setup.ajtai setup.circuitKeys step complete.packageWidth
      control (head (played prover sample.1).first).statement
      (finalProof head kind proofs prover budget sample.1 sample.2)
    · refine Or.inr ?_
      show IvcSecurity.FalseAcceptance _ _ _ _ ((ivc).output sample)
      rw [show (ivc).output sample = _ from ite_eq_left accepted]
      exact ⟨holds_of_verifies verifies, unreached⟩
    · exact Or.inl ⟨accepted, verifies⟩
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
  finalAccepts head kind accepts (played prover ((start admitted depth tape : (ivc).Tape).1)) ∧
    ¬ IvcWalk.ReachedBy exports.program
      (head (played prover ((start admitted depth tape : (ivc).Tape).1)).first).statement
      (IvcSecurity.extracted setup.ajtai setup.circuitKeys exports
        ((finalStages admitted depth).1.input tape) ((finalStages admitted depth).1.results tape))

/-- The knowledge form of `history_sound`, in the form of `Claims.Final.history_knowledge`: the
probability that the final verifier accepts and the witnesses that the reverse extractor returns
are not a history of the accepted statement is at most the sum of the bounds of the shapes plus
`IvcSecurity.walkBound`. A tape with `Rejected` at the start of the walk has the bound of
`rejected_le`. For each other tape the extraction fails at the start of the walk
(`IvcSecurity.extraction_bound`). -/
theorem history_knowledge {bounds : ι → ℝ}
    (each : ∀ index,
      𝔼 table, 𝔼 tape : Tape (Address Wire total) (Option Coord) (CodeSlot K) budget total,
        (if accepts index (played prover table) ∧
            ¬ Verifies setup.ajtai setup.circuitKeys step complete.packageWidth control
              (head (played prover table).first).statement (proofs index budget table tape)
          then (1 : ℝ) else 0) ≤ bounds index)
    (keysShaped : ∀ key, (setup.circuitKeys key).digest.length = 4)
    (admitted : StageAdmitted
      (HyperNova.ReverseStages.Stage.start (IvcSecurity.walkAdversary ivc))) (depth : ℕ)
    (depthBound : ∀ table, finalAccepts head kind accepts (played prover table) →
      (head (played prover table).first).statement.iteration ≤ depth) :
    ((finalStages admitted depth).1.tape.toOuterMeasure {tape |
        AdviceFailure setup step head kind accepts proofs prover budget exports assumption
          closed admitted depth tape}).toReal ≤
      ∑ index, bounds index +
        IvcSecurity.walkBound setup.ajtai setup.circuitKeys exports assumption closed ivc admitted
          depth := by
  have rejected := rejected_le setup complete.packageWidth step control head kind accepts
    proofs prover budget each
  have failed := IvcSecurity.extraction_bound setup.ajtai setup.circuitKeys exports assumption
    closed ivc admitted keysShaped control depth fun sample _ =>
      adversary_depth head kind accepts proofs prover budget depth depthBound sample
  have cover : {tape | AdviceFailure setup step head kind accepts proofs prover budget
        exports assumption closed admitted depth tape} ⊆
      start admitted depth ⁻¹' {sample : (ivc).Tape | rejects sample} ∪
        {tape | IvcSecurity.ExtractionFailure setup.ajtai setup.circuitKeys exports control
          ((finalStages admitted depth).1.input tape)
          ((finalStages admitted depth).1.results tape)} := by
    rintro tape ⟨accepted, unreached⟩
    by_cases verifies : Verifies setup.ajtai setup.circuitKeys step complete.packageWidth
      control
      (head (played prover ((start admitted depth tape : (ivc).Tape).1)).first).statement
      (finalProof head kind proofs prover budget (start admitted depth tape : (ivc).Tape).1
        (start admitted depth tape : (ivc).Tape).2)
    · have input : (finalStages admitted depth).1.input tape = IvcWalk.initial
          ((head (played prover ((start admitted depth tape : (ivc).Tape).1)).first).statement,
            finalProof head kind proofs prover budget (start admitted depth tape : (ivc).Tape).1
              (start admitted depth tape : (ivc).Tape).2) := by
        rw [HyperNova.ReverseStages.input_origin]
        show IvcWalk.initial ((ivc).output (start admitted depth tape)) = _
        rw [show (ivc).output (start admitted depth tape) = _ from ite_eq_left accepted]
      refine Or.inr ⟨?_, ?_⟩
      · rw [input]
        exact holds_of_verifies verifies
      · rw [show ((finalStages admitted depth).1.input tape).statement = _ from
          congrArg IvcWalk.Visit.statement input]
        exact unreached
    · exact Or.inl ⟨accepted, verifies⟩
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

end NightstreamFPrime.Export.BoundedDepth.FinalShapes
