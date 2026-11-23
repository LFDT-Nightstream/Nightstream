import NightstreamFPrime.Export.BoundedDepth.FinalShapes
import NightstreamFPrime.Export.BoundedDepth.NebulaSecurity

/-! The memory verifier of a final proof over a family of verifiers of the proofs with steps, in
the game of `FinalShapes.lean` (SPEC.md, Algorithms 8 and 10; `SECURITY_MODEL.md`, adversaries 5
and 6). It accepts when the final verifier `FinalShapes.finalAccepts` accepts `(i, z0, zi)` and
the checks of Algorithm 8 hold for the memory statement and `(i, z0, zi)`.

The prover of `FinalShapes.adversary` names a memory statement with each table of the oracle
(`claim`, a law for each table). With that statement on its tape it is an IVC adversary
(`adversary`), whose output is the output of `FinalShapes.adversary`. A false memory statement is
a tape of the walk's last stage on which the final verifier accepts `(i, z0, zi)`, the checks of
Algorithm 8 hold, and the history that the walk decodes does not attest the memory statement
(`FalseMemory`). On such a tape `Verifies` rejects the proof of the extractor
(`FinalShapes.Rejected`), or the sample is bad for `NebulaSecurity.checked_le`. So its
probability is at most the sum of the bounds of the shapes plus the bound of
`NebulaSecurity.checked_le` for this adversary (`memory_sound`). The walk has `S_max · N` stages:
the checks of Algorithm 8 accept at most `S_max · N` steps
(`EndToEnd.VerifierChecks.iteration_le`), so no premise bounds the iteration of the accepted
statements. -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.BoundedDepth.FinalMemory

open scoped BigOperators
open NightstreamFPrime.Spec NightstreamFPrime.Spec.FieldExtensionRing
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Nebula
open NightstreamFPrime.Spec.HyperNova.Construction2.Paper
open NightstreamFPrime.Spec.HyperNova.ReverseStages
open NightstreamFPrime.Spec.Compression.Production
open NightstreamFPrime.Spec.Compression.Akita NightstreamFPrime.Spec.Compression.Akita.Extract
open NightstreamFPrime.Spec.Compression.Game (map_fst_bind)
open NightstreamFPrime.Spec.Checkpoint
open NightstreamFPrime.Spec.RandomOracle (OracleComp)
open NightstreamFPrime.Lifecycle NightstreamFPrime.Lifecycle.BoundedDepth
open NightstreamFPrime.Lifecycle.Nebula
open NightstreamFPrime.Lifecycle.Nebula.EndToEnd (MemoryStatement VerifierChecks History)
open NightstreamFPrime.Export.BoundedDepth.RunCompleteness
open NightstreamFPrime.Export.BoundedDepth.IvcWalk
open NightstreamFPrime.Export.BoundedDepth.IvcSecurity (Output walkAdversary walkBound)
open NightstreamFPrime.Export.BoundedDepth.FinalVerifier (Mode)
open NightstreamFPrime.Export.BoundedDepth.FinalAkita
open NightstreamFPrime.Export.BoundedDepth.NebulaSecurity
open FinalKey20 (played)

attribute [local instance low] Classical.propDecidable

attribute [local instance] NightstreamFPrime.Export.BoundedDepth.FinalKey20.instAlgebraZqK
  NightstreamFPrime.Export.BoundedDepth.FinalKey20.instDegree2K

variable {logicalWidth : Nat}
  {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
    RingCarrierLayout.carrierWidth logicalWidth}
  {mode : Mode} (setup : FinalVerifier.Setup mode logicalWidth publicFits)
  (step : AppState → AppWitness → AppState) (cap : ℕ)
  (control : AppState → AppWitness → Fin 2) {Wire Coord : Type} {total : ℕ}
  (head : Wire → First logicalWidth publicFits)
  [Fintype Coord] [DecidableEq Coord] {ι : Type} [Fintype ι]
  (kind : Wire → Option ι) (accepts : ι → Prefix Wire (Option Coord → K) total → Prop)
  (proofs : ι → (budget : ℕ) → (Address Wire total → Option Coord → K) →
    Tape (Address Wire total) (Option Coord) K budget total →
      Terminal.ProofEnvelope (logicalWidth := logicalWidth) (publicFits := publicFits))
  (prover : OracleComp (Address Wire total) (Option Coord → K) (ℕ → Wire))
  [Finite Wire] (budget : ℕ) {p : Plan} (app : PortApp p.toGeometry)

local notation "shapes" => FinalShapes.adversary head kind accepts proofs prover budget

/-- The prover of `FinalShapes.adversary` with the memory statement of law `claim table` for the
table of the oracle, as an IVC adversary. Its tape is the tape of `FinalShapes.adversary` and the
memory statement. Its output is the output of `FinalShapes.adversary` for the first part. -/
noncomputable def adversary
    (claim : (Address Wire total → Option Coord → K) →
      PMF (MemoryStatement app.width)) :
    IvcAdversary (Output logicalWidth publicFits) where
  Tape := (shapes).Tape × MemoryStatement app.width
  tape := (shapes).tape.bind fun sample => (claim sample.1).map (sample, ·)
  output tape := (shapes).output tape.1

variable {relation : ProductionKey.LogicalRelation logicalWidth publicFits}
  {cubeFits : RingCarrierLayout.carrierWidth logicalWidth ≤ 2 ^ cubeVariables}
  (exports : TerminalBridge.Exports relation setup.ajtai setup.circuitKeys step cubeFits)

/-- The final verifier accepts `(i, z0, zi)` for the table of the oracle, the checks of SPEC.md
Algorithm 8 hold for it and the sample's memory statement, and the history that the walk decodes
from the sample is not a history of `(i, z0, zi)` that attests the memory statement. On the law
of `memory_sound` the sample's statement is `(i, z0, zi)`. -/
def FalseMemory (table : Address Wire total → Option Coord → K)
    (sample : Sample app.width logicalWidth publicFits) : Prop :=
  FinalShapes.finalAccepts head kind accepts (played prover table) ∧
    VerifierChecks p (head (played prover table).first).statement sample.1 ∧
    ¬ (History (MemoryApp.program p.toGeometry app) (head (played prover table).first).statement
        ((experiment app setup.ajtai setup.circuitKeys exports).states sample)
        ((experiment app setup.ajtai setup.circuitKeys exports).witness sample) ∧
      (experiment app setup.ajtai setup.circuitKeys exports).Attested app sample)

variable {Admitted : NifsAdversary (WalkAttempt logicalWidth publicFits) → Prop}
  {StageAdmitted : Stage (Visit logicalWidth publicFits) (Answer logicalWidth publicFits) → Prop}
  {Efficient : (adversary : NifsAdversary (WalkAttempt logicalWidth publicFits)) →
    NifsExtractor (Answer logicalWidth publicFits) adversary → Prop}
  {error : NifsAdversary (WalkAttempt logicalWidth publicFits) → ℝ}
  (assumption : Assumption1 (WalkSucceeds setup.ajtai) (WalkExtracted setup.ajtai) Admitted
    Efficient error)
  (closed : Closed (attemptOf setup.ajtai setup.circuitKeys exports)
    (back setup.ajtai setup.circuitKeys exports) Admitted StageAdmitted Efficient)
  (claim : (Address Wire total → Option Coord → K) →
    PMF (MemoryStatement app.width))
  (admitted : StageAdmitted
    (Stage.start (walkAdversary (adversary head kind accepts proofs prover budget app claim))))

local notation "memory" => adversary head kind accepts proofs prover budget app claim

local notation "stages" => reverseStages assumption closed (walkAdversary memory) admitted

/-- The prover's tape inside a tape of the walk's stage `depth`. -/
noncomputable def tapeOf (depth : ℕ) (tape : (stages depth).1.Tape) :
    (shapes).Tape × MemoryStatement app.width :=
  origin assumption closed (walkAdversary memory) admitted depth tape

/-- The walk's stage `depth` with the memory statement of the prover's tape. -/
noncomputable def claimed (depth : ℕ) :
    ClaimedStage app (logicalWidth := logicalWidth) (publicFits := publicFits) :=
  ⟨(stages depth).1, fun tape => (tapeOf setup step head kind accepts proofs prover budget app
    exports assumption closed claim admitted depth tape).2⟩

/-- The sample of a tape of the walk's stage `depth`: the memory statement, the walk's start and
the extractor's answers. `(claimed …).law` is its law. -/
noncomputable def sampleOf (depth : ℕ) (tape : (stages depth).1.Tape) :
    Sample app.width logicalWidth publicFits :=
  ((tapeOf setup step head kind accepts proofs prover budget app exports assumption closed claim
      admitted depth tape).2,
    (stages depth).1.input tape, (stages depth).1.results tape)

local notation "tapeAt" =>
  tapeOf setup step head kind accepts proofs prover budget app exports assumption closed claim
    admitted (p.sMax * p.n)

local notation "sampleAt" =>
  sampleOf setup step head kind accepts proofs prover budget app exports assumption closed claim
    admitted (p.sMax * p.n)

local notation "last" =>
  claimed setup step head kind accepts proofs prover budget app exports assumption closed claim
    admitted (p.sMax * p.n)

/-- The memory verifier against a false memory statement, in the form of
`Claims.Final.memory_sound`. Over the table of the oracle, the tape of the extractor and the
memory statement, and the walk's `S_max · N` stages (`p.sMax * p.n`), the probability that the
final verifier accepts, the checks of SPEC.md Algorithm 8 hold, and the decoded history does not
attest the memory statement is at most the sum of the bounds of the shapes plus the terms of
`NebulaSecurity.checked_le` for the IVC adversary `adversary`: `IvcSecurity.walkBound`, the
state-link and chain collision masses, and the memory-round error. `each`: the bound of each shape
for its verifier and its proof, at the widths of the packages. The checks of Algorithm 8 bound
the iteration by `S_max · N` (`EndToEnd.VerifierChecks.iteration_le`), so the answers of the last
stage cover the iteration of each statement in the event. -/
theorem memory_sound (complete : StepExtend.Complete exports) {bounds : ι → ℝ}
    (each : ∀ index,
      𝔼 table, 𝔼 tape : Tape (Address Wire total) (Option Coord) K budget total,
        (if accepts index (played prover table) ∧
            ¬ Verifies setup.ajtai setup.circuitKeys step complete.packageWidth cap control
              (head (played prover table).first).statement (proofs index budget table tape)
          then (1 : ℝ) else 0) ≤ bounds index)
    (valid : p.Valid) (keysShaped : ∀ key, (setup.circuitKeys key).digest.length = 4)
    (programEq : exports.program = MemoryApp.program p.toGeometry app)
    {memoryError : ClaimedStage app (logicalWidth := logicalWidth) (publicFits := publicFits) → ℝ}
    (round : MemoryRound app setup.ajtai setup.circuitKeys exports StageAdmitted memoryError) :
    ((stages (p.sMax * p.n)).1.tape.toOuterMeasure
        {tape | FalseMemory setup step head kind accepts prover app exports (tapeAt tape).1.1
          (sampleAt tape)}).toReal ≤
      ∑ index, bounds index +
        (walkBound setup.ajtai setup.circuitKeys exports assumption closed memory admitted
            (p.sMax * p.n) +
          ((last).law.toOuterMeasure
            {sample | (experiment app setup.ajtai setup.circuitKeys exports).StateLinkCollision app
              sample}).toReal +
          ((last).law.toOuterMeasure
            {sample | (experiment app setup.ajtai setup.circuitKeys exports).ChainCollision app
              sample}).toReal +
          memoryError last) := by
  let stage := (stages (p.sMax * p.n)).1
  let rejects :=
    FinalShapes.Rejected setup complete.packageWidth step cap control head kind accepts proofs
      prover budget
  let Bad : Sample app.width logicalWidth publicFits → Prop :=
    (experiment app setup.ajtai setup.circuitKeys exports).Bad app
      (Checked app setup.ajtai setup.circuitKeys (step := step) control cap)
  have origins : stage.tape.map tapeAt = (memory).tape :=
    tape_map_origin assumption closed (walkAdversary memory) admitted (p.sMax * p.n)
  have firsts : (memory).tape.map Prod.fst = (shapes).tape := map_fst_bind _ _
  have cover : {tape | FalseMemory setup step head kind accepts prover app exports
        (tapeAt tape).1.1 (sampleAt tape)} ⊆
      {tape | rejects (tapeAt tape).1} ∪ {tape | Bad (sampleAt tape)} := by
    rintro tape ⟨accepted, checks, notGood⟩
    by_cases verifies : Verifies setup.ajtai setup.circuitKeys step complete.packageWidth cap
      control (head (played prover (tapeAt tape).1.1).first).statement
      (FinalShapes.finalProof head kind proofs prover budget (tapeAt tape).1.1 (tapeAt tape).1.2)
    · have input : stage.input tape = initial
          ((head (played prover (tapeAt tape).1.1).first).statement,
            FinalShapes.finalProof head kind proofs prover budget (tapeAt tape).1.1
              (tapeAt tape).1.2) := by
        rw [input_origin assumption closed (walkAdversary memory) admitted (p.sMax * p.n) tape]
        show initial ((shapes).output (tapeAt tape).1) = _
        rw [show (shapes).output (tapeAt tape).1 = _ from ite_eq_left accepted]
      have statement : (sampleAt tape).2.1.statement =
          (head (played prover (tapeAt tape).1.1).first).statement :=
        congrArg Visit.statement input
      refine Or.inr ⟨⟨?_, congrArg Visit.bounds input⟩, ?_, ?_⟩
      · change Terminal.HoldsFor cap setup.ajtai setup.circuitKeys step control
          (stage.input tape).statement (stage.input tape).proof
        rw [input]
        exact holds_of_verifies verifies
      · change VerifierChecks p (sampleAt tape).2.1.statement (sampleAt tape).1
        rw [statement]
        exact checks
      · change ¬ (History (MemoryApp.program p.toGeometry app) (sampleAt tape).2.1.statement _ _ ∧
          _)
        rw [statement]
        exact notGood
    · exact Or.inl ⟨accepted, verifies⟩
  have rejected : (stage.tape.toOuterMeasure {tape | rejects (tapeAt tape).1}).toReal ≤
      ∑ index, bounds index := by
    have moved : stage.tape.toOuterMeasure {tape | rejects (tapeAt tape).1} =
        (shapes).tape.toOuterMeasure {sample | rejects sample} := by
      rw [← firsts, ← origins, PMF.toOuterMeasure_map_apply, PMF.toOuterMeasure_map_apply]
      rfl
    rw [moved]
    exact FinalShapes.rejected_le setup complete.packageWidth step cap control head kind accepts
      proofs prover budget each
  have bad := checked_le app setup.ajtai setup.circuitKeys exports assumption closed memory
    admitted valid keysShaped programEq control cap Prod.snd round
  have sampled : stage.tape.toOuterMeasure {tape | Bad (sampleAt tape)} =
      (last).law.toOuterMeasure {sample | Bad sample} := by
    rw [ClaimedStage.law, PMF.toOuterMeasure_map_apply]
    rfl
  calc (stage.tape.toOuterMeasure {tape | FalseMemory setup step head kind accepts prover app
          exports (tapeAt tape).1.1 (sampleAt tape)}).toReal
      ≤ (stage.tape.toOuterMeasure {tape | rejects (tapeAt tape).1} +
          stage.tape.toOuterMeasure {tape | Bad (sampleAt tape)}).toReal :=
        ENNReal.toReal_mono (ENNReal.add_ne_top.mpr ⟨event_ne_top _ _, event_ne_top _ _⟩)
          ((MeasureTheory.measure_mono cover).trans (MeasureTheory.measure_union_le _ _))
    _ = (stage.tape.toOuterMeasure {tape | rejects (tapeAt tape).1}).toReal +
          (stage.tape.toOuterMeasure {tape | Bad (sampleAt tape)}).toReal :=
        ENNReal.toReal_add (event_ne_top _ _) (event_ne_top _ _)
    _ ≤ _ := add_le_add rejected (by rw [sampled]; exact bad)

end NightstreamFPrime.Export.BoundedDepth.FinalMemory
