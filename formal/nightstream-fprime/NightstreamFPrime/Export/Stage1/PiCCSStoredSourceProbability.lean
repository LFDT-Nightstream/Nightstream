import NightstreamFPrime.Export.Stage1.SecurityInstance
import NightstreamFPrime.Export.Stage1.PiCCSStoredWitnessCheck
import NightstreamFPrime.Lifecycle.Nifs.InteractiveOutput

/-!
Connect the existing functional B.1 source-return probability to the selected
stored checked return. Storage preserves all values, typed coins and raw
certificate lists. The caller supplies declared clocks; no bound or runtime
refinement is asserted. Matrix values remain the existing selected functions.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiCCSStoredSourceProbability

open _root_.NightstreamFPrime.Spec
open _root_.NightstreamFPrime.Spec.Folding
open _root_.NightstreamFPrime.Spec.Folding.Nifs
open _root_.NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open StrongReduction ConcreteCarrier CheckedWitnessExtraction
open _root_.NightstreamFPrime.Lifecycle
open _root_.NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtractionWork (Primitives)
open _root_.NightstreamFPrime.Spec.Folding.PiRLC.CoordinateForkLaw (Challenge)
open PiCCSStoredWitnessCheck (carrier commit statement)

variable (inst : SecurityInstance)

abbrev FunctionalWitness := OutputWitness productionShape (carrier inst).carrierWidth
abbrev FunctionalCandidate := Probe K productionShape × FunctionalWitness inst
abbrev CheckClock := FunctionalCandidate inst → Nat
abbrev AccessClock := FunctionalWitness inst → Fin productionShape.sourceCount → Fin
    (carrier inst).carrierWidth → Nat

private def storeProbe {shape : Shape} (probe : Probe K shape) : StoredProbe shape where
  coins := probe.coins
  certificate := probe.response.rounds
  pad := Vector.ofFn fun source => Vector.ofFn (probe.response.fullOutput.padCoordinate source)
  matrix := Vector.ofFn fun source => Vector.ofFn fun matrix =>
    Vector.ofFn (probe.response.fullOutput.matrixCoordinate source matrix)

private theorem storeProbe_view {shape : Shape} (probe : Probe K shape) :
    (storeProbe probe).view = probe := by
  cases probe with
  | mk coins response =>
      cases response with
      | mk rounds output =>
          cases output with
          | mk pad matrix =>
              apply congrArg (Probe.mk coins)
              apply congrArg (Response.mk rounds)
              apply congrArg₂ FullOutputCoordinates.FullOutput.mk
              · funext source coefficient
                simp [storeProbe, StoredProbe.view, Vector.get]
              · funext source port coefficient
                simp [storeProbe, StoredProbe.view, Vector.get]

private def storeWitness {shape : Shape} {columns : Nat}
    (witness : OutputWitness shape columns) : Vector (Vector F columns) shape.sourceCount :=
  Vector.ofFn fun source => Vector.ofFn (witness.assignments source)

private theorem storeWitness_view {shape : Shape} {carrier : Phi81Relation.Shape}
    (witness : OutputWitness shape carrier.carrierWidth) :
    StoredWitnessProjection.view (shape := shape) (carrier := carrier) (storeWitness witness) = witness := by
  cases witness with
  | mk assignments =>
      apply congrArg OutputWitness.mk
      funext source column
      simp [storeWitness, StoredWitnessProjection.view, Vector.get]

private def storeCandidate (candidate : FunctionalCandidate inst) : PiCCSStoredWitnessCheck.Candidate inst :=
  (storeProbe candidate.1, storeWitness candidate.2)

/-- Convert the actual returned values to the existing arrays. Abort stays
abort; no certificate parsing, filtering or replacement takes place here. -/
def storeOutcome (outcome : Outcome productionShape (carrier inst)) : StoredOutcome productionShape
    (carrier inst) :=
  outcome.map (storeCandidate inst)

theorem storeOutcome_view (outcome : Outcome productionShape (carrier inst)) :
    storedView (storeOutcome inst outcome) = outcome := by
  cases outcome with
  | none => rfl
  | some candidate =>
      rcases candidate with ⟨probe, witness⟩
      simp only [storeOutcome, storeCandidate, storedView, Option.map_some,
        storeProbe_view, storeWitness_view]

/-- The functional consumer executes the same selected Bool checker and
reads the same witness. Both counters are caller-owned declared values;
this definition supplies no constant-cost or runtime bound. -/
def sourceProgram (input : PiCCSInputCheck.Input) (checkClock : CheckClock inst)
    (accessClock : AccessClock inst) :
    CheckedWitnessExtraction.Program productionShape (carrier inst) where
  check := fun candidate =>
      ⟨PiCCSStoredWitnessCheck.check inst input (storeCandidate inst candidate), checkClock candidate⟩
  access := fun witness source column => ⟨witness.assignments source column, accessClock witness source column⟩

/-- Exact check and access refinement for all functional candidates, including
malformed raw certificates. Neither a matrix nor a checker premise remains. -/
theorem sourceProgram_correct (input : PiCCSInputCheck.Input)
    (checkClock : CheckClock inst) (accessClock : AccessClock inst) :
    CheckedWitnessExtraction.Correct (width := 9) (sourceProgram inst input checkClock accessClock)
      (commit inst) productionGlobalParams (statement inst input) := by
  constructor
  · intro probe witness
    change StoredWitnessCheck.check (commit inst) productionGlobalParams (statement inst input)
      (ProductionKey.degreeBound inst.relation)
      ((storeProbe probe).view, storeWitness witness) = true ↔ _
    rw [storeProbe_view, StoredWitnessCheck.check_eq_true_iff,
      storeWitness_view, ProductionKey.degreeBound_eq]
  · intro witness source column
    rfl

private theorem sourceProjection_value (input : PiCCSInputCheck.Input)
    (checkClock : CheckClock inst) (accessClock : AccessClock inst) (witness : FunctionalWitness inst) :
    (CostedWitnessProjection.project (sourceProgram inst input checkClock accessClock).access witness).value =
      (StoredWitnessProjection.project (storeWitness witness)).value := by
  rw [CostedWitnessProjection.project_value _
      (sourceProgram_correct inst input checkClock accessClock).access,
    StoredWitnessProjection.project_value, storeWitness_view]

/-- Caller clocks do not change the actual returned source values. -/
theorem finish_value (input : PiCCSInputCheck.Input) (checkClock : CheckClock inst)
    (accessClock : AccessClock inst)
    (outcome : Outcome productionShape (carrier inst)) :
    (CheckedWitnessExtraction.finish (sourceProgram inst input checkClock accessClock) outcome).value =
      PiCCSStoredWitnessCheck.finishValue inst input (storeOutcome inst outcome) := by
  cases outcome with
  | none => rfl
  | some candidate =>
      rcases candidate with ⟨probe, witness⟩
      apply Option.ext
      intro values
      rw [CheckedWitnessExtraction.finish_return_iff]
      change (∃ otherProbe otherWitness,
        some (probe, witness) = some (otherProbe, otherWitness) ∧
          PiCCSStoredWitnessCheck.check inst input (storeCandidate inst (otherProbe, otherWitness)) = true ∧
            (CostedWitnessProjection.project
                (sourceProgram inst input checkClock accessClock).access otherWitness).value = values) ↔
        (if PiCCSStoredWitnessCheck.check inst input (storeCandidate inst (probe, witness)) then
          some (StoredWitnessProjection.project (storeWitness witness)).value else none) = some values
      simp only [sourceProjection_value inst]
      cases checked : PiCCSStoredWitnessCheck.check inst input (storeCandidate inst (probe, witness)) <;>
        simp [checked]

section Probability

variable {Context State Tape : Type*}
  (inputs : Context → PiCCSInputCheck.Input)
  [DecidableEq RingF]
  [Fintype (Challenge (ProductionKey.key inst.relation inst.ajtai).piRlcAlgebra)]
  [Nonempty (Challenge (ProductionKey.key inst.relation inst.ajtai).piRlcAlgebra)]
  (contexts : PMF Context)
  (originalFirstPhase : Context → InteractivePrefix.Prover State productionShape 9)
  (publicCheck : Context → Probe K productionShape → Bool)
  (continuation : ∀ context (coins : PublicCoins K productionShape)
    (output : FullOutputCoordinates.FullOutput K productionShape), State →
      Lifecycle.Nifs.WeakExtraction.Continuation Tape inst.relation
        inst.ajtai
        (inst.running (inputs context)) (inst.fresh (inputs context)) coins output)
  (primitives : Primitives RingF
    (PaperAlgebra.Assignment (logicalWidth := inst.logicalWidth) (publicFits := inst.publicFits)))
  (checkClock : Context → CheckClock inst) (accessClock : Context → AccessClock inst)

/-- The existing sequential source-return probability counts exactly this
stored checked return. The original prefix, captured state, suffix law and
consumed weak endpoint are identical on both sides. -/
theorem returnedSourceProbability_eq_finishValue :
    Lifecycle.Nifs.InteractiveOutput.returnedSourceProbability
      inst.relation inst.ajtai
      (fun context => inst.running (inputs context))
      (fun context => inst.fresh (inputs context))
      originalFirstPhase publicCheck continuation primitives
      (fun context => sourceProgram inst (inputs context) (checkClock context)
          (accessClock context)) contexts =
    PaperCompositionProbability.eventProbability contexts
      (Lifecycle.Nifs.InteractiveComposition.firstPhase originalFirstPhase publicCheck)
      (Lifecycle.Nifs.InteractiveComposition.suffixLaw inst.relation
        inst.ajtai
        (fun context => inst.running (inputs context))
        (fun context => inst.fresh (inputs context)) continuation)
      (Lifecycle.Nifs.InteractiveComposition.consume inst.relation
        inst.ajtai primitives)
      (fun context outcome => SourceReturned (commit inst) productionGlobalParams
          (statement inst (inputs context))
        (PiCCSStoredWitnessCheck.finishValue inst (inputs context) (storeOutcome inst outcome))) := by
  unfold Lifecycle.Nifs.InteractiveOutput.returnedSourceProbability
  congr 1
  funext context outcome
  rw [finish_value inst]
  rfl

/-- Instantiate sourceCorrect in the existing probability consumer. This
is the same sourceProbability used by SupportedExtraction, with no free
check/access refinement and no work or security-probability bound added. -/
theorem returnedSourceProbability_eq_sourceProbability :
    Lifecycle.Nifs.InteractiveOutput.returnedSourceProbability
      inst.relation inst.ajtai
      (fun context => inst.running (inputs context))
      (fun context => inst.fresh (inputs context))
      originalFirstPhase publicCheck continuation primitives
      (fun context => sourceProgram inst (inputs context) (checkClock context)
          (accessClock context)) contexts =
    PaperCompositionProbability.sourceProbability contexts
      (Lifecycle.Nifs.InteractiveComposition.firstPhase originalFirstPhase publicCheck)
      (Lifecycle.Nifs.InteractiveComposition.suffixLaw inst.relation
        inst.ajtai
        (fun context => inst.running (inputs context))
        (fun context => inst.fresh (inputs context)) continuation)
      (Lifecycle.Nifs.InteractiveComposition.consume inst.relation
        inst.ajtai primitives)
      (fun _ => PaperAlgebra.openingMaps inst.ajtai) productionGlobalParams
      (fun context => statement inst (inputs context)) := by
  exact Lifecycle.Nifs.InteractiveOutput.returnedSourceProbability_eq
    inst.relation inst.ajtai
    (fun context => inst.running (inputs context))
    (fun context => inst.fresh (inputs context))
    originalFirstPhase publicCheck continuation primitives
    (fun context => sourceProgram inst (inputs context) (checkClock context) (accessClock context))
    (fun context => sourceProgram_correct inst (inputs context) (checkClock context)
        (accessClock context)) contexts

end Probability

end NightstreamFPrime.Export.Stage1.PiCCSStoredSourceProbability
