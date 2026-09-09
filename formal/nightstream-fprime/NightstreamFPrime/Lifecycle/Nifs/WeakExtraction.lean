import NightstreamFPrime.Lifecycle.XOut
import NightstreamFPrime.Lifecycle.ProductionKey
import NightstreamFPrime.Spec.Folding.Nifs.PaperWeakAlgorithm
import NightstreamFPrime.Spec.Folding.Nifs.PaperWeakOutput
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.StrongProbability

/-!
The actual weak suffix for the selected production key. Its batch depends on
the public coins and full PiCCS output, and its endpoint consumer uses the
literal returned list. The local probability bound is derived from that
continuation; it is not supplied as a composition premise.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Lifecycle.Nifs.WeakExtraction

attribute [local instance] Classical.propDecidable
open scoped BigOperators
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open StrongReduction ConcreteCarrier
open _root_.NightstreamFPrime.Spec.Folding.Nifs
open _root_.NightstreamFPrime.Lifecycle.PaperAlgebra
open _root_.NightstreamFPrime.Lifecycle.ProductionKey
open PiRLC.PaperForkExtraction PiRLC.CoordinateForkLaw
open PiRLC.PaperForkExtractionWork

variable (Tape : Type*) {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}
  (relation : LogicalRelation logicalWidth publicFits)
  (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
  (running : Lifecycle.Running (logicalWidth := logicalWidth) (publicFits := publicFits))
  (fresh : Lifecycle.Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))

/-- The weak input uses only the public output fields. Empty rounds here are
a batch view; no acceptance claim is made about this auxiliary probe. -/
noncomputable def batchForOutput (coins : PublicCoins K productionShape)
    (output : FullOutputCoordinates.FullOutput K productionShape) :=
  PaperStrongInterface.piRlcBatchForProbe (ProductionKey.key relation ajtai) running fresh {
    coins := coins
    response := { rounds := ⟨[]⟩, fullOutput := output }
  }

/-- Every actual probe with these public fields has this exact weak batch. -/
theorem batchForOutput_eq_probe (probe : Probe K productionShape) :
    batchForOutput relation ajtai running fresh probe.coins probe.response.fullOutput =
      PaperStrongInterface.piRlcBatchForProbe (ProductionKey.key relation ajtai) running fresh probe := rfl

/-- One captured continuation at the selected key and this exact C output. -/
abbrev Continuation (coins : PublicCoins K productionShape)
    (output : FullOutputCoordinates.FullOutput K productionShape) :=
  PaperWeakAlgorithm.Algorithm Tape (ProductionKey.key relation ajtai).piRlcAlgebra
    (batchForOutput relation ajtai running fresh coins output)
    (ProductionKey.key relation ajtai).piDecAlgebra
    (ProductionKey.key relation ajtai).piDecPublicInputSplit
    (ProductionKey.key relation ajtai).piDecEvaluationArity

variable {Tape}
  [DecidableEq RingF]
  [Fintype (Challenge (ProductionKey.key relation ajtai).piRlcAlgebra)]
  [Nonempty (Challenge (ProductionKey.key relation ajtai).piRlcAlgebra)]
  [Fintype (PaperAlgebra.Assignment (logicalWidth := logicalWidth) (publicFits := publicFits))]

/-- The finite endpoint law is derived from the actual charged continuation. -/
noncomputable def endpointLaw {coins : PublicCoins K productionShape}
    {output : FullOutputCoordinates.FullOutput K productionShape}
    (continuation : Continuation Tape relation ajtai running fresh coins output) :=
  PaperWeakLaw.law continuation.chargedOracle continuation.check

/-- Decode the actual coordinate terminal return into the selected C witness. -/
noncomputable def consume
    (program : Primitives RingF
      (PaperAlgebra.Assignment (logicalWidth := logicalWidth) (publicFits := publicFits)))
    (endpoint : PaperWeakLaw.Endpoint (Fin (ProductionKey.key relation ajtai).arity.total)
      (Challenge (ProductionKey.key relation ajtai).piRlcAlgebra)
      (PaperAlgebra.Assignment (logicalWidth := logicalWidth) (publicFits := publicFits))) :=
  PaperWeakOutput.endpointWitness (ProductionKey.key relation ajtai) program endpoint

omit [DecidableEq RingF]
  [Fintype (PaperAlgebra.Assignment (logicalWidth := logicalWidth) (publicFits := publicFits))] in
/-- The original suffix event is a probability under the same public/private
coin law used by the extractor. -/
theorem continuation_success_range {coins : PublicCoins K productionShape}
    {output : FullOutputCoordinates.FullOutput K productionShape}
    (continuation : Continuation Tape relation ajtai running fresh coins output) :
    0 ≤ continuation.successProbability ∧ continuation.successProbability ≤ 1 := by
  rw [← continuation.base_rate (ProductionKey.key relation ajtai).kPositive]
  exact ⟨PiRLC.CoordinateRetry.Line.rate_nonnegative _, PiRLC.CoordinateRetry.Line.rate_le_one _⟩

private theorem relaxedSuccess_map_iff
    (probe : Probe K productionShape)
    (accepted : probe.FixedWidthAccepted extensionOps K.embed
      ((ProductionKey.key relation ajtai).statement running fresh) 9)
    (outcome : Option (OutputWitness productionShape (Phi81CarrierLayout.carrierWidth logicalWidth))) :
    StrongProbability.RelaxedSuccess (width := 9) (PaperAlgebra.openingMaps ajtai) productionGlobalParams
      ((ProductionKey.key relation ajtai).statement running fresh) (outcome.map (fun witness => (probe, witness))) ↔
      ∃ witness, outcome = some witness ∧
        AmbientOutputHolds extensionOps K.embed (PaperAlgebra.openingMaps ajtai) productionGlobalParams
          ((ProductionKey.key relation ajtai).statement running fresh) probe witness := by
  cases outcome with
  | none => simp [StrongProbability.RelaxedSuccess]
  | some witness =>
      simp only [StrongProbability.RelaxedSuccess, Option.map_some]
      constructor
      · rintro ⟨otherProbe, otherWitness, equal, _checked, valid⟩
        have pair := Option.some.inj equal
        have sameProbe := congrArg Prod.fst pair
        have sameWitness := congrArg Prod.snd pair
        dsimp only at sameProbe sameWitness
        subst otherProbe
        subst otherWitness
        exact ⟨witness, rfl, valid⟩
      · rintro ⟨otherWitness, equal, valid⟩
        have same := Option.some.inj equal
        subst otherWitness
        exact ⟨probe, witness, rfl, accepted, valid⟩

/-- The actual decoded weak return supplies C relaxed success, with the paper
coordinate-collision loss from the original final-output success event. The
selected arity is definitionally 17. No local success-bound premise is used. -/
theorem weak_relaxed_success_bound
    (laws : ExtractionAlgebra (ProductionKey.key relation ajtai).piRlcSemantics
      (ProductionKey.key relation ajtai).params (ProductionKey.key relation ajtai).piRlcAlgebra)
    (strongSet : StrongSetUnits laws.ring (ProductionKey.key relation ajtai).piRlcAlgebra.challengeValid)
    (program : Primitives RingF
      (PaperAlgebra.Assignment (logicalWidth := logicalWidth) (publicFits := publicFits)))
    (correct : Correct laws.ring laws.assignmentModule program)
    (bounds : PrimitiveBounds) (bounded : Bounded laws.ring program bounds)
    (probe : Probe K productionShape)
    (accepted : probe.FixedWidthAccepted extensionOps K.embed
      ((ProductionKey.key relation ajtai).statement running fresh) 9)
    (continuation : Continuation Tape relation ajtai running fresh probe.coins probe.response.fullOutput) :
    continuation.successProbability - ((ProductionKey.key relation ajtai).arity.total : ℝ) /
        Fintype.card (Challenge (ProductionKey.key relation ajtai).piRlcAlgebra) ≤
      ∑ endpoint, (endpointLaw relation ajtai running fresh continuation endpoint).toReal *
        (if StrongProbability.RelaxedSuccess (width := 9) (PaperAlgebra.openingMaps ajtai)
          productionGlobalParams ((ProductionKey.key relation ajtai).statement running fresh)
          ((consume relation ajtai program endpoint).map (fun witness => (probe, witness)))
          then (1 : ℝ) else 0) := by
  have checkSpec : ∀ response, (continuation.parentChecker response).accepted = true ↔
      response.Success (ProductionKey.key relation ajtai).piRlcSemantics
        (ProductionKey.key relation ajtai).params (ProductionKey.key relation ajtai).piRlcAlgebra
        (PaperStrongInterface.piRlcBatchForProbe (ProductionKey.key relation ajtai) running fresh probe) := by
    intro response
    rw [← batchForOutput_eq_probe relation ajtai running fresh probe]
    exact continuation.parentChecker_spec response
  have lower := continuation.weak_success_bound laws strongSet
    (ProductionKey.key relation ajtai).kPositive program correct bounds bounded
  rw [PaperWeakLaw.successProbability_eq_returningProbability] at lower
  have mean := PaperWeakOutput.valid_mean_eq_returningProbability
    (ProductionKey.key relation ajtai) running fresh laws strongSet probe continuation.chargedOracle
    continuation.parentChecker checkSpec program correct
  unfold PaperWeakAlgorithm.Algorithm.check at lower
  rw [← mean] at lower
  refine lower.trans_eq ?_
  apply Finset.sum_congr rfl
  intro endpoint _
  have event := relaxedSuccess_map_iff relation ajtai running fresh probe accepted
    (consume relation ajtai program endpoint)
  change (endpointLaw relation ajtai running fresh continuation endpoint).toReal *
    (if ∃ witness, consume relation ajtai program endpoint = some witness ∧
      AmbientOutputHolds extensionOps K.embed (PaperAlgebra.openingMaps ajtai) productionGlobalParams
        ((ProductionKey.key relation ajtai).statement running fresh) probe witness
      then (1 : ℝ) else 0) = _
  simp only [← event]

end NightstreamFPrime.Lifecycle.Nifs.WeakExtraction
