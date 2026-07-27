import Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionMixingBoundary.IdealInteractiveFeMixing

/-!
Finite probability bound for the exact production FE mixing-root event.

Assurance tier: model-level registered-deviation refinement.

Owns: selector marginals in exact engine order, conditional gamma root
counting for the dense production FE polynomial, exact transport to the
repository's named `Fe.MixingSoundness.MixingRoot`, and the explicit
row/lane/gamma union bound.

Does not own: NC mixing, SumCheck collisions, Fiat--Shamir, Poseidon2,
closed field certificates, Rust/R1CS, artifacts, costs, or rows.

Emits constraints: no.

| Boundary | Owned equation |
| --- | --- |
| Exact event | Boolean event is equivalent to `Fe.MixingSoundness.MixingRoot` |
| Causal sampling | Each selector/gamma marginal retains the selected transcript prefix |
| Union | FE failure probability is bounded by row + lane + gamma ratios |
-/

set_option autoImplicit false
set_option maxHeartbeats 1000000

namespace Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionMixingBoundary.IdealInteractiveFeSoundness

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial
open Nightstream.SuperNeo.InteractiveReduction.CoordinateForking.FiniteWords
open Nightstream.SuperNeo.InteractiveReduction.FiniteUniform
open Nightstream.SuperNeo.SumCheck.Finite
open IdealInteractiveCarrier
open IdealInteractiveRootCounting
open IdealInteractiveFeMixing

private abbrev ops := ConcreteCarrier.extensionOps
private abbrev laws := ConcreteCarrier.extensionLaws

private noncomputable def propositionEvent (proposition : Prop) : Bool :=
  @ite Bool proposition (Classical.propDecidable proposition) true false

private theorem false_probability
    {Seed : Type}
    (support : Support Seed) :
    support.uniform.probabilityBool (fun _seed => false) = 0 := by
  unfold Experiment.probabilityBool Experiment.countBool
  simp [Rat.div_def]

private theorem probabilityBool_or_le_of_bounds
    {Outcome : Type}
    (experiment : Experiment Outcome)
    (left right : Outcome -> Bool)
    (leftBudget rightBudget : Rat)
    (leftBound : experiment.probabilityBool left <= leftBudget)
    (rightBound : experiment.probabilityBool right <= rightBudget) :
    experiment.probabilityBool
        (fun outcome => left outcome || right outcome) <=
      leftBudget + rightBudget := by
  exact Rat.le_trans
    (experiment.probabilityBool_or_le left right)
    (Rat.le_trans
      ((Rat.add_le_add_right
        (c := experiment.probabilityBool right)).mpr leftBound)
      ((Rat.add_le_add_left (c := leftBudget)).mpr rightBound))

/-- Fresh-row selector bound over the exact alpha, betaA, betaR engine
prefix. The event is false for a carried controller. -/
theorem freshSelectorBad_probability_le
    {shape : SemanticShape}
    {profile : Fe.SupportedProfile shape PiCcsDomains.production.fe}
    {data : Data shape}
    (controller : Controller profile data)
    (noZeroDivisors : FiniteRootCounting.NoZeroDivisors ops)
    (alphabet : Support K) :
    ((enginePrefixSupport (shape := shape) alphabet).uniform
      ).probabilityBool (freshSelectorBad controller) <=
      ratio shape.rowVariables alphabet.cardinality := by
  cases controller with
  | fresh source tableNonzero =>
      let laneSupport :=
        Support.challengeVectors alphabet
          PiCcsDomains.production.fe.laneVariables
      let rowSupport :=
        Support.challengeVectors alphabet shape.rowVariables
      calc
        ((enginePrefixSupport (shape := shape) alphabet).uniform
            ).probabilityBool
              (freshSelectorBad (.fresh source tableNonzero)) =
          rowSupport.uniform.probabilityBool
            (fun betaR =>
              decide
                ((freshTable data source).evaluateCoordinates ops
                  (List.ofFn betaR) = ops.zero)) := by
            simpa [enginePrefixSupport, laneSupport, rowSupport,
              freshSelectorBad] using
              Support.product_uniform_probabilityBool_second
                (laneSupport.product laneSupport) rowSupport
                (fun betaR =>
                  decide
                    ((freshTable data source).evaluateCoordinates ops
                      (List.ofFn betaR) = ops.zero))
        _ <= ratio shape.rowVariables alphabet.cardinality :=
          multilinearZero_probability_le ops laws noZeroDivisors
            (freshTable data source) alphabet tableNonzero
  | carried coordinate tableNonzero =>
      have eventFalse :
          freshSelectorBad
              (.carried (profile := profile) coordinate tableNonzero) =
            fun _engineHead => false := by
        funext engineHead
        rfl
      rw [eventFalse, false_probability]
      exact ratio_nonneg _ alphabet.cardinality_pos

/-- Carried-lane selector bound over the exact engine prefix. The event
depends on alpha, while betaA and betaR remain sampled marginals. -/
theorem carriedSelectorBad_probability_le
    {shape : SemanticShape}
    {profile : Fe.SupportedProfile shape PiCcsDomains.production.fe}
    {data : Data shape}
    (controller : Controller profile data)
    (noZeroDivisors : FiniteRootCounting.NoZeroDivisors ops)
    (alphabet : Support K) :
    ((enginePrefixSupport (shape := shape) alphabet).uniform
      ).probabilityBool (carriedSelectorBad controller) <=
      ratio PiCcsDomains.production.fe.laneVariables
        alphabet.cardinality := by
  cases controller with
  | fresh source tableNonzero =>
      have eventFalse :
          carriedSelectorBad
              (.fresh (profile := profile) source tableNonzero) =
            fun _engineHead => false := by
        funext engineHead
        rfl
      rw [eventFalse, false_probability]
      exact ratio_nonneg _ alphabet.cardinality_pos
  | carried coordinate tableNonzero =>
      let laneSupport :=
        Support.challengeVectors alphabet
          PiCcsDomains.production.fe.laneVariables
      let rowSupport :=
        Support.challengeVectors alphabet shape.rowVariables
      let laneEvent : AlphaWord -> Bool := fun alpha =>
        decide
          ((carriedTable profile data coordinate.running coordinate.matrix
            ).evaluateCoordinates ops (List.ofFn alpha) = ops.zero)
      calc
        ((enginePrefixSupport (shape := shape) alphabet).uniform
            ).probabilityBool
              (carriedSelectorBad (.carried coordinate tableNonzero)) =
          (laneSupport.product laneSupport).uniform.probabilityBool
            (fun pair => laneEvent pair.1) := by
              simpa [enginePrefixSupport, laneSupport, rowSupport,
                laneEvent, carriedSelectorBad] using
                Support.product_uniform_probabilityBool_first
                  (laneSupport.product laneSupport) rowSupport
                  (fun pair => laneEvent pair.1)
        _ = laneSupport.uniform.probabilityBool laneEvent :=
          Support.product_uniform_probabilityBool_first
            laneSupport laneSupport laneEvent
        _ <= ratio PiCcsDomains.production.fe.laneVariables
              alphabet.cardinality := by
          exact multilinearZero_probability_le ops laws noZeroDivisors
            (carriedTable profile data coordinate.running coordinate.matrix)
            alphabet tableNonzero

/-- Gamma-root event only when the exact coefficient list is nonzero. This
guard makes the conditional root theorem total on every engine prefix. -/
noncomputable def gammaRootEvent
    {shape : SemanticShape}
    (profile : Fe.SupportedProfile shape PiCcsDomains.production.fe)
    (data : Data shape)
    (engineHead : (AlphaWord × BetaAWord) × BetaRWord shape)
    (gamma : K) : Bool :=
  @ite Bool
    (CoefficientRootCounting.AllZero ops
      (gammaCoefficients profile data engineHead.1.1 engineHead.2))
    (Classical.propDecidable _)
    false
    (decide
      (Message.evaluateCoefficients ops.toOps gamma
        (gammaCoefficients profile data engineHead.1.1 engineHead.2) =
          ops.zero))

theorem gammaRoot_component_probability_le
    {shape : SemanticShape}
    (profile : Fe.SupportedProfile shape PiCcsDomains.production.fe)
    (data : Data shape)
    (engineHead : (AlphaWord × BetaAWord) × BetaRWord shape)
    (noZeroDivisors : FiniteRootCounting.NoZeroDivisors ops)
    (alphabet : Support K) :
    alphabet.uniform.probabilityBool
        (gammaRootEvent profile data engineHead) <=
      ratio (gammaDegree shape) alphabet.cardinality := by
  classical
  by_cases allZero :
      CoefficientRootCounting.AllZero ops
        (gammaCoefficients profile data engineHead.1.1 engineHead.2)
  · have eventFalse :
        gammaRootEvent profile data engineHead =
          fun _gamma => false := by
      funext gamma
      simp [gammaRootEvent, allZero]
    rw [eventFalse, false_probability]
    exact ratio_nonneg _ alphabet.cardinality_pos
  · have eventEq :
        gammaRootEvent profile data engineHead =
          fun gamma =>
            decide
              (Message.evaluateCoefficients ops.toOps gamma
                (gammaCoefficients profile data engineHead.1.1
                  engineHead.2) = ops.zero) := by
      funext gamma
      simp [gammaRootEvent, allZero]
    rw [eventEq]
    exact coefficientZero_probability_le ops laws noZeroDivisors
      (gammaDegree shape)
      (gammaCoefficients profile data engineHead.1.1 engineHead.2)
      (gammaCoefficients_count_eq_degree_add_one
        profile data engineHead.1.1 engineHead.2)
      alphabet allZero

/-- Conditional gamma bound averaged over every prior engine coordinate.
Gamma is sampled once, after alpha, betaA, and betaR. -/
theorem gammaRoot_engine_probability_le
    {shape : SemanticShape}
    (profile : Fe.SupportedProfile shape PiCcsDomains.production.fe)
    (data : Data shape)
    (noZeroDivisors : FiniteRootCounting.NoZeroDivisors ops)
    (alphabet : Support K) :
    ((engineSupport (shape := shape) alphabet).uniform).probabilityBool
        (fun engine =>
          gammaRootEvent profile data engine.1 engine.2) <=
      ratio (gammaDegree shape) alphabet.cardinality := by
  unfold engineSupport
  exact product_probabilityBool_le_of_components
    (enginePrefixSupport (shape := shape) alphabet) alphabet
    (gammaRootEvent profile data)
    (ratio (gammaDegree shape) alphabet.cardinality)
    (fun engineHead _member =>
      gammaRoot_component_probability_le profile data engineHead
        noZeroDivisors alphabet)

/-- Exact repository FE mixing-root event on the selected engine seed. -/
noncomputable def mixingRootEvent
    {shape : SemanticShape}
    (profile : Fe.SupportedProfile shape PiCcsDomains.production.fe)
    (data : Data shape)
    (engine : EngineSeed shape) : Bool :=
  propositionEvent
    (Fe.MixingSoundness.MixingRoot profile data
      (coins engine.1.1.1 engine.1.1.2 engine.1.2 engine.2))

@[simp] theorem mixingRootEvent_eq_true_iff
    {shape : SemanticShape}
    (profile : Fe.SupportedProfile shape PiCcsDomains.production.fe)
    (data : Data shape)
    (engine : EngineSeed shape) :
    mixingRootEvent profile data engine = true ↔
      Fe.MixingSoundness.MixingRoot profile data
        (coins engine.1.1.1 engine.1.1.2 engine.1.2 engine.2) := by
  simp [mixingRootEvent, propositionEvent]

private theorem mixingRoot_implies_union
    {shape : SemanticShape}
    (profile : Fe.SupportedProfile shape PiCcsDomains.production.fe)
    (data : Data shape)
    (controller : Controller profile data)
    (engine : EngineSeed shape)
    (rootTrue : mixingRootEvent profile data engine = true) :
    (freshSelectorBad controller engine.1 ||
      (carriedSelectorBad controller engine.1 ||
        gammaRootEvent profile data engine.1 engine.2)) = true := by
  have root :
      Fe.MixingSoundness.MixingRoot profile data
        (coins engine.1.1.1 engine.1.1.2 engine.1.2 engine.2) := by
    simpa [mixingRootEvent, propositionEvent] using rootTrue
  cases freshBad : freshSelectorBad controller engine.1 with
  | true => simp
  | false =>
      cases carriedBad : carriedSelectorBad controller engine.1 with
      | true => simp
      | false =>
          have coefficientsNonzero :=
            Controller.controls controller engine.1 freshBad carriedBad
          have evaluationZero :
              Message.evaluateCoefficients ops.toOps engine.2
                  (gammaCoefficients profile data engine.1.1.1
                    engine.1.2) =
                ops.zero := by
            rw [gammaCoefficients_evaluate profile data engine.1.1.1
              engine.1.1.2 engine.1.2 engine.2]
            exact root.compressedZero
          have gammaBad :
              gammaRootEvent profile data engine.1 engine.2 = true := by
            simp [gammaRootEvent, coefficientsNonzero, evaluationZero]
          simp [gammaBad]

/-- Explicit finite union bound for the actual bundled FE `MixingRoot`.
The loss order is row selector, lane selector, then shared gamma. -/
theorem mixingRoot_probability_le
    {shape : SemanticShape}
    (profile : Fe.SupportedProfile shape PiCcsDomains.production.fe)
    (data : Data shape)
    (noZeroDivisors : FiniteRootCounting.NoZeroDivisors ops)
    (alphabet : Support K) :
    ((engineSupport (shape := shape) alphabet).uniform).probabilityBool
        (mixingRootEvent profile data) <=
      ratio shape.rowVariables alphabet.cardinality +
        (ratio PiCcsDomains.production.fe.laneVariables
            alphabet.cardinality +
          ratio (gammaDegree shape) alphabet.cardinality) := by
  classical
  by_cases residualsZero : Semantics.Fe.ResidualsZero data
  · have eventFalse :
        mixingRootEvent profile data = fun _engine => false := by
      funext engine
      apply Bool.eq_false_iff.mpr
      intro rootTrue
      have root :
          Fe.MixingSoundness.MixingRoot profile data
            (coins engine.1.1.1 engine.1.1.2 engine.1.2 engine.2) := by
        simpa [mixingRootEvent, propositionEvent] using rootTrue
      exact root.residualsNonzero residualsZero
    rw [eventFalse, false_probability]
    exact Rat.add_nonneg
      (ratio_nonneg _ alphabet.cardinality_pos)
      (Rat.add_nonneg
        (ratio_nonneg _ alphabet.cardinality_pos)
        (ratio_nonneg _ alphabet.cardinality_pos))
  · let controller :=
      controllerOfResidualsNonzero profile data residualsZero
    let experiment := (engineSupport (shape := shape) alphabet).uniform
    let freshEvent : EngineSeed shape -> Bool := fun engine =>
      freshSelectorBad controller engine.1
    let carriedEvent : EngineSeed shape -> Bool := fun engine =>
      carriedSelectorBad controller engine.1
    let gammaEvent : EngineSeed shape -> Bool := fun engine =>
      gammaRootEvent profile data engine.1 engine.2
    have eventToUnion :
        ∀ engine,
          mixingRootEvent profile data engine = true ->
            (freshEvent engine ||
              (carriedEvent engine || gammaEvent engine)) = true := by
      intro engine rootTrue
      simpa [freshEvent, carriedEvent, gammaEvent] using
        mixingRoot_implies_union profile data controller engine rootTrue
    have freshBound :
        experiment.probabilityBool freshEvent <=
          ratio shape.rowVariables alphabet.cardinality := by
      rw [show
        experiment.probabilityBool freshEvent =
          ((enginePrefixSupport (shape := shape) alphabet).uniform
            ).probabilityBool (freshSelectorBad controller) by
          exact Support.product_uniform_probabilityBool_first
            (enginePrefixSupport (shape := shape) alphabet) alphabet
            (freshSelectorBad controller)]
      exact freshSelectorBad_probability_le controller noZeroDivisors
        alphabet
    have carriedBound :
        experiment.probabilityBool carriedEvent <=
          ratio PiCcsDomains.production.fe.laneVariables
            alphabet.cardinality := by
      rw [show
        experiment.probabilityBool carriedEvent =
          ((enginePrefixSupport (shape := shape) alphabet).uniform
            ).probabilityBool (carriedSelectorBad controller) by
          exact Support.product_uniform_probabilityBool_first
            (enginePrefixSupport (shape := shape) alphabet) alphabet
            (carriedSelectorBad controller)]
      exact carriedSelectorBad_probability_le controller noZeroDivisors
        alphabet
    have gammaBound :
        experiment.probabilityBool gammaEvent <=
          ratio (gammaDegree shape) alphabet.cardinality := by
      simpa [experiment, gammaEvent] using
        gammaRoot_engine_probability_le profile data noZeroDivisors alphabet
    have carriedGamma :
        experiment.probabilityBool
            (fun engine => carriedEvent engine || gammaEvent engine) <=
          ratio PiCcsDomains.production.fe.laneVariables
              alphabet.cardinality +
            ratio (gammaDegree shape) alphabet.cardinality :=
      probabilityBool_or_le_of_bounds experiment carriedEvent gammaEvent
        (ratio PiCcsDomains.production.fe.laneVariables
          alphabet.cardinality)
        (ratio (gammaDegree shape) alphabet.cardinality)
        carriedBound gammaBound
    have allRoots :
        experiment.probabilityBool
            (fun engine =>
              freshEvent engine ||
                (carriedEvent engine || gammaEvent engine)) <=
          ratio shape.rowVariables alphabet.cardinality +
            (ratio PiCcsDomains.production.fe.laneVariables
                alphabet.cardinality +
              ratio (gammaDegree shape) alphabet.cardinality) :=
      probabilityBool_or_le_of_bounds experiment freshEvent
        (fun engine => carriedEvent engine || gammaEvent engine)
        (ratio shape.rowVariables alphabet.cardinality)
        (ratio PiCcsDomains.production.fe.laneVariables
            alphabet.cardinality +
          ratio (gammaDegree shape) alphabet.cardinality)
        freshBound carriedGamma
    exact Rat.le_trans
      (Experiment.probabilityBool_mono experiment eventToUnion)
      allRoots

/-- Delayed pre-SumCheck coordinates are a pure marginal for the FE event. -/
theorem mixingRoot_pre_probability_le
    {shape : SemanticShape}
    (profile : Fe.SupportedProfile shape PiCcsDomains.production.fe)
    (data : Data shape)
    (noZeroDivisors : FiniteRootCounting.NoZeroDivisors ops)
    (alphabet : Support K) :
    ((preSupport (shape := shape) alphabet).uniform).probabilityBool
        (fun seed => mixingRootEvent profile data seed.1) <=
      ratio shape.rowVariables alphabet.cardinality +
        (ratio PiCcsDomains.production.fe.laneVariables
            alphabet.cardinality +
          ratio (gammaDegree shape) alphabet.cardinality) := by
  rw [show
    ((preSupport (shape := shape) alphabet).uniform).probabilityBool
        (fun seed => mixingRootEvent profile data seed.1) =
      ((engineSupport (shape := shape) alphabet).uniform).probabilityBool
        (mixingRootEvent profile data) by
      exact Support.product_uniform_probabilityBool_first
        (engineSupport (shape := shape) alphabet)
        (delayedSupport alphabet) (mixingRootEvent profile data)]
  exact mixingRoot_probability_le profile data noZeroDivisors alphabet

end Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionMixingBoundary.IdealInteractiveFeSoundness
