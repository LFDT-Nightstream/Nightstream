import Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionMixingBoundary.IdealInteractiveExecution
import Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionMixingBoundary.IdealInteractiveFeSoundness
import Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionMixingBoundary.IdealInteractiveNcMixing
import Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionMixingBoundary.Algebra

/-!
Concrete ideal-interactive soundness bound for production Split-NC.

Assurance tier: model-level registered-deviation refinement.

Owns: the exact FE and NC mixing-root union, the existing physical FE/NC
SumCheck collision union, transport to the actual production
`FeFailure`/`NcFailure` families, and the frozen `(mixing + SumCheck)` loss
grouping over one transcript-ordered finite support.

Does not own: Fiat--Shamir, Poseidon2, a bounded production sampler, closed
Goldilocks arithmetic certificates, Rust/R1CS, artifacts, costs, or rows.

Emits constraints: no.

| Boundary | Owned equation |
| --- | --- |
| Event transport | Algebraic event equals the actual `FeFailure ∨ NcFailure` event |
| Collision | Existing split FE/NC round-collision bound is reused unchanged |
| Total loss | `(feMixing + ncMixing) + splitCollision`, without reassociation |
-/

set_option autoImplicit false
set_option maxHeartbeats 1000000

namespace Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionMixingBoundary.IdealInteractiveSoundness

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.SuperNeo.InteractiveReduction.FiniteUniform
open IdealInteractiveCarrier
open IdealInteractiveRootCounting

private abbrev ops := ConcreteCarrier.extensionOps

universe uState

private noncomputable def propositionEvent (proposition : Prop) : Bool :=
  @ite Bool proposition (Classical.propDecidable proposition) true false

@[simp] private theorem propositionEvent_eq_true_iff
    (proposition : Prop) :
    propositionEvent proposition = true ↔ proposition := by
  simp [propositionEvent]

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

/-! ## Frozen-ordered loss expression -/

/-- Exact FE compression loss: row selector, lane selector, shared gamma. -/
def feMixingBudget (shape : SemanticShape) (cardinality : Nat) : Rat :=
  ratio shape.rowVariables cardinality +
    (ratio PiCcsDomains.production.fe.laneVariables cardinality +
      ratio
        (IdealInteractiveFeMixing.gammaDegree shape)
        cardinality)

/-- Exact NC compression loss in production constructor order. -/
def ncMixingBudget (shape : SemanticShape) (cardinality : Nat) : Rat :=
  ratio PiCcsDomains.production.nc.laneVariables cardinality +
    (ratio PiCcsDomains.production.nc.blockVariables cardinality +
      (ratio (shape.sourceCount - 1) cardinality +
        ratio 1 cardinality))

/-- Existing physical FE-plus-NC SumCheck collision loss. -/
def splitCollisionBudget
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits : ringDegree * publicRingColumns <= shape.carrierWidth}
    (baseInput : ProductionRefinement.AuthoritativeInput
      (shape := shape) (State := State)
      (publicRingColumns := publicRingColumns)
      (verifierRows := verifierRows)
      (publicFits := publicFits))
    (cardinality : Nat) : Rat :=
  ratio
      (CausalSoundness.Generic.feRoundCount shape
          PiCcsDomains.production.fe *
        SumCheck.Fe.Drow (PublicInput.ofSources baseInput.data))
      cardinality +
    ratio
      (CausalSoundness.ncRoundCount *
        Polynomial.Nc.Degree.ncSumcheckDegreeBound)
      cardinality

/-- Frozen production loss grouping: all mixing roots, then SumCheck. -/
def totalBudget
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits : ringDegree * publicRingColumns <= shape.carrierWidth}
    (baseInput : ProductionRefinement.AuthoritativeInput
      (shape := shape) (State := State)
      (publicRingColumns := publicRingColumns)
      (verifierRows := verifierRows)
      (publicFits := publicFits))
    (cardinality : Nat) : Rat :=
  (feMixingBudget shape cardinality +
    ncMixingBudget shape cardinality) +
  splitCollisionBudget baseInput cardinality

/-! ## Exact events -/

/-- The two exact pre-SumCheck mixing families. FE and NC read the same
sampled gamma from `seed`. -/
noncomputable def mixingEvent
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits : ringDegree * publicRingColumns <= shape.carrierWidth}
    (baseInput : ProductionRefinement.AuthoritativeInput
      (shape := shape) (State := State)
      (publicRingColumns := publicRingColumns)
      (verifierRows := verifierRows)
      (publicFits := publicFits))
    (seed : PreSeed shape) : Bool :=
  IdealInteractiveFeSoundness.mixingRootEvent
      baseInput.full.profile baseInput.data seed.1 ||
    IdealInteractiveNcMixing.ncMixingRootEvent
      baseInput.full.covers baseInput.data
      (ProductionProjection.productionWeights baseInput.full)
      baseInput.full.pending seed

/-- Exact existing physical FE-or-NC SumCheck collision. -/
noncomputable def splitCollisionEvent
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits : ringDegree * publicRingColumns <= shape.carrierWidth}
    (alphabet : Support K)
    (baseInput : ProductionRefinement.AuthoritativeInput
      (shape := shape) (State := State)
      (publicRingColumns := publicRingColumns)
      (verifierRows := verifierRows)
      (publicFits := publicFits))
    (strategy : IdealInteractiveExecution.Strategy baseInput)
    (seed : Seed shape) : Bool :=
  propositionEvent
    (IdealInteractiveExecution.SplitCollision
      alphabet baseInput strategy seed)

/-- The concrete algebraic event, preserving frozen mixing-then-SumCheck
grouping without reassociation. -/
noncomputable def algebraicFailureEvent
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits : ringDegree * publicRingColumns <= shape.carrierWidth}
    (alphabet : Support K)
    (baseInput : ProductionRefinement.AuthoritativeInput
      (shape := shape) (State := State)
      (publicRingColumns := publicRingColumns)
      (verifierRows := verifierRows)
      (publicFits := publicFits))
    (strategy : IdealInteractiveExecution.Strategy baseInput)
    (seed : Seed shape) : Bool :=
  mixingEvent baseInput seed.1 ||
    splitCollisionEvent alphabet baseInput strategy seed

/-- The actual production failure family for the replayed physical
certificate. This is not a replacement algebraic event. -/
def NamedFailure
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits : ringDegree * publicRingColumns <= shape.carrierWidth}
    (alphabet : Support K)
    (baseInput : ProductionRefinement.AuthoritativeInput
      (shape := shape) (State := State)
      (publicRingColumns := publicRingColumns)
      (verifierRows := verifierRows)
      (publicFits := publicFits))
    (strategy : IdealInteractiveExecution.Strategy baseInput)
    (suffix : Seed shape ->
      IdealInteractiveExecution.Suffix shape publicRingColumns verifierRows
        publicFits)
    (seed : Seed shape) : Prop :=
  ProductionRefinement.FeFailure
      (input alphabet baseInput seed)
      (IdealInteractiveExecution.certificate alphabet baseInput strategy
        suffix seed) ∨
    ProductionRefinement.NcFailure
      (input alphabet baseInput seed)
      (IdealInteractiveExecution.certificate alphabet baseInput strategy
        suffix seed)

/-- Boolean monitor for the literal production `FeFailure ∨ NcFailure`. -/
noncomputable def namedFailureEvent
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits : ringDegree * publicRingColumns <= shape.carrierWidth}
    (alphabet : Support K)
    (baseInput : ProductionRefinement.AuthoritativeInput
      (shape := shape) (State := State)
      (publicRingColumns := publicRingColumns)
      (verifierRows := verifierRows)
      (publicFits := publicFits))
    (strategy : IdealInteractiveExecution.Strategy baseInput)
    (suffix : Seed shape ->
      IdealInteractiveExecution.Suffix shape publicRingColumns verifierRows
        publicFits)
    (seed : Seed shape) : Bool :=
  propositionEvent
    (NamedFailure alphabet baseInput strategy suffix seed)

/-- Every monitored root or physical collision constructs the corresponding
actual production failure constructor. -/
theorem algebraicFailure_implies_namedFailure
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits : ringDegree * publicRingColumns <= shape.carrierWidth}
    (alphabet : Support K)
    (baseInput : ProductionRefinement.AuthoritativeInput
      (shape := shape) (State := State)
      (publicRingColumns := publicRingColumns)
      (verifierRows := verifierRows)
      (publicFits := publicFits))
    (strategy : IdealInteractiveExecution.Strategy baseInput)
    (suffix : Seed shape ->
      IdealInteractiveExecution.Suffix shape publicRingColumns verifierRows
        publicFits)
    (seed : Seed shape)
    (failure :
      algebraicFailureEvent alphabet baseInput strategy seed = true) :
    NamedFailure alphabet baseInput strategy suffix seed := by
  rw [algebraicFailureEvent, Bool.or_eq_true] at failure
  rcases failure with mixing | collision
  · rw [mixingEvent, Bool.or_eq_true] at mixing
    rcases mixing with fe | nc
    · apply Or.inl
      refine .sumcheck
        (input alphabet baseInput seed).publicInput_eq_sources
        (.mixingRoot ?_)
      have root :=
        (IdealInteractiveFeSoundness.mixingRootEvent_eq_true_iff
          baseInput.full.profile baseInput.data seed.1.1).mp fe
      simpa [IdealInteractiveFeMixing.coins,
        IdealInteractiveCarrier.challenges] using root
    · have roots :=
        (IdealInteractiveNcMixing.ncMixingRootEvent_eq_true_iff
          baseInput.full.covers baseInput.data
          (ProductionProjection.productionWeights baseInput.full)
          baseInput.full.pending seed.1).mp nc
      apply Or.inr
      rcases roots with lane | block | gamma | residual
      · apply ProductionRefinement.NcFailure.laneSelectorRoot
        simpa [IdealInteractiveNcMixing.coins,
          IdealInteractiveCarrier.challenges] using lane
      · apply ProductionRefinement.NcFailure.blockSelectorRoot
        simpa [IdealInteractiveNcMixing.coins,
          IdealInteractiveCarrier.challenges] using block
      · apply ProductionRefinement.NcFailure.gammaPolynomialRoot
        simpa [IdealInteractiveNcMixing.coins,
          IdealInteractiveCarrier.challenges] using gamma
      · rcases residual with ⟨pending, pendingEq, root⟩
        refine ProductionRefinement.NcFailure.residualWeightRoot
          pending ?_ ?_
        · simpa using pendingEq
        · simpa [IdealInteractiveNcMixing.coins,
            IdealInteractiveCarrier.challenges] using root
  · exact IdealInteractiveExecution.splitCollision_implies_namedFailure
      alphabet baseInput strategy suffix seed
      ((propositionEvent_eq_true_iff _).mp collision)

/-- Conversely, every actual production algebraic failure is one of the
monitored mixing roots or physical collisions. -/
theorem namedFailure_implies_algebraicFailure
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits : ringDegree * publicRingColumns <= shape.carrierWidth}
    (alphabet : Support K)
    (baseInput : ProductionRefinement.AuthoritativeInput
      (shape := shape) (State := State)
      (publicRingColumns := publicRingColumns)
      (verifierRows := verifierRows)
      (publicFits := publicFits))
    (strategy : IdealInteractiveExecution.Strategy baseInput)
    (suffix : Seed shape ->
      IdealInteractiveExecution.Suffix shape publicRingColumns verifierRows
        publicFits)
    (seed : Seed shape)
    (failure : NamedFailure alphabet baseInput strategy suffix seed) :
    algebraicFailureEvent alphabet baseInput strategy seed = true := by
  rcases failure with fe | nc
  · cases fe with
    | sumcheck bound bad =>
        cases bad with
        | mixingRoot root =>
            rw [algebraicFailureEvent, Bool.or_eq_true]
            apply Or.inl
            rw [mixingEvent, Bool.or_eq_true]
            apply Or.inl
            apply
              (IdealInteractiveFeSoundness.mixingRootEvent_eq_true_iff
                baseInput.full.profile baseInput.data seed.1.1).2
            simpa [IdealInteractiveFeMixing.coins,
              IdealInteractiveCarrier.challenges] using root
        | roundCollision round collision =>
            rw [algebraicFailureEvent, Bool.or_eq_true]
            apply Or.inr
            apply (propositionEvent_eq_true_iff _).2
            apply Or.inl
            refine ⟨round, ?_⟩
            rw [ProductionRefinement.Certificate.fePoint_materialize] at collision
            rw [IdealInteractiveExecution.certificate_fe_coordinates] at collision
            have physicalCertificate :
                (Protocol.BlockLane.certificateAtSources
                  (input alphabet baseInput seed).data
                  (IdealInteractiveExecution.certificate alphabet baseInput
                    strategy suffix seed).materialize.piCcs bound).fe =
                  (strategy seed.1).fe.physicalCertificate seed.feWord := by
              have boundEq : bound = rfl := Subsingleton.elim _ _
              rw [boundEq]
              rfl
            rw [physicalCertificate] at collision
            exact collision
  · cases nc with
    | laneSelectorRoot root =>
        rw [algebraicFailureEvent, Bool.or_eq_true]
        apply Or.inl
        rw [mixingEvent, Bool.or_eq_true]
        apply Or.inr
        apply
          (IdealInteractiveNcMixing.ncMixingRootEvent_eq_true_iff
            baseInput.full.covers baseInput.data
            (ProductionProjection.productionWeights baseInput.full)
            baseInput.full.pending seed.1).2
        apply Or.inl
        simpa [IdealInteractiveNcMixing.coins,
          IdealInteractiveCarrier.challenges] using root
    | blockSelectorRoot root =>
        rw [algebraicFailureEvent, Bool.or_eq_true]
        apply Or.inl
        rw [mixingEvent, Bool.or_eq_true]
        apply Or.inr
        apply
          (IdealInteractiveNcMixing.ncMixingRootEvent_eq_true_iff
            baseInput.full.covers baseInput.data
            (ProductionProjection.productionWeights baseInput.full)
            baseInput.full.pending seed.1).2
        apply Or.inr
        apply Or.inl
        simpa [IdealInteractiveNcMixing.coins,
          IdealInteractiveCarrier.challenges] using root
    | gammaPolynomialRoot root =>
        rw [algebraicFailureEvent, Bool.or_eq_true]
        apply Or.inl
        rw [mixingEvent, Bool.or_eq_true]
        apply Or.inr
        apply
          (IdealInteractiveNcMixing.ncMixingRootEvent_eq_true_iff
            baseInput.full.covers baseInput.data
            (ProductionProjection.productionWeights baseInput.full)
            baseInput.full.pending seed.1).2
        apply Or.inr
        apply Or.inr
        apply Or.inl
        simpa [IdealInteractiveNcMixing.coins,
          IdealInteractiveCarrier.challenges] using root
    | residualWeightRoot pending pendingEq root =>
        rw [algebraicFailureEvent, Bool.or_eq_true]
        apply Or.inl
        rw [mixingEvent, Bool.or_eq_true]
        apply Or.inr
        apply
          (IdealInteractiveNcMixing.ncMixingRootEvent_eq_true_iff
            baseInput.full.covers baseInput.data
            (ProductionProjection.productionWeights baseInput.full)
            baseInput.full.pending seed.1).2
        apply Or.inr
        apply Or.inr
        apply Or.inr
        refine ⟨pending, ?_, ?_⟩
        · simpa using pendingEq
        · simpa [IdealInteractiveNcMixing.coins,
            IdealInteractiveCarrier.challenges] using root
    | roundCollision round collision =>
        rw [algebraicFailureEvent, Bool.or_eq_true]
        apply Or.inr
        apply (propositionEvent_eq_true_iff _).2
        apply Or.inr
        refine ⟨round, ?_⟩
        rw [ProductionRefinement.Certificate.ncPoint_materialize] at collision
        rw [IdealInteractiveExecution.certificate_nc_coordinates] at collision
        simpa [IdealInteractiveExecution.certificate] using collision

/-- Exact event transport: the finite monitor accepts precisely the literal
production algebraic failure family for the replayed certificate. -/
theorem algebraicFailureEvent_eq_namedFailureEvent
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits : ringDegree * publicRingColumns <= shape.carrierWidth}
    (alphabet : Support K)
    (baseInput : ProductionRefinement.AuthoritativeInput
      (shape := shape) (State := State)
      (publicRingColumns := publicRingColumns)
      (verifierRows := verifierRows)
      (publicFits := publicFits))
    (strategy : IdealInteractiveExecution.Strategy baseInput)
    (suffix : Seed shape ->
      IdealInteractiveExecution.Suffix shape publicRingColumns verifierRows
        publicFits)
    (seed : Seed shape) :
    algebraicFailureEvent alphabet baseInput strategy seed =
      namedFailureEvent alphabet baseInput strategy suffix seed := by
  apply Bool.eq_iff_iff.mpr
  rw [namedFailureEvent, propositionEvent_eq_true_iff]
  exact ⟨
    algebraicFailure_implies_namedFailure alphabet baseInput strategy suffix
      seed,
    namedFailure_implies_algebraicFailure alphabet baseInput strategy suffix
      seed⟩

/-! ## Probability composition -/

theorem mixingEvent_probability_le
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits : ringDegree * publicRingColumns <= shape.carrierWidth}
    (baseInput : ProductionRefinement.AuthoritativeInput
      (shape := shape) (State := State)
      (publicRingColumns := publicRingColumns)
      (verifierRows := verifierRows)
      (publicFits := publicFits))
    (noZeroDivisors : FiniteRootCounting.NoZeroDivisors ops)
    (alphabet : Support K) :
    ((preSupport (shape := shape) alphabet).uniform).probabilityBool
        (mixingEvent baseInput) <=
      feMixingBudget shape alphabet.cardinality +
        ncMixingBudget shape alphabet.cardinality := by
  let experiment := (preSupport (shape := shape) alphabet).uniform
  let feEvent : PreSeed shape -> Bool := fun seed =>
    IdealInteractiveFeSoundness.mixingRootEvent
      baseInput.full.profile baseInput.data seed.1
  let ncEvent : PreSeed shape -> Bool :=
    IdealInteractiveNcMixing.ncMixingRootEvent
      baseInput.full.covers baseInput.data
      (ProductionProjection.productionWeights baseInput.full)
      baseInput.full.pending
  have feBound :
      experiment.probabilityBool feEvent <=
        feMixingBudget shape alphabet.cardinality := by
    simpa [experiment, feEvent, feMixingBudget] using
      IdealInteractiveFeSoundness.mixingRoot_pre_probability_le
        baseInput.full.profile baseInput.data noZeroDivisors alphabet
  have ncBound :
      experiment.probabilityBool ncEvent <=
        ncMixingBudget shape alphabet.cardinality := by
    simpa [experiment, ncEvent, ncMixingBudget] using
      IdealInteractiveNcMixing.ncMixingRoot_probability_le
        baseInput.full.covers baseInput.data
        (ProductionProjection.productionWeights baseInput.full)
        baseInput.full.pending noZeroDivisors alphabet
  simpa [experiment, feEvent, ncEvent, mixingEvent] using
    probabilityBool_or_le_of_bounds experiment feEvent ncEvent
      (feMixingBudget shape alphabet.cardinality)
      (ncMixingBudget shape alphabet.cardinality)
      feBound ncBound

/-- An anchor carries the fixed pre-SumCheck seed while the existing
collision theorem quantifies over the later FE/NC words. -/
def anchorSeed
    {shape : SemanticShape}
    (seed : PreSeed shape) : Seed shape :=
  (seed, (fun _ => K.zero, fun _ => K.zero))

theorem splitCollision_probability_le
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits : ringDegree * publicRingColumns <= shape.carrierWidth}
    (alphabet : Support K)
    (baseInput : ProductionRefinement.AuthoritativeInput
      (shape := shape) (State := State)
      (publicRingColumns := publicRingColumns)
      (verifierRows := verifierRows)
      (publicFits := publicFits))
    (strategy : IdealInteractiveExecution.Strategy baseInput)
    (noZeroDivisors : FiniteRootCounting.NoZeroDivisors ops) :
    ((support (shape := shape) alphabet).uniform).probabilityBool
        (splitCollisionEvent alphabet baseInput strategy) <=
      splitCollisionBudget baseInput alphabet.cardinality := by
  unfold support
  refine product_probabilityBool_le_of_components
    (preSupport (shape := shape) alphabet)
    (sumCheckSupport (shape := shape) alphabet)
    (fun preSeed sumCheckSeed =>
      splitCollisionEvent alphabet baseInput strategy
        (preSeed, sumCheckSeed))
    (splitCollisionBudget baseInput alphabet.cardinality) ?_
  intro preSeed _preSeedMember
  let anchoredInput :=
    input alphabet baseInput (anchorSeed preSeed)
  let experiment :=
    (sumCheckSupport (shape := shape) alphabet).uniform
  have eventEq :
      (fun sumCheckSeed =>
        propositionEvent
          (IdealInteractiveExecution.SplitCollision alphabet baseInput
            strategy (preSeed, sumCheckSeed))) =
      (fun sumCheckSeed =>
        propositionEvent
          (CausalSoundness.SplitCollision anchoredInput
            (strategy preSeed) sumCheckSeed)) := by
    funext sumCheckSeed
    rfl
  change experiment.probabilityBool
      (fun sumCheckSeed =>
        propositionEvent
          (IdealInteractiveExecution.SplitCollision alphabet baseInput
            strategy (preSeed, sumCheckSeed))) <=
    splitCollisionBudget baseInput alphabet.cardinality
  rw [eventEq]
  have propositionProbability :
      experiment.probabilityBool
          (fun sumCheckSeed =>
            propositionEvent
              (CausalSoundness.SplitCollision anchoredInput
                (strategy preSeed) sumCheckSeed)) =
        experiment.probability
          (CausalSoundness.SplitCollision anchoredInput
            (strategy preSeed)) := by
    rw [← experiment.probability_bool_event]
    congr 1
    funext sumCheckSeed
    apply propext
    exact propositionEvent_eq_true_iff _
  rw [propositionProbability]
  simpa [experiment, sumCheckSupport, anchoredInput, anchorSeed,
    splitCollisionBudget] using
    CausalSoundness.splitCollision_probability_le anchoredInput
      (strategy preSeed) noZeroDivisors alphabet rfl

/-- Final concrete ideal-interactive algebraic bound. No
`SumCheckSoundnessContract` or generic mixing contract is a premise. -/
theorem algebraicFailure_probability_le
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits : ringDegree * publicRingColumns <= shape.carrierWidth}
    (alphabet : Support K)
    (baseInput : ProductionRefinement.AuthoritativeInput
      (shape := shape) (State := State)
      (publicRingColumns := publicRingColumns)
      (verifierRows := verifierRows)
      (publicFits := publicFits))
    (strategy : IdealInteractiveExecution.Strategy baseInput)
    (noZeroDivisors : FiniteRootCounting.NoZeroDivisors ops) :
    ((support (shape := shape) alphabet).uniform).probabilityBool
        (algebraicFailureEvent alphabet baseInput strategy) <=
      totalBudget baseInput alphabet.cardinality := by
  let experiment := (support (shape := shape) alphabet).uniform
  let fullMixing : Seed shape -> Bool := fun seed =>
    mixingEvent baseInput seed.1
  let collision : Seed shape -> Bool :=
    splitCollisionEvent alphabet baseInput strategy
  have mixingBound :
      experiment.probabilityBool fullMixing <=
        feMixingBudget shape alphabet.cardinality +
          ncMixingBudget shape alphabet.cardinality := by
    rw [show
      experiment.probabilityBool fullMixing =
        ((preSupport (shape := shape) alphabet).uniform).probabilityBool
          (mixingEvent baseInput) by
      exact Support.product_uniform_probabilityBool_first
        (preSupport (shape := shape) alphabet)
        (sumCheckSupport (shape := shape) alphabet)
        (mixingEvent baseInput)]
    exact mixingEvent_probability_le baseInput noZeroDivisors alphabet
  have collisionBound :
      experiment.probabilityBool collision <=
        splitCollisionBudget baseInput alphabet.cardinality := by
    simpa [experiment, collision] using
      splitCollision_probability_le alphabet baseInput strategy
        noZeroDivisors
  simpa [experiment, fullMixing, collision, algebraicFailureEvent,
    totalBudget] using
    probabilityBool_or_le_of_bounds experiment fullMixing collision
      (feMixingBudget shape alphabet.cardinality +
        ncMixingBudget shape alphabet.cardinality)
      (splitCollisionBudget baseInput alphabet.cardinality)
      mixingBound collisionBound

/-- Headline production theorem: the probability of the actual
`FeFailure ∨ NcFailure` family for the replayed physical certificate is
bounded directly by finite root counting and the existing causal SumCheck
theorem. Neither `SumCheckSoundnessContract` nor a generic mixing contract is
a premise. -/
theorem namedFailure_probability_le
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits : ringDegree * publicRingColumns <= shape.carrierWidth}
    (alphabet : Support K)
    (baseInput : ProductionRefinement.AuthoritativeInput
      (shape := shape) (State := State)
      (publicRingColumns := publicRingColumns)
      (verifierRows := verifierRows)
      (publicFits := publicFits))
    (strategy : IdealInteractiveExecution.Strategy baseInput)
    (suffix : Seed shape ->
      IdealInteractiveExecution.Suffix shape publicRingColumns verifierRows
        publicFits)
    (noZeroDivisors : FiniteRootCounting.NoZeroDivisors ops) :
    ((support (shape := shape) alphabet).uniform).probabilityBool
        (namedFailureEvent alphabet baseInput strategy suffix) <=
      totalBudget baseInput alphabet.cardinality := by
  have eventEq :
      namedFailureEvent alphabet baseInput strategy suffix =
        algebraicFailureEvent alphabet baseInput strategy := by
    funext seed
    exact
      (algebraicFailureEvent_eq_namedFailureEvent alphabet baseInput strategy
        suffix seed).symm
  rw [eventEq]
  exact algebraicFailure_probability_le alphabet baseInput strategy
    noZeroDivisors

/-- Concrete algebra corollary. The only field premises are the repository's
exact Goldilocks Euclid and `u² = 7` nonresidue facts; no abstract
no-zero-divisors argument remains. -/
theorem namedFailure_probability_le_of_productionField
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits : ringDegree * publicRingColumns <= shape.carrierWidth}
    (alphabet : Support K)
    (baseInput : ProductionRefinement.AuthoritativeInput
      (shape := shape) (State := State)
      (publicRingColumns := publicRingColumns)
      (verifierRows := verifierRows)
      (publicFits := publicFits))
    (strategy : IdealInteractiveExecution.Strategy baseInput)
    (suffix : Seed shape ->
      IdealInteractiveExecution.Suffix shape publicRingColumns verifierRows
        publicFits)
    (euclid : NormRange.GoldilocksModulusEuclid)
    (sevenNonresidue : ConcreteCarrier.SevenProjectiveNonresidue) :
    ((support (shape := shape) alphabet).uniform).probabilityBool
        (namedFailureEvent alphabet baseInput strategy suffix) <=
      totalBudget baseInput alphabet.cardinality :=
  namedFailure_probability_le alphabet baseInput strategy suffix
    (ProductionMixingBoundary.productionExtensionNoZeroDivisors
      euclid sevenNonresidue)

end Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionMixingBoundary.IdealInteractiveSoundness
