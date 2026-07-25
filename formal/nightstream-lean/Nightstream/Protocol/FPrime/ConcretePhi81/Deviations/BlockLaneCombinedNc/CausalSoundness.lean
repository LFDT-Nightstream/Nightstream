import Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionRefinement
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.BlockLane.CausalSoundness

/-!
Concrete ideal-interactive SumCheck bound for production Split-NC.

Assurance tier: model-level registered-deviation refinement.

Owns: prefix-local quartic representability of production's exact
ordinary-or-delayed raw NC polynomial, exact physical FE and block/lane NC
collision events, transport to the two causal monitors, challenge-support
cardinality alignment, and the explicit FE-plus-NC probability bound.

Does not own: alpha/gamma mixing bounds, Fiat--Shamir, Poseidon2, random-oracle
programming, Goldilocks primality, Rust/R1CS, artifacts, costs, or rows.

Emits constraints: none.

`SplitCollision` uses the repository's existing `FixedPhase.BadChallenge`
predicate on the exact mixed-width FE certificate and exact five-slot NC
certificate. `SumCheckSoundnessContract` is neither imported as authority nor
retained as a premise.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `pi_ccs.production.nc.degree` | represent the exact ordinary-or-delayed NC polynomial at quartic degree | derived | `rawRoundRepresentable` |
| `pi_ccs.production.nc.certificate` | project the physical block/lane certificate to the exact fixed-phase certificate | computed | `ncCertificate`, `ncCertificate_toSumCheck` |
| `pi_ccs.production.collision` | retain the exact FE and NC repository collision predicates | checked | `FeRoundCollision`, `NcRoundCollision`, `SplitCollision` |
| `pi_ccs.production.collision.transport` | map each physical collision into its causal detector | derived | `splitCollision_implies_detects` |
| `pi_ccs.production.collision.probability` | prove the explicit FE-plus-NC ideal-interactive union bound | derived | `splitCollision_probability_le` |
-/

set_option autoImplicit false

namespace Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.CausalSoundness

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.BlockLane
open Nightstream.SuperNeo.InteractiveReduction.FiniteUniform
open Nightstream.SuperNeo.SumCheck.Finite

namespace Generic

export Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.BlockLane.CausalSoundness
  (SplitStrategy feRoundCount feDetects ncDetects
    fe_roundCollision_implies_detects nc_roundCollision_implies_detects
    split_detects_probability_le)

end Generic

private abbrev ops := ConcreteCarrier.extensionOps

universe uState

variable
  {shape : SemanticShape}
  {State : Type uState}
  {publicRingColumns verifierRows : Nat}
  {publicFits : ringDegree * publicRingColumns <= shape.carrierWidth}

/-- Exact production block-plus-lane NC round count. -/
abbrev ncRoundCount : Nat :=
  Transcript.Nc.BlockLane.roundCount PiCcsDomains.production.nc

/-- The exact production raw polynomial has one five-slot representation at
every prior NC prefix, in both the base and delayed branches. -/
theorem rawRoundRepresentable
    (input : ProductionRefinement.AuthoritativeInput
      (shape := shape) (State := State)
      (publicRingColumns := publicRingColumns)
      (verifierRows := verifierRows)
      (publicFits := publicFits)) :
    FixedPhase.Sequential.RoundRepresentable ops.toOps
      (ProductionPiCcs.rawPolynomial input.full input.data)
      Polynomial.Nc.Degree.ncSumcheckDegreeBound ncRoundCount := by
  cases pendingEq : input.full.pending with
  | none =>
      simpa [ProductionPiCcs.rawPolynomial, pendingEq, ncRoundCount] using
        (Nc.BlockLane.HonestProver.roundRepresentable
          input.full.covers input.data input.full.ncCoins)
  | some pending =>
      simpa [ProductionPiCcs.rawPolynomial, pendingEq, ncRoundCount] using
        (Nc.BlockLane.DelayedHonestProver.roundRepresentable
          input.full.covers input.data input.full.ncCoins
          (ProductionProjection.productionWeights input.full)
          input.full.producerBeta input.full.batchWeight pending.oldBlock)

/-- Ideal-interactive prover strategy at the exact production arities. -/
abbrev Strategy
    (input : ProductionRefinement.AuthoritativeInput
      (shape := shape) (State := State)
      (publicRingColumns := publicRingColumns)
      (verifierRows := verifierRows)
      (publicFits := publicFits)) :=
  Generic.SplitStrategy (PublicInput.ofSources input.data)
    PiCcsDomains.production.fe ncRoundCount

/-- Exact block/lane message-only NC certificate generated causally from a
complete ideal challenge word. -/
def ncCertificate
    {input : ProductionRefinement.AuthoritativeInput
      (shape := shape) (State := State)
      (publicRingColumns := publicRingColumns)
      (verifierRows := verifierRows)
      (publicFits := publicFits)}
    (strategy : Strategy input)
    (feWord :
      Fin (Generic.feRoundCount shape PiCcsDomains.production.fe) -> K)
    (ncWord : Fin ncRoundCount -> K) :
    Transcript.Nc.BlockLane.Certificate PiCcsDomains.production.nc where
  rounds := fun round =>
    (strategy.nc feWord).message round.val round.isLt
      (SumCheck.CausalFixedPhase.prefixAt ncWord round.val
        (Nat.le_of_lt round.isLt))

/-- The physical block/lane projection is the exact generic five-slot
certificate used by root counting. -/
@[simp] theorem ncCertificate_toSumCheck
    {input : ProductionRefinement.AuthoritativeInput
      (shape := shape) (State := State)
      (publicRingColumns := publicRingColumns)
      (verifierRows := verifierRows)
      (publicFits := publicFits)}
    (strategy : Strategy input)
    (feWord :
      Fin (Generic.feRoundCount shape PiCcsDomains.production.fe) -> K)
    (ncWord : Fin ncRoundCount -> K) :
    (ncCertificate strategy feWord ncWord).toSumCheck =
      (strategy.nc feWord).certificate ncWord := by
  rfl

/-- Exact repository FE collision event for the production mixed-width
certificate. -/
def FeRoundCollision
    (input : ProductionRefinement.AuthoritativeInput
      (shape := shape) (State := State)
      (publicRingColumns := publicRingColumns)
      (verifierRows := verifierRows)
      (publicFits := publicFits))
    (strategy : Strategy input)
    (feWord :
      Fin (Generic.feRoundCount shape PiCcsDomains.production.fe) -> K) :
    Prop :=
  ∃ round,
    FixedPhase.BadChallenge ops.toOps
      (Polynomial.Fe.InitialSum.sumcheckPolynomial input.full.profile
        input.data input.full.feCoins)
      (SumCheck.Fe.Drow (PublicInput.ofSources input.data))
      input.full.challengeSetSize
      (Polynomial.Fe.initial input.full.profile
        (PublicInput.ofSources input.data) input.full.feCoins)
      (List.ofFn feWord)
      {
        rounds := (strategy.fe.physicalCertificate feWord).uniformRounds
      }
      round

/-- Exact repository NC collision event for production's
ordinary-or-delayed raw polynomial and physical block/lane certificate. -/
def NcRoundCollision
    (input : ProductionRefinement.AuthoritativeInput
      (shape := shape) (State := State)
      (publicRingColumns := publicRingColumns)
      (verifierRows := verifierRows)
      (publicFits := publicFits))
    (strategy : Strategy input)
    (seed :
      (Fin (Generic.feRoundCount shape PiCcsDomains.production.fe) -> K) ×
        (Fin ncRoundCount -> K)) :
    Prop :=
  ∃ round,
    FixedPhase.BadChallenge ops.toOps
      (ProductionPiCcs.rawPolynomial input.full input.data)
      Polynomial.Nc.Degree.ncSumcheckDegreeBound
      input.full.challengeSetSize
      (ProductionPiCcs.rawInitial input.full)
      (List.ofFn seed.2)
      (ncCertificate strategy seed.1 seed.2).toSumCheck
      round

/-- The two SumCheck collision events only. Alpha/gamma mixing events remain
separate named events in the deterministic production reduction. -/
def SplitCollision
    (input : ProductionRefinement.AuthoritativeInput
      (shape := shape) (State := State)
      (publicRingColumns := publicRingColumns)
      (verifierRows := verifierRows)
      (publicFits := publicFits))
    (strategy : Strategy input)
    (seed :
      (Fin (Generic.feRoundCount shape PiCcsDomains.production.fe) -> K) ×
        (Fin ncRoundCount -> K)) :
    Prop :=
  FeRoundCollision input strategy seed.1 ∨
    NcRoundCollision input strategy seed

/-- Exact event transport from both physical repository collision predicates
to the Boolean union monitored by the causal root-counting theorem. -/
theorem splitCollision_implies_detects
    (input : ProductionRefinement.AuthoritativeInput
      (shape := shape) (State := State)
      (publicRingColumns := publicRingColumns)
      (verifierRows := verifierRows)
      (publicFits := publicFits))
    (strategy : Strategy input)
    (seed :
      (Fin (Generic.feRoundCount shape PiCcsDomains.production.fe) -> K) ×
        (Fin ncRoundCount -> K))
    (collision : SplitCollision input strategy seed) :
    (Generic.feDetects input.full.profile input.data input.full.feCoins
          input.full.challengeSetSize strategy.fe seed.1 ||
        Generic.ncDetects
          (ProductionPiCcs.rawPolynomial input.full input.data)
          (ProductionPiCcs.rawInitial input.full)
          input.full.challengeSetSize
          (rawRoundRepresentable input) strategy seed) = true := by
  rcases collision with fe | nc
  · rw [Bool.or_eq_true]
    apply Or.inl
    rcases fe with ⟨round, bad⟩
    exact Generic.fe_roundCollision_implies_detects
      input.full.profile input.data input.full.feCoins
      input.full.challengeSetSize strategy.fe seed.1 round bad
  · rw [Bool.or_eq_true]
    apply Or.inr
    rcases nc with ⟨round, bad⟩
    apply Generic.nc_roundCollision_implies_detects
      (ProductionPiCcs.rawPolynomial input.full input.data)
      (ProductionPiCcs.rawInitial input.full)
      input.full.challengeSetSize
      (rawRoundRepresentable input)
      (strategy.nc seed.1) seed.2 round
    simpa only [ncCertificate_toSumCheck] using bad

/-- Concrete production Split-NC SumCheck theorem. The existing
`SumCheckSoundnessContract` is not a premise: finite root counting,
successive-coordinate independence, exact event transport, and the explicit
FE/NC union bound construct the result directly.

The support equality binds the denominator to the actual sampled alphabet.
Fiat--Shamir remains a separate non-interactivity obligation. -/
theorem splitCollision_probability_le
    (input : ProductionRefinement.AuthoritativeInput
      (shape := shape) (State := State)
      (publicRingColumns := publicRingColumns)
      (verifierRows := verifierRows)
      (publicFits := publicFits))
    (strategy : Strategy input)
    (noZeroDivisors :
      FiniteRootCounting.NoZeroDivisors ops)
    (alphabet : Support K)
    (challengeSetSize_eq :
      input.full.challengeSetSize = alphabet.cardinality) :
    let feWords :=
      Nightstream.SuperNeo.InteractiveReduction.CoordinateForking.FiniteWords.Support.challengeVectors
        alphabet
        (Generic.feRoundCount shape PiCcsDomains.production.fe)
    let ncWords :=
      Nightstream.SuperNeo.InteractiveReduction.CoordinateForking.FiniteWords.Support.challengeVectors
        alphabet ncRoundCount
    ((feWords.product ncWords).uniform).probability
        (SplitCollision input strategy) <=
      ratio
          (Generic.feRoundCount shape PiCcsDomains.production.fe *
            SumCheck.Fe.Drow (PublicInput.ofSources input.data))
          input.full.challengeSetSize +
        ratio
          (ncRoundCount * Polynomial.Nc.Degree.ncSumcheckDegreeBound)
          input.full.challengeSetSize := by
  let feWords :=
    Nightstream.SuperNeo.InteractiveReduction.CoordinateForking.FiniteWords.Support.challengeVectors
      alphabet (Generic.feRoundCount shape PiCcsDomains.production.fe)
  let ncWords :=
    Nightstream.SuperNeo.InteractiveReduction.CoordinateForking.FiniteWords.Support.challengeVectors
      alphabet ncRoundCount
  let experiment := (feWords.product ncWords).uniform
  let detector :
      ((Fin (Generic.feRoundCount shape PiCcsDomains.production.fe) -> K) ×
        (Fin ncRoundCount -> K)) -> Bool :=
    fun seed =>
      Generic.feDetects input.full.profile input.data input.full.feCoins
          input.full.challengeSetSize strategy.fe seed.1 ||
        Generic.ncDetects
          (ProductionPiCcs.rawPolynomial input.full input.data)
          (ProductionPiCcs.rawInitial input.full)
          input.full.challengeSetSize
          (rawRoundRepresentable input) strategy seed
  have eventTransport :
      ∀ seed, SplitCollision input strategy seed ->
        detector seed = true := by
    intro seed collision
    exact splitCollision_implies_detects input strategy seed collision
  have monitorBound :
      experiment.probabilityBool detector <=
        ratio
            (Generic.feRoundCount shape PiCcsDomains.production.fe *
              SumCheck.Fe.Drow (PublicInput.ofSources input.data))
            alphabet.cardinality +
          ratio
            (ncRoundCount * Polynomial.Nc.Degree.ncSumcheckDegreeBound)
            alphabet.cardinality := by
    exact Generic.split_detects_probability_le
      input.full.profile input.data input.full.feCoins
      (ProductionPiCcs.rawPolynomial input.full input.data)
      (ProductionPiCcs.rawInitial input.full)
      input.full.challengeSetSize
      (rawRoundRepresentable input) strategy noZeroDivisors alphabet
  calc
    experiment.probability (SplitCollision input strategy) <=
        experiment.probability (fun seed => detector seed = true) :=
      experiment.probability_mono eventTransport
    _ = experiment.probabilityBool detector :=
      experiment.probability_bool_event detector
    _ <=
        ratio
            (Generic.feRoundCount shape PiCcsDomains.production.fe *
              SumCheck.Fe.Drow (PublicInput.ofSources input.data))
            alphabet.cardinality +
          ratio
            (ncRoundCount * Polynomial.Nc.Degree.ncSumcheckDegreeBound)
            alphabet.cardinality :=
      monitorBound
    _ =
        ratio
            (Generic.feRoundCount shape PiCcsDomains.production.fe *
              SumCheck.Fe.Drow (PublicInput.ofSources input.data))
            input.full.challengeSetSize +
          ratio
            (ncRoundCount * Polynomial.Nc.Degree.ncSumcheckDegreeBound)
            input.full.challengeSetSize := by
      rw [challengeSetSize_eq]

end Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.CausalSoundness
