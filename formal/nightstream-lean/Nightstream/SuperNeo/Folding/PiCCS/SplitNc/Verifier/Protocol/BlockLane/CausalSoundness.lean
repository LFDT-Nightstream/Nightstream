import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Nc.BlockLane.DelayedHonestProver
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.CausalFixedPhase
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Fe.Semantics
import Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.Reindex
import Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.Rejection

/-!
Causal finite-uniform SumCheck soundness for the production-shaped FE/NC
split.

Assurance tier: model-level registered-deviation refinement.

Owns: the physical FE row/lane prover interface, causal conversion to the
uniform proof view, prefix-local FE degree construction, the five-slot NC
causal interface, exact transport of both repository round-collision events,
and their separate finite root-counting bounds.

Does not own: alpha/gamma mixing bounds, Fiat--Shamir, Poseidon2, a random
oracle, production field primality, Rust/R1CS, artifacts, costs, or rows.

Emits constraints: none.

The sampled objects here are ideal interactive challenge words. FE messages
see only the FE prefix. NC messages may be selected after the completed FE
word is fixed, but see only the prior NC prefix. A later Fiat--Shamir theorem
must establish the corresponding oracle experiment; deterministic transcript
replay is not treated as a probability argument.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `pi_ccs.split.fe` | preserve physical row/lane messages and fix them from prior FE challenges | checked | `FeStrategy`, `fe_uniformRounds_eq_generated` |
| `pi_ccs.split.fe.soundness` | transport FE round collisions and apply the exact degree bound | derived | `fe_roundCollision_implies_detects`, `fe_detects_probability_le` |
| `pi_ccs.split.nc.soundness` | fix five-slot NC messages from prior NC challenges and transport collisions | checked/derived | `NcStrategy`, `nc_roundCollision_implies_detects` |
| `pi_ccs.split.union` | sample a fresh NC word after FE while allowing NC to depend on completed FE | derived | `split_detects_probability_le` |
-/

set_option autoImplicit false

namespace Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.BlockLane.CausalSoundness

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.DegreeSupport
open Nightstream.SuperNeo.InteractiveReduction.FiniteUniform
open Nightstream.SuperNeo.SumCheck.Finite

private abbrev ops := ConcreteCarrier.extensionOps
private abbrev laws := ConcreteCarrier.extensionLaws

private theorem fixedCertificate_ext
    {degree : Nat}
    (left right : FixedPhase.Certificate K degree)
    (rounds : left.rounds = right.rounds) :
    left = right := by
  cases left
  cases right
  simp_all

/-- Exact number of challenges in the mixed-width FE phase. -/
abbrev feRoundCount
    (shape : SemanticShape)
    (domain : FlatNcDomain) : Nat :=
  shape.rowVariables + domain.laneVariables

/-- A production-shaped FE prover strategy. Row messages inhabit the
syntax-derived row width. Lane messages inhabit exactly the independent
quadratic width. Both callbacks receive only the prior FE prefix. -/
structure FeStrategy
    {shape : SemanticShape}
    (input : PublicInput shape)
    (domain : FlatNcDomain) where
  rowMessage :
    forall round : Nat,
      round < feRoundCount shape domain ->
      round < shape.rowVariables ->
      SumCheck.CausalFixedPhase.Prefix round ->
      SumCheck.Fe.RowMessage input
  laneMessage :
    forall round : Nat,
      round < feRoundCount shape domain ->
      shape.rowVariables <= round ->
      SumCheck.CausalFixedPhase.Prefix round ->
      SumCheck.Fe.LaneMessage

namespace FeStrategy

/-- Semantic uniform-width view of one causal physical FE strategy. The lane
branch appends only verifier-known high zeros. -/
def claimed
    {shape : SemanticShape}
    {input : PublicInput shape}
    {domain : FlatNcDomain}
    (strategy : FeStrategy input domain)
    (round : Nat)
    (within : round < feRoundCount shape domain)
    (prior : SumCheck.CausalFixedPhase.Prefix round) :
    FixedPolynomial K (SumCheck.Fe.Drow input) :=
  if rowPhase : round < shape.rowVariables then
    strategy.rowMessage round within rowPhase prior
  else
    SumCheck.Fe.laneToUniform input
      (strategy.laneMessage round within (Nat.le_of_not_gt rowPhase) prior)

/-- Physical message-only FE certificate induced by one complete ideal
challenge word. No challenge or transcript state is stored in the
certificate. -/
def physicalCertificate
    {shape : SemanticShape}
    {input : PublicInput shape}
    {domain : FlatNcDomain}
    (strategy : FeStrategy input domain)
    (word : Fin (feRoundCount shape domain) -> K) :
    SumCheck.Fe.Certificate input domain where
  rowRounds := fun round =>
    strategy.rowMessage round.val (by
        simp only [feRoundCount]
        omega)
      round.isLt
      (SumCheck.CausalFixedPhase.prefixAt word round.val (by
        simp only [feRoundCount]
        omega))
  laneRounds := fun round =>
    strategy.laneMessage (shape.rowVariables + round.val)
      (by
        simp only [feRoundCount]
        omega)
      (by omega)
      (SumCheck.CausalFixedPhase.prefixAt word
        (shape.rowVariables + round.val) (by
          simp only [feRoundCount]
          omega))

end FeStrategy

/-- The independent row and lane degree theorems combine into one
prefix-local uniform-degree theorem. Physical lane messages remain quadratic;
only the semantic proof view is widened to the row ceiling. -/
theorem feRoundRepresentable
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (profile : Polynomial.Fe.SupportedProfile shape domain)
    (data : Data shape)
    (coins : Polynomial.Fe.Coins shape domain) :
    FixedPhase.Sequential.RoundRepresentable ops.toOps
      (Polynomial.Fe.InitialSum.sumcheckPolynomial profile data coins)
      (SumCheck.Fe.Drow (PublicInput.ofSources data))
      (feRoundCount shape domain) := by
  intro fixed remaining length
  by_cases rowPhase : fixed.length < shape.rowVariables
  · exact Polynomial.Fe.Degree.expectedRowRound_bounded
      profile data coins fixed remaining rowPhase length
  · have lanePhase : shape.rowVariables <= fixed.length :=
      Nat.le_of_not_gt rowPhase
    rcases Polynomial.Fe.Degree.expectedLaneRound_quadratic
        profile data coins fixed remaining lanePhase length with
      ⟨polynomial, represents⟩
    refine ⟨SumCheck.Fe.laneToUniform
      (PublicInput.ofSources data) polynomial, ?_⟩
    intro point
    rw [SumCheck.Fe.lane_evaluate_uniform, represents point]

/-- Causal uniform-degree generator for the exact physical FE strategy. -/
noncomputable def feGenerator
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (profile : Polynomial.Fe.SupportedProfile shape domain)
    (data : Data shape)
    (coins : Polynomial.Fe.Coins shape domain)
    (challengeSetSize : Nat)
    (strategy : FeStrategy (PublicInput.ofSources data) domain) :
    SumCheck.CausalFixedPhase.Generator
      (SumCheck.Fe.Drow (PublicInput.ofSources data))
      (feRoundCount shape domain) :=
  SumCheck.CausalFixedPhase.Generator.ofRoundRepresentable
    (Polynomial.Fe.InitialSum.sumcheckPolynomial profile data coins)
    (Polynomial.Fe.initial profile (PublicInput.ofSources data) coins)
    challengeSetSize strategy.claimed
    (feRoundRepresentable profile data coins)

/-- The generic generated certificate is exactly the physical FE
certificate's semantic uniform view. This is the bridge that prevents the
root-counting proof from silently replacing the production mixed-width
messages. -/
theorem fe_uniformRounds_eq_generated
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (profile : Polynomial.Fe.SupportedProfile shape domain)
    (data : Data shape)
    (coins : Polynomial.Fe.Coins shape domain)
    (challengeSetSize : Nat)
    (strategy : FeStrategy (PublicInput.ofSources data) domain)
    (word : Fin (feRoundCount shape domain) -> K) :
    (strategy.physicalCertificate word).uniformRounds =
      (SumCheck.CausalFixedPhase.certificate
        (feGenerator profile data coins challengeSetSize strategy) word
        ).rounds := by
  rw [SumCheck.CausalFixedPhase.certificate]
  change
    List.ofFn (strategy.physicalCertificate word).rowRounds ++
        (List.ofFn (strategy.physicalCertificate word).laneRounds).map
          (SumCheck.Fe.laneToUniform (PublicInput.ofSources data)) =
      List.ofFn fun round =>
        strategy.claimed round.val round.isLt
          (SumCheck.CausalFixedPhase.prefixAt word round.val
            (Nat.le_of_lt round.isLt))
  rw [List.ofFn_add]
  congr 1
  · apply congrArg List.ofFn
    funext round
    simp [FeStrategy.physicalCertificate, FeStrategy.claimed, round.isLt]
  · rw [List.map_ofFn]
    apply congrArg List.ofFn
    funext round
    have notRow :
        ¬shape.rowVariables + round.val < shape.rowVariables := by
      omega
    simp [FeStrategy.physicalCertificate, FeStrategy.claimed, notRow]

/-- Exact transport of the repository FE round-collision constructor to the
causal monitor. The mixing-root constructor is intentionally outside this
SumCheck theorem. -/
theorem fe_roundCollision_implies_detects
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (profile : Polynomial.Fe.SupportedProfile shape domain)
    (data : Data shape)
    (coins : Polynomial.Fe.Coins shape domain)
    (challengeSetSize : Nat)
    (strategy : FeStrategy (PublicInput.ofSources data) domain)
    (word : Fin (feRoundCount shape domain) -> K)
    (round : Nightstream.SuperNeo.SumCheck.Round K K)
    (collision :
      FixedPhase.BadChallenge ops.toOps
        (Polynomial.Fe.InitialSum.sumcheckPolynomial profile data coins)
        (SumCheck.Fe.Drow (PublicInput.ofSources data))
        challengeSetSize
        (Polynomial.Fe.initial profile (PublicInput.ofSources data) coins)
        (List.ofFn word)
        {
          rounds :=
            (strategy.physicalCertificate word).uniformRounds
        }
        round) :
    PaperJoint.CausalSumCheckBound.detects ops
      (SumCheck.CausalFixedPhase.process
        (feGenerator profile data coins challengeSetSize strategy)) word =
        true := by
  apply SumCheck.CausalFixedPhase.badChallenge_implies_detects
    (feGenerator profile data coins challengeSetSize strategy) word
  change
    ∃ candidate,
      FixedPhase.BadChallenge ops.toOps
        (Polynomial.Fe.InitialSum.sumcheckPolynomial profile data coins)
        (SumCheck.Fe.Drow (PublicInput.ofSources data))
        challengeSetSize
        (Polynomial.Fe.initial profile (PublicInput.ofSources data) coins)
        (List.ofFn word)
        (SumCheck.CausalFixedPhase.certificate
          (feGenerator profile data coins challengeSetSize strategy) word)
        candidate
  refine ⟨round, ?_⟩
  let generic :=
    SumCheck.CausalFixedPhase.certificate
      (feGenerator profile data coins challengeSetSize strategy) word
  have roundsEqual :
      (strategy.physicalCertificate word).uniformRounds = generic.rounds := by
    exact fe_uniformRounds_eq_generated profile data coins challengeSetSize
      strategy word
  have certificateEqual :
      ({
        rounds := (strategy.physicalCertificate word).uniformRounds
      } : FixedPhase.Certificate K
        (SumCheck.Fe.Drow (PublicInput.ofSources data))) = generic := by
    exact fixedCertificate_ext _ _ roundsEqual
  change
    FixedPhase.BadChallenge ops.toOps
      (Polynomial.Fe.InitialSum.sumcheckPolynomial profile data coins)
      (SumCheck.Fe.Drow (PublicInput.ofSources data))
      challengeSetSize
      (Polynomial.Fe.initial profile (PublicInput.ofSources data) coins)
      (List.ofFn word) generic round
  rw [← certificateEqual]
  exact collision

/-- Finite root counting and successive-coordinate independence bound every
causal FE round collision. -/
theorem fe_detects_probability_le
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (profile : Polynomial.Fe.SupportedProfile shape domain)
    (data : Data shape)
    (coins : Polynomial.Fe.Coins shape domain)
    (challengeSetSize : Nat)
    (strategy : FeStrategy (PublicInput.ofSources data) domain)
    (noZeroDivisors :
      PaperJoint.FiniteRootCounting.NoZeroDivisors ops)
    (alphabet : Support K) :
    ((Nightstream.SuperNeo.InteractiveReduction.CoordinateForking.FiniteWords.Support.challengeVectors
        alphabet (feRoundCount shape domain)).uniform).probabilityBool
        (PaperJoint.CausalSumCheckBound.detects ops
          (SumCheck.CausalFixedPhase.process
            (feGenerator profile data coins challengeSetSize strategy))) <=
      ratio
        (feRoundCount shape domain *
          SumCheck.Fe.Drow (PublicInput.ofSources data))
        alphabet.cardinality := by
  exact SumCheck.CausalFixedPhase.detects_probability_le
    (feGenerator profile data coins challengeSetSize strategy)
    noZeroDivisors alphabet

/-- A five-slot NC strategy selected after all prior phases are fixed. The
callback sees only the prior NC challenge prefix. -/
structure NcStrategy (rounds : Nat) where
  message :
    forall round : Nat, round < rounds ->
      SumCheck.CausalFixedPhase.Prefix round ->
      SumCheck.Nc.RoundMessage

namespace NcStrategy

/-- Fixed-phase certificate induced by one complete ideal NC word. -/
def certificate
    {rounds : Nat}
    (strategy : NcStrategy rounds)
    (word : Fin rounds -> K) :
    FixedPhase.Certificate K Polynomial.Nc.Degree.ncSumcheckDegreeBound where
  rounds := List.ofFn fun round =>
    strategy.message round.val round.isLt
      (SumCheck.CausalFixedPhase.prefixAt word round.val
        (Nat.le_of_lt round.isLt))

end NcStrategy

/-- Causal generator for a concrete NC polynomial with its independently
proved quartic round bound. -/
noncomputable def ncGenerator
    {rounds : Nat}
    (q : List K -> K)
    (initial : K)
    (challengeSetSize : Nat)
    (represented :
      FixedPhase.Sequential.RoundRepresentable ops.toOps q
        Polynomial.Nc.Degree.ncSumcheckDegreeBound rounds)
    (strategy : NcStrategy rounds) :
    SumCheck.CausalFixedPhase.Generator
      Polynomial.Nc.Degree.ncSumcheckDegreeBound rounds :=
  SumCheck.CausalFixedPhase.Generator.ofRoundRepresentable q initial
    challengeSetSize
    strategy.message represented

@[simp] theorem nc_generated_certificate_eq
    {rounds : Nat}
    (q : List K -> K)
    (initial : K)
    (challengeSetSize : Nat)
    (represented :
      FixedPhase.Sequential.RoundRepresentable ops.toOps q
        Polynomial.Nc.Degree.ncSumcheckDegreeBound rounds)
    (strategy : NcStrategy rounds)
    (word : Fin rounds -> K) :
    SumCheck.CausalFixedPhase.certificate
        (ncGenerator q initial challengeSetSize represented strategy) word =
      strategy.certificate word := by
  rfl

/-- Exact transport of the repository five-slot NC collision event. -/
theorem nc_roundCollision_implies_detects
    {rounds : Nat}
    (q : List K -> K)
    (initial : K)
    (challengeSetSize : Nat)
    (represented :
      FixedPhase.Sequential.RoundRepresentable ops.toOps q
        Polynomial.Nc.Degree.ncSumcheckDegreeBound rounds)
    (strategy : NcStrategy rounds)
    (word : Fin rounds -> K)
    (round : Nightstream.SuperNeo.SumCheck.Round K K)
    (collision :
      FixedPhase.BadChallenge ops.toOps q
        Polynomial.Nc.Degree.ncSumcheckDegreeBound challengeSetSize initial
        (List.ofFn word) (strategy.certificate word) round) :
    PaperJoint.CausalSumCheckBound.detects ops
      (SumCheck.CausalFixedPhase.process
        (ncGenerator q initial challengeSetSize represented strategy)) word =
        true := by
  apply SumCheck.CausalFixedPhase.badChallenge_implies_detects
    (ncGenerator q initial challengeSetSize represented strategy) word
  exact ⟨round, collision⟩

/-- Finite root counting bound for the causal five-slot NC phase. -/
theorem nc_detects_probability_le
    {rounds : Nat}
    (q : List K -> K)
    (initial : K)
    (challengeSetSize : Nat)
    (represented :
      FixedPhase.Sequential.RoundRepresentable ops.toOps q
        Polynomial.Nc.Degree.ncSumcheckDegreeBound rounds)
    (strategy : NcStrategy rounds)
    (noZeroDivisors :
      PaperJoint.FiniteRootCounting.NoZeroDivisors ops)
    (alphabet : Support K) :
    ((Nightstream.SuperNeo.InteractiveReduction.CoordinateForking.FiniteWords.Support.challengeVectors
        alphabet rounds).uniform).probabilityBool
        (PaperJoint.CausalSumCheckBound.detects ops
          (SumCheck.CausalFixedPhase.process
            (ncGenerator q initial challengeSetSize represented strategy))) <=
      ratio
        (rounds * Polynomial.Nc.Degree.ncSumcheckDegreeBound)
        alphabet.cardinality := by
  exact SumCheck.CausalFixedPhase.detects_probability_le
    (ncGenerator q initial challengeSetSize represented strategy)
    noZeroDivisors alphabet

/-- Full ideal-interactive prover strategy for the two physical SumChecks.
The NC strategy may depend on the completed FE word, matching the phase
ordering, but never on its current or future NC challenges. -/
structure SplitStrategy
    {shape : SemanticShape}
    (input : PublicInput shape)
    (feDomain : FlatNcDomain)
    (ncRounds : Nat) where
  fe : FeStrategy input feDomain
  nc : (Fin (feRoundCount shape feDomain) -> K) -> NcStrategy ncRounds

private theorem mixture_probabilityBool_le_of_components
    {Prefix Outcome : Type}
    (mixture : Mixture Prefix Outcome)
    (event : Outcome -> Bool)
    (bound : Rat)
    (componentBound : forall outer,
      outer ∈ mixture.prefixes.values ->
        (mixture.component outer).probabilityBool event <= bound) :
    mixture.probabilityBool event <= bound := by
  rw [← mixture.probability_bool_event]
  apply Mixture.probability_le_of_components
  intro outer member
  rw [(mixture.component outer).probability_bool_event]
  exact componentBound outer member

/-- Exact FE detector on a complete ideal FE word. -/
noncomputable def feDetects
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (profile : Polynomial.Fe.SupportedProfile shape domain)
    (data : Data shape)
    (coins : Polynomial.Fe.Coins shape domain)
    (challengeSetSize : Nat)
    (strategy : FeStrategy (PublicInput.ofSources data) domain)
    (word : Fin (feRoundCount shape domain) -> K) : Bool :=
  PaperJoint.CausalSumCheckBound.detects ops
    (SumCheck.CausalFixedPhase.process
      (feGenerator profile data coins challengeSetSize strategy)) word

/-- Exact NC detector after the completed FE word has selected the causal NC
strategy. -/
noncomputable def ncDetects
    {shape : SemanticShape}
    {input : PublicInput shape}
    {feDomain : FlatNcDomain}
    {ncRounds : Nat}
    (q : List K -> K)
    (initial : K)
    (challengeSetSize : Nat)
    (represented :
      FixedPhase.Sequential.RoundRepresentable ops.toOps q
        Polynomial.Nc.Degree.ncSumcheckDegreeBound ncRounds)
    (strategy : SplitStrategy input feDomain ncRounds)
    (seed :
      (Fin (feRoundCount shape feDomain) -> K) ×
        (Fin ncRounds -> K)) : Bool :=
  PaperJoint.CausalSumCheckBound.detects ops
    (SumCheck.CausalFixedPhase.process
      (ncGenerator q initial challengeSetSize represented
        (strategy.nc seed.1))) seed.2

/-- Explicit two-phase union bound. The product support makes every NC word
fresh after the complete FE word. The component proof permits the NC prover
strategy to depend on that FE word; no FE/NC strategy independence premise is
used. -/
theorem split_detects_probability_le
    {shape : SemanticShape}
    {feDomain : FlatNcDomain}
    {ncRounds : Nat}
    (profile : Polynomial.Fe.SupportedProfile shape feDomain)
    (data : Data shape)
    (feCoins : Polynomial.Fe.Coins shape feDomain)
    (q : List K -> K)
    (initial : K)
    (challengeSetSize : Nat)
    (represented :
      FixedPhase.Sequential.RoundRepresentable ops.toOps q
        Polynomial.Nc.Degree.ncSumcheckDegreeBound ncRounds)
    (strategy :
      SplitStrategy (PublicInput.ofSources data) feDomain ncRounds)
    (noZeroDivisors :
      PaperJoint.FiniteRootCounting.NoZeroDivisors ops)
    (alphabet : Support K) :
    let feWords :=
      Nightstream.SuperNeo.InteractiveReduction.CoordinateForking.FiniteWords.Support.challengeVectors
        alphabet (feRoundCount shape feDomain)
    let ncWords :=
      Nightstream.SuperNeo.InteractiveReduction.CoordinateForking.FiniteWords.Support.challengeVectors
        alphabet ncRounds
    ((feWords.product ncWords).uniform).probabilityBool
        (fun seed =>
          feDetects profile data feCoins challengeSetSize strategy.fe seed.1 ||
            ncDetects q initial challengeSetSize represented strategy seed) <=
      ratio
          (feRoundCount shape feDomain *
            SumCheck.Fe.Drow (PublicInput.ofSources data))
          alphabet.cardinality +
        ratio
          (ncRounds * Polynomial.Nc.Degree.ncSumcheckDegreeBound)
          alphabet.cardinality := by
  let feWords :=
    Nightstream.SuperNeo.InteractiveReduction.CoordinateForking.FiniteWords.Support.challengeVectors
      alphabet (feRoundCount shape feDomain)
  let ncWords :=
    Nightstream.SuperNeo.InteractiveReduction.CoordinateForking.FiniteWords.Support.challengeVectors
      alphabet ncRounds
  let experiment := (feWords.product ncWords).uniform
  let feEvent :
      ((Fin (feRoundCount shape feDomain) -> K) ×
        (Fin ncRounds -> K)) -> Bool :=
    fun seed =>
      feDetects profile data feCoins challengeSetSize strategy.fe seed.1
  let ncEvent :
      ((Fin (feRoundCount shape feDomain) -> K) ×
        (Fin ncRounds -> K)) -> Bool :=
    ncDetects q initial challengeSetSize represented strategy
  let feBudget :=
    ratio
      (feRoundCount shape feDomain *
        SumCheck.Fe.Drow (PublicInput.ofSources data))
      alphabet.cardinality
  let ncBudget :=
    ratio
      (ncRounds * Polynomial.Nc.Degree.ncSumcheckDegreeBound)
      alphabet.cardinality
  have feBound :
      experiment.probabilityBool feEvent <= feBudget := by
    calc
      experiment.probabilityBool feEvent =
          feWords.uniform.probabilityBool
            (feDetects profile data feCoins challengeSetSize strategy.fe) := by
        simpa only [experiment, feEvent] using
          Support.product_uniform_probabilityBool_first feWords ncWords
            (feDetects profile data feCoins challengeSetSize strategy.fe)
      _ <= feBudget := by
        exact fe_detects_probability_le profile data feCoins challengeSetSize
          strategy.fe noZeroDivisors alphabet
  let ncMixture :
      Mixture (Fin (feRoundCount shape feDomain) -> K)
        ((Fin (feRoundCount shape feDomain) -> K) ×
          (Fin ncRounds -> K)) := {
    prefixes := feWords
    component := fun feWord => {
      Seed := Fin ncRounds -> K
      support := ncWords
      outcome := fun ncWord => (feWord, ncWord)
    }
  }
  have ncComponentBound :
      forall feWord,
        feWord ∈ ncMixture.prefixes.values ->
          (ncMixture.component feWord).probabilityBool ncEvent <=
            ncBudget := by
    intro feWord _member
    change
      ncWords.uniform.probabilityBool
          (fun ncWord =>
            ncDetects q initial challengeSetSize represented strategy
              (feWord, ncWord)) <=
        ncBudget
    exact nc_detects_probability_le q initial challengeSetSize represented
      (strategy.nc feWord) noZeroDivisors alphabet
  have ncMixtureBound :
      ncMixture.probabilityBool ncEvent <= ncBudget :=
    mixture_probabilityBool_le_of_components ncMixture ncEvent ncBudget
      ncComponentBound
  have productEquality :=
    Mixture.sharedSupport_probabilityBool_eq_product
      feWords ncWords (fun feWord ncWord => (feWord, ncWord)) ncEvent
  have ncBound :
      experiment.probabilityBool ncEvent <= ncBudget := by
    calc
      experiment.probabilityBool ncEvent =
          ncMixture.probabilityBool ncEvent := by
        simpa only [experiment, ncMixture] using productEquality.symm
      _ <= ncBudget := ncMixtureBound
  exact Rat.le_trans
    (experiment.probabilityBool_or_le feEvent ncEvent)
    (Rat.le_trans
      ((Rat.add_le_add_right
        (c := experiment.probabilityBool ncEvent)).mpr feBound)
      ((Rat.add_le_add_left (c := feBudget)).mpr ncBound))

end Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.BlockLane.CausalSoundness
