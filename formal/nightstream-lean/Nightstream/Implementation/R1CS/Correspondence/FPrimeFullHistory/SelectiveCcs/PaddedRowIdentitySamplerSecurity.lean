import Mathlib.FieldTheory.Finite.Basic
import Mathlib.Tactic.NormNum
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.PaddedRowIdentityPoseidon2
import Nightstream.SuperNeo.InteractiveReduction.FiniteUniform

/-!
Contract: finite failure bound for the selected bounded full-field `Pi_RLC`
sampler.

Owns:
- the exact 15-source, 54-coefficient, three-attempt parameter tuple;
- the equivalence between one coefficient failure and three rejections;
- the finite ideal experiment for three independent uniform Goldilocks
  candidates;
- the exact `1/q^3` ideal exhaustion probability;
- the exact `810/q^3` per-fold exhaustion expression; and
- the finite union bound and proved 182-bit rational upper bound.

Does not own: a proof that each concrete Poseidon2 candidate triple has the
ideal joint distribution, collision resistance, the random-oracle assumption,
low-norm invertibility, Rust, or R1CS correspondence.

Assurance tier: security-reduced. The finite rational arithmetic is proved in
Lean. `Poseidon2IdealSamplerTransfer` is the precise remaining distribution
premise. It equates each concrete indexed candidate triple with the finite
ideal experiment. Lean derives the complete sampler bound from that local
transfer. It does not accept the final bound as a premise.
-/

set_option autoImplicit false
set_option maxRecDepth 10000

namespace Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentitySamplerSecurity

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.Nifs
open Nightstream.SuperNeo.InteractiveReduction.FiniteUniform
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityPoseidon2

/-- Exact event that all three indexed candidates for one coefficient reject. -/
def ThreeRejections
    (state : State)
    (source : Fin PaperProfile.arity.total)
    (coefficient : Fin samplerCoefficientCount) : Prop :=
  candidateAccepted
      (candidateValue state source coefficient firstAttempt) = false /\
    candidateAccepted
      (candidateValue state source coefficient secondAttempt) = false /\
    candidateAccepted
      (candidateValue state source coefficient thirdAttempt) = false

theorem sampleCoefficient_eq_none_iff_threeRejections
    (state : State)
    (source : Fin PaperProfile.arity.total)
    (coefficient : Fin samplerCoefficientCount) :
    sampleCoefficient state source coefficient = none ↔
      ThreeRejections state source coefficient := by
  simpa [sampleCoefficient, ThreeRejections, candidateAccepted, candidateValue,
    firstAttempt, secondAttempt, thirdAttempt] using
    Nightstream.Implementation.Transcript.Construction3Poseidon2.sampleCoefficient_eq_none_iff
      state source coefficient

/-- A shortfall witness names one coefficient whose three attempts reject. -/
theorem shortfall_requires_three_rejections
    {state : State} (shortfall : SamplerShortfall state) :
    Exists fun source : Fin PaperProfile.arity.total =>
      Exists fun coefficient : Fin samplerCoefficientCount =>
        ThreeRejections state source coefficient := by
  rcases shortfall with ⟨source, coefficient, failed⟩
  exact ⟨source, coefficient,
    (sampleCoefficient_eq_none_iff_threeRejections
      state source coefficient).mp failed⟩

/-- One uniform candidate rejects with probability exactly `1/q`. -/
def singleCandidateRejectionProbability : Rat :=
  (1 : Rat) / (goldilocksModulus : Rat)

/-- Three independent candidate rejections exhaust one coefficient. -/
def singleCoefficientExhaustionBound : Rat :=
  (1 : Rat) / ((goldilocksModulus ^ samplerAttemptCount : Nat) : Rat)

/-- There are exactly `15 * 54 = 810` sampled coefficients per fold. -/
def sampledCoefficientCount : Nat :=
  PaperProfile.arity.total * samplerCoefficientCount

/-- Union bound for all coefficient-exhaustion events in one fold. -/
def completeSamplerShortfallBound : Rat :=
  (sampledCoefficientCount : Rat) * singleCoefficientExhaustionBound

/-- Target used to report the exact selected loss in whole security bits. -/
def samplerSecurityTarget : Rat :=
  (1 : Rat) / (((2 : Nat) ^ 182 : Nat) : Rat)

theorem selected_sampler_parameters :
    PaperProfile.arity.total = 15 /\
    samplerCoefficientCount = 54 /\
    samplerAttemptCount = 3 /\
    sampledCoefficientCount = 810 := by
  decide

/-- The exact accepted-domain size is divisible by five. This is the
arithmetic premise behind uniform mod-5 digits after rejection. -/
theorem accepted_domain_divisible_by_five :
    goldilocksModulus - 1 = acceptedQuotientCount * 5 :=
  acceptedDomain_factorization

/-! ## Finite ideal random-oracle experiment -/

/-- The three candidates used for one coefficient. The nested product order is
first, second, then third attempt. -/
abbrev CandidateTriple := F × (F × F)

/-- Duplicate-free full support for an inhabited finite type. -/
private noncomputable def finiteTypeSupport
    (Value : Type) [Fintype Value] [DecidableEq Value] [Nonempty Value] :
    Support Value where
  values := Finset.univ.toList
  nodup := Finset.nodup_toList Finset.univ
  nonempty := by
    intro empty
    let value : Value := Classical.choice inferInstance
    have member : value ∈ (Finset.univ : Finset Value).toList := by simp
    rw [empty] at member
    exact List.not_mem_nil member

private theorem finiteTypeSupport_count_singleton
    {Value : Type} [Fintype Value] [DecidableEq Value] [Nonempty Value]
    (target : Value) :
    (finiteTypeSupport Value).values.countP
        (fun value => decide (value = target)) = 1 := by
  have countP_eq_count : forall values : List Value,
      values.countP (fun value => decide (value = target)) =
        values.count target := by
    intro values
    induction values with
    | nil => rfl
    | cons head tail inductionHypothesis =>
        by_cases equal : head = target <;>
          simp [equal, inductionHypothesis]
  rw [countP_eq_count]
  exact List.count_eq_one_of_mem
    (finiteTypeSupport Value).nodup (by simp [finiteTypeSupport])

private theorem finiteTypeSupport_probability_singleton
    {Value : Type} [Fintype Value] [DecidableEq Value] [Nonempty Value]
    (target : Value) :
    (finiteTypeSupport Value).uniform.probability
        (fun value => value = target) =
      (1 : Rat) / (Fintype.card Value : Rat) := by
  have eventEq :
      (fun value : Value => value = target) =
        (fun value => decide (value = target) = true) := by
    funext value
    apply propext
    simp
  rw [eventEq, Experiment.probability_bool_event]
  change
    (((finiteTypeSupport Value).values.countP
        (fun value => decide (value = target)) : Nat) : Rat) /
      ((finiteTypeSupport Value).cardinality : Rat) =
        (1 : Rat) / (Fintype.card Value : Rat)
  rw [finiteTypeSupport_count_singleton target]
  congr 1
  simp [finiteTypeSupport, Support.cardinality]

/-- All triples of Goldilocks candidates, each present exactly once. -/
noncomputable def idealCandidateTripleSupport : Support CandidateTriple :=
  finiteTypeSupport CandidateTriple

/-- One ideal random-oracle coordinate: three jointly uniform candidates. -/
noncomputable def idealCandidateTripleExperiment :
    Experiment CandidateTriple :=
  idealCandidateTripleSupport.uniform

/-- The ideal joint support has exactly `q^3` points. -/
theorem idealCandidateTripleSupport_cardinality :
    idealCandidateTripleSupport.cardinality = goldilocksModulus ^ 3 := by
  simp [idealCandidateTripleSupport, finiteTypeSupport, Support.cardinality,
    CandidateTriple, F, Nat.pow_succ, Nat.mul_assoc]

/-- Exact joint law of the three ideal candidates. This singleton law is the
finite statement that all three attempts are uniform and independent. -/
theorem idealCandidateTriple_joint_probability
    (triple : CandidateTriple) :
    idealCandidateTripleExperiment.probability
        (fun sampled => sampled = triple) =
      (1 : Rat) / ((goldilocksModulus ^ 3 : Nat) : Rat) := by
  rw [show idealCandidateTripleExperiment =
      idealCandidateTripleSupport.uniform by rfl]
  rw [show idealCandidateTripleSupport =
      finiteTypeSupport CandidateTriple by rfl]
  rw [finiteTypeSupport_probability_singleton triple]
  rw [show Fintype.card CandidateTriple = goldilocksModulus ^ 3 by
    simp [CandidateTriple, F, Nat.pow_succ, Nat.mul_assoc]]

/-- The only rejected Goldilocks residue. -/
def rejectedCandidate : F :=
  ⟨goldilocksModulus - 1, by
    simp only [goldilocksModulus]
    omega⟩

def rejectedTriple : CandidateTriple :=
  (rejectedCandidate, (rejectedCandidate, rejectedCandidate))

/-- Ideal exhaustion rejects all three candidates. -/
def IdealThreeRejections (triple : CandidateTriple) : Prop :=
  candidateAccepted triple.1 = false /\
    candidateAccepted triple.2.1 = false /\
    candidateAccepted triple.2.2 = false

theorem idealThreeRejections_iff_eq_rejectedTriple
    (triple : CandidateTriple) :
    IdealThreeRejections triple ↔ triple = rejectedTriple := by
  rcases triple with ⟨first, second, third⟩
  simp only [IdealThreeRejections, candidateAccepted_eq_false_iff]
  constructor
  · rintro ⟨firstRejected, secondRejected, thirdRejected⟩
    apply Prod.ext
    · apply Fin.ext
      exact firstRejected
    · apply Prod.ext <;> apply Fin.ext
      · exact secondRejected
      · exact thirdRejected
  · intro equal
    have firstEqual := congrArg (fun value : CandidateTriple => value.1.val) equal
    have secondEqual := congrArg
      (fun value : CandidateTriple => value.2.1.val) equal
    have thirdEqual := congrArg
      (fun value : CandidateTriple => value.2.2.val) equal
    simpa [rejectedTriple, rejectedCandidate] using
      And.intro firstEqual (And.intro secondEqual thirdEqual)

/-- Three ideal rejections occur with exact probability `1/q^3`. -/
theorem idealThreeRejections_probability :
    idealCandidateTripleExperiment.probability IdealThreeRejections =
      singleCoefficientExhaustionBound := by
  have eventEq : IdealThreeRejections =
      (fun triple => triple = rejectedTriple) := by
    funext triple
    apply propext
    exact idealThreeRejections_iff_eq_rejectedTriple triple
  rw [eventEq, idealCandidateTriple_joint_probability rejectedTriple]
  rfl

/-- Concrete Poseidon2 candidates for one indexed coefficient. -/
def candidateTriple
    (state : State)
    (source : Fin PaperProfile.arity.total)
    (coefficient : Fin samplerCoefficientCount) : CandidateTriple :=
  (candidateValue state source coefficient firstAttempt,
    (candidateValue state source coefficient secondAttempt,
      candidateValue state source coefficient thirdAttempt))

theorem threeRejections_iff_ideal
    (state : State)
    (source : Fin PaperProfile.arity.total)
    (coefficient : Fin samplerCoefficientCount) :
    ThreeRejections state source coefficient ↔
      IdealThreeRejections (candidateTriple state source coefficient) := by
  rfl

/-- Explicit Poseidon2-to-ideal boundary. For each indexed coefficient, every
event over its three candidates has the same probability as in the exact
finite ideal experiment. Independence between different coefficients is not
required for the union bound. -/
def Poseidon2IdealSamplerTransfer
    (experiment : Experiment State) : Prop :=
  forall source coefficient event,
    experiment.probability
        (fun state => event (candidateTriple state source coefficient)) =
      idealCandidateTripleExperiment.probability event

theorem threeRejections_probability_eq
    (experiment : Experiment State)
    (transfer : Poseidon2IdealSamplerTransfer experiment)
    (source : Fin PaperProfile.arity.total)
    (coefficient : Fin samplerCoefficientCount) :
    experiment.probability
        (fun state => ThreeRejections state source coefficient) =
      singleCoefficientExhaustionBound := by
  have eventEq :
      (fun state => ThreeRejections state source coefficient) =
        (fun state =>
          IdealThreeRejections (candidateTriple state source coefficient)) := by
    funext state
    apply propext
    exact threeRejections_iff_ideal state source coefficient
  rw [eventEq, transfer source coefficient IdealThreeRejections,
    idealThreeRejections_probability]

private theorem probability_exists_fin_le_mul
    {Outcome : Type}
    (experiment : Experiment Outcome)
    {count : Nat}
    (event : Fin count -> Outcome -> Prop)
    (bound : Rat)
    (eventBound : forall index,
      experiment.probability (event index) <= bound) :
    experiment.probability
        (fun outcome => Exists fun index => event index outcome) <=
      (count : Rat) * bound := by
  induction count with
  | zero =>
      have impossible :
          (fun outcome => Exists fun index : Fin 0 => event index outcome) =
            (fun _ => False) := by
        funext outcome
        apply propext
        simp
      rw [impossible, experiment.probability_false]
      simp
  | succ smaller inductionHypothesis =>
      let head : Outcome -> Prop := event 0
      let tail : Fin smaller -> Outcome -> Prop := fun index => event index.succ
      have split :
          (fun outcome => Exists fun index : Fin (smaller + 1) =>
            event index outcome) =
          (fun outcome => head outcome \/
            Exists fun index : Fin smaller => tail index outcome) := by
        funext outcome
        apply propext
        exact Fin.exists_fin_succ
      rw [split]
      calc
        experiment.probability
              (fun outcome => head outcome \/
                Exists fun index : Fin smaller => tail index outcome) <=
            experiment.probability head +
              experiment.probability
                (fun outcome => Exists fun index : Fin smaller =>
                  tail index outcome) :=
          experiment.probability_or_le _ _
        _ <= bound + (smaller : Rat) * bound := by
          exact Rat.le_trans
            ((Rat.add_le_add_right
              (c := experiment.probability
                (fun outcome => Exists fun index : Fin smaller =>
                  tail index outcome))).mpr (eventBound 0))
            ((Rat.add_le_add_left (c := bound)).mpr
              (inductionHypothesis tail
                (fun index => eventBound index.succ)))
        _ = (smaller + 1 : Nat) * bound := by
          rw [Rat.natCast_add, Rat.natCast_ofNat, Rat.add_mul, Rat.one_mul]
          exact Rat.add_comm _ _

/-- The complete per-fold sampler loss is at most `2^-182`. -/
theorem completeSamplerShortfallBound_le_target :
    completeSamplerShortfallBound <= samplerSecurityTarget := by
  norm_num [completeSamplerShortfallBound, sampledCoefficientCount,
    singleCoefficientExhaustionBound, samplerSecurityTarget,
    samplerAttemptCount, samplerCoefficientCount, PaperProfile.arity,
    Nightstream.Implementation.Transcript.Construction3Poseidon2.samplerAttemptCount,
    Nightstream.Implementation.Transcript.Construction3Poseidon2.samplerCoefficientCount,
    goldilocksModulus]

theorem samplerShortfall_probability_le
    (experiment : Experiment State)
    (transfer : Poseidon2IdealSamplerTransfer experiment) :
    experiment.probability SamplerShortfall <=
      completeSamplerShortfallBound := by
  have shortfallEq : SamplerShortfall =
      (fun state =>
        Exists fun source : Fin PaperProfile.arity.total =>
          Exists fun coefficient : Fin samplerCoefficientCount =>
            ThreeRejections state source coefficient) := by
    funext state
    apply propext
    constructor
    · exact shortfall_requires_three_rejections
    · rintro ⟨source, coefficient, rejected⟩
      exact ⟨source, coefficient,
        (sampleCoefficient_eq_none_iff_threeRejections
          state source coefficient).mpr rejected⟩
  rw [shortfallEq]
  have perSource : forall source : Fin PaperProfile.arity.total,
      experiment.probability
          (fun state => Exists fun coefficient : Fin samplerCoefficientCount =>
            ThreeRejections state source coefficient) <=
        (samplerCoefficientCount : Rat) *
          singleCoefficientExhaustionBound := by
    intro source
    apply probability_exists_fin_le_mul experiment
      (fun coefficient state => ThreeRejections state source coefficient)
      singleCoefficientExhaustionBound
    intro coefficient
    rw [threeRejections_probability_eq experiment transfer source coefficient]
  have outerBound := probability_exists_fin_le_mul experiment
    (fun source state =>
      Exists fun coefficient : Fin samplerCoefficientCount =>
        ThreeRejections state source coefficient)
    ((samplerCoefficientCount : Rat) * singleCoefficientExhaustionBound)
    perSource
  simpa [completeSamplerShortfallBound, sampledCoefficientCount,
    Rat.natCast_mul, Rat.mul_assoc] using outerBound

/-- Under the named full-field random-oracle premise, the complete bounded
sampler fails with probability at most `2^-182`. -/
theorem samplerShortfall_probability_le_182_bits
    (experiment : Experiment State)
    (transfer : Poseidon2IdealSamplerTransfer experiment) :
    experiment.probability SamplerShortfall <= samplerSecurityTarget :=
  Rat.le_trans
    (samplerShortfall_probability_le experiment transfer)
    completeSamplerShortfallBound_le_target

end Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentitySamplerSecurity
