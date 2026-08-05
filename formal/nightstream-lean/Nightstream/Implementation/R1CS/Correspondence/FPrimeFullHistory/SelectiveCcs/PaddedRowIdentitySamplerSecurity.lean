import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Tactic.NormNum
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.PaddedRowIdentityPoseidon2
import Nightstream.SuperNeo.InteractiveReduction.FiniteUniform

/-!
Contract: finite failure bound for the selected 54-of-64 `Pi_RLC` sampler.

Owns:
- the proof that one scalar shortfalls only after at least eleven rejected
  chunks;
- the Goldilocks-word conservative fixed-set and union-bound constants;
- the union bound over all fifteen sampled scalars; and
- the exact 121-bit rational security inequality.

Does not own: a proof that concrete Poseidon2 outputs are independent uniform
Goldilocks words, collision resistance, the random-oracle assumption,
low-norm invertibility, Rust, or R1CS correspondence.

Assurance tier: security-reduced. The finite counting and rational arithmetic
are proved in Lean. `GoldilocksRandomOracleSamplerContract` is the precise
remaining distribution premise: it must be discharged by the selected
Poseidon2/random-oracle model, not by protocol code.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentitySamplerSecurity

open Nightstream.SuperNeo.Folding.Nifs
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler
open Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionAlphabet
open Nightstream.SuperNeo.InteractiveReduction.FiniteUniform
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityPoseidon2

/-- Rejected candidates in one finite sampler prefix. -/
def rejectedCount (candidates : List Chunk) : Nat :=
  candidates.countP fun chunk => !verifier.accepts chunk

theorem acceptedCount_add_rejectedCount (candidates : List Chunk) :
    Nightstream.SuperNeo.Sampling.FirstAccepted.acceptedCount verifier candidates +
        rejectedCount candidates = candidates.length := by
  induction candidates with
  | nil =>
      simp [Nightstream.SuperNeo.Sampling.FirstAccepted.acceptedCount,
        Nightstream.SuperNeo.Sampling.FirstAccepted.acceptedCandidates,
        rejectedCount]
  | cons head tail inductionHypothesis =>
      unfold Nightstream.SuperNeo.Sampling.FirstAccepted.acceptedCount
        Nightstream.SuperNeo.Sampling.FirstAccepted.acceptedCandidates
        rejectedCount at inductionHypothesis ⊢
      cases accepted : verifier.accepts head <;>
        simp [accepted] <;> omega

/-- A 54-of-64 shortfall requires at least eleven rejected chunks. -/
theorem shortfall_requires_eleven_rejections
    {candidates : List Chunk}
    (lengthExact : candidates.length = candidateBound)
    (shortfall : Nightstream.SuperNeo.Sampling.FirstAccepted.Shortfall
      verifier coefficientCount candidates) :
    11 <= rejectedCount candidates := by
  have partition := acceptedCount_add_rejectedCount candidates
  have acceptedLt :
      Nightstream.SuperNeo.Sampling.FirstAccepted.acceptedCount
        verifier candidates < 54 := by
    simpa [coefficientCount] using shortfall
  have lengthValue : candidates.length = 64 := by
    simpa [candidateBound] using lengthExact
  omega

/-- One rejected 16-bit chunk fixes one of four digits in a Goldilocks word. -/
def chunkBase : Nat := chunkModulus

/-- A shortfall has eleven fixed rejected positions. For each occupied
Goldilocks word, the elementary preimage bound loses at most a factor two
because `q > 2^63`. There are at most eleven occupied words. -/
def fixedElevenRejectionBound : Rat :=
  ((2 ^ 11 : Nat) : Rat) / ((chunkBase ^ 11 : Nat) : Rat)

/-- Union bound over all choices of eleven rejected positions in one
64-candidate scalar. -/
def singleScalarShortfallBound : Rat :=
  ((Nat.choose candidateBound 11 : Nat) : Rat) * fixedElevenRejectionBound

/-- Union bound over the selected fifteen `Pi_RLC` scalar challenges. -/
def completeSamplerShortfallBound : Rat :=
  ((PaperProfile.arity.total : Nat) : Rat) * singleScalarShortfallBound

/-- Target used to report the sampler loss in security bits. -/
def samplerSecurityTarget : Rat :=
  (1 : Rat) / (((2 : Nat) ^ 121 : Nat) : Rat)

theorem selected_sampler_parameters :
    PaperProfile.arity.total = 15 /\
    candidateBound = 64 /\
    coefficientCount = 54 /\
    chunkBase = 65536 := by
  decide

/-- Goldilocks is larger than half of the 64-bit word space. This is the
source of the conservative factor two per occupied word. -/
theorem goldilocks_exceeds_half_word :
    (2 : Nat) ^ 63 < goldilocksModulus := by
  decide

theorem choose_64_11_value :
    Nat.choose 64 11 = 743595781824 := by
  decide

/-- The complete conservative sampler loss is at most `2^-121`. -/
theorem completeSamplerShortfallBound_le_target :
    completeSamplerShortfallBound <= samplerSecurityTarget := by
  norm_num [completeSamplerShortfallBound, singleScalarShortfallBound,
    fixedElevenRejectionBound, samplerSecurityTarget, chunkBase,
    chunkModulus, candidateBound, PaperProfile.arity, choose_64_11_value]

/-- Shortfall at one scalar coordinate in an outcome state. -/
def scalarShortfallAt (coordinate : Nat) (state : State) : Prop :=
  ShortfallAt samplerSpecification candidateBound state coordinate

/-- Existence of a shortfall below a natural coordinate bound. -/
def shortfallBelow (count : Nat) (state : State) : Prop :=
  Exists fun coordinate : Nat =>
    coordinate < count /\ scalarShortfallAt coordinate state

theorem shortfallBelow_zero_iff (state : State) :
    shortfallBelow 0 state <-> False := by
  simp [shortfallBelow]

theorem shortfallBelow_succ_iff (count : Nat) (state : State) :
    shortfallBelow (count + 1) state <->
      shortfallBelow count state \/ scalarShortfallAt count state := by
  constructor
  · rintro ⟨coordinate, before, shortfall⟩
    rcases Nat.lt_or_eq_of_le (Nat.le_of_lt_succ before) with earlier | equal
    · exact Or.inl ⟨coordinate, earlier, shortfall⟩
    · exact Or.inr (equal ▸ shortfall)
  · rintro (⟨coordinate, before, shortfall⟩ | shortfall)
    · exact ⟨coordinate, Nat.lt_succ_of_lt before, shortfall⟩
    · exact ⟨count, Nat.lt_add_one count, shortfall⟩

theorem samplerShortfall_iff_shortfallBelow (state : State) :
    SamplerShortfall state <-> shortfallBelow 15 state := by
  constructor
  · rintro ⟨coordinate, shortfall⟩
    exact ⟨coordinate.val, by simpa using coordinate.isLt, shortfall⟩
  · rintro ⟨coordinate, before, shortfall⟩
    exact ⟨⟨coordinate, by simpa using before⟩, shortfall⟩

/-- Exact random-oracle distribution premise needed by the concrete sampler.
It is per scalar, so the complete-batch union is derived below rather than
assumed. -/
def GoldilocksRandomOracleSamplerContract
    (experiment : Experiment State) : Prop :=
  forall coordinate,
    coordinate < PaperProfile.arity.total ->
      experiment.probability (scalarShortfallAt coordinate) <=
        singleScalarShortfallBound

private theorem shortfallBelow_probability_le
    (experiment : Experiment State)
    (distribution : GoldilocksRandomOracleSamplerContract experiment) :
    forall count,
      count <= PaperProfile.arity.total ->
      experiment.probability (shortfallBelow count) <=
        ((count : Nat) : Rat) * singleScalarShortfallBound := by
  intro count
  induction count with
  | zero =>
      intro _
      have falseEvent : shortfallBelow 0 = (fun _ : State => False) := by
        funext state
        exact propext (shortfallBelow_zero_iff state)
      rw [falseEvent, Experiment.probability_false]
      simp
  | succ count inductionHypothesis =>
      intro within
      have countWithin : count <= PaperProfile.arity.total :=
        Nat.le_trans (Nat.le_succ count) within
      have coordinateWithin : count < PaperProfile.arity.total := by
        exact Nat.lt_of_succ_le within
      have eventEq :
          shortfallBelow (count + 1) =
            (fun state => shortfallBelow count state \/
              scalarShortfallAt count state) := by
        funext state
        exact propext (shortfallBelow_succ_iff count state)
      rw [eventEq]
      calc
        experiment.probability
              (fun state => shortfallBelow count state \/
                scalarShortfallAt count state) <=
            experiment.probability (shortfallBelow count) +
              experiment.probability (scalarShortfallAt count) :=
          experiment.probability_or_le _ _
        _ <= ((count : Nat) : Rat) * singleScalarShortfallBound +
              singleScalarShortfallBound :=
          scaleLaws.add_mono
            (inductionHypothesis countWithin)
            (distribution count coordinateWithin)
        _ = (((count + 1 : Nat) : Nat) : Rat) *
              singleScalarShortfallBound := by
          simp [Rat.add_mul]

/-- Under the exact per-scalar random-oracle distribution contract, the
complete selected sampler shortfalls with probability at most the proved
batch bound. -/
theorem samplerShortfall_probability_le
    (experiment : Experiment State)
    (distribution : GoldilocksRandomOracleSamplerContract experiment) :
    experiment.probability SamplerShortfall <=
      completeSamplerShortfallBound := by
  have eventEq : SamplerShortfall = shortfallBelow 15 := by
    funext state
    exact propext (samplerShortfall_iff_shortfallBelow state)
  rw [eventEq]
  have bound := shortfallBelow_probability_le experiment distribution 15 (by decide)
  simpa [completeSamplerShortfallBound, PaperProfile.arity] using bound

/-- Headline sampler-security result: under the named random-oracle
distribution premise, the full fifteen-scalar bounded sampler loss is at most
`2^-121`. -/
theorem samplerShortfall_probability_le_121_bits
    (experiment : Experiment State)
    (distribution : GoldilocksRandomOracleSamplerContract experiment) :
    experiment.probability SamplerShortfall <= samplerSecurityTarget :=
  Rat.le_trans
    (samplerShortfall_probability_le experiment distribution)
    completeSamplerShortfallBound_le_target

end Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentitySamplerSecurity
