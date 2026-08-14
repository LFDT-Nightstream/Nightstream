import Nightstream.Implementation.R1CS.Canonical.KPiCcsPaperFiatShamir
import Nightstream.Implementation.R1CS.Canonical.Poseidon2CanonicalConstants
import Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.Challenge
import Nightstream.SuperNeo.Folding.Nifs.PaperProfile
import Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionStrongSet

/-!
Contract: shared HyperNova Construction 3 Poseidon2 framing and bounded
PiRLC sampling.

Assurance tier: executable transcript model.

Owns the fixed Construction 3 labels, event descriptor encoding, selected
Poseidon2 constants, challenge squeeze, canonical field serialization, and
the 15-by-54 three-attempt PiRLC sampler.

Does not own a protocol profile's event schedule, public statement encoding,
PiCCS output encoding, generated rows, Rust conformance, or a security
reduction.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 10000

namespace Nightstream.Implementation.Transcript.Construction3Poseidon2

open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS.Canonical.KPiCcsPaperFiatShamir
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.Nifs
open Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionStrongSet
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

abbrev State := Poseidon2Duplex.State
/-- Four Goldilocks output lanes from one Poseidon2 digest. -/
abbrev StatementId := Fin 4 -> F
/-- Selected, Lean-owned width-8 Poseidon2 constants. -/
def constants : Poseidon2Schedule.Constants :=
  Poseidon2CanonicalConstants.selected

/-- Empty state before the statement identifier starts one NIFS transcript. -/
def initialState : State := Poseidon2Duplex.empty

/-! ## Canonical field serialization -/

/-- Numeric words are reduced exactly as the existing one-joint schedule. -/
def word (value : Nat) : Nat := value % goldilocksModulus

/-- Canonical UTF-8 bytes for
`HyperNova/MultiFold/Fiat-Shamir/v2`. -/
def construction3DomainBytes : List Nat :=
  [72, 121, 112, 101, 114, 78, 111, 118, 97, 47, 77, 117, 108, 116,
    105, 70, 111, 108, 100, 47, 70, 105, 97, 116, 45, 83, 104, 97,
    109, 105, 114, 47, 118, 50]

/-- Canonical UTF-8 bytes for Construction 3's `statement-id` label. -/
def statementIdLabelBytes : List Nat :=
  [115, 116, 97, 116, 101, 109, 101, 110, 116, 45, 105, 100]

/-- Canonical UTF-8 bytes for Construction 3's `proof` label. -/
def proofLabelBytes : List Nat := [112, 114, 111, 111, 102]

/-- Canonical UTF-8 bytes for Construction 3's `prover-message` label. -/
def proverMessageLabelBytes : List Nat :=
  [112, 114, 111, 118, 101, 114, 45, 109, 101, 115, 115, 97, 103, 101]

/-- Canonical UTF-8 bytes for Construction 3's `verifier-challenge` label. -/
def verifierChallengeLabelBytes : List Nat :=
  [118, 101, 114, 105, 102, 105, 101, 114, 45, 99, 104, 97, 108,
    108, 101, 110, 103, 101]

/-- Type-and-length frame for one Construction 3 string. -/
def stringFields (bytes : List Nat) : List Nat :=
  [word 32, word bytes.length] ++ bytes.map word

def construction3DomainFields : List Nat :=
  stringFields construction3DomainBytes

def statementIdLabelFields : List Nat :=
  stringFields statementIdLabelBytes

def proofLabelFields : List Nat := stringFields proofLabelBytes

def proverMessageLabelFields : List Nat :=
  stringFields proverMessageLabelBytes

def verifierChallengeLabelFields : List Nat :=
  stringFields verifierChallengeLabelBytes

/-- Construction 3 domain tag for the fixed-length statement identifier. -/
def statementIdentifierTag : Nat := 39

/-- Exact selected event descriptor. Indices are one-based, as in
Construction 3. `fieldCount` fixes the declared message or challenge space. -/
inductive Event where
  | proverMessage
      (eventIndex messageIndex messageType fieldCount : Nat)
  | verifierCoins
      (eventIndex challengeIndex challengeType fieldCount : Nat)
deriving Repr, DecidableEq

def Event.fields : Event -> List Nat
  | .proverMessage eventIndex messageIndex messageType fieldCount =>
      [word 34, word eventIndex, word messageIndex, word messageType,
        word fieldCount]
  | .verifierCoins eventIndex challengeIndex challengeType fieldCount =>
      [word 35, word eventIndex, word challengeIndex, word challengeType,
        word fieldCount]

/-- Production transcript tag for one indexed PiRLC candidate. -/
def piRlcCandidateTag : Nat := 1314062624

/-- The final PiDEC prover message has its own fixed type tag. -/
def piDecOutputTag : Nat := 48
/-- A base-field element has one canonical Goldilocks coordinate. -/
def fFields (value : F) : List Nat := [value.val]

/-- A quadratic-extension element is low limb followed by high limb. -/
def kFields (value : K) : List Nat := [value.c0.val, value.c1.val]

/-- Encode a finite function in increasing `Fin` order. -/
def finFields
    {count : Nat} {Value : Type}
    (encode : Value -> List Nat) (values : Fin count -> Value) : List Nat :=
  (canonicalFinIndices count).flatMap fun index => encode (values index)

/-- Ring coefficients use increasing polynomial degree. -/
def ringFFields (value : RingF) : List Nat :=
  finFields fFields value
/-- Interpret the first two freshly permuted lanes as the selected concrete
quadratic extension. -/
def challengeValue (state : State) : K where
  c0 := ⟨state.lanes ⟨0, by decide⟩ % goldilocksModulus,
    Nat.mod_lt _ (by decide)⟩
  c1 := ⟨state.lanes ⟨1, by decide⟩ % goldilocksModulus,
    Nat.mod_lt _ (by decide)⟩

def squeezeK (state : State) : K × State :=
  let next := Poseidon2Duplex.gate constants state
  (challengeValue next, next)

/-- Construction 3 challenge frame. The domain, literal challenge label,
event index, challenge index, declared type, and domain-expansion coordinates
are absorbed before the concrete squeeze. -/
def verifierChallengeFields
    (eventIndex challengeIndex challengeType : Nat)
    (coordinates : List Nat) : List Nat :=
  construction3DomainFields ++ verifierChallengeLabelFields ++
    [word eventIndex, word challengeIndex, word challengeType,
      word coordinates.length] ++ coordinates.map word

def squeezeVerifierChallenge
    (eventIndex challengeIndex challengeType : Nat)
    (coordinates : List Nat) (state : State) : K × State :=
  squeezeK (Poseidon2Duplex.absorbList constants
    (verifierChallengeFields eventIndex challengeIndex challengeType coordinates)
    state)
/-! ## Exact bounded full-field PiRLC sampling -/

abbrev Coefficient :=
  Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionAlphabet.Coefficient

abbrev Scalar :=
  Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionStrongSet.Scalar

/-- Every ring coefficient gets at most three full-field candidates. -/
def samplerAttemptCount : Nat := 3

/-- Canonical attempt indices. Named values keep the attempt identity stable
across the sampler and its security proof. -/
def firstAttempt : Fin samplerAttemptCount := ⟨0, by decide⟩

def secondAttempt : Fin samplerAttemptCount := ⟨1, by decide⟩

def thirdAttempt : Fin samplerAttemptCount := ⟨2, by decide⟩

/-- Exact number of coefficients in one Phi81 challenge. -/
def samplerCoefficientCount : Nat := 54

/-- Exact source-major, coefficient-major, attempt-minor candidate index. -/
def candidateFlat
    (source : Fin PaperProfile.arity.total)
    (coefficient : Fin samplerCoefficientCount)
    (attempt : Fin samplerAttemptCount) : Nat :=
  (source.val * samplerCoefficientCount + coefficient.val) *
      samplerAttemptCount + attempt.val

/-- Exact fixed-width domain frame for one PiRLC candidate fork.

The complete post-PiCCS state already binds the statement, profile, event
schedule, proof prefix, and PiCCS output. One unique candidate tag plus the
injective flat candidate index therefore separates every candidate without
reabsorbing the text-form Construction-3 labels on each of the 2,430 forks.
The fixed two-field arity and index order are verifier-key data. -/
def candidateFields
    (source : Fin PaperProfile.arity.total)
    (coefficient : Fin samplerCoefficientCount)
    (attempt : Fin samplerAttemptCount) : List Nat :=
  [word piRlcCandidateTag, word (candidateFlat source coefficient attempt)]

@[simp] theorem candidateFields_length
    (source : Fin PaperProfile.arity.total)
    (coefficient : Fin samplerCoefficientCount)
    (attempt : Fin samplerAttemptCount) :
    (candidateFields source coefficient attempt).length = 2 := by
  rfl

/-- One indexed full-field candidate derived from the fixed post-PiCCS state.
The candidate is a complete canonical Goldilocks element, not a 16-bit chunk. -/
def candidateValue
    (state : State)
    (source : Fin PaperProfile.arity.total)
    (coefficient : Fin samplerCoefficientCount)
    (attempt : Fin samplerAttemptCount) : F :=
  let tagged := Poseidon2Duplex.absorbList constants
    (candidateFields source coefficient attempt) state
  let sampled := Poseidon2Duplex.challengeField constants tagged
  ⟨sampled.1 % goldilocksModulus,
    Nat.mod_lt _ (by decide)⟩

/-- Reject only the final Goldilocks residue `q-1`. -/
def candidateAccepted (candidate : F) : Bool :=
  decide (candidate.val < goldilocksModulus - 1)

/-- Accepted residues map in order to the five centered digits. -/
def candidateDigit (candidate : F) : Coefficient :=
  ⟨candidate.val % 5, Nat.mod_lt _ (by decide)⟩

@[simp] theorem candidateAccepted_eq_true_iff (candidate : F) :
    candidateAccepted candidate = true ↔
      candidate.val < goldilocksModulus - 1 := by
  simp [candidateAccepted]

@[simp] theorem candidateAccepted_eq_false_iff (candidate : F) :
    candidateAccepted candidate = false ↔
      candidate.val = goldilocksModulus - 1 := by
  simp only [candidateAccepted, decide_eq_false_iff_not]
  have upper := candidate.isLt
  simp only [goldilocksModulus] at upper ⊢
  omega

/-- One coefficient uses the first accepted candidate and fails after exactly
three rejections. Unused later attempts have no authority. -/
def sampleCoefficient
    (state : State)
    (source : Fin PaperProfile.arity.total)
    (coefficient : Fin samplerCoefficientCount) : Option Coefficient :=
  let first := candidateValue state source coefficient firstAttempt
  if candidateAccepted first then
    some (candidateDigit first)
  else
    let second := candidateValue state source coefficient secondAttempt
    if candidateAccepted second then
      some (candidateDigit second)
    else
      let third := candidateValue state source coefficient thirdAttempt
      if candidateAccepted third then some (candidateDigit third) else none

theorem sampleCoefficient_eq_none_iff
    (state : State)
    (source : Fin PaperProfile.arity.total)
    (coefficient : Fin samplerCoefficientCount) :
    sampleCoefficient state source coefficient = none ↔
      candidateAccepted
          (candidateValue state source coefficient firstAttempt) = false /\
        candidateAccepted
          (candidateValue state source coefficient secondAttempt) = false /\
        candidateAccepted
          (candidateValue state source coefficient thirdAttempt) = false := by
  cases first : candidateAccepted
      (candidateValue state source coefficient firstAttempt) <;>
    cases second : candidateAccepted
      (candidateValue state source coefficient secondAttempt) <;>
    cases third : candidateAccepted
      (candidateValue state source coefficient thirdAttempt) <;>
    simp [sampleCoefficient, first, second, third]

/-- Exact proof-rejection event: at least one of the 15x54 coefficients
exhausts all three attempts. -/
def SamplerShortfall (state : State) : Prop :=
  Exists fun source : Fin PaperProfile.arity.total =>
    Exists fun coefficient : Fin samplerCoefficientCount =>
      sampleCoefficient state source coefficient = none

def SamplerAvailable (state : State) : Prop :=
  ¬ SamplerShortfall state

/-- Executable gate used by the selected verifier. -/
noncomputable def samplerSucceeded (state : State) : Bool := by
  classical
  exact decide (SamplerAvailable state)

@[simp] theorem samplerSucceeded_eq_true_iff (state : State) :
    samplerSucceeded state = true ↔ SamplerAvailable state := by
  classical
  simp [samplerSucceeded]

@[simp] theorem samplerSucceeded_eq_false_iff (state : State) :
    samplerSucceeded state = false ↔ SamplerShortfall state := by
  classical
  simp [samplerSucceeded, SamplerAvailable]

theorem available_or_shortfall (state : State) :
    SamplerAvailable state \/ SamplerShortfall state := by
  rcases Classical.em (SamplerShortfall state) with shortfall | available
  · exact Or.inr shortfall
  · exact Or.inl available

theorem available_excludes_shortfall
    {state : State} (available : SamplerAvailable state) :
    ¬ SamplerShortfall state :=
  available

theorem not_available_iff_shortfall (state : State) :
    ¬ SamplerAvailable state ↔ SamplerShortfall state := by
  simp only [SamplerAvailable, Classical.not_not]

/-- Centered zero is symbol `2`, since the semantic value is `symbol - 2`. -/
def zeroCoefficient : Coefficient := ⟨2, by decide⟩

def zeroScalar : Scalar := fun _ => zeroCoefficient

/-- The generic key needs a total scalar function. Failed coordinates use a
fixed internal zero only so the carrier is total. `samplerSucceeded` prevents
the selected verifier from accepting any execution that reaches this case. -/
def scalarResponse
    (state : State) (source : Fin PaperProfile.arity.total) : Scalar :=
  fun coefficient =>
    (sampleCoefficient state source
      (Fin.cast (by rfl) coefficient)).getD zeroCoefficient

/-- Ring-valued response consumed by the generic paper-key carrier. -/
def piRlcResponse
    (state : State) (source : Fin PaperProfile.arity.total) : RingF :=
  Phi81StrongSet.embedScalar (scalarResponse state source)

theorem piRlcResponse_valid (state : State)
    (source : Fin PaperProfile.arity.total) :
    Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.Challenge.challengeValid
      (piRlcResponse state source) := by
  exact
    Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.Challenge.embedScalar_valid _

/-- Pointwise refinement to the exact successful three-attempt sampler. -/
def ResponseRefinesAt
    (response : State -> Fin PaperProfile.arity.total -> Scalar)
    (state : State) : Prop :=
  forall source coefficient,
    sampleCoefficient state source (Fin.cast (by rfl) coefficient) =
      some (response state source coefficient)

theorem piRlcResponse_refines_of_available
    {state : State} (available : SamplerAvailable state) :
    ResponseRefinesAt scalarResponse state := by
  intro source coefficient
  have succeeds :
      sampleCoefficient state source (Fin.cast (by rfl) coefficient) ≠ none := by
    intro failed
    exact available ⟨source, Fin.cast (by rfl) coefficient, failed⟩
  unfold scalarResponse
  cases sampled : sampleCoefficient state source (Fin.cast (by rfl) coefficient) with
  | none => exact False.elim (succeeds sampled)
  | some value => simp [sampled]

theorem piRlcResponse_refines_of_no_shortfall
    {state : State} (noShortfall : ¬ SamplerShortfall state) :
    ResponseRefinesAt scalarResponse state :=
  piRlcResponse_refines_of_available noShortfall

/-! ## Exact balance of the accepted field domain -/

def acceptedQuotientCount : Nat := 3689348813882916864

theorem acceptedDomain_factorization :
    goldilocksModulus - 1 = acceptedQuotientCount * 5 := by
  decide

abbrev AcceptedCandidate :=
  { candidate : F // candidate.val < goldilocksModulus - 1 }

def factorAccepted (candidate : AcceptedCandidate) :
    Fin acceptedQuotientCount × Coefficient :=
  let quotient := candidate.val.val / 5
  have quotientLt : quotient < acceptedQuotientCount := by
    have accepted := candidate.property
    simp only [goldilocksModulus, acceptedQuotientCount] at accepted ⊢
    omega
  ⟨⟨quotient, quotientLt⟩, candidateDigit candidate.val⟩

def combineAccepted
    (coordinates : Fin acceptedQuotientCount × Coefficient) :
    AcceptedCandidate :=
  let value := coordinates.1.val * 5 + coordinates.2.val
  have accepted : value < goldilocksModulus - 1 := by
    have quotientLt := coordinates.1.isLt
    have residueLt : coordinates.2.val < 5 := by
      simpa [Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionAlphabet.alphabetSize]
        using coordinates.2.isLt
    simp only [acceptedQuotientCount] at quotientLt
    change coordinates.1.val * 5 + coordinates.2.val <
      goldilocksModulus - 1
    simp only [goldilocksModulus]
    omega
  have canonical : value < goldilocksModulus := by
    omega
  ⟨⟨value, canonical⟩, accepted⟩

theorem combineAccepted_factorAccepted (candidate : AcceptedCandidate) :
    combineAccepted (factorAccepted candidate) = candidate := by
  apply Subtype.ext
  apply Fin.ext
  change candidate.val.val / 5 * 5 + candidate.val.val % 5 = candidate.val.val
  simpa [Nat.mul_comm] using Nat.div_add_mod candidate.val.val 5

theorem factorAccepted_combineAccepted
    (coordinates : Fin acceptedQuotientCount × Coefficient) :
    factorAccepted (combineAccepted coordinates) = coordinates := by
  rcases coordinates with ⟨quotient, residue⟩
  have residueLt : residue.val < 5 := by
    exact residue.isLt
  apply Prod.ext
  · apply Fin.ext
    change (quotient.val * 5 + residue.val) / 5 = quotient.val
    omega
  · apply Fin.ext
    change (quotient.val * 5 + residue.val) % 5 = residue.val
    omega

/-- The accepted candidate domain is exactly a product with `Fin 5`.
Therefore a uniform full-field candidate, conditioned on acceptance, gives an
exactly uniform centered digit. -/
theorem acceptedCandidate_exactly_balanced :
    (forall candidate, combineAccepted (factorAccepted candidate) = candidate) /\
      (forall coordinates, factorAccepted (combineAccepted coordinates) = coordinates) :=
  ⟨combineAccepted_factorAccepted, factorAccepted_combineAccepted⟩

/-- Concrete transcript-security event added by the bounded sampler. The four
paper transcript collision classes remain those in
`PaperNonInteractive.TranscriptSecurityEvent`. -/
inductive Poseidon2SecurityEvent (state : State) where
  | boundedSamplerShortfall (failure : SamplerShortfall state)

end Nightstream.Implementation.Transcript.Construction3Poseidon2
