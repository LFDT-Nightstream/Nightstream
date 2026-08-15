import Nightstream.SuperNeo.CheckPlan

/-!
Exact control-flow model for the three terminal proof-boundary guards.

Assurance tier: model-level.

Owns: the expected-image check, backend-success check, rebuilt-public-statement
check, their executable conjunction, and one removal counterexample for each
guard.

Does not own: Spartan or WHIR soundness, proof parsing, Rust refinement, the
terminal R1CS, verifier-native statement checks, or a deployed verifier key.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Assurance.TerminalProofBoundary

open Nightstream.SuperNeo.CheckPlan

/-- Exact proof-boundary guard vocabulary used by the Rust terminal verifier. -/
inductive Guard where
  | expectedPublicImage
  | backendVerification
  | publicStatement
deriving DecidableEq, Repr

def guardName : Guard → String
  | .expectedPublicImage => "terminal.proof.expected_public_image"
  | .backendVerification => "terminal.proof.spartan_verification"
  | .publicStatement => "terminal.proof.public_statement"

def guards : List Guard :=
  [.expectedPublicImage, .backendVerification, .publicStatement]

def guardNames : List String := guards.map guardName

theorem guardNames_exact :
    guardNames = [
      "terminal.proof.expected_public_image",
      "terminal.proof.spartan_verification",
      "terminal.proof.public_statement"
    ] := by
  rfl

/-- Minimal data visible at the proof-verification boundary. `none` means that
the backend rejected the proof. -/
structure Candidate where
  expectedImage : Bool
  statementImage : Bool
  expectedPublic : Bool
  backendResult : Option Bool
deriving DecidableEq, Repr

/-- One independent predicate for each Rust proof-boundary guard. -/
def semantics : Guard → Candidate → Prop
  | .expectedPublicImage, candidate =>
      candidate.statementImage = candidate.expectedImage
  | .backendVerification, candidate =>
      candidate.backendResult.isSome = true
  | .publicStatement, candidate =>
      match candidate.backendResult with
      | none => True
      | some verifiedPublic => verifiedPublic = candidate.expectedPublic

/-- Independent target relation for the complete proof boundary. -/
def Target (candidate : Candidate) : Prop :=
  candidate.statementImage = candidate.expectedImage ∧
    match candidate.backendResult with
    | none => False
    | some verifiedPublic => verifiedPublic = candidate.expectedPublic

/-- The same branch order as `verify_spartan`: bind the expected image, reject
backend failure, then compare the verified and rebuilt public vectors. -/
def verify (candidate : Candidate) : Bool :=
  decide (candidate.statementImage = candidate.expectedImage) &&
    match candidate.backendResult with
    | none => false
    | some verifiedPublic => decide (verifiedPublic = candidate.expectedPublic)

theorem accepts_iff_target (candidate : Candidate) :
    Accepts semantics guards candidate ↔ Target candidate := by
  rcases candidate with ⟨expectedImage, statementImage, expectedPublic,
    backendResult⟩
  cases backendResult <;>
    simp [Accepts, guards, semantics, Target]

theorem verify_eq_true_iff_target (candidate : Candidate) :
    verify candidate = true ↔ Target candidate := by
  rcases candidate with ⟨expectedImage, statementImage, expectedPublic,
    backendResult⟩
  cases backendResult <;>
    simp [verify, Target]

def expectedPublicImageWitness : Candidate where
  expectedImage := false
  statementImage := true
  expectedPublic := false
  backendResult := some false

def backendVerificationWitness : Candidate where
  expectedImage := false
  statementImage := false
  expectedPublic := false
  backendResult := none

def publicStatementWitness : Candidate where
  expectedImage := false
  statementImage := false
  expectedPublic := false
  backendResult := some true

theorem expectedPublicImage_necessary :
    NecessaryForSoundness semantics Target guards .expectedPublicImage := by
  refine ⟨expectedPublicImageWitness, ?_, ?_⟩
  · simp [Accepts, without, guards, semantics, expectedPublicImageWitness]
  · simp [Target, expectedPublicImageWitness]

theorem backendVerification_necessary :
    NecessaryForSoundness semantics Target guards .backendVerification := by
  refine ⟨backendVerificationWitness, ?_, ?_⟩
  · simp [Accepts, without, guards, semantics, backendVerificationWitness]
  · simp [Target, backendVerificationWitness]

theorem publicStatement_necessary :
    NecessaryForSoundness semantics Target guards .publicStatement := by
  refine ⟨publicStatementWitness, ?_, ?_⟩
  · simp [Accepts, without, guards, semantics, publicStatementWitness]
  · simp [Target, publicStatementWitness]

theorem retained_necessary
    (guard : Guard)
    (_member : guard ∈ guards) :
    NecessaryForSoundness semantics Target guards guard := by
  cases guard with
  | expectedPublicImage => exact expectedPublicImage_necessary
  | backendVerification => exact backendVerification_necessary
  | publicStatement => exact publicStatement_necessary

/-- Model-level inclusion-minimality of the exact three-check proof boundary. -/
theorem inclusionMinimalSound :
    InclusionMinimalSound semantics Target guards := by
  apply inclusionMinimalSound_of_witnesses
  · intro candidate accepted
    exact (accepts_iff_target candidate).1 accepted
  · exact retained_necessary

end Nightstream.Assurance.TerminalProofBoundary
