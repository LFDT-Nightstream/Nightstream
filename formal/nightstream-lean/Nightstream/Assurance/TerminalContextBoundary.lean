import Nightstream.SuperNeo.CheckPlan

/-!
Exact control-flow model for the four terminal context guards.

Assurance tier: model-level.

Owns: recursive-induction certification, plain-chain selection, public-width
agreement, relation-structure agreement, and one removal counterexample for
each guard.

Does not own: construction of the recursive relation, Rust preprocessing
refinement, manifest parsing, exact matrix equality, or a deployed verifier
key. Separate artifact and Rust-conformance theorems must supply those facts.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Assurance.TerminalContextBoundary

open Nightstream.SuperNeo.CheckPlan

inductive Guard where
  | induction
  | plainChain
  | publicWidth
  | relationStructure
deriving DecidableEq, Repr

def guardName : Guard → String
  | .induction => "terminal.context.induction"
  | .plainChain => "terminal.context.plain_chain"
  | .publicWidth => "terminal.context.public_width"
  | .relationStructure => "terminal.context.relation_structure"

def guards : List Guard :=
  [.induction, .plainChain, .publicWidth, .relationStructure]

def guardNames : List String := guards.map guardName

theorem guardNames_exact :
    guardNames = [
      "terminal.context.induction",
      "terminal.context.plain_chain",
      "terminal.context.public_width",
      "terminal.context.relation_structure"
    ] := by
  rfl

structure Candidate where
  inductionCertified : Bool
  nebulaPresent : Bool
  publicWidth : Nat
  expectedPublicWidth : Nat
  relationStructureMatches : Bool
deriving DecidableEq, Repr

def semantics : Guard → Candidate → Prop
  | .induction, candidate => candidate.inductionCertified = true
  | .plainChain, candidate => candidate.nebulaPresent = false
  | .publicWidth, candidate =>
      candidate.publicWidth = candidate.expectedPublicWidth
  | .relationStructure, candidate =>
      candidate.relationStructureMatches = true

def Target (candidate : Candidate) : Prop :=
  candidate.inductionCertified = true ∧
  candidate.nebulaPresent = false ∧
  candidate.publicWidth = candidate.expectedPublicWidth ∧
  candidate.relationStructureMatches = true

private instance targetDecidable (candidate : Candidate) :
    Decidable (Target candidate) := by
  unfold Target
  infer_instance

def verify (candidate : Candidate) : Bool := decide (Target candidate)

theorem accepts_iff_target (candidate : Candidate) :
    Accepts semantics guards candidate ↔ Target candidate := by
  simp [Accepts, guards, semantics, Target]

theorem verify_eq_true_iff_target (candidate : Candidate) :
    verify candidate = true ↔ Target candidate := by
  simp [verify]

def valid : Candidate where
  inductionCertified := true
  nebulaPresent := false
  publicWidth := 270
  expectedPublicWidth := 270
  relationStructureMatches := true

def removalWitness : Guard → Candidate
  | .induction => { valid with inductionCertified := false }
  | .plainChain => { valid with nebulaPresent := true }
  | .publicWidth => { valid with publicWidth := 269 }
  | .relationStructure => { valid with relationStructureMatches := false }

theorem removalWitness_accepts_without (removed : Guard) :
    Accepts semantics (without guards removed) (removalWitness removed) := by
  cases removed <;>
    intro retained member <;>
    cases retained <;>
    simp [without, guards, semantics, removalWitness, valid] at member ⊢

theorem removalWitness_rejects_target (removed : Guard) :
    ¬ Target (removalWitness removed) := by
  cases removed <;> simp [Target, removalWitness, valid]

theorem retained_necessary (guard : Guard) :
    NecessaryForSoundness semantics Target guards guard :=
  ⟨removalWitness guard, removalWitness_accepts_without guard,
    removalWitness_rejects_target guard⟩

/-- Model-level inclusion-minimality of the exact four-check context boundary. -/
theorem inclusionMinimalSound :
    InclusionMinimalSound semantics Target guards := by
  apply inclusionMinimalSound_of_witnesses
  · intro candidate accepted
    exact (accepts_iff_target candidate).1 accepted
  · intro guard _member
    exact retained_necessary guard

end Nightstream.Assurance.TerminalContextBoundary
