import Nightstream.Assurance.TerminalContextBoundary
import Nightstream.Assurance.TerminalProofBoundary
import Nightstream.Assurance.TerminalStatementBoundary
import Nightstream.Implementation.R1CS.Artifacts.TerminalVerifierNativeGuards
import Nightstream.Implementation.Rust.TerminalNativeGuards

/-!
Combined inclusion-minimal plan for all verifier-native terminal guards.

Assurance tier: model-level for semantic exactness and removal witnesses;
artifact-checked for equality with the ordered Rust-generated guard ledger.

Owns: the combined 18-guard vocabulary, exact acceptance target, Rust-shaped
execution-order refinement, and one isolated removal witness for each guard.

Does not own: universal Rust source refinement, expected-value computation,
the terminal R1CS, Spartan or WHIR soundness, digest security, or a deployed
verifier key.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Assurance.TerminalNativeGuardPlan

open Nightstream.SuperNeo.CheckPlan
open Nightstream.Implementation.Rust

inductive Guard where
  | context (guard : TerminalContextBoundary.Guard)
  | statement (guard : TerminalStatementBoundary.Guard)
  | proof (guard : TerminalProofBoundary.Guard)
deriving DecidableEq, Repr

structure Candidate where
  context : TerminalContextBoundary.Candidate
  statement : TerminalStatementBoundary.Candidate
  proof : TerminalProofBoundary.Candidate
deriving DecidableEq, Repr

def semantics : Guard → Candidate → Prop
  | .context guard, candidate =>
      TerminalContextBoundary.semantics guard candidate.context
  | .statement guard, candidate =>
      TerminalStatementBoundary.semantics guard candidate.statement
  | .proof guard, candidate =>
      TerminalProofBoundary.semantics guard candidate.proof

def guards : List Guard :=
  TerminalContextBoundary.guards.map .context ++
    TerminalStatementBoundary.guards.map .statement ++
      TerminalProofBoundary.guards.map .proof

def guardName : Guard → String
  | .context guard => TerminalContextBoundary.guardName guard
  | .statement guard => TerminalStatementBoundary.guardName guard
  | .proof guard => TerminalProofBoundary.guardName guard

def guardNames : List String := guards.map guardName

def Target (candidate : Candidate) : Prop :=
  TerminalContextBoundary.Target candidate.context ∧
    TerminalStatementBoundary.Target candidate.statement ∧
      TerminalProofBoundary.Target candidate.proof

def contextChecks
    (candidate : TerminalContextBoundary.Candidate) :
    TerminalNativeGuards.ContextChecks where
  inductionCertified := candidate.inductionCertified
  plainChain := !candidate.nebulaPresent
  publicWidth := decide
    (candidate.publicWidth = candidate.expectedPublicWidth)
  relationStructure := candidate.relationStructureMatches

def statementChecks
    (candidate : TerminalStatementBoundary.Candidate) :
    TerminalNativeGuards.StatementChecks where
  runningClaimCount := decide
    (candidate.runningClaimCount = candidate.expectedRunningClaimCount)
  verifierKey := decide
    (candidate.verifierKey = candidate.expectedVerifierKey)
  initialSemanticState := decide
    (candidate.initialSemanticState = candidate.expectedInitialSemanticState)
  initialBoundary := decide
    (candidate.initialBoundary = candidate.expectedInitialBoundary)
  programCounter := decide (candidate.programCounter = 1)
  counters := decide
    (candidate.chunkCount ≠ 0 ∧ candidate.stepCount ≠ 0 ∧
      candidate.chunkCount = candidate.stepCount)
  freshBoundary := decide
    (candidate.freshBoundary = candidate.expectedFreshBoundary ∧
      candidate.publicTrace = candidate.expectedFreshBoundary)
  runningAccumulator := decide
    (candidate.runningAccumulator = candidate.expectedRunningAccumulator)
  semanticState := decide
    (candidate.stateless = false ∨
      candidate.semanticState = candidate.runningAccumulator)
  stateXOut := decide
    (candidate.stateXOut = candidate.expectedStateXOut)
  freshPublicLink := candidate.freshPublicLinkAccepted

def proofChecks
    (candidate : TerminalProofBoundary.Candidate) :
    TerminalNativeGuards.ProofChecks where
  expectedPublicImage := decide
    (candidate.statementImage = candidate.expectedImage)
  spartanVerification := candidate.backendResult.isSome
  publicStatement :=
    match candidate.backendResult with
    | none => true
    | some verifiedPublic => decide
        (verifiedPublic = candidate.expectedPublic)

def verify (candidate : Candidate) : Except TerminalNativeGuards.Error Unit :=
  TerminalNativeGuards.verify
    (contextChecks candidate.context)
    (statementChecks candidate.statement)
    (proofChecks candidate.proof)

theorem accepts_iff_target (candidate : Candidate) :
    Accepts semantics guards candidate ↔ Target candidate := by
  constructor
  · intro accepted
    have contextAccepted :
        Accepts TerminalContextBoundary.semantics
          TerminalContextBoundary.guards candidate.context := by
      intro guard member
      exact accepted (.context guard) (by simp [guards, member])
    have statementAccepted :
        Accepts TerminalStatementBoundary.semantics
          TerminalStatementBoundary.guards candidate.statement := by
      intro guard member
      exact accepted (.statement guard) (by simp [guards, member])
    have proofAccepted :
        Accepts TerminalProofBoundary.semantics
          TerminalProofBoundary.guards candidate.proof := by
      intro guard member
      exact accepted (.proof guard) (by simp [guards, member])
    exact ⟨
      (TerminalContextBoundary.accepts_iff_target candidate.context).1
        contextAccepted,
      (TerminalStatementBoundary.accepts_iff_target candidate.statement).1
        statementAccepted,
      (TerminalProofBoundary.accepts_iff_target candidate.proof).1
        proofAccepted⟩
  · rintro ⟨contextTarget, statementTarget, proofTarget⟩ guard _member
    have contextAccepted :=
      (TerminalContextBoundary.accepts_iff_target candidate.context).2
        contextTarget
    have statementAccepted :=
      (TerminalStatementBoundary.accepts_iff_target candidate.statement).2
        statementTarget
    have proofAccepted :=
      (TerminalProofBoundary.accepts_iff_target candidate.proof).2 proofTarget
    cases guard with
    | context guard =>
        exact contextAccepted guard (by
          cases guard <;> simp [TerminalContextBoundary.guards])
    | statement guard =>
        exact statementAccepted guard (by
          cases guard <;> simp [TerminalStatementBoundary.guards])
    | proof guard =>
        exact proofAccepted guard (by
          cases guard <;> simp [TerminalProofBoundary.guards])

theorem contextChecks_accepted_iff_target
    (candidate : TerminalContextBoundary.Candidate) :
    TerminalNativeGuards.ContextAccepted (contextChecks candidate) ↔
      TerminalContextBoundary.Target candidate := by
  simp [TerminalNativeGuards.ContextAccepted, contextChecks,
    TerminalContextBoundary.Target]

theorem statementChecks_accepted_iff_target
    (candidate : TerminalStatementBoundary.Candidate) :
    TerminalNativeGuards.StatementAccepted (statementChecks candidate) ↔
      TerminalStatementBoundary.Target candidate := by
  simp [TerminalNativeGuards.StatementAccepted, statementChecks,
    TerminalStatementBoundary.Target, and_assoc]

theorem proofChecks_accepted_iff_target
    (candidate : TerminalProofBoundary.Candidate) :
    TerminalNativeGuards.ProofAccepted (proofChecks candidate) ↔
      TerminalProofBoundary.Target candidate := by
  rcases candidate with
    ⟨expectedImage, statementImage, expectedPublic, backendResult⟩
  cases backendResult <;>
    simp [TerminalNativeGuards.ProofAccepted, proofChecks,
      TerminalProofBoundary.Target]

/-- Model-level: the Rust-shaped first-error program accepts exactly the
combined target, including the interleaved expected-image check. -/
theorem verify_eq_ok_iff_target (candidate : Candidate) :
    verify candidate = .ok () ↔ Target candidate := by
  rw [verify, TerminalNativeGuards.verify_eq_ok_iff]
  rw [contextChecks_accepted_iff_target,
    statementChecks_accepted_iff_target,
    proofChecks_accepted_iff_target]
  rfl

def proofValid : TerminalProofBoundary.Candidate where
  expectedImage := false
  statementImage := false
  expectedPublic := false
  backendResult := some false

def valid : Candidate where
  context := TerminalContextBoundary.valid
  statement := TerminalStatementBoundary.valid
  proof := proofValid

def removalWitness : Guard → Candidate
  | .context guard =>
      { valid with context := TerminalContextBoundary.removalWitness guard }
  | .statement guard =>
      { valid with statement := TerminalStatementBoundary.removalWitness guard }
  | .proof .expectedPublicImage =>
      { valid with proof := TerminalProofBoundary.expectedPublicImageWitness }
  | .proof .backendVerification =>
      { valid with proof := TerminalProofBoundary.backendVerificationWitness }
  | .proof .publicStatement =>
      { valid with proof := TerminalProofBoundary.publicStatementWitness }

theorem removalWitness_accepts_without (removed : Guard) :
    Accepts semantics (without guards removed) (removalWitness removed) := by
  cases removed with
  | context guard =>
      cases guard <;>
        simp [Accepts, without, guards, semantics, removalWitness, valid,
          proofValid, TerminalContextBoundary.guards,
          TerminalContextBoundary.semantics,
          TerminalContextBoundary.removalWitness,
          TerminalContextBoundary.valid,
          TerminalStatementBoundary.guards,
          TerminalStatementBoundary.semantics,
          TerminalStatementBoundary.valid,
          TerminalProofBoundary.guards,
          TerminalProofBoundary.semantics]
  | statement guard =>
      cases guard <;>
        simp [Accepts, without, guards, semantics, removalWitness, valid,
          proofValid, TerminalContextBoundary.guards,
          TerminalContextBoundary.semantics,
          TerminalContextBoundary.valid,
          TerminalStatementBoundary.guards,
          TerminalStatementBoundary.semantics,
          TerminalStatementBoundary.removalWitness,
          TerminalStatementBoundary.valid,
          TerminalProofBoundary.guards,
          TerminalProofBoundary.semantics]
  | proof guard =>
      cases guard <;>
        simp [Accepts, without, guards, semantics, removalWitness, valid,
          proofValid, TerminalContextBoundary.guards,
          TerminalContextBoundary.semantics,
          TerminalContextBoundary.valid,
          TerminalStatementBoundary.guards,
          TerminalStatementBoundary.semantics,
          TerminalStatementBoundary.valid,
          TerminalProofBoundary.guards,
          TerminalProofBoundary.semantics,
          TerminalProofBoundary.expectedPublicImageWitness,
          TerminalProofBoundary.backendVerificationWitness,
          TerminalProofBoundary.publicStatementWitness]

theorem removalWitness_rejects_target (removed : Guard) :
    ¬ Target (removalWitness removed) := by
  cases removed with
  | context guard =>
      cases guard <;>
        simp [Target, removalWitness, valid, proofValid,
          TerminalContextBoundary.Target,
          TerminalContextBoundary.removalWitness,
          TerminalContextBoundary.valid]
  | statement guard =>
      cases guard <;>
        simp [Target, removalWitness, valid, proofValid,
          TerminalStatementBoundary.Target,
          TerminalStatementBoundary.removalWitness,
          TerminalStatementBoundary.valid]
  | proof guard =>
      cases guard <;>
        simp [Target, removalWitness, valid, proofValid,
          TerminalProofBoundary.Target,
          TerminalProofBoundary.expectedPublicImageWitness,
          TerminalProofBoundary.backendVerificationWitness,
          TerminalProofBoundary.publicStatementWitness]

theorem retained_necessary (guard : Guard) :
    NecessaryForSoundness semantics Target guards guard :=
  ⟨removalWitness guard, removalWitness_accepts_without guard,
    removalWitness_rejects_target guard⟩

/-- Model-level inclusion-minimality of the combined 18-check native plan. -/
theorem inclusionMinimalSound :
    InclusionMinimalSound semantics Target guards := by
  apply inclusionMinimalSound_of_witnesses
  · intro candidate accepted
    exact (accepts_iff_target candidate).1 accepted
  · intro guard _member
    exact retained_necessary guard

/-- Artifact-checked: the reviewed plan names and order are the exact
Rust-generated terminal native guard ledger. -/
theorem artifact_guard_names_exact :
    Nightstream.Implementation.R1CS.Artifacts.TerminalVerifierNativeGuards.names =
      guardNames := by
  rfl

/-- Artifact-checked name binding plus a model-level checked removal witness
for every name in the Rust-generated ledger. -/
theorem artifact_name_has_removal_witness
    (name : String)
    (member : name ∈
      Nightstream.Implementation.R1CS.Artifacts.TerminalVerifierNativeGuards.names) :
    ∃ guard, guard ∈ guards ∧ guardName guard = name ∧
      NecessaryForSoundness semantics Target guards guard := by
  rw [artifact_guard_names_exact] at member
  rcases List.mem_map.mp member with ⟨guard, guardMember, nameEqual⟩
  exact ⟨guard, guardMember, nameEqual, retained_necessary guard⟩

end Nightstream.Assurance.TerminalNativeGuardPlan
