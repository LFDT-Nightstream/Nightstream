/-!
Rust-shaped control flow for verifier-native terminal guards.

Assurance tier: model-level. This module fixes the check order and first-error
behavior used by the terminal verifier. The Rust-to-Lean translation remains a
source-review and drift boundary.

Owns: the four context results, eleven statement results, three proof results,
their named rejection classes, and the exact verification order.

Does not own: the computation of a result, Rust source refinement, the
terminal R1CS, Spartan or WHIR soundness, or cryptographic assumptions.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Rust.TerminalNativeGuards

inductive Error where
  | contextInduction
  | contextPlainChain
  | contextPublicWidth
  | contextRelationStructure
  | statementRunningClaimCount
  | statementVerifierKey
  | statementInitialSemanticState
  | statementInitialBoundary
  | statementProgramCounter
  | statementCounters
  | statementFreshBoundary
  | statementRunningAccumulator
  | statementSemanticState
  | statementStateXOut
  | statementFreshPublicLink
  | proofExpectedPublicImage
  | proofSpartanVerification
  | proofPublicStatement
deriving DecidableEq, Repr

def runChecks : List (Bool × Error) → Except Error Unit
  | [] => .ok ()
  | (true, _) :: tail => runChecks tail
  | (false, error) :: _ => .error error

theorem runChecks_eq_ok_iff (checks : List (Bool × Error)) :
    runChecks checks = .ok () ↔
      checks.all (fun check => check.1) = true := by
  induction checks with
  | nil => simp [runChecks]
  | cons check tail inductionHypothesis =>
      rcases check with ⟨accepted, error⟩
      cases accepted <;> simp [runChecks, inductionHypothesis]

structure ContextChecks where
  inductionCertified : Bool
  plainChain : Bool
  publicWidth : Bool
  relationStructure : Bool
deriving DecidableEq, Repr

structure StatementChecks where
  runningClaimCount : Bool
  verifierKey : Bool
  initialSemanticState : Bool
  initialBoundary : Bool
  programCounter : Bool
  counters : Bool
  freshBoundary : Bool
  runningAccumulator : Bool
  semanticState : Bool
  stateXOut : Bool
  freshPublicLink : Bool
deriving DecidableEq, Repr

structure ProofChecks where
  expectedPublicImage : Bool
  spartanVerification : Bool
  publicStatement : Bool
deriving DecidableEq, Repr

def contextProgram (checks : ContextChecks) : List (Bool × Error) := [
  (checks.inductionCertified, .contextInduction),
  (checks.plainChain, .contextPlainChain),
  (checks.publicWidth, .contextPublicWidth),
  (checks.relationStructure, .contextRelationStructure)
]

def statementProgram (checks : StatementChecks) : List (Bool × Error) := [
  (checks.runningClaimCount, .statementRunningClaimCount),
  (checks.verifierKey, .statementVerifierKey),
  (checks.initialSemanticState, .statementInitialSemanticState),
  (checks.initialBoundary, .statementInitialBoundary),
  (checks.programCounter, .statementProgramCounter),
  (checks.counters, .statementCounters),
  (checks.freshBoundary, .statementFreshBoundary),
  (checks.runningAccumulator, .statementRunningAccumulator),
  (checks.semanticState, .statementSemanticState),
  (checks.stateXOut, .statementStateXOut),
  (checks.freshPublicLink, .statementFreshPublicLink)
]

/-- Exact `verify_spartan` guard order. Statement validation occurs after the
expected-image check and before backend verification. -/
def verifierProgram
    (context : ContextChecks)
    (statement : StatementChecks)
    (proof : ProofChecks) : List (Bool × Error) :=
  contextProgram context ++
    [(proof.expectedPublicImage, .proofExpectedPublicImage)] ++
      statementProgram statement ++
        [ (proof.spartanVerification, .proofSpartanVerification)
        , (proof.publicStatement, .proofPublicStatement)
        ]

def verify
    (context : ContextChecks)
    (statement : StatementChecks)
    (proof : ProofChecks) : Except Error Unit :=
  runChecks (verifierProgram context statement proof)

def ContextAccepted (context : ContextChecks) : Prop :=
  context.inductionCertified = true ∧
    context.plainChain = true ∧
      context.publicWidth = true ∧
        context.relationStructure = true

def StatementAccepted (statement : StatementChecks) : Prop :=
  statement.runningClaimCount = true ∧
    statement.verifierKey = true ∧
      statement.initialSemanticState = true ∧
        statement.initialBoundary = true ∧
          statement.programCounter = true ∧
            statement.counters = true ∧
              statement.freshBoundary = true ∧
                statement.runningAccumulator = true ∧
                  statement.semanticState = true ∧
                    statement.stateXOut = true ∧
                      statement.freshPublicLink = true

def ProofAccepted (proof : ProofChecks) : Prop :=
  proof.expectedPublicImage = true ∧
    proof.spartanVerification = true ∧
      proof.publicStatement = true

theorem verify_eq_ok_iff
    (context : ContextChecks)
    (statement : StatementChecks)
    (proof : ProofChecks) :
    verify context statement proof = .ok () ↔
      ContextAccepted context ∧
        StatementAccepted statement ∧
          ProofAccepted proof := by
  simp [verify, runChecks_eq_ok_iff, verifierProgram, contextProgram,
    statementProgram, ContextAccepted, StatementAccepted, ProofAccepted,
    and_assoc, and_left_comm, and_comm]

end Nightstream.Implementation.Rust.TerminalNativeGuards
