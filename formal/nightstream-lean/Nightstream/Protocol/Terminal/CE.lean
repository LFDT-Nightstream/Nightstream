import Nightstream.SuperNeo.Relations

/-!
Contract: terminal CE authority checked by the native verifier and the
terminal decider relation.

`verifierChildren` are the claims produced by verifier-driven terminal NIFS.
`recordedClaims` and `witnesses` are prover-carried data. Acceptance requires
exact child identity before opening every child against its witness: commitment,
public projection and width, verifier-owned norm bound, evaluation-point shape,
all ring evaluations, constant terms, and implementation sidecars.

The checker contains no field that states its own conclusion. Every conjunct is
computed through `Semantics`, so `terminalCE_sound` is a real executable-checker
theorem rather than an `accepted_implies_valid` assumption.

Maps to:
- `lifecycle::verify::check_running_witnesses_authority`;
- `paper::decider_ce_relation::enforce_final_ce_relations`;
- `engine::decider::enforce_child_core_equal_running` for non-terminal core
  continuity, followed by terminal sidecar attachment and direct validation.
-/

namespace Nightstream.Protocol.TerminalCE

universe uStructure uAssignment uPublicInput uPoint uEvaluation uCommitment
  uScalar uSidecar

/-- Operations executed at the terminal witness-authority boundary. -/
structure Semantics
    (Structure : Type uStructure)
    (Assignment : Type uAssignment)
    (PublicInput : Type uPublicInput)
    (Point : Type uPoint)
    (Evaluation : Type uEvaluation)
    (Commitment : Type uCommitment)
    (Scalar : Type uScalar)
    (Sidecar : Type uSidecar) where
  commit : Assignment → Commitment
  /-- `none` rejects malformed assignment/projection shapes. -/
  projectPublicInput : Nat → Assignment → Option PublicInput
  normBounded : Nat → Assignment → Bool
  evaluationPointValid : Structure → Point → Bool
  /-- `none` rejects malformed matrix/evaluation shapes. -/
  evaluations : Structure → Assignment → Point → Option (List Evaluation)
  constantTerm : Evaluation → Scalar
  /-- NC-channel recomputation, Nebula slice openings, and unsupported-field rejection. -/
  sidecarValid : Structure → Assignment → Sidecar → Bool

/-- One verifier-derived terminal CE child. -/
structure Claim
    (PublicInput : Type uPublicInput)
    (Point : Type uPoint)
    (Evaluation : Type uEvaluation)
    (Commitment : Type uCommitment)
    (Scalar : Type uScalar)
    (Sidecar : Type uSidecar) where
  commitment : Commitment
  publicWidth : Nat
  publicInput : PublicInput
  point : Point
  evaluations : List Evaluation
  constantTerms : List Scalar
  sidecar : Sidecar
deriving Repr, DecidableEq

/-- Verifier-owned relation parameters. The norm bound is not statement data. -/
structure Context (Structure : Type uStructure) where
  relation : Structure
  normBound : Nat
  /-- `none` permits any well-formed width; `some n` pins the deployment width. -/
  expectedPublicWidth : Option Nat

def PublicWidthHolds (expected : Option Nat) (actual : Nat) : Prop :=
  match expected with
  | none => True
  | some width => actual = width

def checkPublicWidth (expected : Option Nat) (actual : Nat) : Bool :=
  match expected with
  | none => true
  | some width => decide (actual = width)

theorem checkPublicWidth_eq_true_iff (expected : Option Nat) (actual : Nat) :
    checkPublicWidth expected actual = true ↔ PublicWidthHolds expected actual := by
  cases expected <;> simp [checkPublicWidth, PublicWidthHolds]

/-- All authority obligations for one child/witness pair. -/
def ClaimHolds
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {Sidecar : Type uSidecar}
    (semantics : Semantics Structure Assignment PublicInput Point Evaluation
      Commitment Scalar Sidecar)
    (context : Context Structure)
    (claim : Claim PublicInput Point Evaluation Commitment Scalar Sidecar)
    (witness : Assignment) : Prop :=
  PublicWidthHolds context.expectedPublicWidth claim.publicWidth ∧
  semantics.commit witness = claim.commitment ∧
  semantics.projectPublicInput claim.publicWidth witness = some claim.publicInput ∧
  semantics.normBounded context.normBound witness = true ∧
  semantics.evaluationPointValid context.relation claim.point = true ∧
  semantics.evaluations context.relation witness claim.point = some claim.evaluations ∧
  claim.evaluations.map semantics.constantTerm = claim.constantTerms ∧
  semantics.sidecarValid context.relation witness claim.sidecar = true

/-- Executable counterpart of `ClaimHolds`, in native Rust rejection order. -/
def checkClaim
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {Sidecar : Type uSidecar}
    [DecidableEq PublicInput]
    [DecidableEq Evaluation]
    [DecidableEq Commitment]
    [DecidableEq Scalar]
    (semantics : Semantics Structure Assignment PublicInput Point Evaluation
      Commitment Scalar Sidecar)
    (context : Context Structure)
    (claim : Claim PublicInput Point Evaluation Commitment Scalar Sidecar)
    (witness : Assignment) : Bool :=
  checkPublicWidth context.expectedPublicWidth claim.publicWidth &&
  decide (semantics.commit witness = claim.commitment) &&
  decide (semantics.projectPublicInput claim.publicWidth witness = some claim.publicInput) &&
  semantics.normBounded context.normBound witness &&
  semantics.evaluationPointValid context.relation claim.point &&
  decide (semantics.evaluations context.relation witness claim.point = some claim.evaluations) &&
  decide (claim.evaluations.map semantics.constantTerm = claim.constantTerms) &&
  semantics.sidecarValid context.relation witness claim.sidecar

theorem checkClaim_eq_true_iff
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {Sidecar : Type uSidecar}
    [DecidableEq PublicInput]
    [DecidableEq Evaluation]
    [DecidableEq Commitment]
    [DecidableEq Scalar]
    (semantics : Semantics Structure Assignment PublicInput Point Evaluation
      Commitment Scalar Sidecar)
    (context : Context Structure)
    (claim : Claim PublicInput Point Evaluation Commitment Scalar Sidecar)
    (witness : Assignment) :
    checkClaim semantics context claim witness = true ↔
      ClaimHolds semantics context claim witness := by
  simp [checkClaim, ClaimHolds, checkPublicWidth_eq_true_iff, and_assoc]

/-- Exact length-preserving child/witness relation. -/
def PairsHold
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {Sidecar : Type uSidecar}
    (semantics : Semantics Structure Assignment PublicInput Point Evaluation
      Commitment Scalar Sidecar)
    (context : Context Structure) :
    List (Claim PublicInput Point Evaluation Commitment Scalar Sidecar) →
      List Assignment → Prop
  | [], [] => True
  | claim :: claims, witness :: witnesses =>
      ClaimHolds semantics context claim witness ∧
        PairsHold semantics context claims witnesses
  | _, _ => False

def checkPairs
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {Sidecar : Type uSidecar}
    [DecidableEq PublicInput]
    [DecidableEq Evaluation]
    [DecidableEq Commitment]
    [DecidableEq Scalar]
    (semantics : Semantics Structure Assignment PublicInput Point Evaluation
      Commitment Scalar Sidecar)
    (context : Context Structure) :
    List (Claim PublicInput Point Evaluation Commitment Scalar Sidecar) →
      List Assignment → Bool
  | [], [] => true
  | claim :: claims, witness :: witnesses =>
      checkClaim semantics context claim witness &&
        checkPairs semantics context claims witnesses
  | _, _ => false

theorem checkPairs_eq_true_iff
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {Sidecar : Type uSidecar}
    [DecidableEq PublicInput]
    [DecidableEq Evaluation]
    [DecidableEq Commitment]
    [DecidableEq Scalar]
    (semantics : Semantics Structure Assignment PublicInput Point Evaluation
      Commitment Scalar Sidecar)
    (context : Context Structure)
    (claims : List (Claim PublicInput Point Evaluation Commitment Scalar Sidecar))
    (witnesses : List Assignment) :
    checkPairs semantics context claims witnesses = true ↔
      PairsHold semantics context claims witnesses := by
  induction claims generalizing witnesses with
  | nil => cases witnesses <;> simp [checkPairs, PairsHold]
  | cons claim claims inductionHypothesis =>
      cases witnesses with
      | nil => simp [checkPairs, PairsHold]
      | cons witness witnesses =>
          simp [checkPairs, PairsHold, checkClaim_eq_true_iff,
            inductionHypothesis]

/-- Full terminal input. `recordedClaims` cannot substitute for verifier children. -/
structure Instance
    (Structure : Type uStructure)
    (Assignment : Type uAssignment)
    (PublicInput : Type uPublicInput)
    (Point : Type uPoint)
    (Evaluation : Type uEvaluation)
    (Commitment : Type uCommitment)
    (Scalar : Type uScalar)
    (Sidecar : Type uSidecar) where
  context : Context Structure
  verifierChildren : List (Claim PublicInput Point Evaluation Commitment Scalar Sidecar)
  recordedClaims : List (Claim PublicInput Point Evaluation Commitment Scalar Sidecar)
  witnesses : List Assignment

/-- Terminal acceptance truth: child authority plus every CE opening. -/
def Holds
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {Sidecar : Type uSidecar}
    (semantics : Semantics Structure Assignment PublicInput Point Evaluation
      Commitment Scalar Sidecar)
    (terminal : Instance Structure Assignment PublicInput Point Evaluation
      Commitment Scalar Sidecar) : Prop :=
  terminal.recordedClaims = terminal.verifierChildren ∧
  PairsHold semantics terminal.context terminal.verifierChildren terminal.witnesses

/-- Executable terminal checker used as the theorem-facing acceptance surface. -/
def check
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {Sidecar : Type uSidecar}
    [DecidableEq Assignment]
    [DecidableEq PublicInput]
    [DecidableEq Point]
    [DecidableEq Evaluation]
    [DecidableEq Commitment]
    [DecidableEq Scalar]
    [DecidableEq Sidecar]
    (semantics : Semantics Structure Assignment PublicInput Point Evaluation
      Commitment Scalar Sidecar)
    (terminal : Instance Structure Assignment PublicInput Point Evaluation
      Commitment Scalar Sidecar) : Bool :=
  decide (terminal.recordedClaims = terminal.verifierChildren) &&
    checkPairs semantics terminal.context terminal.verifierChildren terminal.witnesses

theorem check_eq_true_iff
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {Sidecar : Type uSidecar}
    [DecidableEq Assignment]
    [DecidableEq PublicInput]
    [DecidableEq Point]
    [DecidableEq Evaluation]
    [DecidableEq Commitment]
    [DecidableEq Scalar]
    [DecidableEq Sidecar]
    (semantics : Semantics Structure Assignment PublicInput Point Evaluation
      Commitment Scalar Sidecar)
    (terminal : Instance Structure Assignment PublicInput Point Evaluation
      Commitment Scalar Sidecar) :
    check semantics terminal = true ↔ Holds semantics terminal := by
  simp [check, Holds, checkPairs_eq_true_iff]

/-- `TERM-CE`: computed terminal acceptance establishes every named authority obligation. -/
theorem terminalCE_sound
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {Sidecar : Type uSidecar}
    [DecidableEq Assignment]
    [DecidableEq PublicInput]
    [DecidableEq Point]
    [DecidableEq Evaluation]
    [DecidableEq Commitment]
    [DecidableEq Scalar]
    [DecidableEq Sidecar]
    (semantics : Semantics Structure Assignment PublicInput Point Evaluation
      Commitment Scalar Sidecar)
    (terminal : Instance Structure Assignment PublicInput Point Evaluation
      Commitment Scalar Sidecar)
    (accepted : check semantics terminal = true) :
    Holds semantics terminal :=
  (check_eq_true_iff semantics terminal).1 accepted

theorem terminalCE_complete
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {Sidecar : Type uSidecar}
    [DecidableEq Assignment]
    [DecidableEq PublicInput]
    [DecidableEq Point]
    [DecidableEq Evaluation]
    [DecidableEq Commitment]
    [DecidableEq Scalar]
    [DecidableEq Sidecar]
    (semantics : Semantics Structure Assignment PublicInput Point Evaluation
      Commitment Scalar Sidecar)
    (terminal : Instance Structure Assignment PublicInput Point Evaluation
      Commitment Scalar Sidecar)
    (holds : Holds semantics terminal) :
    check semantics terminal = true :=
  (check_eq_true_iff semantics terminal).2 holds

end Nightstream.Protocol.TerminalCE
