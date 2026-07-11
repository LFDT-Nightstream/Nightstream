import Nightstream.Protocol.Terminal.CE

/-!
Contract: Rust-shaped control-flow refinement for terminal CE verification.

Unlike the Boolean theorem-facing checker, `verify` retains the first named
rejection class in the same authority order as the native verifier. The two
independent executable presentations are proved equivalent to the same
`TerminalCE.Holds` predicate. This covers both success and rejection paths;
there is no catch-all success assumption.

The Rust-to-Lean translation of primitive operations remains a trusted M5
boundary. Production conformance tests and source drift hashes own that
translation boundary.
-/

namespace Nightstream.Implementation.Rust.Terminal

open Nightstream.Protocol

universe uStructure uAssignment uPublicInput uPoint uEvaluation uCommitment
  uScalar uSidecar

/-- Stable rejection classes exposed by the refinement model. -/
inductive Error where
  | childAuthority
  | witnessCount
  | publicWidth
  | commitment
  | publicProjection
  | norm
  | evaluationPoint
  | evaluations
  | constantTerms
  | sidecar
deriving Repr, DecidableEq

section

variable
  {Structure : Type uStructure}
  {Assignment : Type uAssignment}
  {PublicInput : Type uPublicInput}
  {Point : Type uPoint}
  {Evaluation : Type uEvaluation}
  {Commitment : Type uCommitment}
  {Scalar : Type uScalar}
  {Sidecar : Type uSidecar}

abbrev RustSemantics
    (Structure : Type uStructure)
    (Assignment : Type uAssignment)
    (PublicInput : Type uPublicInput)
    (Point : Type uPoint)
    (Evaluation : Type uEvaluation)
    (Commitment : Type uCommitment)
    (Scalar : Type uScalar)
    (Sidecar : Type uSidecar) :=
  TerminalCE.Semantics Structure Assignment PublicInput Point Evaluation
    Commitment Scalar Sidecar

abbrev RustClaim
    (PublicInput : Type uPublicInput)
    (Point : Type uPoint)
    (Evaluation : Type uEvaluation)
    (Commitment : Type uCommitment)
    (Scalar : Type uScalar)
    (Sidecar : Type uSidecar) :=
  TerminalCE.Claim PublicInput Point Evaluation Commitment Scalar Sidecar

abbrev RustInstance
    (Structure : Type uStructure)
    (Assignment : Type uAssignment)
    (PublicInput : Type uPublicInput)
    (Point : Type uPoint)
    (Evaluation : Type uEvaluation)
    (Commitment : Type uCommitment)
    (Scalar : Type uScalar)
    (Sidecar : Type uSidecar) :=
  TerminalCE.Instance Structure Assignment PublicInput Point Evaluation
    Commitment Scalar Sidecar

/-- Execute ordered Boolean guards and retain the first named failure. -/
def runChecks : List (Bool × Error) → Except Error Unit
  | [] => .ok ()
  | (true, _) :: checks => runChecks checks
  | (false, error) :: _ => .error error

theorem runChecks_eq_ok_iff (checks : List (Bool × Error)) :
    runChecks checks = .ok () ↔ checks.all (fun check => check.1) = true := by
  induction checks with
  | nil => simp [runChecks]
  | cons check checks inductionHypothesis =>
      cases check with
      | mk passes error =>
          cases passes <;> simp [runChecks, inductionHypothesis]

/-- Ordered native checks for one terminal claim opening. -/
def claimChecks
    [DecidableEq PublicInput]
    [DecidableEq Evaluation]
    [DecidableEq Commitment]
    [DecidableEq Scalar]
    (semantics : RustSemantics Structure Assignment PublicInput Point Evaluation
      Commitment Scalar Sidecar)
    (context : TerminalCE.Context Structure)
    (claim : RustClaim PublicInput Point Evaluation Commitment Scalar Sidecar)
    (witness : Assignment) : List (Bool × Error) :=
  [ (TerminalCE.checkPublicWidth context.expectedPublicWidth claim.publicWidth,
      .publicWidth)
  , (decide (semantics.commit witness = claim.commitment), .commitment)
  , (decide (semantics.projectPublicInput claim.publicWidth witness = some claim.publicInput),
      .publicProjection)
  , (semantics.normBounded context.normBound witness, .norm)
  , (semantics.evaluationPointValid context.relation claim.point, .evaluationPoint)
  , (decide (semantics.evaluations context.relation witness claim.point = some claim.evaluations),
      .evaluations)
  , (decide (claim.evaluations.map semantics.constantTerm = claim.constantTerms),
      .constantTerms)
  , (semantics.sidecarValid context.relation witness claim.sidecar, .sidecar)
  ]

/-- One native claim-opening check, with the first failure retained. -/
def verifyClaim
    [DecidableEq PublicInput]
    [DecidableEq Evaluation]
    [DecidableEq Commitment]
    [DecidableEq Scalar]
    (semantics : RustSemantics Structure Assignment PublicInput Point Evaluation
      Commitment Scalar Sidecar)
    (context : TerminalCE.Context Structure)
    (claim : RustClaim PublicInput Point Evaluation Commitment Scalar Sidecar)
    (witness : Assignment) : Except Error Unit :=
  runChecks (claimChecks semantics context claim witness)

theorem verifyClaim_eq_ok_iff
    [DecidableEq PublicInput]
    [DecidableEq Evaluation]
    [DecidableEq Commitment]
    [DecidableEq Scalar]
    (semantics : RustSemantics Structure Assignment PublicInput Point Evaluation
      Commitment Scalar Sidecar)
    (context : TerminalCE.Context Structure)
    (claim : RustClaim PublicInput Point Evaluation Commitment Scalar Sidecar)
    (witness : Assignment) :
    verifyClaim semantics context claim witness = .ok () ↔
      TerminalCE.ClaimHolds semantics context claim witness := by
  simp [verifyClaim, runChecks_eq_ok_iff, claimChecks,
    TerminalCE.ClaimHolds, TerminalCE.checkPublicWidth_eq_true_iff]

/-- Length-preserving Rust loop over terminal children and witnesses. -/
def verifyPairs
    [DecidableEq PublicInput]
    [DecidableEq Evaluation]
    [DecidableEq Commitment]
    [DecidableEq Scalar]
    (semantics : RustSemantics Structure Assignment PublicInput Point Evaluation
      Commitment Scalar Sidecar)
    (context : TerminalCE.Context Structure) :
    List (RustClaim PublicInput Point Evaluation Commitment Scalar Sidecar) →
      List Assignment → Except Error Unit
  | [], [] => .ok ()
  | claim :: claims, witness :: witnesses =>
      match verifyClaim semantics context claim witness with
      | .ok _ => verifyPairs semantics context claims witnesses
      | .error error => .error error
  | _, _ => .error .witnessCount

theorem verifyPairs_eq_ok_iff
    [DecidableEq PublicInput]
    [DecidableEq Evaluation]
    [DecidableEq Commitment]
    [DecidableEq Scalar]
    (semantics : RustSemantics Structure Assignment PublicInput Point Evaluation
      Commitment Scalar Sidecar)
    (context : TerminalCE.Context Structure)
    (claims : List (RustClaim PublicInput Point Evaluation Commitment Scalar Sidecar))
    (witnesses : List Assignment) :
    verifyPairs semantics context claims witnesses = .ok () ↔
      TerminalCE.PairsHold semantics context claims witnesses := by
  induction claims generalizing witnesses with
  | nil => cases witnesses <;> simp [verifyPairs, TerminalCE.PairsHold]
  | cons claim claims inductionHypothesis =>
      cases witnesses with
      | nil => simp [verifyPairs, TerminalCE.PairsHold]
      | cons witness witnesses =>
          cases result : verifyClaim semantics context claim witness with
          | error error =>
              have claimInvalid :
                  ¬ TerminalCE.ClaimHolds semantics context claim witness := by
                intro claimHolds
                have accepted :=
                  (verifyClaim_eq_ok_iff semantics context claim witness).2 claimHolds
                rw [result] at accepted
                contradiction
              simp [verifyPairs, result, TerminalCE.PairsHold, claimInvalid]
          | ok value =>
              cases value
              have claimHolds :=
                (verifyClaim_eq_ok_iff semantics context claim witness).1 result
              simp [verifyPairs, result, TerminalCE.PairsHold, claimHolds,
                inductionHypothesis]

/-- Rust-shaped terminal entrypoint. Verifier children are authoritative. -/
def verify
    [DecidableEq Assignment]
    [DecidableEq PublicInput]
    [DecidableEq Point]
    [DecidableEq Evaluation]
    [DecidableEq Commitment]
    [DecidableEq Scalar]
    [DecidableEq Sidecar]
    (semantics : RustSemantics Structure Assignment PublicInput Point Evaluation
      Commitment Scalar Sidecar)
    (terminal : RustInstance Structure Assignment PublicInput Point Evaluation
      Commitment Scalar Sidecar) : Except Error Unit :=
  if terminal.recordedClaims = terminal.verifierChildren then
    verifyPairs semantics terminal.context terminal.verifierChildren terminal.witnesses
  else
    .error .childAuthority

/-- Universal success-path refinement for the native terminal verifier. -/
theorem verify_eq_ok_iff
    [DecidableEq Assignment]
    [DecidableEq PublicInput]
    [DecidableEq Point]
    [DecidableEq Evaluation]
    [DecidableEq Commitment]
    [DecidableEq Scalar]
    [DecidableEq Sidecar]
    (semantics : RustSemantics Structure Assignment PublicInput Point Evaluation
      Commitment Scalar Sidecar)
    (terminal : RustInstance Structure Assignment PublicInput Point Evaluation
      Commitment Scalar Sidecar) :
    verify semantics terminal = .ok () ↔ TerminalCE.Holds semantics terminal := by
  by_cases sameChildren : terminal.recordedClaims = terminal.verifierChildren
  · simp [verify, sameChildren, TerminalCE.Holds, verifyPairs_eq_ok_iff]
  · simp [verify, sameChildren, TerminalCE.Holds]

theorem success_refines_terminalCE
    [DecidableEq Assignment]
    [DecidableEq PublicInput]
    [DecidableEq Point]
    [DecidableEq Evaluation]
    [DecidableEq Commitment]
    [DecidableEq Scalar]
    [DecidableEq Sidecar]
    (semantics : RustSemantics Structure Assignment PublicInput Point Evaluation
      Commitment Scalar Sidecar)
    (terminal : RustInstance Structure Assignment PublicInput Point Evaluation
      Commitment Scalar Sidecar)
    (accepted : verify semantics terminal = .ok ()) :
    TerminalCE.Holds semantics terminal :=
  (verify_eq_ok_iff semantics terminal).1 accepted

/-- Rejection is complete: invalid terminal authority always has a named error. -/
theorem invalid_has_named_rejection
    [DecidableEq Assignment]
    [DecidableEq PublicInput]
    [DecidableEq Point]
    [DecidableEq Evaluation]
    [DecidableEq Commitment]
    [DecidableEq Scalar]
    [DecidableEq Sidecar]
    (semantics : RustSemantics Structure Assignment PublicInput Point Evaluation
      Commitment Scalar Sidecar)
    (terminal : RustInstance Structure Assignment PublicInput Point Evaluation
      Commitment Scalar Sidecar)
    (invalid : ¬ TerminalCE.Holds semantics terminal) :
    ∃ error, verify semantics terminal = .error error := by
  have notAccepted : verify semantics terminal ≠ .ok () := by
    intro accepted
    exact invalid ((verify_eq_ok_iff semantics terminal).1 accepted)
  cases result : verify semantics terminal with
  | ok value =>
      cases value
      exact False.elim (notAccepted result)
  | error error => exact ⟨error, by simp⟩

/-- The Rust-shaped and theorem-facing executable checkers accept exactly the same inputs. -/
theorem verify_ok_iff_check
    [DecidableEq Assignment]
    [DecidableEq PublicInput]
    [DecidableEq Point]
    [DecidableEq Evaluation]
    [DecidableEq Commitment]
    [DecidableEq Scalar]
    [DecidableEq Sidecar]
    (semantics : RustSemantics Structure Assignment PublicInput Point Evaluation
      Commitment Scalar Sidecar)
    (terminal : RustInstance Structure Assignment PublicInput Point Evaluation
      Commitment Scalar Sidecar) :
    verify semantics terminal = .ok () ↔ TerminalCE.check semantics terminal = true := by
  rw [verify_eq_ok_iff, TerminalCE.check_eq_true_iff]

end

end Nightstream.Implementation.Rust.Terminal
