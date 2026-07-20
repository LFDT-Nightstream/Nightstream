import Nightstream.Implementation.R1CS.Correspondence.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionTerminal
import Nightstream.Implementation.Rust.Terminal
import Nightstream.Protocol.Terminal.CE

/-!
Concrete terminal-CE adapter for the delayed packed-`yZcol` terminal.

Assurance tier: model-level, pending the Rust `WitnessMat` decoder and native
primitive-operation refinement.

Owns: the exact Phi81 specialization of terminal child-opening validation;
canonical `Fin 14` to list order; recovery of every `ProductionTerminal`
child obligation from terminal-CE acceptance; and composition with the
independent delayed projection Boolean check.

Does not own: the native `WitnessMat` decoder, the missing Rust projection
comparison, Ajtai binding, terminal circuit rows, Poseidon2, costs, or row
removal.

Emits constraints: none. This correspondence leaf specializes typed terminal
semantics and refines verifier control flow; concrete terminal rows remain an
explicit downstream obligation.

Authority boundary: terminal CE acceptance supplies only child-opening
authority. It cannot establish the old-parent projection because neither the
pending parent vector nor its old block point occurs in `TerminalCE.Holds`.
That equality is recomputed by `ProductionTerminal.projectionCheck` over the
same ordered raw assignments.

| Stage path | Mathematical obligation | Authority class |
|---|---|---|
| `nifs.production.terminal.ce.children` | all fourteen verifier children open against their raw assignments in canonical order | checked/refined |
| `nifs.production.terminal.ce.projection` | the delayed parent equals the old-point projection of the raw-child radix recomposition | checked by separate component |
-/

namespace Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionTerminal.TerminalCEBridge

open Nightstream.Protocol
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier

universe uState

variable
  {shape : SemanticShape}
  {State : Type uState}
  {publicRingColumns verifierRows : Nat}
  {publicFits : ringDegree * publicRingColumns <= shape.carrierWidth}

/-- Turn an exact executable carrier comparison into the corresponding
decidable equality. This keeps the Rust-shaped loop on the same finite
comparators as the canonical concrete verifier. -/
private def exactDecidableEq {Value : Type}
    (equal : Value -> Value -> Bool)
    (exact : forall left right, equal left right = true <-> left = right) :
    DecidableEq Value :=
  fun left right =>
    if accepted : equal left right = true then
      isTrue ((exact left right).mp accepted)
    else
      isFalse fun equality => accepted ((exact left right).mpr equality)

local instance publicInputDecidableEq :
    DecidableEq
      (Phi81Relation.PublicInput
        (RelationShape shape publicRingColumns publicFits)) :=
  exactDecidableEq CarrierEquality.publicInputEqual
    CarrierEquality.publicInputEqual_eq_true_iff

local instance evaluationDecidableEq : DecidableEq Phi81Relation.Evaluation :=
  exactDecidableEq CarrierEquality.ringKEqual
    CarrierEquality.ringKEqual_eq_true_iff

local instance commitmentDecidableEq :
    DecidableEq (CommitmentValue verifierRows) :=
  exactDecidableEq CarrierEquality.commitmentEqual
    CarrierEquality.commitmentEqual_eq_true_iff

local instance scalarDecidableEq : DecidableEq K :=
  exactDecidableEq CarrierEquality.extensionEqual
    CarrierEquality.extensionEqual_eq_true_iff

/-- Terminal-CE primitive operations specialized to the typed Phi81 child
carrier. Width, point shape, constant terms, and the implementation sidecar
are extra checks at the native boundary; this specialization retains exactly
the four fields needed by `ProductionTerminal.ChildAccepted`. -/
def semantics
    (context : Context shape State publicRingColumns publicFits verifierRows) :
    TerminalCE.Semantics
      (Phi81Relation.Structure
        (RelationShape shape publicRingColumns publicFits))
      (Phi81Relation.Assignment
        (RelationShape shape publicRingColumns publicFits))
      (Phi81Relation.PublicInput
        (RelationShape shape publicRingColumns publicFits))
      (Phi81Relation.Point
        (RelationShape shape publicRingColumns publicFits))
      Phi81Relation.Evaluation (CommitmentValue verifierRows) K Unit where
  commit := commit context.key
  projectPublicInput := fun _width assignment =>
    some (Phi81Relation.projectPublicInput assignment)
  normBounded := ProductionTerminal.assignmentNormCheck
  evaluationPointValid := fun _system _point => true
  evaluations := fun system assignment point =>
    some (Phi81Relation.evaluations system assignment point).toList
  constantTerm := fun evaluation => evaluation ⟨0, by decide⟩
  sidecarValid := fun _system _assignment _sidecar => true

/-- Verifier-owned terminal relation. Every canonical Π_DEC child inherits
this structure and the fresh norm stage from the computed parent. -/
def terminalContext
    (context : Context shape State publicRingColumns publicFits verifierRows)
    (certificate : Certificate context) :
    TerminalCE.Context
      (Phi81Relation.Structure
        (RelationShape shape publicRingColumns publicFits)) where
  relation := (derive context certificate).piRlcOutput.constraintSystem
  normBound := NormStage.fresh.bound productionGlobalParams
  expectedPublicWidth :=
    some (RelationShape shape publicRingColumns publicFits).publicWidth

/-- One verifier-derived terminal child encoded in the generic terminal-CE
surface. The child index is not prover data: `terminalInstance` enumerates
`Fin 14` in canonical order. -/
def childClaim
    (context : Context shape State publicRingColumns publicFits verifierRows)
    (certificate : Certificate context)
    (child : Fin productionGlobalParams.k) :
    TerminalCE.Claim
      (Phi81Relation.PublicInput
        (RelationShape shape publicRingColumns publicFits))
      (Phi81Relation.Point
        (RelationShape shape publicRingColumns publicFits))
      Phi81Relation.Evaluation (CommitmentValue verifierRows) K Unit :=
  let statement := outputChildren context certificate child
  { commitment := statement.commitment
    publicWidth :=
      (RelationShape shape publicRingColumns publicFits).publicWidth
    publicInput := statement.publicInput
    point := statement.point
    evaluations := statement.evaluations.toList
    constantTerms := statement.evaluations.toList.map
      (fun evaluation => evaluation ⟨0, by decide⟩)
    sidecar := () }

/-- Canonical terminal-CE instance over the exact ordered raw child family.
The recorded list is definitionally the verifier list; prover-controlled
claim substitution therefore cannot enter this adapter. -/
def terminalInstance
    (context : Context shape State publicRingColumns publicFits verifierRows)
    (certificate : Certificate context)
    (rawChildren : Fin productionGlobalParams.k ->
      Phi81Relation.Assignment
        (RelationShape shape publicRingColumns publicFits)) :
    TerminalCE.Instance
      (Phi81Relation.Structure
        (RelationShape shape publicRingColumns publicFits))
      (Phi81Relation.Assignment
        (RelationShape shape publicRingColumns publicFits))
      (Phi81Relation.PublicInput
        (RelationShape shape publicRingColumns publicFits))
      (Phi81Relation.Point
        (RelationShape shape publicRingColumns publicFits))
      Phi81Relation.Evaluation (CommitmentValue verifierRows) K Unit :=
  let claims := List.ofFn (childClaim context certificate)
  { context := terminalContext context certificate
    verifierChildren := claims
    recordedClaims := claims
    witnesses := List.ofFn rawChildren }

/-- Rust-shaped ordered child-opening loop on the exact canonical terminal
lists. This is the implementation-facing acceptance surface for the child
component; delayed projection is deliberately absent. -/
def rustVerifyPairs
    (context : Context shape State publicRingColumns publicFits verifierRows)
    (certificate : Certificate context)
    (rawChildren : Fin productionGlobalParams.k ->
      Phi81Relation.Assignment
        (RelationShape shape publicRingColumns publicFits)) :
    Except Nightstream.Implementation.Rust.Terminal.Error Unit :=
  Nightstream.Implementation.Rust.Terminal.verifyPairs
    (semantics context) (terminalContext context certificate)
    (List.ofFn (childClaim context certificate))
    (List.ofFn rawChildren)

/-- A concrete terminal-CE child opening is exactly the compact child
obligation used by the delayed terminal checker. -/
theorem claimHolds_iff_childAccepted
    (context : Context shape State publicRingColumns publicFits verifierRows)
    (certificate : Certificate context)
    (rawChildren : Fin productionGlobalParams.k ->
      Phi81Relation.Assignment
        (RelationShape shape publicRingColumns publicFits))
    (child : Fin productionGlobalParams.k) :
    TerminalCE.ClaimHolds (semantics context)
        (terminalContext context certificate)
        (childClaim context certificate child) (rawChildren child) <->
      ProductionTerminal.ChildAccepted context certificate
        (rawChildren child) child := by
  simp only [TerminalCE.ClaimHolds, TerminalCE.PublicWidthHolds, semantics,
    terminalContext, childClaim,
    ProductionTerminal.assignmentNormCheck_eq_true_iff, true_and,
    Option.some.injEq]
  have systemEq :
      (outputChildren context certificate child).constraintSystem =
        (derive context certificate).piRlcOutput.constraintSystem := by
    rfl
  have stageEq :
      (outputChildren context certificate child).stage = .fresh := by
    rfl
  constructor
  · rintro ⟨commits, publicInput, norm, evaluationsList, _constantTerms⟩
    refine {
      commits := commits
      publicInput := publicInput
      norm := ?_
      evaluations := ?_ }
    · simpa only [stageEq] using norm
    · apply Array.toList_inj.mp
      simpa only [systemEq] using evaluationsList
  · intro accepted
    refine ⟨accepted.commits, accepted.publicInput, ?_, ?_, trivial⟩
    · simpa only [stageEq] using accepted.norm
    · simpa only [systemEq] using congrArg Array.toList accepted.evaluations

/-- Pointwise extraction from the recursive `PairsHold` list relation. -/
private theorem pairsHold_at
    {Structure Assignment PublicInput Point Evaluation Commitment Scalar Sidecar : Type}
    (terminalSemantics : TerminalCE.Semantics Structure Assignment PublicInput
      Point Evaluation Commitment Scalar Sidecar)
    (context : TerminalCE.Context Structure)
    (claims : List
      (TerminalCE.Claim PublicInput Point Evaluation Commitment Scalar Sidecar))
    (witnesses : List Assignment)
    (holds : TerminalCE.PairsHold terminalSemantics context claims witnesses)
    (index : Nat)
    (claimBound : index < claims.length)
    (witnessBound : index < witnesses.length) :
    TerminalCE.ClaimHolds terminalSemantics context
      claims[index] witnesses[index] := by
  induction claims generalizing witnesses index with
  | nil => simp at claimBound
  | cons claim claims inductionHypothesis =>
      cases witnesses with
      | nil => simp at witnessBound
      | cons witness witnesses =>
          simp only [TerminalCE.PairsHold] at holds
          cases index with
          | zero => exact holds.1
          | succ index =>
              exact inductionHypothesis witnesses holds.2 index
                (by simpa using claimBound) (by simpa using witnessBound)

/-- Terminal-CE acceptance derives every ordered raw-child opening; it does
not use, and cannot derive, the delayed parent projection. -/
theorem holds_implies_childrenCheck
    (context : Context shape State publicRingColumns publicFits verifierRows)
    (certificate : Certificate context)
    (rawChildren : Fin productionGlobalParams.k ->
      Phi81Relation.Assignment
        (RelationShape shape publicRingColumns publicFits))
    (accepted : TerminalCE.Holds (semantics context)
      (terminalInstance context certificate rawChildren)) :
    ProductionTerminal.childrenCheck context certificate rawChildren = true := by
  apply (ProductionTerminal.childrenCheck_eq_true_iff context certificate
    rawChildren).2
  intro child
  apply (claimHolds_iff_childAccepted context certificate rawChildren child).1
  have pairAccepted := accepted.2
  have childAccepted :=
    pairsHold_at (semantics context) (terminalContext context certificate)
      (List.ofFn (childClaim context certificate))
      (List.ofFn rawChildren) pairAccepted child.val
      (by simp) (by simp)
  have claimAt :
      (List.ofFn (childClaim context certificate))[child.val] =
        childClaim context certificate child := by
    exact List.getElem_ofFn (f := childClaim context certificate) child.isLt
  have witnessAt :
      (List.ofFn rawChildren)[child.val] = rawChildren child := by
    exact List.getElem_ofFn (f := rawChildren) child.isLt
  rw [claimAt, witnessAt] at childAccepted
  exact childAccepted

/-- Success of the Rust-shaped ordered terminal loop derives the same child
checker. This is control-flow refinement over the concrete Phi81 operations;
translation of native Rust primitives and `WitnessMat` decoding remains the
external implementation boundary. -/
theorem rustVerifyPairsSuccess_implies_childrenCheck
    (context : Context shape State publicRingColumns publicFits verifierRows)
    (certificate : Certificate context)
    (rawChildren : Fin productionGlobalParams.k ->
      Phi81Relation.Assignment
        (RelationShape shape publicRingColumns publicFits))
    (accepted : rustVerifyPairs context certificate rawChildren =
        (Except.ok () : Except Nightstream.Implementation.Rust.Terminal.Error Unit)) :
    ProductionTerminal.childrenCheck context certificate rawChildren = true := by
  apply holds_implies_childrenCheck context certificate rawChildren
  exact ⟨rfl,
    (Nightstream.Implementation.Rust.Terminal.verifyPairs_eq_ok_iff
      (semantics context) (terminalContext context certificate)
      (List.ofFn (childClaim context certificate))
      (List.ofFn rawChildren)).mp (by
        simpa only [rustVerifyPairs] using accepted)⟩

/-- Concrete terminal composition seam. One actual terminal-CE success
supplies raw-child opening authority; the separate delayed projection check
then yields the full terminal relation. -/
theorem accepted_of_terminalCE_and_projectionCheck
    (context : Context shape State publicRingColumns publicFits verifierRows)
    (certificate : Certificate context)
    (rawChildren : Fin productionGlobalParams.k ->
      Phi81Relation.Assignment
        (RelationShape shape publicRingColumns publicFits))
    (terminalAccepted : TerminalCE.Holds (semantics context)
      (terminalInstance context certificate rawChildren))
    (projectionAccepted :
      ProductionTerminal.projectionCheck context certificate rawChildren = true) :
    ProductionTerminal.Accepted context certificate rawChildren := by
  exact ProductionTerminal.accepted_of_component_checks context certificate
    rawChildren
    (holds_implies_childrenCheck context certificate rawChildren
      terminalAccepted)
    projectionAccepted

/-- Rust-shaped terminal child verification plus the independent delayed
projection comparison is sufficient for the complete terminal relation. -/
theorem accepted_of_rustVerifyPairs_and_projectionCheck
    (context : Context shape State publicRingColumns publicFits verifierRows)
    (certificate : Certificate context)
    (rawChildren : Fin productionGlobalParams.k ->
      Phi81Relation.Assignment
        (RelationShape shape publicRingColumns publicFits))
    (terminalAccepted : rustVerifyPairs context certificate rawChildren =
        (Except.ok () : Except Nightstream.Implementation.Rust.Terminal.Error Unit))
    (projectionAccepted :
      ProductionTerminal.projectionCheck context certificate rawChildren = true) :
    ProductionTerminal.Accepted context certificate rawChildren := by
  exact ProductionTerminal.accepted_of_component_checks context certificate
    rawChildren
    (rustVerifyPairsSuccess_implies_childrenCheck context certificate
      rawChildren terminalAccepted)
    projectionAccepted

end Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionTerminal.TerminalCEBridge
