import Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.PackedBlockAction.Linear
import Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.PiRLCFinite

/-!
Contract: lift the one-source packed block action law to the exact finite
challenge fold used by Π_RLC.

Owns: the finite-source packed Π_RLC evaluation homomorphism.

Does not own: augmented CE membership, challenge validity or sampling,
commitments, public inputs, norm growth, Π_DEC, transcripts, Rust, R1CS,
costs, or row removal.

Emits constraints: no.

Authority boundary: source assignments and challenges are explicit typed
inputs. The combined assignment and evaluation use the same canonical
head-first `Fin n` traversal. This proves algebraic transport only; it does
not prove that a public output was transcript- or commitment-bound.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_rlc.verify.authority.packed_y_zcol.finite.combine` | the exact finite `RingF` assignment fold equals the identical evaluation fold | derived | `packedYZcol_combine` |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.PackedBlockAction.Finite

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism
open Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.BaseLinear
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open PackedBlockAction
open Linear

/-- The canonical finite Π_RLC assignment combination and packed evaluation
combination agree exactly. -/
theorem packedYZcol_combine
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    {count : Nat}
    (covers : domain.Covers shape)
    (challenges : Fin count -> RingF)
    (assignments : Fin count -> SemanticAssignment shape)
    (point : CubePoint K domain.blockVariables) :
    packedYZcol covers
        (PiRLCFinite.Raw.combineAssignments challenges assignments) point =
      PiRLCFinite.combineEvaluation challenges fun index =>
        packedYZcol covers (assignments index) point := by
  induction count with
  | zero =>
      rw [PiRLCFinite.Raw.combineAssignments,
        PiRLCFinite.combineEvaluation]
      exact packedYZcol_zero covers point
  | succ count inductionHypothesis =>
      rw [PiRLCFinite.Raw.combineAssignments,
        PiRLCFinite.combineEvaluation]
      calc
        packedYZcol covers
            (BaseLinear.Raw.assignmentAdd
              (CarrierAction.act (challenges 0) (assignments 0))
              (PiRLCFinite.Raw.combineAssignments
                (fun index => challenges index.succ)
                (fun index => assignments index.succ))) point =
          ringKAdd
            (packedYZcol covers
              (CarrierAction.act (challenges 0) (assignments 0)) point)
            (packedYZcol covers
              (PiRLCFinite.Raw.combineAssignments
                (fun index => challenges index.succ)
                (fun index => assignments index.succ)) point) :=
          packedYZcol_add covers _ _ point
        _ = _ := by
          rw [packedYZcol_ringAction]
          rw [inductionHypothesis
            (fun index => challenges index.succ)
            (fun index => assignments index.succ)]

end Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.PackedBlockAction.Finite
