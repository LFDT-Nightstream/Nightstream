import Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.PiDEC
import Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.Semantics

/-!
Base-field linearity of the source-derived SplitNC `yZcol` projection.

Protocol: SuperNeo `Pi_CCS` output claims followed by `Pi_DEC` recomposition.
Phase: complete-carrier assignment recomposition at the verifier-owned column
point.
Constraint family: semantic projection only; this file emits no rows.

Owns: zero, addition, base-field scaling, and finite-combination preservation
for the independently defined `yZcolForAssignment`; the exact production
`Pi_DEC` radix specialization; and transport of every canonical source/lane
coordinate under an explicitly supplied source-assignment equality.

Does not own: proof that NIFS or `Pi_DEC` acceptance supplies that assignment
equality, SplitNC terminal acceptance, transcript derivation of `sPrime`, norm
bounds, commitments, output-message authority, Rust, R1CS, row removal, or
constraint counts.

Emits constraints: no.

Authority boundary: every projected value is recomputed from the authoritative
complete assignment and verifier-owned point. The production transport theorem
requires `data.assignment source = PiDEC.recomposeAssignment children` as an
explicit premise. It does not infer that hard fact from acceptance or from a
digest. Consequently this is a model-level homomorphism theorem, not full
`yZcol` authority closure.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.output.y_zcol.diagonal` | the one-live-lane table is base-`F` linear in the complete assignment | computed | `diagonal_zero`, `diagonal_add`, `diagonal_scale` |
| `nifs.pi_ccs.output.y_zcol.projection` | the carrier-column fold preserves those operations at fixed `sPrime` | derived | `yZcolForAssignment_zero`, `yZcolForAssignment_add`, `yZcolForAssignment_scale` |
| `nifs.pi_dec.verify.recomposition.y_zcol.combine` | all 54 lanes use the same finite weights as the assignment | derived | `yZcolEvaluation_combine` |
| `nifs.pi_dec.verify.recomposition.y_zcol.radix` | specialize to verifier-fixed `b = 2`, `k = 14` | computed | `yZcolEvaluation_piDecRecompose` |
| `nifs.pi_ccs.output.y_zcol.authority` | every source/lane transports from the recomposed parent assignment | checked premise | `canonicalYZcol_product_piDec_transport` |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.BaseLinear

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.BaseLinear
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc

/-! ## Shared assignment carrier -/

/-- Forget batch arity while preserving the logical and complete carrier
widths used by both canonical `yZcol` and the typed Phi81 relation. The zero
public width is inert: this file proves only the independent column sidecar. -/
def relationShape (shape : SemanticShape) : Phi81Relation.Shape :=
  Phi81Relation.Shape.ofSemantic shape 0 (by simp)

/-- The adapter preserves the exact complete-carrier width definitionally. -/
@[simp] theorem relationShape_carrierWidth (shape : SemanticShape) :
    (relationShape shape).carrierWidth = shape.carrierWidth := by
  rfl

/-! ## Diagonal leaves -/

/-- The raw SplitNC diagonal of the zero assignment is zero in every lane. -/
theorem diagonal_zero
    {shape : SemanticShape}
    (column : Fin shape.carrierWidth) (lane : Fin ringDegree) :
    SplitNc.Semantics.Nc.diagonal
        (assignmentZero (shape := relationShape shape)) column lane = 0 := by
  by_cases selected : lane.val = column.val % ringDegree
  · simp [SplitNc.Semantics.Nc.diagonal, assignmentZero, selected]
  · simp [SplitNc.Semantics.Nc.diagonal, selected]

/-- The one-live-lane diagonal is additive in the complete assignment. -/
theorem diagonal_add
    {shape : SemanticShape}
    (left right : Phi81Relation.Assignment (relationShape shape))
    (column : Fin shape.carrierWidth) (lane : Fin ringDegree) :
    SplitNc.Semantics.Nc.diagonal
        (assignmentAdd (shape := relationShape shape) left right) column lane =
      SplitNc.Semantics.Nc.diagonal left column lane +
        SplitNc.Semantics.Nc.diagonal right column lane := by
  by_cases selected : lane.val = column.val % ringDegree
  · simp [SplitNc.Semantics.Nc.diagonal, assignmentAdd, selected]
  · simp [SplitNc.Semantics.Nc.diagonal, selected]

/-- The one-live-lane diagonal commutes with a base-field scalar. -/
theorem diagonal_scale
    {shape : SemanticShape}
    (scalar : F)
    (assignment : Phi81Relation.Assignment (relationShape shape))
    (column : Fin shape.carrierWidth) (lane : Fin ringDegree) :
    SplitNc.Semantics.Nc.diagonal
        (assignmentScale (shape := relationShape shape) scalar assignment)
        column lane =
      scalar * SplitNc.Semantics.Nc.diagonal assignment column lane := by
  by_cases selected : lane.val = column.val % ringDegree
  · simp [SplitNc.Semantics.Nc.diagonal, assignmentScale, selected]
  · simp only [SplitNc.Semantics.Nc.diagonal, selected, ↓reduceIte]
    exact (Fin.mul_zero scalar).symm

/-! ## One weighted column contribution -/

/-- A zero assignment contributes zero to every column/lane leaf. -/
theorem yZcolTerm_zero
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (covers : domain.Covers shape)
    (sPrime : CubePoint K domain.columnVariables)
    (lane : Fin ringDegree) (column : Fin shape.carrierWidth) :
    yZcolTerm covers (assignmentZero (shape := relationShape shape))
        sPrime lane column = K.zero := by
  unfold yZcolTerm
  rw [diagonal_zero]
  change K.mul K.zero (columnWeight sPrime (domain.carrierColumn covers column)) =
    K.zero
  calc
    K.mul K.zero (columnWeight sPrime (domain.carrierColumn covers column)) =
        K.mul (columnWeight sPrime (domain.carrierColumn covers column)) K.zero :=
      ConcreteCarrier.extensionLaws.mul_comm _ _
    _ = K.zero := ConcreteCarrier.extensionLaws.mul_zero _

/-- One weighted column contribution is additive in the assignment. -/
theorem yZcolTerm_add
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (covers : domain.Covers shape)
    (left right : Phi81Relation.Assignment (relationShape shape))
    (sPrime : CubePoint K domain.columnVariables)
    (lane : Fin ringDegree) (column : Fin shape.carrierWidth) :
    yZcolTerm covers
        (assignmentAdd (shape := relationShape shape) left right)
        sPrime lane column =
      K.add (yZcolTerm covers left sPrime lane column)
        (yZcolTerm covers right sPrime lane column) := by
  unfold yZcolTerm
  rw [diagonal_add]
  have embedAdd :
      K.embed
          (SplitNc.Semantics.Nc.diagonal left column lane +
            SplitNc.Semantics.Nc.diagonal right column lane) =
        K.add (K.embed (SplitNc.Semantics.Nc.diagonal left column lane))
          (K.embed (SplitNc.Semantics.Nc.diagonal right column lane)) := by
    simpa only [ConcreteCarrier.baseOps, ConcreteCarrier.extensionOps] using
      (ConcreteCarrier.embed_add
        (SplitNc.Semantics.Nc.diagonal left column lane)
        (SplitNc.Semantics.Nc.diagonal right column lane))
  rw [embedAdd]
  simpa only [ConcreteCarrier.extensionOps] using
    (ConcreteCarrier.extensionLaws.right_distrib
      (K.embed (SplitNc.Semantics.Nc.diagonal left column lane))
      (K.embed (SplitNc.Semantics.Nc.diagonal right column lane))
      (columnWeight sPrime (domain.carrierColumn covers column)))

/-- One weighted column contribution commutes with an embedded base scalar. -/
theorem yZcolTerm_scale
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (covers : domain.Covers shape)
    (scalar : F)
    (assignment : Phi81Relation.Assignment (relationShape shape))
    (sPrime : CubePoint K domain.columnVariables)
    (lane : Fin ringDegree) (column : Fin shape.carrierWidth) :
    yZcolTerm covers
        (assignmentScale (shape := relationShape shape) scalar assignment)
        sPrime lane column =
      K.mul (K.embed scalar)
        (yZcolTerm covers assignment sPrime lane column) := by
  unfold yZcolTerm
  rw [diagonal_scale]
  have embedScale :
      K.embed
          (scalar * SplitNc.Semantics.Nc.diagonal assignment column lane) =
        K.mul (K.embed scalar)
          (K.embed (SplitNc.Semantics.Nc.diagonal assignment column lane)) := by
    simpa only [ConcreteCarrier.baseOps, ConcreteCarrier.extensionOps] using
      (ConcreteCarrier.embed_mul scalar
        (SplitNc.Semantics.Nc.diagonal assignment column lane))
  rw [embedScale]
  simpa only [ConcreteCarrier.extensionOps] using
    (ConcreteCarrier.extensionLaws.mul_assoc (K.embed scalar)
      (K.embed (SplitNc.Semantics.Nc.diagonal assignment column lane))
      (columnWeight sPrime (domain.carrierColumn covers column)))

/-! ## Complete column fold -/

private theorem foldl_zero_terms
    {Index : Type}
    (indices : List Index) :
    indices.foldl (fun accumulated _ => K.add accumulated K.zero) K.zero =
      K.zero := by
  induction indices with
  | nil => rfl
  | cons _ indices inductionHypothesis =>
      simp only [List.foldl_cons]
      have zeroAdd : K.add K.zero K.zero = K.zero := by
        simpa only [ConcreteCarrier.extensionOps] using
          (ConcreteCarrier.extensionLaws.zero_add K.zero)
      rw [zeroAdd]
      exact inductionHypothesis

private theorem foldl_add_terms
    {Index : Type}
    (indices : List Index)
    (left right : Index -> K)
    (leftInitial rightInitial : K) :
    indices.foldl
        (fun accumulated index =>
          K.add accumulated (K.add (left index) (right index)))
        (K.add leftInitial rightInitial) =
      K.add
        (indices.foldl
          (fun accumulated index => K.add accumulated (left index))
          leftInitial)
        (indices.foldl
          (fun accumulated index => K.add accumulated (right index))
          rightInitial) := by
  induction indices generalizing leftInitial rightInitial with
  | nil => rfl
  | cons index indices inductionHypothesis =>
      simp only [List.foldl_cons]
      have rearrange :
          K.add (K.add leftInitial rightInitial)
              (K.add (left index) (right index)) =
            K.add (K.add leftInitial (left index))
              (K.add rightInitial (right index)) := by
        letI : Std.Associative K.add :=
          ⟨ConcreteCarrier.extensionLaws.add_assoc⟩
        letI : Std.Commutative K.add :=
          ⟨ConcreteCarrier.extensionLaws.add_comm⟩
        ac_rfl
      rw [rearrange]
      exact inductionHypothesis _ _

private theorem foldl_scale_terms
    {Index : Type}
    (indices : List Index)
    (scalar : K)
    (term : Index -> K)
    (initial : K) :
    indices.foldl
        (fun accumulated index =>
          K.add accumulated (K.mul scalar (term index)))
        (K.mul scalar initial) =
      K.mul scalar
        (indices.foldl
          (fun accumulated index => K.add accumulated (term index)) initial) := by
  induction indices generalizing initial with
  | nil => rfl
  | cons index indices inductionHypothesis =>
      simp only [List.foldl_cons]
      have distribute :
          K.add (K.mul scalar initial) (K.mul scalar (term index)) =
            K.mul scalar (K.add initial (term index)) := by
        simpa only [ConcreteCarrier.extensionOps] using
          (ConcreteCarrier.extensionLaws.left_distrib
            scalar initial (term index)).symm
      calc
        indices.foldl
            (fun accumulated next =>
              K.add accumulated (K.mul scalar (term next)))
            (K.add (K.mul scalar initial) (K.mul scalar (term index))) =
          indices.foldl
            (fun accumulated next =>
              K.add accumulated (K.mul scalar (term next)))
            (K.mul scalar (K.add initial (term index))) := by
              rw [distribute]
        _ = K.mul scalar
            (indices.foldl
              (fun accumulated next => K.add accumulated (term next))
              (K.add initial (term index))) :=
          inductionHypothesis _

/-- The full-carrier projection of the zero assignment is zero. -/
theorem yZcolForAssignment_zero
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (covers : domain.Covers shape)
    (sPrime : CubePoint K domain.columnVariables)
    (lane : Fin ringDegree) :
    yZcolForAssignment covers
        (assignmentZero (shape := relationShape shape)) sPrime lane = K.zero := by
  unfold yZcolForAssignment
  simp only [yZcolTerm_zero]
  exact foldl_zero_terms (canonicalFinIndices shape.carrierWidth)

/-- The full-carrier projection is additive in the complete assignment. -/
theorem yZcolForAssignment_add
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (covers : domain.Covers shape)
    (left right : Phi81Relation.Assignment (relationShape shape))
    (sPrime : CubePoint K domain.columnVariables)
    (lane : Fin ringDegree) :
    yZcolForAssignment covers
        (assignmentAdd (shape := relationShape shape) left right)
        sPrime lane =
      K.add (yZcolForAssignment covers left sPrime lane)
        (yZcolForAssignment covers right sPrime lane) := by
  unfold yZcolForAssignment
  simp only [yZcolTerm_add]
  simpa only [ConcreteCarrier.extensionOps] using
    (foldl_add_terms (canonicalFinIndices shape.carrierWidth)
      (fun column => yZcolTerm covers left sPrime lane column)
      (fun column => yZcolTerm covers right sPrime lane column)
      K.zero K.zero)

/-- The full-carrier projection commutes with an embedded base scalar. -/
theorem yZcolForAssignment_scale
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (covers : domain.Covers shape)
    (scalar : F)
    (assignment : Phi81Relation.Assignment (relationShape shape))
    (sPrime : CubePoint K domain.columnVariables)
    (lane : Fin ringDegree) :
    yZcolForAssignment covers
        (assignmentScale (shape := relationShape shape) scalar assignment)
        sPrime lane =
      K.mul (K.embed scalar)
        (yZcolForAssignment covers assignment sPrime lane) := by
  unfold yZcolForAssignment
  simp only [yZcolTerm_scale]
  simpa only [ConcreteCarrier.extensionOps] using
    (foldl_scale_terms (canonicalFinIndices shape.carrierWidth)
      (K.embed scalar)
      (fun column => yZcolTerm covers assignment sPrime lane column)
      K.zero)

/-! ## Ring-shaped and production PiDEC transport -/

/-- Package all 54 independently recomputed sidecar coefficients as one
`RingK`, solely to state exact recomposition with the existing CE operation. -/
def yZcolEvaluation
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (covers : domain.Covers shape)
    (assignment : Phi81Relation.Assignment (relationShape shape))
    (sPrime : CubePoint K domain.columnVariables) : RingK :=
  fun lane => yZcolForAssignment covers assignment sPrime lane

/-- All 54 sidecar lanes of the zero assignment are zero. -/
theorem yZcolEvaluation_zero
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (covers : domain.Covers shape)
    (sPrime : CubePoint K domain.columnVariables) :
    yZcolEvaluation covers
        (assignmentZero (shape := relationShape shape)) sPrime =
      evaluationZero := by
  funext lane
  exact yZcolForAssignment_zero covers sPrime lane

/-- All 54 sidecar lanes preserve assignment addition. -/
theorem yZcolEvaluation_add
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (covers : domain.Covers shape)
    (left right : Phi81Relation.Assignment (relationShape shape))
    (sPrime : CubePoint K domain.columnVariables) :
    yZcolEvaluation covers
        (assignmentAdd (shape := relationShape shape) left right) sPrime =
      evaluationAdd (yZcolEvaluation covers left sPrime)
        (yZcolEvaluation covers right sPrime) := by
  funext lane
  exact yZcolForAssignment_add covers left right sPrime lane

/-- All 54 sidecar lanes preserve embedded base-field scaling. -/
theorem yZcolEvaluation_scale
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (covers : domain.Covers shape)
    (scalar : F)
    (assignment : Phi81Relation.Assignment (relationShape shape))
    (sPrime : CubePoint K domain.columnVariables) :
    yZcolEvaluation covers
        (assignmentScale (shape := relationShape shape) scalar assignment)
        sPrime =
      evaluationScale scalar (yZcolEvaluation covers assignment sPrime) := by
  funext lane
  exact yZcolForAssignment_scale covers scalar assignment sPrime lane

/-- Exact finite base-field combination theorem shared with the typed Phi81
CE evaluator. This is the algebraic core needed by `Pi_DEC`. -/
theorem yZcolEvaluation_combine
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {count : Nat}
    (covers : domain.Covers shape)
    (weights : Fin count -> F)
    (assignments : Fin count ->
      Phi81Relation.Assignment (relationShape shape))
    (sPrime : CubePoint K domain.columnVariables) :
    yZcolEvaluation covers (combineAssignments weights assignments) sPrime =
      combineEvaluations weights
        (fun index => yZcolEvaluation covers (assignments index) sPrime) := by
  induction count with
  | zero => exact yZcolEvaluation_zero covers sPrime
  | succ count inductionHypothesis =>
      rw [combineAssignments, combineEvaluations, yZcolEvaluation_add,
        yZcolEvaluation_scale, inductionHypothesis]

/-- Production specialization: `yZcol` uses the same `b = 2`, `k = 14`
recomposition as the independently defined typed Phi81 `PiDEC` parent. -/
theorem yZcolEvaluation_piDecRecompose
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (covers : domain.Covers shape)
    (assignments : Fin productionGlobalParams.k ->
      Phi81Relation.Assignment (relationShape shape))
    (sPrime : CubePoint K domain.columnVariables) :
    yZcolEvaluation covers
        (Phi81Relation.EvaluationHomomorphism.PiDEC.recomposeAssignment
          assignments)
        sPrime =
      combineEvaluations
        Phi81Relation.EvaluationHomomorphism.PiDEC.radixWeight
        (fun index => yZcolEvaluation covers (assignments index) sPrime) := by
  unfold Phi81Relation.EvaluationHomomorphism.PiDEC.recomposeAssignment
  exact yZcolEvaluation_combine covers
    Phi81Relation.EvaluationHomomorphism.PiDEC.radixWeight assignments sPrime

/-- Canonical source-derived 54-lane sidecar for one active `Pi_CCS` source. -/
def canonicalYZcolEvaluation
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (covers : domain.Covers shape)
    (data : SplitNc.Sources.Data shape)
    (points : VerifierPoints shape domain)
    (source : Fin shape.sourceCount) : RingK :=
  fun lane => canonicalYZcol covers data points source lane

/-- Transport one canonical source-derived sidecar through production
`PiDEC` recomposition. The source/recomposition equality is intentionally an
explicit hard premise, not a conclusion of acceptance. -/
theorem canonicalYZcol_piDec_transport
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (covers : domain.Covers shape)
    (data : SplitNc.Sources.Data shape)
    (points : VerifierPoints shape domain)
    (source : Fin shape.sourceCount)
    (children : Fin productionGlobalParams.k ->
      Phi81Relation.Assignment (relationShape shape))
    (sourceRecomposition :
      data.assignment source =
        Phi81Relation.EvaluationHomomorphism.PiDEC.recomposeAssignment
          children) :
    canonicalYZcolEvaluation covers data points source =
      combineEvaluations
        Phi81Relation.EvaluationHomomorphism.PiDEC.radixWeight
        (fun index => yZcolEvaluation covers (children index) points.sPrime) := by
  change yZcolEvaluation covers (data.assignment source) points.sPrime = _
  rw [sourceRecomposition]
  exact yZcolEvaluation_piDecRecompose covers children points.sPrime

/-- Product-level form. For any five-source batch this covers all
`5 * 54 = 270` active `yZcol` coordinates, but the theorem remains general in
the source count. It still requires one independently justified assignment
recomposition equality per source. -/
theorem canonicalYZcol_product_piDec_transport
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (covers : domain.Covers shape)
    (data : SplitNc.Sources.Data shape)
    (points : VerifierPoints shape domain)
    (children : Fin shape.sourceCount -> Fin productionGlobalParams.k ->
      Phi81Relation.Assignment (relationShape shape))
    (sourceRecomposition : forall source,
      data.assignment source =
        Phi81Relation.EvaluationHomomorphism.PiDEC.recomposeAssignment
          (children source)) :
    forall source lane,
      canonicalYZcol covers data points source lane =
        combineEvaluations
          Phi81Relation.EvaluationHomomorphism.PiDEC.radixWeight
          (fun index =>
            yZcolEvaluation covers (children source index) points.sPrime)
          lane := by
  intro source lane
  exact congrFun
    (canonicalYZcol_piDec_transport covers data points source
      (children source) (sourceRecomposition source)) lane

end Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.BaseLinear
