import Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.Authority.DelayedPackedProjection

/-!
Two-limb evaluation of the packed `yZcol` projection.

Protocol: SuperNeo `Pi_RLC` packed-output authority.
Phase: evaluate one 54-coefficient `K` polynomial through its two base-field
coefficient limbs.
Constraint family: fixed affine recombination; this file emits no rows.

Assurance tier: model-level.

Owns: the canonical `K = F[u]/(u^2 - 7)` limb split and the proof that the
two production-shaped limb evaluations recombine to the original packed
projection. The result applies to any coefficient list and specializes to the
fixed-width `RingK` codec.

Does not own: generated columns, transcript-derived beta, parent commitment
opening, Rust/R1CS refinement, physical costs, or row removal.

Emits constraints: no.

Authority boundary: this theorem proves only algebraic recombination. It does
not identify either limb evaluation with a physical column or prove that any
column is commitment-bound.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_rlc.verify.identities.y_zcol.limb0` | embed every `c0` coefficient in `K` | computed | `projectionC0Coefficients` |
| `nifs.pi_rlc.verify.identities.y_zcol.limb1` | embed every `c1` coefficient in `K` | computed | `projectionC1Coefficients` |
| `nifs.pi_rlc.verify.identities.y_zcol.recombine` | `eval(c) = eval(c0) + u * eval(c1)` | derived | `projectionEval_eq_limbEvaluations` |
| `nifs.pi_rlc.verify.authority.packed_y_zcol.parent_scalar` | specialize the split to one fixed-width packed value | derived | `projectedValue_eq_limbEvaluations` |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.Authority.DelayedPackedProjection.LimbDecomposition

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

private abbrev laws := ConcreteCarrier.extensionLaws

local instance : Std.Associative K.add :=
  ⟨laws.add_assoc⟩

local instance : Std.Commutative K.add :=
  ⟨laws.add_comm⟩

local instance : Std.Associative K.mul :=
  ⟨laws.mul_assoc⟩

local instance : Std.Commutative K.mul :=
  ⟨laws.mul_comm⟩

private theorem k_add_zero (value : K) :
    K.add value K.zero = value :=
  laws.add_zero value

private theorem k_mul_zero (value : K) :
    K.mul value K.zero = K.zero :=
  laws.mul_zero value

private theorem k_mul_add (left middle right : K) :
    K.mul left (K.add middle right) =
      K.add (K.mul left middle) (K.mul left right) :=
  laws.left_distrib left middle right

/-- Basis element `u = (0, 1)` of the production quadratic extension. -/
def extensionGenerator : K := ⟨0, 1⟩

/-- Lift each first base-field coefficient limb into `K`. -/
def projectionC0Coefficients (values : List K) : List K :=
  values.map fun value => K.embed value.c0

/-- Lift each second base-field coefficient limb into `K`. -/
def projectionC1Coefficients (values : List K) : List K :=
  values.map fun value => K.embed value.c1

private theorem k_eq_limb_decomposition (value : K) :
    value = K.add (K.embed value.c0)
      (K.mul extensionGenerator (K.embed value.c1)) := by
  rcases value with ⟨value0, value1⟩
  simp only [K.add, K.mul, K.embed, extensionGenerator, K.mk.injEq]
  constructor
  · apply Fin.ext
    simp [Fin.val_add, Fin.val_mul, Nat.mod_eq_of_lt value0.isLt]
  · apply Fin.ext
    simp only [Fin.val_add, Fin.val_mul]
    have oneVal : ((1 : F).val) = 1 := by rfl
    have zeroVal : ((0 : F).val) = 0 := by rfl
    rw [oneVal, zeroVal]
    simp [Nat.mod_eq_of_lt value1.isLt]

private theorem k_hornerStep_eq_limbSteps
    (head producerBeta tail0 tail1 : K) :
    K.add head
        (K.add (K.mul producerBeta tail0)
          (K.mul producerBeta (K.mul extensionGenerator tail1))) =
      K.add
        (K.add (K.embed head.c0) (K.mul producerBeta tail0))
        (K.add (K.mul extensionGenerator (K.embed head.c1))
          (K.mul extensionGenerator (K.mul producerBeta tail1))) := by
  calc
    K.add head
        (K.add (K.mul producerBeta tail0)
          (K.mul producerBeta (K.mul extensionGenerator tail1))) =
      K.add
        (K.add (K.embed head.c0)
          (K.mul extensionGenerator (K.embed head.c1)))
        (K.add (K.mul producerBeta tail0)
          (K.mul producerBeta (K.mul extensionGenerator tail1))) :=
      congrArg
        (fun value => K.add value
          (K.add (K.mul producerBeta tail0)
            (K.mul producerBeta (K.mul extensionGenerator tail1))))
        (k_eq_limb_decomposition head)
    _ = _ := by ac_rfl

/-- Constant-first Horner evaluation commutes with the production two-limb
coefficient split. Recombination is a fixed linear map, not another
degree-53 polynomial evaluation. -/
theorem projectionEval_eq_limbEvaluations
    (values : List K) (producerBeta : K) :
    ProjectionCheck.eval projectionOps values producerBeta =
      K.add
        (ProjectionCheck.eval projectionOps
          (projectionC0Coefficients values) producerBeta)
        (K.mul extensionGenerator
          (ProjectionCheck.eval projectionOps
            (projectionC1Coefficients values) producerBeta)) := by
  induction values with
  | nil =>
      change K.zero = K.add K.zero (K.mul extensionGenerator K.zero)
      rw [k_mul_zero, k_add_zero]
  | cons head tail inductionHypothesis =>
      change K.add head
          (K.mul producerBeta
            (ProjectionCheck.eval projectionOps tail producerBeta)) =
        K.add
          (K.add (K.embed head.c0)
            (K.mul producerBeta
              (ProjectionCheck.eval projectionOps
                (projectionC0Coefficients tail) producerBeta)))
          (K.mul extensionGenerator
            (K.add (K.embed head.c1)
              (K.mul producerBeta
                (ProjectionCheck.eval projectionOps
                  (projectionC1Coefficients tail) producerBeta))))
      rw [inductionHypothesis, k_mul_add, k_mul_add]
      exact k_hornerStep_eq_limbSteps head producerBeta
        (ProjectionCheck.eval projectionOps
          (projectionC0Coefficients tail) producerBeta)
        (ProjectionCheck.eval projectionOps
          (projectionC1Coefficients tail) producerBeta)

/-- Fixed-width specialization used by the parent-projection authority
boundary. -/
theorem projectedValue_eq_limbEvaluations
    (value : RingK) (producerBeta : K) :
    projectedValue value producerBeta =
      K.add
        (ProjectionCheck.eval projectionOps
          (projectionC0Coefficients (coefficients value)) producerBeta)
        (K.mul extensionGenerator
          (ProjectionCheck.eval projectionOps
            (projectionC1Coefficients (coefficients value)) producerBeta)) :=
  projectionEval_eq_limbEvaluations (coefficients value) producerBeta

end Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.Authority.DelayedPackedProjection.LimbDecomposition
