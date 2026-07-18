import Nightstream.Implementation.R1CS.Correspondence.PiCcsNc.Authority.DelayedResidual
import Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.Authority.DelayedPackedProjection.LimbDecomposition
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier.Algebra
import Nightstream.SuperNeo.ProjectionCheck

/-!
Contract: define the fixed-width one-point projection identity used to carry
the delayed Π_CCS NC residual compactly.

Assurance tier: model-level.

Owns: the 54 active child coefficients, their canonical 64-lane padded shape,
the two base-limb evaluations, their zero-row recombination, and the exact-or-
degree-53-bad-root result for one accepted parent/child projection identity.

Does not own: generated parent output columns, beta-ladder rows, transcript
timing, parent commitment opening, Π_DEC child authority, concrete R1CS rows,
physical cost, or row-removal permission.

Emits constraints: no.

Authority boundary: the raw children and claimed parent coefficients must be
fixed independently before `producerBeta` is sampled. A matching scalar is
authority only up to the named bad-root event; neither a role label nor a
self-consistent digest binds the claimed coefficients to the parent opening.

| Stage path | Mathematical obligation | Authority class | Lean owner | Permits row removal? |
|---|---|---|---|---|
| `pi_ccs.nc.delayed_projection.child_coefficients` | radix-recompose exactly 54 active child coefficients | computed | `rawChildProjectionCoefficients` | no |
| `pi_ccs.nc.delayed_projection.padding` | append exactly ten canonical-zero lanes | computed | `paddedRawChildProjectionCoefficients_drop_active` | no |
| `pi_ccs.nc.delayed_projection.child_limb0` | evaluate the child `c0` coefficient limb at producer beta | computed | `childC0ProjectionEvaluation` | no |
| `pi_ccs.nc.delayed_projection.child_limb1` | evaluate the child `c1` coefficient limb at producer beta | computed | `childC1ProjectionEvaluation` | no |
| `pi_ccs.nc.delayed_projection.recombine` | recover the child scalar as `E0 + u * E1` | derived | `compactOldPointEvaluation_eq_childLimbEvaluations` | no |
| `pi_ccs.nc.delayed_projection.identity` | accepted parent/child scalar equality is exact or a degree-53 bad root | security boundary | `acceptedProjectionIdentity_implies_exact_or_badRoot` | no |
-/

namespace Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual

open Nightstream.Implementation.R1CS.PiCcsNc
open Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedParentProjection
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.Authority.DelayedPackedProjection
open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.Authority.DelayedPackedProjection.LimbDecomposition

private abbrev laws :=
  Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier.extensionLaws

local instance : Std.Associative K.add :=
  ⟨laws.add_assoc⟩

local instance : Std.Commutative K.add :=
  ⟨laws.add_comm⟩

local instance : Std.Associative K.mul :=
  ⟨laws.mul_assoc⟩

local instance : Std.Commutative K.mul :=
  ⟨laws.mul_comm⟩

private theorem k_mul_add (left middle right : K) :
    K.mul left (K.add middle right) =
      K.add (K.mul left middle) (K.mul left right) :=
  laws.left_distrib left middle right

/-- Production active-width radix-combined child `y_zcol` coefficient vector.

Π_RLC evaluates exactly `D = 54` coefficients and separately constrains the
`D .. d_pad` tail to zero. The NC lift still ranges over `shape.laneDomain`;
the concrete padded-tail row refinement remains an explicit open obligation. -/
def rawChildProjectionCoefficients
    (shape : Shape) (radix : F) (rawChildren : List (List F))
    (s : List K) : List K :=
  (List.range ringDegree).map fun lane =>
    radixWeightedChildProjection shape radix rawChildren s lane

/-- Child-side evaluation of the first `y_zcol` coefficient limb.

This is not the retained production `YZColLimb` output column: that column
evaluates the claimed combined parent vector. A separate refinement must
instantiate `projectionEval_eq_limbEvaluations` on the parent coefficients and
identify its two limb evaluations with those retained columns. -/
def childC0ProjectionEvaluation
    (shape : Shape) (radix : F) (rawChildren : List (List F))
    (producerBeta : K) (s : List K) : K :=
  Nightstream.SuperNeo.ProjectionCheck.eval projectionOps
    (projectionC0Coefficients
      (rawChildProjectionCoefficients shape radix rawChildren s))
    producerBeta

/-- Child-side evaluation of the second `y_zcol` coefficient limb.

As above, this is the radix-recomposed child vector, not the retained combined
parent output column. -/
def childC1ProjectionEvaluation
    (shape : Shape) (radix : F) (rawChildren : List (List F))
    (producerBeta : K) (s : List K) : K :=
  Nightstream.SuperNeo.ProjectionCheck.eval projectionOps
    (projectionC1Coefficients
      (rawChildProjectionCoefficients shape radix rawChildren s))
    producerBeta

/-- Production's padded ring-column width: active lanes `0 .. 53`, followed
by zero lanes `54 .. 63`. -/
def projectionPaddedWidth : Nat := 64

/-- Model-level production-shaped `y_zcol`: the active delayed-projection
coefficients followed by the ten canonical-zero padding lanes. -/
def paddedRawChildProjectionCoefficients
    (shape : Shape) (radix : F) (rawChildren : List (List F))
    (s : List K) : List K :=
  rawChildProjectionCoefficients shape radix rawChildren s ++
    List.replicate (projectionPaddedWidth - ringDegree) K.zero

/-- The model-level padded vector has the exact production width 64. -/
theorem paddedRawChildProjectionCoefficients_length
    (shape : Shape) (radix : F) (rawChildren : List (List F))
    (s : List K) :
    (paddedRawChildProjectionCoefficients
      shape radix rawChildren s).length = 64 := by
  simp [paddedRawChildProjectionCoefficients,
    rawChildProjectionCoefficients, projectionPaddedWidth, ringDegree]

/-- Lanes `54 .. 63` of the reconstructed child vector are exactly the
ten-zero suffix. This matches the fixed profile's padding geometry, but does
not identify the generated Rust padding pins, which constrain separate PiRLC
input and combined-parent vectors. -/
theorem paddedRawChildProjectionCoefficients_drop_active
    (shape : Shape) (radix : F) (rawChildren : List (List F))
    (s : List K) :
    (paddedRawChildProjectionCoefficients
      shape radix rawChildren s).drop ringDegree =
      List.replicate 10 K.zero := by
  simp [paddedRawChildProjectionCoefficients,
    rawChildProjectionCoefficients, projectionPaddedWidth, ringDegree]

private theorem projectionEval_append_single
    (coefficients : List K) (coefficient producerBeta : K) :
    Nightstream.SuperNeo.ProjectionCheck.eval projectionOps
        (coefficients ++ [coefficient]) producerBeta =
      K.add
        (Nightstream.SuperNeo.ProjectionCheck.eval projectionOps
          coefficients producerBeta)
        (K.mul coefficient (powK producerBeta coefficients.length)) := by
  induction coefficients with
  | nil =>
      change K.add coefficient (K.mul producerBeta K.zero) =
        K.add K.zero (K.mul coefficient K.one)
      rw [mul_zero, add_zero, mul_one, zero_add]
  | cons head tail inductionHypothesis =>
      change K.add head
          (K.mul producerBeta
            (Nightstream.SuperNeo.ProjectionCheck.eval projectionOps
              (tail ++ [coefficient]) producerBeta)) =
        K.add
          (K.add head
            (K.mul producerBeta
              (Nightstream.SuperNeo.ProjectionCheck.eval projectionOps
                tail producerBeta)))
          (K.mul coefficient (powK producerBeta (tail.length + 1)))
      rw [inductionHypothesis, k_mul_add, powK]
      ac_rfl

private theorem projectionEval_range
    (count : Nat) (coefficient : Nat → K) (producerBeta : K) :
    Nightstream.SuperNeo.ProjectionCheck.eval projectionOps
        ((List.range count).map coefficient) producerBeta =
      sumRange count fun lane =>
        K.mul (coefficient lane) (powK producerBeta lane) := by
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      rw [List.range_succ, List.map_append]
      simp only [List.map_singleton]
      rw [projectionEval_append_single, inductionHypothesis]
      simp only [List.length_map, List.length_range]
      rw [sumRange]

/-- The child-side scalar of the fixed-width delayed projection identity at
`producerBeta`. Current production rows retain the analogous parent-side
scalar; their concrete connection remains a separate refinement. -/
def compactOldPointEvaluation
    (shape : Shape) (radix : F) (rawChildren : List (List F))
    (producerBeta : K) (s : List K) : K :=
  Nightstream.SuperNeo.ProjectionCheck.eval projectionOps
    (rawChildProjectionCoefficients shape radix rawChildren s)
    producerBeta

/-- The delayed child-side compact scalar is recoverable from its two
base-limb evaluations by the fixed linear extension-basis map. No second
coefficient traversal or degree-53 evaluation is mathematically required.

This theorem does not identify those child-side evaluations with production's
retained parent-output columns. That later bridge must instantiate the generic
limb theorem on the claimed parent coefficients and then use the delayed
projection identity to relate the parent and child evaluations. -/
theorem compactOldPointEvaluation_eq_childLimbEvaluations
    (shape : Shape) (radix : F) (rawChildren : List (List F))
    (producerBeta : K) (s : List K) :
    compactOldPointEvaluation
        shape radix rawChildren producerBeta s =
      K.add
        (childC0ProjectionEvaluation
          shape radix rawChildren producerBeta s)
        (K.mul extensionGenerator
          (childC1ProjectionEvaluation
            shape radix rawChildren producerBeta s)) := by
  exact projectionEval_eq_limbEvaluations
    (rawChildProjectionCoefficients shape radix rawChildren s)
    producerBeta

/-- Horner evaluation of the fixed-width projection vector is exactly the
active `sum_{lane < 54} y_zcol[lane] * beta^lane` used by the delayed cube
normalization theorem. -/
theorem compactOldPointEvaluation_eq_active
    (shape : Shape) (radix : F) (rawChildren : List (List F))
    (producerBeta : K) (s : List K) :
    compactOldPointEvaluation
        shape radix rawChildren producerBeta s =
      activeRawProjectionAtProducerBeta
        shape radix rawChildren producerBeta s := by
  unfold compactOldPointEvaluation rawChildProjectionCoefficients
    activeRawProjectionAtProducerBeta
  exact projectionEval_range ringDegree
    (fun lane =>
      radixWeightedChildProjection shape radix rawChildren s lane)
    producerBeta

/-- The production-shaped cube theorem expressed directly through the scalar
evaluated by the fixed-width projection identity. -/
theorem delayedResidualCubeSum_eq_weightedProjectionEvaluation
    (shape : Shape) (radix : F) (rawChildren : List (List F))
    (producerBeta batchWeight : K) (oldS : List K)
    (wellShaped : DelayedResidualShape shape oldS rawChildren) :
    delayedResidualCubeSum shape radix rawChildren
        producerBeta batchWeight oldS =
      K.mul batchWeight
        (compactOldPointEvaluation
          shape radix rawChildren producerBeta oldS) := by
  rw [compactOldPointEvaluation_eq_active]
  exact delayedResidualCubeSum_eq_weightedCompactOldProjection
    shape radix rawChildren producerBeta batchWeight oldS wellShaped

/-- The bounded polynomial identity whose one-point evaluation can be carried
as the compact delayed handle. The claimed parent vector is an explicit input;
constructing this value does not prove it was accumulator-bound before
`producerBeta` was sampled. -/
def projectionIdentity
    (shape : Shape) (radix : F) (rawChildren : List (List F))
    (s : List K) (claimedParentCoefficients : List K)
    (producerBeta : K) :
    Nightstream.SuperNeo.ProjectionCheck.Identity K where
  lhs := claimedParentCoefficients
  rhs := rawChildProjectionCoefficients shape radix rawChildren s
  beta := producerBeta
  maxDegree := ringDegree - 1

/-- The identity's child-side scalar is the compact old-point evaluation used
by the delayed residual. -/
theorem projectionIdentity_rhsEvaluation_eq_compact
    (shape : Shape) (radix : F) (rawChildren : List (List F))
    (s : List K) (claimedParentCoefficients : List K)
    (producerBeta : K) :
    Nightstream.SuperNeo.ProjectionCheck.eval projectionOps
        (projectionIdentity shape radix rawChildren s
          claimedParentCoefficients producerBeta).rhs
        producerBeta =
      compactOldPointEvaluation
        shape radix rawChildren producerBeta s := by
  rfl

/-- The concrete production identity has 54 coefficients and maximum degree
53. -/
theorem projectionIdentity_activeWidth
    (shape : Shape) (radix : F) (rawChildren : List (List F))
    (s : List K) (claimedParentCoefficients : List K)
    (producerBeta : K) :
    (projectionIdentity shape radix rawChildren s
        claimedParentCoefficients producerBeta).rhs.length = 54 ∧
      (projectionIdentity shape radix rawChildren s
        claimedParentCoefficients producerBeta).maxDegree = 53 := by
  unfold projectionIdentity rawChildProjectionCoefficients ringDegree
  simp

/-- With the expected fixed width, the compact projection identity satisfies
the generic degree/representation side condition. -/
theorem projectionIdentity_wellFormed
    (shape : Shape) (radix : F) (rawChildren : List (List F))
    (s : List K) (claimedParentCoefficients : List K)
    (producerBeta : K)
    (claimedLength : claimedParentCoefficients.length = ringDegree) :
    (projectionIdentity shape radix rawChildren s
      claimedParentCoefficients producerBeta).WellFormed := by
  unfold Nightstream.SuperNeo.ProjectionCheck.Identity.WellFormed
    projectionIdentity rawChildProjectionCoefficients
  simp only [List.length_map, List.length_range, claimedLength, true_and]
  have positive : 0 < ringDegree := by decide
  omega

/-- Reuse of the generic one-point soundness boundary: acceptance yields exact
fixed-width coefficients or identifies the producer challenge as a bad root.
No premise asserts either branch of the conclusion. -/
theorem acceptedProjectionIdentity_implies_exact_or_badRoot
    (shape : Shape) (radix : F) (rawChildren : List (List F))
    (s : List K) (claimedParentCoefficients : List K)
    (producerBeta : K)
    (accepted : Nightstream.SuperNeo.ProjectionCheck.Accepted projectionOps
      (projectionIdentity shape radix rawChildren s
        claimedParentCoefficients producerBeta)) :
    (projectionIdentity shape radix rawChildren s
        claimedParentCoefficients producerBeta).Exact ∨
      Nightstream.SuperNeo.ProjectionCheck.BadRoot projectionOps
        (projectionIdentity shape radix rawChildren s
          claimedParentCoefficients producerBeta) :=
  Nightstream.SuperNeo.ProjectionCheck.accepted_implies_exact_or_badRoot
    projectionOps
    (projectionIdentity shape radix rawChildren s
      claimedParentCoefficients producerBeta)
    accepted

end Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual
