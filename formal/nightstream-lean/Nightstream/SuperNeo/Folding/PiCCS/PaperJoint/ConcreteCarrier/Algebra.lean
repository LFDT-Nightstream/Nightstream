import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConstraintPolynomialLift.Evaluation
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.NormResidualTable

/-!
Concrete carrier algebra for paper joint `Pi_CCS`.

Protocol: SuperNeo `Pi_CCS` (Section 7.3 / Appendix D.4).
Phase: base/extension arithmetic and algebraic carrier placement.
Constraint family: mathematical field/ring operations only; this file emits
no rows.

Owns: canonical `InterpolationOps` for Goldilocks `F` and the quadratic
extension `K`; their interpolation laws; derived extension subtraction;
base-zero agreement; zero/one/add/mul preservation by `K.embed`; and the
direct sparse-polynomial `EvaluationLaws` instance for that embedding.

Does not own: protocol-level zero reflection, strict-norm compatibility,
`ProtocolDataRefinement.ProtocolLift`, verifier composition, a proof of the
Goldilocks modulus Euclid property, transcript hashing, SumCheck degree,
Rust/R1CS refinement, row removal, or counts.

Emits constraints: no.

Authority boundary: all operations are definitions over the already-owned
concrete `F` and `K` types. The embedding laws are proved directly from those
definitions; callers do not supply arithmetic or evaluation callbacks.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `pi_ccs.carrier.base.ops` | Goldilocks zero/one/add/mul/neg | computed | `baseOps` |
| `pi_ccs.carrier.base.laws` | interpolation algebra over `F` | derived | `baseLaws` |
| `pi_ccs.carrier.extension.ops` | quadratic-extension zero/one/add/mul/neg | computed | `extensionOps` |
| `pi_ccs.carrier.extension.laws` | interpolation algebra over `K` | derived | `extensionLaws` |
| `pi_ccs.carrier.extension.sub` | derived subtraction equals concrete `K.sub` | derived | `derived_sub_eq_concrete_sub` |
| `pi_ccs.carrier.embed.algebra` | embedding preserves zero/one/add/mul | derived | `embed_zero`, `embed_one`, `embed_add`, `embed_mul` |
| `pi_ccs.ccs.lift.algebra` | concrete sparse-polynomial evaluation laws | derived | `constraintEvaluationLaws` |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

/-- Canonical paper arithmetic over the production Goldilocks residue type. -/
def baseOps : InterpolationOps F where
  zero := 0
  one := 1
  add := (· + ·)
  mul := (· * ·)
  neg := fun value => 0 - value

/-- Canonical paper arithmetic over the production quadratic extension. -/
def extensionOps : InterpolationOps K where
  zero := K.zero
  one := K.one
  add := K.add
  mul := K.mul
  neg := fun value => K.sub K.zero value

/-- The semantic base zero is definitionally the concrete Goldilocks zero. -/
def baseZeroAgreement : NormResidualTable.BaseZeroAgreement baseOps where
  zero_eq := rfl

private theorem fadd_assoc (left middle right : F) :
    (left + middle) + right = left + (middle + right) :=
  Lean.Grind.Fin.add_assoc _ _ _

private theorem fadd_comm (left right : F) : left + right = right + left :=
  Lean.Grind.Fin.add_comm _ _

private theorem fmul_assoc (left middle right : F) :
    (left * middle) * right = left * (middle * right) :=
  Fin.mul_assoc _ _ _

private theorem fmul_comm (left right : F) : left * right = right * left :=
  Fin.mul_comm _ _

private theorem fmul_add (left middle right : F) :
    left * (middle + right) = left * middle + left * right :=
  Lean.Grind.Fin.left_distrib _ _ _

private theorem fadd_mul (left middle right : F) :
    (left + middle) * right = left * right + middle * right := by
  calc
    (left + middle) * right = right * (left + middle) :=
      fmul_comm _ _
    _ = right * left + right * middle := fmul_add _ _ _
    _ = left * right + middle * right := by
      rw [fmul_comm right left, fmul_comm right middle]

private theorem fadd_neg_cancel (value : F) : value + -value = 0 := by
  rw [fadd_comm]
  exact Lean.Grind.Fin.neg_add_cancel value

private theorem fneg_add (left right : F) :
    -(left + right) = -left + -right :=
  Lean.Grind.AddCommGroup.neg_add _ _

private theorem fmul_neg (left right : F) :
    left * -right = -(left * right) := by
  calc
    left * -right = (-right) * left := fmul_comm _ _
    _ = -(right * left) := Lean.Grind.Fin.neg_mul _ _
    _ = -(left * right) := congrArg Neg.neg (fmul_comm _ _)

local instance : Std.Associative (fun (left right : F) => left + right) :=
  ⟨fadd_assoc⟩

local instance : Std.Commutative (fun (left right : F) => left + right) :=
  ⟨fadd_comm⟩

local instance : Std.Associative (fun (left right : F) => left * right) :=
  ⟨fmul_assoc⟩

local instance : Std.Commutative (fun (left right : F) => left * right) :=
  ⟨fmul_comm⟩

/-- The concrete Goldilocks operations satisfy every algebraic law needed by
finite matrix images and coefficient-kernel expansion. -/
theorem baseLaws : InterpolationEvaluationLaws baseOps := by
  constructor
  · exact fadd_assoc
  · exact fadd_comm
  · exact Fin.zero_add
  · exact Fin.add_zero
  · exact fmul_assoc
  · exact fmul_comm
  · exact Fin.one_mul
  · exact Fin.mul_one
  · exact Fin.mul_zero
  · exact fmul_add
  · exact fadd_mul
  · exact fadd_neg_cancel
  · exact fneg_add
  · exact Lean.Grind.Fin.neg_mul

private theorem k_mul_assoc (left middle right : K) :
    K.mul (K.mul left middle) right = K.mul left (K.mul middle right) := by
  rcases left with ⟨left0, left1⟩
  rcases middle with ⟨middle0, middle1⟩
  rcases right with ⟨right0, right1⟩
  simp only [K.mul, K.mk.injEq]
  constructor <;> simp only [fmul_add, fadd_mul, fmul_assoc] <;> ac_rfl

private theorem k_mul_comm (left right : K) :
    K.mul left right = K.mul right left := by
  rcases left with ⟨left0, left1⟩
  rcases right with ⟨right0, right1⟩
  simp only [K.mul, K.mk.injEq]
  constructor <;> ac_rfl

private theorem k_mul_add (left middle right : K) :
    K.mul left (K.add middle right) =
      K.add (K.mul left middle) (K.mul left right) := by
  rcases left with ⟨left0, left1⟩
  rcases middle with ⟨middle0, middle1⟩
  rcases right with ⟨right0, right1⟩
  simp only [K.mul, K.add, K.mk.injEq]
  constructor <;> simp only [fmul_add] <;> ac_rfl

private theorem k_add_mul (left middle right : K) :
    K.mul (K.add left middle) right =
      K.add (K.mul left right) (K.mul middle right) := by
  rw [k_mul_comm, k_mul_add]
  congr 1 <;> rw [k_mul_comm]

/-- Subtraction derived by `InterpolationOps` agrees with the concrete
quadratic-extension subtraction on both coefficients. -/
theorem derived_sub_eq_concrete_sub (left right : K) :
    extensionOps.sub left right = K.sub left right := by
  rcases left with ⟨left0, left1⟩
  rcases right with ⟨right0, right1⟩
  simp only [InterpolationOps.sub, extensionOps, K.add, K.sub, K.zero,
    Fin.sub_eq_add_neg, Fin.zero_add]

/-- The concrete quadratic-extension operations satisfy every algebraic law
used by Boolean interpolation, SumCheck truth, and carried evaluation. -/
theorem extensionLaws : InterpolationEvaluationLaws extensionOps := by
  constructor
  · intro left middle right
    rcases left with ⟨left0, left1⟩
    rcases middle with ⟨middle0, middle1⟩
    rcases right with ⟨right0, right1⟩
    simp only [extensionOps, K.add, K.mk.injEq]
    exact ⟨fadd_assoc _ _ _, fadd_assoc _ _ _⟩
  · intro left right
    rcases left with ⟨left0, left1⟩
    rcases right with ⟨right0, right1⟩
    simp only [extensionOps, K.add, K.mk.injEq]
    exact ⟨fadd_comm _ _, fadd_comm _ _⟩
  · intro value
    rcases value with ⟨value0, value1⟩
    simp only [extensionOps, K.add, K.zero, Fin.zero_add]
  · intro value
    rcases value with ⟨value0, value1⟩
    simp only [extensionOps, K.add, K.zero, Fin.add_zero]
  · intro left middle right
    exact k_mul_assoc left middle right
  · intro left right
    exact k_mul_comm left right
  · intro value
    rcases value with ⟨value0, value1⟩
    simp only [extensionOps, K.mul, K.one, Fin.one_mul,
      Fin.zero_mul, Fin.mul_zero, Fin.add_zero]
  · intro value
    rcases value with ⟨value0, value1⟩
    simp only [extensionOps, K.mul, K.one, Fin.mul_one,
      Fin.mul_zero, Fin.add_zero, Fin.zero_add]
  · intro value
    rcases value with ⟨value0, value1⟩
    simp only [extensionOps, K.mul, K.zero, Fin.mul_zero,
      Fin.add_zero]
  · intro left middle right
    exact k_mul_add left middle right
  · intro left middle right
    exact k_add_mul left middle right
  · intro value
    rcases value with ⟨value0, value1⟩
    simp only [extensionOps, K.add, K.sub, K.zero, K.mk.injEq,
      Fin.sub_eq_add_neg, Fin.zero_add]
    exact ⟨fadd_neg_cancel _, fadd_neg_cancel _⟩
  · intro left right
    rcases left with ⟨left0, left1⟩
    rcases right with ⟨right0, right1⟩
    simp only [extensionOps, K.add, K.sub, K.zero, K.mk.injEq,
      Fin.sub_eq_add_neg, Fin.zero_add]
    exact ⟨fneg_add _ _, fneg_add _ _⟩
  · intro left right
    rcases left with ⟨left0, left1⟩
    rcases right with ⟨right0, right1⟩
    simp only [extensionOps, K.mul, K.sub, K.zero,
      Fin.sub_eq_add_neg, Fin.zero_add, Lean.Grind.Fin.neg_mul, fmul_neg,
      fneg_add]

/-- The zero laws used by the coefficient transform are a direct projection
of the stronger concrete evaluation laws. -/
def extensionZeroLaws : InterpolationZeroLaws extensionOps where
  add_zero := extensionLaws.add_zero
  neg_zero := by
    simp only [extensionOps, K.sub, K.zero, Fin.sub_self]

/-- The concrete embedding maps base zero to extension zero. -/
theorem embed_zero : K.embed baseOps.zero = extensionOps.zero := by
  rfl

/-- The concrete embedding maps the base unit to the extension unit. -/
theorem embed_one : K.embed baseOps.one = extensionOps.one := by
  rfl

/-- The concrete embedding commutes with addition. -/
theorem embed_add (left right : F) :
    K.embed (baseOps.add left right) =
      extensionOps.add (K.embed left) (K.embed right) := by
  simp only [baseOps, extensionOps, K.embed, K.add,
    Fin.add_zero]

/-- The concrete embedding commutes with multiplication. -/
theorem embed_mul (left right : F) :
    K.embed (baseOps.mul left right) =
      extensionOps.mul (K.embed left) (K.embed right) := by
  simp only [baseOps, extensionOps, K.embed, K.mul,
    Fin.mul_zero, Fin.zero_mul, Fin.add_zero]

/-- Direct concrete algebraic contract for structurally lifting sparse CCS
polynomials from `F` to `K`. This contract is independent of norm semantics
and the high-level protocol verifier. -/
def constraintEvaluationLaws :
    ConstraintPolynomialLift.Evaluation.EvaluationLaws
      baseOps extensionOps K.embed where
  map_zero := embed_zero
  map_one := embed_one
  map_add := embed_add
  map_mul := embed_mul

end Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier
