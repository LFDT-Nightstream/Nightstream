import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.UnifiedProtocolVerifier

/-!
Concrete Goldilocks-to-quadratic-extension carrier for paper joint `Pi_CCS`.

Protocol: SuperNeo `Pi_CCS` (Section 7.3 / Appendix D.4).
Phase: concrete instantiation of the semantic arithmetic carrier.
Constraint family: mathematical field/ring operations only; this file emits
no rows.

Owns: the canonical `InterpolationOps` records for Goldilocks `F` and the
quadratic extension `K = F[X]/(X^2 - 7)`; the algebraic laws used by the
paper-polynomial verifier; exact base-zero agreement; and discharge of every
`ProtocolDataRefinement.ProtocolLift` field for `K.embed`.

Does not own: a proof of the modulus-level Euclid property used to exclude
Goldilocks zero divisors, proof that Rust field arithmetic refines these
definitions, coefficient-expanded matrix derivation, transcript hashing,
SumCheck degree bounds, output CE projection, Pi_RLC handoff, R1CS, row
removal, or counts.

Emits constraints: no.

Authority boundary: operations are definitions over the already-owned
concrete `F` and `K` types. No caller supplies algebraic law proofs or a
base-to-extension function in the concrete theorem path. The still-unproved
Goldilocks modulus Euclid property remains an explicit premise rather than
being hidden inside the carrier instance.

| Protocol | Phase | Family | Concrete owner / result |
|---|---|---|---|
| `Pi_CCS` | base arithmetic | Goldilocks add/mul/neg | `baseOps` |
| assurance | base algebra | interpolation evaluation laws | `baseLaws` |
| `Pi_CCS` | extension arithmetic | `K` add/mul/neg | `extensionOps` |
| assurance | extension algebra | interpolation evaluation laws | `extensionLaws` |
| assurance | base norm root | semantic zero is Goldilocks zero | `baseZeroAgreement` |
| assurance | carrier placement | `F -> K` | `protocolLift` |
| open arithmetic | no zero divisors | modulus-level Euclid property | explicit theorem premise |
| assurance | unified verifier | no abstract lift/law premise | `check_implies_semanticTruth_or_badEvent` |
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

private theorem k_embed_strictNorm (value : F) :
    K.mul (K.mul (K.add (K.embed value) K.one) (K.embed value))
        (K.sub (K.embed value) K.one) =
      K.embed (NormRange.cubicResidual value) := by
  rw [show K.one = K.embed 1 from rfl]
  exact NormRange.embed_cubicResidual value

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

/-- The concrete embedding preserves and reflects the semantic zero. -/
def zeroReflectingLift :
    ConcreteJointData.ZeroReflectingLift baseOps extensionOps K.embed where
  zero_iff := by
    intro value
    constructor
    · intro equal
      have component := congrArg K.c0 equal
      simpa only [K.embed, K.zero, baseOps, extensionOps] using component
    · intro equal
      subst equal
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

/-- Applying the extension-carrier norm polynomial after embedding is exactly
embedding the independent concrete base cubic. -/
theorem embed_strictNorm (value : F) :
    ProtocolPolynomial.strictNormResidual extensionOps (K.embed value) =
      K.embed (NormRange.cubicResidual value) := by
  unfold ProtocolPolynomial.strictNormResidual
  rw [derived_sub_eq_concrete_sub]
  exact k_embed_strictNorm value

/-- Concrete `F -> K` placement assembled solely from the named leaf
theorems above. -/
def protocolLift :
    ProtocolDataRefinement.ProtocolLift baseOps extensionOps K.embed where
  toZeroReflectingLift := zeroReflectingLift
  map_one := embed_one
  map_add := embed_add
  map_mul := embed_mul
  map_strictNorm := embed_strictNorm

/-- Concrete deterministic semantic soundness. Verifier context, certificate,
source data, the explicit modulus-Euclid premise, degree/cardinality
parameters, and acceptance remain as inputs; the carrier and its placement
laws are fixed here. -/
theorem check_implies_semanticTruth_or_badEvent
    {Context : Type}
    {State : Type}
    {shape : Shape}
    {columns : Nat}
    (oracle : ProtocolVerifier.Oracle Context K State shape)
    (context : Context)
    (data : UnifiedSources.UnifiedInputs K shape columns)
    (euclid : NormRange.GoldilocksModulusEuclid)
    (maxDegree challengeSetSize : Nat)
    (certificate : ProtocolVerifier.Certificate K shape)
    (checked : ProtocolVerifier.check oracle context extensionOps
      (ProtocolDataRefinement.toProtocolData baseOps K.embed data)
      maxDegree certificate = true) :
    let protocolData :=
      ProtocolDataRefinement.toProtocolData baseOps K.embed data
    let execution := ProtocolVerifier.derive oracle context certificate
    data.SemanticTruth baseOps extensionOps K.embed \/
      SignedCoefficientObject.MixingRoot extensionOps
        (protocolData.toJointData extensionOps)
        execution.coins.alpha execution.coins.gamma \/
      (exists round,
        Nightstream.SuperNeo.SumCheck.BadChallenge
          (SumCheckInitial.symbolicInstance extensionOps
            (protocolData.toJointData extensionOps)
            execution.coins.alpha execution.coins.gamma maxDegree
            challengeSetSize execution.coins.roundPoint.coordinates
            (ProtocolPolynomial.terminalFromMessage extensionOps protocolData
              execution.coins.alpha execution.coins.gamma
              execution.coins.roundPoint certificate.output)
            certificate.toFinite
            (ProtocolPolynomial.canonicalExpected extensionOps protocolData
              execution.coins.alpha execution.coins.gamma
              execution.coins.roundPoint.coordinates))
          round) \/
      ProtocolPolynomial.OutputMismatch extensionOps protocolData
        execution.coins.alpha execution.coins.gamma
        execution.coins.roundPoint certificate.output := by
  exact UnifiedProtocolVerifier.check_implies_semanticTruth_or_badEvent
    oracle context baseOps baseZeroAgreement
    (NormRange.baseFieldNoZeroDivisors_of_modulusEuclid euclid) extensionOps
    extensionLaws extensionZeroLaws K.embed protocolLift data
    maxDegree challengeSetSize certificate checked

end Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier
