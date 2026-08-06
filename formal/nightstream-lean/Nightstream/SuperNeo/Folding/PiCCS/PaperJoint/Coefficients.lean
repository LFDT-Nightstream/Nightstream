import Nightstream.SuperNeo.SumCheck.Polynomial

/-!
Finite coefficient ownership for the paper's single joint `Pi_CCS` polynomial.

Owns: the selected Section 7.3 / Appendix D.4 gamma-block index layout, finite
alpha coefficient vectors, the three formula-agnostic residual families, and
the theorem reducing joint coefficient truth to three independently stated
obligation families.

Does not own: construction of residuals from CCS matrices or witnesses,
SumCheck rounds, sampled challenges, root counting, Fiat--Shamir, the
production data-layout refinement, Rust, R1CS, or constraint counts.

Emits constraints: no.

Authority boundary: a polynomial is finite coefficient data over one
verifier-owned alpha monomial basis. No evaluation function or degree claim is
prover data. The block positions do not construct any CCS, norm, target, or
evaluation formula and do not prove the signed joint identity.
`ResidualizationBoundary` is an explicit, unclosed assumption relating
caller-stated semantic obligations to these residual coefficients; it is not
itself a proof of Appendix D.4, Lemma 7.

| Gamma coefficient block | Selected positions | Stored data | Zero obligation |
|---|---|---:|---|
| CCS | `0 .. K - 1` | alpha-polynomial residuals | every entry is zero |
| norm | `K .. 2K + k - 1` | alpha-polynomial residuals | every entry is zero |
| carried evaluation | `2K+k .. 2K+k+ktd-1` | scalar residuals | every entry is zero |

The corrected paper defines the local helper `T_local` and selects
`T_abs = C^(2K+k) * T_local` for the joint identity. `TargetConvention`
records this distinction. This file does not formalize either target or the
joint polynomial. No theorem here is literal Lemma 7.
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

open Nightstream.SuperNeo.SumCheck

universe uField uLeft uRight

/-- Paper dimensions used by the coefficient layout.

`freshCount`, `runningCount`, `matrixCount`, and `coefficientCount` correspond
to `K`, `k`, `t`, and `d`. `cubeVariables` is `log m`. This is deliberately
paper-shaped rather than production-mode-shaped. -/
structure Shape where
  cubeVariables : Nat
  freshCount : Nat
  runningCount : Nat
  matrixCount : Nat
  coefficientCount : Nat
deriving Repr, DecidableEq

namespace Shape

/-- Number of CCS/CE sources whose norms are checked. -/
def sourceCount (shape : Shape) : Nat :=
  shape.freshCount + shape.runningCount

/-- Number of scalar carried-evaluation residuals (`k * t * d`). -/
def carriedEvaluationCount (shape : Shape) : Nat :=
  shape.runningCount * shape.matrixCount * shape.coefficientCount

/-- First gamma exponent owned by the norm residual block. -/
def normOffset (shape : Shape) : Nat :=
  shape.freshCount

/-- First gamma exponent owned by the carried-evaluation block. -/
def carriedEvaluationOffset (shape : Shape) : Nat :=
  shape.freshCount + shape.sourceCount

/-- Number of gamma coefficients in the complete joint residual polynomial. -/
def jointCoefficientCount (shape : Shape) : Nat :=
  shape.freshCount + shape.sourceCount + shape.carriedEvaluationCount

/-- The paper's `2K + k` carried-evaluation offset. -/
theorem carriedEvaluationOffset_eq (shape : Shape) :
    shape.carriedEvaluationOffset = 2 * shape.freshCount + shape.runningCount := by
  simp only [carriedEvaluationOffset, sourceCount]
  omega

end Shape

/-- A zero-based coordinate for the paper's carried-evaluation coefficient
`I(i,j,l)`. `running`, `matrix`, and `coefficient` encode respectively
`i-(K+1)`, `j-1`, and `l-1`. -/
structure CarriedCoordinate (shape : Shape) where
  running : Fin shape.runningCount
  matrix : Fin shape.matrixCount
  coefficient : Fin shape.coefficientCount
deriving Repr, DecidableEq

namespace CarriedCoordinate

/-- The paper's local exponent
`I(i,j,l) = (i-(K+1)) + k*(j-1) + k*t*(l-1)`, with zero-based fields. -/
def localGammaExponent {shape : Shape}
    (coordinate : CarriedCoordinate shape) : Nat :=
  coordinate.running.val + shape.runningCount * coordinate.matrix.val +
    shape.runningCount * shape.matrixCount * coordinate.coefficient.val

/-- The absolute gamma exponent after the paper's `2K+k` block shift. -/
def gammaExponent {shape : Shape}
    (coordinate : CarriedCoordinate shape) : Nat :=
  shape.carriedEvaluationOffset + coordinate.localGammaExponent

end CarriedCoordinate

/-- Canonical increasing enumeration of a finite index type. -/
def canonicalFinIndices (count : Nat) : List (Fin count) :=
  List.ofFn id

/-- The canonical finite-index enumeration has the declared size. -/
theorem canonicalFinIndices_length (count : Nat) :
    (canonicalFinIndices count).length = count := by
  simp [canonicalFinIndices]

/-- The canonical finite-index enumeration contains no repeated coordinate. -/
theorem canonicalFinIndices_nodup (count : Nat) :
    (canonicalFinIndices count).Nodup := by
  induction count with
  | zero => simp [canonicalFinIndices]
  | succ count inductionHypothesis =>
      rw [canonicalFinIndices, List.ofFn_succ]
      apply List.nodup_cons.mpr
      constructor
      · intro member
        rcases List.mem_ofFn.mp member with ⟨index, equal⟩
        exact Fin.succ_ne_zero index equal
      · have mapped :
            ((canonicalFinIndices count).map Fin.succ).Nodup := by
          exact inductionHypothesis.map Fin.succ (by
            intro left right different
            intro equal
            exact different (Fin.succ_inj.mp equal))
        simpa [canonicalFinIndices, Function.comp_def] using mapped

/-- The typed finite-index traversal is exactly `0, ..., count - 1` after
forgetting the bounds. This is the positional fact used by gamma blocks. -/
theorem canonicalFinIndices_values (count : Nat) :
    (canonicalFinIndices count).map (fun index => index.val) =
      List.range count := by
  apply List.ext_get
  · simp [canonicalFinIndices]
  · intro index leftLt rightLt
    simp [canonicalFinIndices]

private theorem canonical_length_flatMap_constant
    {Index : Type uLeft}
    {Value : Type uRight}
    (indices : List Index)
    (values : Index -> List Value)
    (count : Nat)
    (eachLength : forall index, (values index).length = count) :
    (indices.flatMap values).length = indices.length * count := by
  induction indices with
  | nil => simp
  | cons index indices inductionHypothesis =>
      simp [eachLength, inductionHypothesis, Nat.add_mul, Nat.add_comm]

/-- The sole canonical traversal of carried coordinates. `running` changes
fastest, then `matrix`, then `coefficient`, matching the paper's zero-based
`I(i,j,l)` formula. -/
def canonicalCarriedCoordinates (shape : Shape) :
    List (CarriedCoordinate shape) :=
  (canonicalFinIndices shape.coefficientCount).flatMap fun coefficient =>
    (canonicalFinIndices shape.matrixCount).flatMap fun matrix =>
      (canonicalFinIndices shape.runningCount).map fun running =>
        { running := running
          matrix := matrix
          coefficient := coefficient }

/-- The canonical carried traversal has exactly `k * t * d` entries. -/
theorem canonicalCarriedCoordinates_length (shape : Shape) :
    (canonicalCarriedCoordinates shape).length =
      shape.carriedEvaluationCount := by
  unfold canonicalCarriedCoordinates
  calc
    _ = (canonicalFinIndices shape.coefficientCount).length *
        (shape.matrixCount * shape.runningCount) := by
      apply canonical_length_flatMap_constant
      intro coefficient
      calc
        _ = (canonicalFinIndices shape.matrixCount).length *
            shape.runningCount := by
          apply canonical_length_flatMap_constant
          intro matrix
          simp [canonicalFinIndices]
        _ = shape.matrixCount * shape.runningCount := by
          simp [canonicalFinIndices]
    _ = shape.carriedEvaluationCount := by
      simp [canonicalFinIndices, Shape.carriedEvaluationCount,
        Nat.mul_comm, Nat.mul_left_comm]

private theorem flatMap_congr_local
    {Index : Type uLeft}
    {Value : Type uRight}
    {indices : List Index}
    {left right : Index -> List Value}
    (equal : forall index, index ∈ indices -> left index = right index) :
    indices.flatMap left = indices.flatMap right := by
  induction indices with
  | nil => rfl
  | cons index indices inductionHypothesis =>
      simp only [List.flatMap_cons]
      rw [equal index (by simp)]
      congr 1
      exact inductionHypothesis fun prior member =>
        equal prior (by simp [member])

/-- Consecutive fixed-width ranges concatenate to one consecutive range. This
is the arithmetic fact used to audit the nested carried-coordinate order. -/
private theorem range_flatMap_chunks
    (start chunk blockCount : Nat) :
    (List.range blockCount).flatMap (fun block =>
        List.range' (start + chunk * block) chunk) =
      List.range' start (blockCount * chunk) := by
  induction blockCount with
  | zero => simp
  | succ blockCount inductionHypothesis =>
      rw [List.range_succ, List.flatMap_append]
      simp only [List.flatMap_cons, List.flatMap_nil, List.append_nil]
      rw [inductionHypothesis]
      have blockStart :
          start + chunk * blockCount = start + blockCount * chunk := by
        rw [Nat.mul_comm chunk blockCount]
      have totalWidth :
          (blockCount + 1) * chunk = blockCount * chunk + chunk := by
        rw [Nat.add_mul]
        simp
      rw [blockStart, totalWidth]
      exact List.range'_append_1

private theorem runningExponentBlock
    (shape : Shape)
    (matrix : Fin shape.matrixCount)
    (coefficient : Fin shape.coefficientCount) :
    (canonicalFinIndices shape.runningCount).map (fun running =>
        running.val + shape.runningCount * matrix.val +
          shape.runningCount * shape.matrixCount * coefficient.val) =
      List.range'
        (shape.runningCount * matrix.val +
          shape.runningCount * shape.matrixCount * coefficient.val)
        shape.runningCount := by
  rw [List.range'_eq_map_range, ← canonicalFinIndices_values]
  rw [List.map_map]
  apply List.map_congr_left
  intro running _
  simp only [Function.comp_apply]
  omega

private theorem matrixExponentBlock
    (shape : Shape)
    (coefficient : Fin shape.coefficientCount) :
    (canonicalFinIndices shape.matrixCount).flatMap (fun matrix =>
        (canonicalFinIndices shape.runningCount).map (fun running =>
          running.val + shape.runningCount * matrix.val +
            shape.runningCount * shape.matrixCount * coefficient.val)) =
      List.range'
        (shape.runningCount * shape.matrixCount * coefficient.val)
        (shape.runningCount * shape.matrixCount) := by
  calc
    _ = (canonicalFinIndices shape.matrixCount).flatMap (fun matrix =>
        List.range'
          (shape.runningCount * matrix.val +
            shape.runningCount * shape.matrixCount * coefficient.val)
          shape.runningCount) := by
      apply flatMap_congr_local
      intro matrix _
      exact runningExponentBlock shape matrix coefficient
    _ = (canonicalFinIndices shape.matrixCount).flatMap (fun matrix =>
        List.range'
          (shape.runningCount * shape.matrixCount * coefficient.val +
            shape.runningCount * matrix.val)
          shape.runningCount) := by
      apply flatMap_congr_local
      intro matrix _
      exact congrArg (fun start => List.range' start shape.runningCount)
        (Nat.add_comm
          (shape.runningCount * matrix.val)
          (shape.runningCount * shape.matrixCount * coefficient.val))
    _ = ((canonicalFinIndices shape.matrixCount).map (fun matrix =>
          matrix.val)).flatMap (fun matrix =>
        List.range'
          (shape.runningCount * shape.matrixCount * coefficient.val +
            shape.runningCount * matrix)
          shape.runningCount) := by
      exact (List.flatMap_map
        (fun matrix : Fin shape.matrixCount => matrix.val)
        (fun matrix =>
          List.range'
            (shape.runningCount * shape.matrixCount * coefficient.val +
              shape.runningCount * matrix)
            shape.runningCount)
        (canonicalFinIndices shape.matrixCount)).symm
    _ = (List.range shape.matrixCount).flatMap (fun matrix =>
        List.range'
          (shape.runningCount * shape.matrixCount * coefficient.val +
            shape.runningCount * matrix)
          shape.runningCount) := by
      rw [canonicalFinIndices_values]
    _ = List.range'
        (shape.runningCount * shape.matrixCount * coefficient.val)
        (shape.matrixCount * shape.runningCount) :=
      range_flatMap_chunks
        (shape.runningCount * shape.matrixCount * coefficient.val)
        shape.runningCount shape.matrixCount
    _ = List.range'
        (shape.runningCount * shape.matrixCount * coefficient.val)
        (shape.runningCount * shape.matrixCount) := by
      rw [Nat.mul_comm shape.matrixCount shape.runningCount]

/-- The sole carried-coordinate traversal induces exactly the consecutive
local gamma exponents `0, ..., k*t*d - 1`. Thus the declared formula and the
serialized list order cannot silently disagree by a gap, overlap, or
permutation. -/
theorem canonicalCarriedCoordinates_localGammaExponents (shape : Shape) :
    (canonicalCarriedCoordinates shape).map
        CarriedCoordinate.localGammaExponent =
      List.range shape.carriedEvaluationCount := by
  unfold canonicalCarriedCoordinates CarriedCoordinate.localGammaExponent
  simp only [List.map_flatMap, List.map_map]
  calc
    _ = (canonicalFinIndices shape.coefficientCount).flatMap
        (fun coefficient =>
          List.range'
            (shape.runningCount * shape.matrixCount * coefficient.val)
            (shape.runningCount * shape.matrixCount)) := by
      apply flatMap_congr_local
      intro coefficient _
      exact matrixExponentBlock shape coefficient
    _ = ((canonicalFinIndices shape.coefficientCount).map (fun coefficient =>
          coefficient.val)).flatMap (fun coefficient =>
        List.range'
          (0 + shape.runningCount * shape.matrixCount * coefficient)
          (shape.runningCount * shape.matrixCount)) := by
      simpa only [Nat.zero_add] using (List.flatMap_map
        (fun coefficient : Fin shape.coefficientCount => coefficient.val)
        (fun coefficient =>
          List.range'
            (0 + shape.runningCount * shape.matrixCount * coefficient)
            (shape.runningCount * shape.matrixCount))
        (canonicalFinIndices shape.coefficientCount)).symm
    _ = (List.range shape.coefficientCount).flatMap (fun coefficient =>
        List.range'
          (0 + shape.runningCount * shape.matrixCount * coefficient)
          (shape.runningCount * shape.matrixCount)) := by
      rw [canonicalFinIndices_values]
    _ = List.range' 0
        (shape.coefficientCount *
          (shape.runningCount * shape.matrixCount)) :=
      range_flatMap_chunks 0
        (shape.runningCount * shape.matrixCount) shape.coefficientCount
    _ = List.range shape.carriedEvaluationCount := by
      rw [List.range_eq_range']
      apply congrArg (fun count => List.range' 0 count)
      unfold Shape.carriedEvaluationCount
      exact Nat.mul_comm
        shape.coefficientCount
        (shape.runningCount * shape.matrixCount)

/-- One alpha monomial, represented by its finite exponent vector. -/
structure AlphaMonomial (shape : Shape) where
  exponents : List Nat
  arity : exponents.length = shape.cubeVariables
deriving Repr, DecidableEq

/-- Verifier-owned finite basis for polynomials in Appendix D.4's indeterminates
`A_1, ..., A_logm`. Distinct basis monomials make coefficient truth
unambiguous. -/
structure AlphaBasis (shape : Shape) where
  monomials : List (AlphaMonomial shape)
  nodup : monomials.Nodup

/-- Finite alpha polynomial. Coefficients align positionally with one fixed
verifier-owned basis; there is no function-valued evaluator or declared
degree. -/
structure AlphaPolynomial
    (Field : Type uField)
    {shape : Shape}
    (basis : AlphaBasis shape) where
  coefficients : List Field
  coefficientCount : coefficients.length = basis.monomials.length

namespace AlphaPolynomial

/-- Coefficient-level zero for an alpha polynomial. -/
def CoefficientZero
    {Field : Type uField}
    {shape : Shape}
    {basis : AlphaBasis shape}
    (ops : SumCheck.Finite.Ops Field)
    (polynomial : AlphaPolynomial Field basis) : Prop :=
  forall coefficient, coefficient ∈ polynomial.coefficients -> coefficient = ops.zero

end AlphaPolynomial

/-- The three finite residual families in the paper's gamma exponent order.
The flat carried-evaluation list uses `running` fastest, then `matrix`, then
`coefficient`, matching `CarriedCoordinate.localGammaExponent`. The concrete
residualization that populates this order remains an open boundary. -/
structure Residuals
    (Field : Type uField)
    (shape : Shape)
    (basis : AlphaBasis shape) where
  ccs : List (AlphaPolynomial Field basis)
  ccsCount : ccs.length = shape.freshCount
  norm : List (AlphaPolynomial Field basis)
  normCount : norm.length = shape.sourceCount
  carriedEvaluation : List Field
  carriedEvaluationCount :
    carriedEvaluation.length = shape.carriedEvaluationCount

/-- Independent semantic obligations which a later concrete paper
arithmetization must residualize. These are propositions, not conclusions
carried by a prover or imported from Rust. -/
structure Obligations (shape : Shape) where
  ccs : List Prop
  ccsCount : ccs.length = shape.freshCount
  norm : List Prop
  normCount : norm.length = shape.sourceCount
  carriedEvaluation : List Prop
  carriedEvaluationCount :
    carriedEvaluation.length = shape.carriedEvaluationCount

namespace Obligations

/-- All paper obligation leaves hold before alpha/gamma compression. -/
def AllHold {shape : Shape} (obligations : Obligations shape) : Prop :=
  (forall obligation, obligation ∈ obligations.ccs -> obligation) ∧
  (forall obligation, obligation ∈ obligations.norm -> obligation) ∧
  forall obligation, obligation ∈ obligations.carriedEvaluation -> obligation

end Obligations

/-- Length-preserving pointwise relation between two finite lists. Lean's core
library intentionally keeps this small relation local to the model. -/
inductive Aligned {Left : Type uLeft} {Right : Type uRight}
    (relation : Left -> Right -> Prop) : List Left -> List Right -> Prop where
  | nil : Aligned relation [] []
  | cons (left : Left) (right : Right) (lefts : List Left)
      (rights : List Right) :
      relation left right -> Aligned relation lefts rights ->
        Aligned relation (left :: lefts) (right :: rights)

/-- Unclosed per-leaf residualization boundary. A future concrete Appendix D.4
model must construct this from independently defined CCS, norm, and carried
evaluation semantics. Supplying caller-selected propositions here is not a
Lemma 7 proof. -/
structure ResidualizationBoundary
    {Field : Type uField}
    {shape : Shape}
    {basis : AlphaBasis shape}
    (ops : SumCheck.Finite.Ops Field)
    (residuals : Residuals Field shape basis)
    (obligations : Obligations shape) : Prop where
  ccs : Aligned
    (fun (residual : AlphaPolynomial Field basis) (obligation : Prop) =>
      residual.CoefficientZero ops ↔ obligation)
    residuals.ccs obligations.ccs
  norm : Aligned
    (fun (residual : AlphaPolynomial Field basis) (obligation : Prop) =>
      residual.CoefficientZero ops ↔ obligation)
    residuals.norm obligations.norm
  carriedEvaluation : Aligned
    (fun (residual : Field) (obligation : Prop) =>
      residual = ops.zero ↔ obligation)
    residuals.carriedEvaluation obligations.carriedEvaluation

/-- One coefficient in the formula-agnostic zero-equivalence serialization.
CCS and norm entries are finite polynomials in alpha; carried-evaluation
entries are alpha-free scalars. -/
inductive JointCoefficient
    (Field : Type uField)
    {shape : Shape}
    (basis : AlphaBasis shape) where
  | alpha (polynomial : AlphaPolynomial Field basis)
  | scalar (value : Field)

namespace JointCoefficient

/-- Coefficient-level zero for one joint gamma coefficient. -/
def Zero
    {Field : Type uField}
    {shape : Shape}
    {basis : AlphaBasis shape}
    (ops : SumCheck.Finite.Ops Field) :
    JointCoefficient Field basis -> Prop
  | .alpha polynomial => polynomial.CoefficientZero ops
  | .scalar value => value = ops.zero

end JointCoefficient

namespace Residuals

/-- Constant-first serialization of the three zero-residual families.
Concatenation implements the selected disjoint gamma shifts. This unsigned
zero-equivalence representation does not itself claim the signed expansion of
`T_abs(C) - sum_x Q(x, A, C)`. `SignedJointIdentity` proves that identity with
the fixed carried orientation `T_local - sum Eval_local`. -/
def jointCoefficients
    {Field : Type uField}
    {shape : Shape}
    {basis : AlphaBasis shape}
    (residuals : Residuals Field shape basis) :
    List (JointCoefficient Field basis) :=
  residuals.ccs.map JointCoefficient.alpha ++
    residuals.norm.map JointCoefficient.alpha ++
    residuals.carriedEvaluation.map JointCoefficient.scalar

/-- Coefficient-level truth of the complete joint polynomial, before sampling
alpha or gamma. -/
def CoefficientTruth
    {Field : Type uField}
    {shape : Shape}
    {basis : AlphaBasis shape}
    (ops : SumCheck.Finite.Ops Field)
    (residuals : Residuals Field shape basis) : Prop :=
  forall coefficient,
    coefficient ∈ residuals.jointCoefficients -> coefficient.Zero ops

/-- The serialized joint coefficient list has exactly the paper's three block
sizes; no hidden coefficient family exists between them. -/
theorem jointCoefficients_length
    {Field : Type uField}
    {shape : Shape}
    {basis : AlphaBasis shape}
    (residuals : Residuals Field shape basis) :
    residuals.jointCoefficients.length = shape.jointCoefficientCount := by
  simp [jointCoefficients, Shape.jointCoefficientCount,
    residuals.ccsCount, residuals.normCount,
    residuals.carriedEvaluationCount, Nat.add_assoc]

/-- Joint coefficient truth is exactly the conjunction of the three serialized
residual families. This is a finite-list concatenation theorem. Interpreting
the positions as the paper's signed gamma-exponent blocks remains part of the
open joint-polynomial identity. -/
theorem coefficientTruth_iff_residualFamilies
    {Field : Type uField}
    {shape : Shape}
    {basis : AlphaBasis shape}
    (ops : SumCheck.Finite.Ops Field)
    (residuals : Residuals Field shape basis) :
    CoefficientTruth ops residuals ↔
      (forall residual, residual ∈ residuals.ccs ->
        residual.CoefficientZero ops) ∧
      (forall residual, residual ∈ residuals.norm ->
        residual.CoefficientZero ops) ∧
      forall residual, residual ∈ residuals.carriedEvaluation ->
        residual = ops.zero := by
  constructor
  · intro truth
    refine ⟨?_, ?_, ?_⟩
    · intro residual member
      exact truth (.alpha residual) (by
        simp [jointCoefficients, member])
    · intro residual member
      exact truth (.alpha residual) (by
        simp [jointCoefficients, member])
    · intro residual member
      exact truth (.scalar residual) (by
        simp [jointCoefficients, member])
  · rintro ⟨ccsTruth, normTruth, carriedTruth⟩ coefficient member
    simp only [jointCoefficients, List.mem_append, List.mem_map] at member
    rcases member with
      (⟨residual, residualMember, rfl⟩ |
        ⟨residual, residualMember, rfl⟩) |
        ⟨residual, residualMember, rfl⟩
    · exact ccsTruth residual residualMember
    · exact normTruth residual residualMember
    · exact carriedTruth residual residualMember

private theorem forall₂_iff_all
    {Left : Type uLeft}
    {Right : Type uRight}
    {left : List Left}
    {right : List Right}
    {leftHolds : Left -> Prop}
    {rightHolds : Right -> Prop}
    (exact : Aligned
      (fun leftValue rightValue =>
        leftHolds leftValue ↔ rightHolds rightValue)
      left right) :
    (forall leftValue, leftValue ∈ left -> leftHolds leftValue) ↔
      forall rightValue, rightValue ∈ right -> rightHolds rightValue := by
  induction exact with
  | nil => simp
  | cons left right lefts rights headExact tailExact inductionHypothesis =>
      simp only [List.mem_cons, forall_eq_or_imp]
      exact and_congr headExact inductionHypothesis

/-- Conditional coefficient-level composition. Given the visibly unclosed
`ResidualizationBoundary`, the paper's disjoint gamma blocks are zero if and
only if every supplied CCS, norm, and carried-evaluation obligation holds.
This theorem does not construct or validate that boundary. -/
theorem coefficientTruth_iff_allObligations
    {Field : Type uField}
    {shape : Shape}
    {basis : AlphaBasis shape}
    (ops : SumCheck.Finite.Ops Field)
    (residuals : Residuals Field shape basis)
    (obligations : Obligations shape)
    (boundary : ResidualizationBoundary ops residuals obligations) :
    residuals.CoefficientTruth ops ↔ obligations.AllHold := by
  rw [coefficientTruth_iff_residualFamilies]
  unfold Obligations.AllHold
  exact and_congr
    (forall₂_iff_all
      (Left := AlphaPolynomial Field basis)
      (Right := Prop)
      (left := residuals.ccs)
      (right := obligations.ccs)
      (leftHolds := fun residual : AlphaPolynomial Field basis =>
        residual.CoefficientZero ops)
      (rightHolds := fun obligation : Prop => obligation)
      boundary.ccs)
    (and_congr
      (forall₂_iff_all
        (Left := AlphaPolynomial Field basis)
        (Right := Prop)
        (left := residuals.norm)
        (right := obligations.norm)
        (leftHolds := fun residual : AlphaPolynomial Field basis =>
          residual.CoefficientZero ops)
        (rightHolds := fun obligation : Prop => obligation)
        boundary.norm)
      (forall₂_iff_all
        (Left := Field)
        (Right := Prop)
        (left := residuals.carriedEvaluation)
        (right := obligations.carriedEvaluation)
        (leftHolds := fun residual : Field => residual = ops.zero)
        (rightHolds := fun obligation : Prop => obligation)
        boundary.carriedEvaluation))

end Residuals

end Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
