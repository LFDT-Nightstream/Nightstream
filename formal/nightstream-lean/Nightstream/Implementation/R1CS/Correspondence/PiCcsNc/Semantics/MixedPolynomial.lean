import Nightstream.Implementation.R1CS.Correspondence.PiCcsNc.Parameters
import Nightstream.Implementation.R1CS.Correspondence.PiCcsNc.Semantics.RangePolynomial

/-!
Contract: state the production-shaped Π_CCS NC polynomial over the direct raw
packed assignment, before transcript or SumCheck refinement.

Owns: extension-field equality and Boolean-basis weights, the direct diagonal
packed table, its authoritative `y_zcol` projection, `zTilde`, the
`gamma^(i+1)` batch mixture, and the one-way zero-initial-sum consequence of
authoritative strict `b = 2` norms.

Does not own: balanced-digit decomposition, prover-carried output authority,
transcript challenge derivation, SumCheck acceptance, terminal R1CS rows, or a
production fixed-shape instantiation.

Emits constraints: no.

Authority boundary: assignments are independent semantic inputs. In
particular, `directDiagonal` reads the raw assignment coordinate at logical
column `s` and packed lane `s mod 54`; it never decomposes that coordinate
into balanced digits.

| Definition / theorem | Mathematical obligation | Assumptions | Rust owner / status | Permits row removal? |
|---|---|---|---|---|
| `chi` / `eqPoint` | Boolean MLE basis and equality factors | matching point lengths for nonzero equality | NC equality / chi arithmetic; row refinement open | no |
| `directDiagonal` | raw `Z[rho, block]` appears only at `rho = column mod D` | direct raw packing | NC oracle source table; source refinement open | no |
| `AssignmentsFitColumnDomain` | every raw assignment coordinate fits the selected column domain | exact production assignment width | required only by the future source/shape bridge | no |
| `authoritativeYZcol` | independently project the direct table at `s` | verifier-selected shape and point | no current verifier authority bridge | no |
| `zTilde` | evaluate that projection at `alpha` | verifier-selected shape and points | NC output evaluation; row refinement open | no |
| `mixedRangePolynomial` | `sum_i gamma^(i+1) R_2(zTilde_i)` | ordered assignment batch | NC weighted range sum; row refinement open | no |
| `qNc_cubePoint_eq_qNcOnCube` | extension polynomial restricts to the direct Boolean table | beta lengths match `ellM`/`ellD` | model-level true-polynomial bridge | no |
| `trueInitial_eq_sum_qNc` | direct initial claim is the Boolean-cube sum of `qNc` | beta lengths match `ellM`/`ellD` | SumCheck truth-path refinement open | no |
| `trueInitial_eq_zero_of_normBounded` | strict input norms imply the Boolean-cube claim is zero | Goldilocks prime and strict `b = 2` norms | fixed-parameter/source refinement open | no |
-/

namespace Nightstream.Implementation.R1CS.PiCcsNc.MixedPolynomial

open Nightstream.Implementation.R1CS.PiCcsNc
open Nightstream.Implementation.R1CS.PiCcsNc.RangePolynomial
open Nightstream.SuperNeo.Concrete

/-- One Boolean-basis factor. Bits are little-endian, matching the production
NC round order. -/
def chiFactor (point : List K) (mask bit : Nat) : K :=
  if Nat.testBit mask bit then
    point.getD bit K.zero
  else
    K.sub K.one (point.getD bit K.zero)

/-- `chi_point(mask)` over the explicitly supplied point coordinates. -/
def chi (point : List K) (mask : Nat) : K :=
  productRange point.length (chiFactor point mask)

/-- One coordinate of the multilinear equality polynomial. -/
def eqFactor (left right : K) : K :=
  K.add (K.mul left right)
    (K.mul (K.sub K.one left) (K.sub K.one right))

/-- Multilinear equality polynomial, with a zero result on shape mismatch. -/
def eqPoint (left right : List K) : K :=
  if left.length = right.length then
    productRange left.length fun index =>
      eqFactor (left.getD index K.zero) (right.getD index K.zero)
  else
    K.zero

/-- Little-endian Boolean point used to state the exact hypercube restriction
of the extension polynomial. -/
def cubePoint (width mask : Nat) : List K :=
  (List.range width).map fun bit =>
    if Nat.testBit mask bit then K.one else K.zero

@[simp] theorem cubePoint_length (width mask : Nat) :
    (cubePoint width mask).length = width := by
  simp [cubePoint]

theorem cubePoint_getD
    (width mask bit : Nat) (bitLt : bit < width) :
    (cubePoint width mask).getD bit K.zero =
      if Nat.testBit mask bit then K.one else K.zero := by
  simp [cubePoint, List.getD_eq_getElem?_getD, bitLt]

private theorem one_sub_zero :
    K.sub K.one K.zero = K.one := by
  rfl

private theorem one_sub_one :
    K.sub K.one K.one = K.zero := by
  rfl

theorem chiFactor_cubePoint
    (width source target bit : Nat) (bitLt : bit < width) :
    chiFactor (cubePoint width source) target bit =
      if Nat.testBit source bit = Nat.testBit target bit then
        K.one
      else
        K.zero := by
  unfold chiFactor
  rw [cubePoint_getD width source bit bitLt]
  cases sourceBit : Nat.testBit source bit <;>
    cases targetBit : Nat.testBit target bit <;>
    simp [one_sub_zero, one_sub_one]

private theorem exists_testBit_ne_of_ne_of_lt_pow
    {width source target : Nat}
    (sourceLt : source < 2 ^ width)
    (targetLt : target < 2 ^ width)
    (different : source ≠ target) :
    ∃ bit, bit < width ∧
      Nat.testBit source bit ≠ Nat.testBit target bit := by
  classical
  exact Classical.byContradiction fun noDifference => by
    apply different
    apply Nat.eq_of_testBit_eq
    intro bit
    by_cases bitLt : bit < width
    · by_cases bitsEqual :
          Nat.testBit source bit = Nat.testBit target bit
      · exact bitsEqual
      · exact False.elim (noDifference ⟨bit, bitLt, bitsEqual⟩)
    · have widthLe : width ≤ bit := Nat.le_of_not_lt bitLt
      have powLe : 2 ^ width ≤ 2 ^ bit :=
        Nat.pow_le_pow_right (by decide : 1 ≤ 2) widthLe
      have sourceBit : Nat.testBit source bit = false :=
        Nat.testBit_lt_two_pow (Nat.lt_of_lt_of_le sourceLt powLe)
      have targetBit : Nat.testBit target bit = false :=
        Nat.testBit_lt_two_pow (Nat.lt_of_lt_of_le targetLt powLe)
      simp [sourceBit, targetBit]

/-- On in-domain Boolean points, `chi` is the exact Kronecker selector. -/
theorem chi_cubePoint_eq_if
    (width source target : Nat)
    (sourceLt : source < 2 ^ width)
    (targetLt : target < 2 ^ width) :
    chi (cubePoint width source) target =
      if source = target then K.one else K.zero := by
  unfold chi
  rw [cubePoint_length]
  by_cases equal : source = target
  · rw [if_pos equal]
    apply productRange_eq_one
    intro bit bitLt
    rw [chiFactor_cubePoint width source target bit bitLt]
    simp [equal]
  · rw [if_neg equal]
    rcases exists_testBit_ne_of_ne_of_lt_pow
      sourceLt targetLt equal with ⟨bit, bitLt, bitDifferent⟩
    apply productRange_eq_zero_of_exists
    refine ⟨bit, bitLt, ?_⟩
    rw [chiFactor_cubePoint width source target bit bitLt]
    simp [bitDifferent]

private theorem eqFactor_cubeBit
    (bit : Bool) (value : K) :
    eqFactor (if bit then K.one else K.zero) value =
      if bit then value else K.sub K.one value := by
  cases bit
  · change eqFactor K.zero value = K.sub K.one value
    simp only [eqFactor, one_sub_zero]
    rw [zero_mul, one_mul, zero_add]
  · change eqFactor K.one value = value
    simp only [eqFactor, one_sub_one]
    rw [one_mul, zero_mul, add_zero]

/-- Equality against a Boolean point is the corresponding `chi` weight. -/
theorem eqPoint_cubePoint_eq_chi
    (width mask : Nat) (point : List K)
    (pointLength : point.length = width) :
    eqPoint (cubePoint width mask) point = chi point mask := by
  unfold eqPoint chi
  rw [if_pos (by simp [pointLength])]
  rw [cubePoint_length, pointLength]
  apply productRange_congr
  intro bit bitLt
  rw [cubePoint_getD width mask bit bitLt]
  unfold chiFactor
  rw [eqFactor_cubeBit]

/-- Raw packed-diagonal NC table entry.

Logical column `column` reads exactly `assignment[column]` at lane
`column mod D`; every other lane and every padded column is zero. -/
def directDiagonal
    (assignment : List F) (column lane : Nat) : K :=
  if column < assignment.length ∧
      lane = column % ringDegree then
    K.embed (assignment.getD column 0)
  else
    K.zero

/-- Functional representation of one `y_zcol` vector. -/
abbrev YZcol := Nat → K

/-- The independently evaluated packed-assignment projection at `s`. -/
def authoritativeYZcol
    (shape : Shape) (assignment : List F) (s : List K) : YZcol :=
  fun lane =>
    if lane < shape.laneDomain then
      sumRange shape.columnDomain fun column =>
        K.mul (directDiagonal assignment column lane) (chi s column)
    else
      K.zero

/-- At an in-domain Boolean column, the authoritative `y_zcol` projection is
exactly the selected raw diagonal entry. -/
theorem authoritativeYZcol_cubePoint
    (shape : Shape) (assignment : List F)
    {column lane : Nat}
    (columnLt : column < shape.columnDomain)
    (laneLt : lane < shape.laneDomain) :
    authoritativeYZcol shape assignment
        (cubePoint shape.ellM column) lane =
      directDiagonal assignment column lane := by
  unfold authoritativeYZcol
  rw [if_pos laneLt]
  calc
    sumRange shape.columnDomain (fun current =>
        K.mul (directDiagonal assignment current lane)
          (chi (cubePoint shape.ellM column) current)) =
        sumRange shape.columnDomain (fun current =>
          if current = column then
            directDiagonal assignment current lane
          else
            K.zero) := by
      apply sumRange_congr
      intro current currentLt
      rw [chi_cubePoint_eq_if shape.ellM column current]
      · by_cases selected : current = column
        · subst current
          simp [mul_one]
        · rw [if_neg selected, if_neg (Ne.symm selected), mul_zero]
      · exact columnLt
      · exact currentLt
    _ = directDiagonal assignment column lane :=
      sumRange_select shape.columnDomain column
        (fun current => directDiagonal assignment current lane) columnLt

/-- Dot one `y_zcol` prefix with `chi_alpha`. -/
def dotChi (shape : Shape) (yZcol : YZcol) (alpha : List K) : K :=
  sumRange shape.laneDomain fun lane =>
    K.mul (yZcol lane) (chi alpha lane)

/-- Direct packed-table multilinear evaluation `Z tilde(s, alpha)`. -/
def zTilde
    (shape : Shape) (assignment : List F)
    (s alpha : List K) : K :=
  dotChi shape (authoritativeYZcol shape assignment s) alpha

/-- At in-domain Boolean points, `zTilde` is the raw packed-diagonal table
entry selected by the column and lane coordinates. -/
theorem zTilde_cubePoint
    (shape : Shape) (assignment : List F)
    {column lane : Nat}
    (columnLt : column < shape.columnDomain)
    (laneLt : lane < shape.laneDomain) :
    zTilde shape assignment
        (cubePoint shape.ellM column)
        (cubePoint shape.ellD lane) =
      directDiagonal assignment column lane := by
  unfold zTilde dotChi
  calc
    sumRange shape.laneDomain (fun current =>
        K.mul
          (authoritativeYZcol shape assignment
            (cubePoint shape.ellM column) current)
          (chi (cubePoint shape.ellD lane) current)) =
        sumRange shape.laneDomain (fun current =>
          if current = lane then
            directDiagonal assignment column current
          else
            K.zero) := by
      apply sumRange_congr
      intro current currentLt
      rw [authoritativeYZcol_cubePoint shape assignment columnLt currentLt]
      rw [chi_cubePoint_eq_if shape.ellD lane current laneLt currentLt]
      by_cases selected : current = lane
      · subst current
        simp [mul_one]
      · rw [if_neg selected, if_neg (Ne.symm selected), mul_zero]
    _ = directDiagonal assignment column lane :=
      sumRange_select shape.laneDomain lane
        (fun current => directDiagonal assignment column current) laneLt

/-- Ordered NC range mixture. Output index zero is weighted by `gamma^1`,
not by `gamma^0`. -/
def mixedRangePolynomial
    (shape : Shape) (assignments : List (List F))
    (gamma : K) (s alpha : List K) : K :=
  sumRange assignments.length fun index =>
    K.mul (powK gamma (index + 1))
      (rangeProductB2
        (zTilde shape (assignments.getD index []) s alpha))

/-- Production-shaped NC polynomial at extension points. -/
def qNc
    (shape : Shape) (betaM betaA : List K) (gamma : K)
    (assignments : List (List F)) (s alpha : List K) : K :=
  K.mul
    (K.mul (eqPoint alpha betaA) (eqPoint s betaM))
    (mixedRangePolynomial shape assignments gamma s alpha)

/-- The direct Boolean-table range mixture before multilinear interpolation. -/
def mixedRangeOnCube
    (assignments : List (List F)) (gamma : K)
    (column lane : Nat) : K :=
  sumRange assignments.length fun index =>
    K.mul (powK gamma (index + 1))
      (rangeProductB2
        (directDiagonal (assignments.getD index []) column lane))

/-- Boolean-cube restriction of the NC table, with the equality selectors
written as their canonical `chi` weights. -/
def qNcOnCube
    (betaM betaA : List K) (gamma : K)
    (assignments : List (List F)) (column lane : Nat) : K :=
  K.mul
    (K.mul (chi betaA lane) (chi betaM column))
    (mixedRangeOnCube assignments gamma column lane)

/-- No raw assignment coordinate lies beyond the selected Boolean column
domain. This is a protocol-shape premise, not a consequence of interpolation. -/
def AssignmentsFitColumnDomain
    (shape : Shape) (assignments : List (List F)) : Prop :=
  ∀ assignment ∈ assignments, assignment.length ≤ shape.columnDomain

/-- The extension polynomial agrees with the direct Boolean table at every
in-domain cube point. The beta-vector lengths are explicit shape premises;
without them `eqPoint` deliberately returns zero on a shape mismatch. -/
theorem qNc_cubePoint_eq_qNcOnCube
    (shape : Shape) (betaM betaA : List K) (gamma : K)
    (assignments : List (List F)) {column lane : Nat}
    (betaMLength : betaM.length = shape.ellM)
    (betaALength : betaA.length = shape.ellD)
    (columnLt : column < shape.columnDomain)
    (laneLt : lane < shape.laneDomain) :
    qNc shape betaM betaA gamma assignments
        (cubePoint shape.ellM column)
        (cubePoint shape.ellD lane) =
      qNcOnCube betaM betaA gamma assignments column lane := by
  unfold qNc qNcOnCube
  rw [eqPoint_cubePoint_eq_chi shape.ellD lane betaA betaALength]
  rw [eqPoint_cubePoint_eq_chi shape.ellM column betaM betaMLength]
  congr 1
  unfold mixedRangePolynomial mixedRangeOnCube
  apply sumRange_congr
  intro index indexLt
  rw [zTilde_cubePoint shape (assignments.getD index []) columnLt laneLt]

/-- The true NC initial claim is the sum of the direct table over both padded
Boolean domains. -/
def trueInitial
    (shape : Shape) (betaM betaA : List K) (gamma : K)
    (assignments : List (List F)) : K :=
  sumRange shape.columnDomain fun column =>
    sumRange shape.laneDomain fun lane =>
      qNcOnCube betaM betaA gamma assignments column lane

/-- `trueInitial` is the exact Boolean-hypercube sum of the extension
polynomial, not merely a separately defined lookup-table claim. -/
theorem trueInitial_eq_sum_qNc
    (shape : Shape) (betaM betaA : List K) (gamma : K)
    (assignments : List (List F))
    (betaMLength : betaM.length = shape.ellM)
    (betaALength : betaA.length = shape.ellD) :
    trueInitial shape betaM betaA gamma assignments =
      sumRange shape.columnDomain fun column =>
        sumRange shape.laneDomain fun lane =>
          qNc shape betaM betaA gamma assignments
            (cubePoint shape.ellM column)
            (cubePoint shape.ellD lane) := by
  unfold trueInitial
  apply sumRange_congr
  intro column columnLt
  apply sumRange_congr
  intro lane laneLt
  exact (qNc_cubePoint_eq_qNcOnCube shape betaM betaA gamma assignments
    betaMLength betaALength columnLt laneLt).symm

/-- Every authoritative assignment has the strict production `b = 2` norm. -/
def InputsNormBoundedTwo (assignments : List (List F)) : Prop :=
  ∀ assignment ∈ assignments, normBounded 2 assignment

private theorem getD_mem_of_lt
    {values : List F} {index : Nat} (indexLt : index < values.length) :
    values.getD index 0 ∈ values := by
  have member := List.getElem_mem (l := values) indexLt
  rwa [List.getElem_eq_getD 0] at member

private theorem assignment_getD_mem_of_lt
    {assignments : List (List F)} {index : Nat}
    (indexLt : index < assignments.length) :
    assignments.getD index [] ∈ assignments := by
  have member := List.getElem_mem (l := assignments) indexLt
  rwa [List.getElem_eq_getD []] at member

/-- Each raw diagonal table entry satisfies the `b = 2` range polynomial
when the underlying assignment is norm bounded. Padded and off-diagonal
entries reduce to the zero root. -/
theorem directDiagonal_rangeProduct_eq_zero
    (prime : EuclidPrime goldilocksP)
    {assignment : List F}
    (norm : normBounded 2 assignment)
    (column lane : Nat) :
    rangeProductB2 (directDiagonal assignment column lane) = K.zero := by
  unfold directDiagonal
  by_cases live : column < assignment.length ∧
      lane = column % ringDegree
  · rw [if_pos live]
    exact (rangeProductB2_embed_eq_zero_iff_normTwo
      prime (assignment.getD column 0)).mpr
        (norm _ (getD_mem_of_lt live.1))
  · rw [if_neg live]
    rfl

/-- Norm-bounded inputs make every direct Boolean-table range mixture zero. -/
theorem mixedRangeOnCube_eq_zero_of_normBounded
    (prime : EuclidPrime goldilocksP)
    {assignments : List (List F)}
    (norms : InputsNormBoundedTwo assignments)
    (gamma : K) (column lane : Nat) :
    mixedRangeOnCube assignments gamma column lane = K.zero := by
  unfold mixedRangeOnCube
  apply sumRange_eq_zero
  intro index indexLt
  have assignmentMember := assignment_getD_mem_of_lt indexLt
  have rangeZero := directDiagonal_rangeProduct_eq_zero prime
    (norms _ assignmentMember) column lane
  rw [rangeZero]
  exact mul_zero _

/-- One-way honest-input theorem: authoritative strict norms force the
production NC initial claim to be zero.

No converse is claimed: sampled mixing can vanish at a bad challenge even
when an input violates the norm. -/
theorem trueInitial_eq_zero_of_normBounded
    (prime : EuclidPrime goldilocksP)
    {assignments : List (List F)}
    (norms : InputsNormBoundedTwo assignments)
    (shape : Shape) (betaM betaA : List K) (gamma : K) :
    trueInitial shape betaM betaA gamma assignments = K.zero := by
  apply sumRange_eq_zero
  intro column columnLt
  apply sumRange_eq_zero
  intro lane laneLt
  unfold qNcOnCube
  rw [mixedRangeOnCube_eq_zero_of_normBounded
    prime norms gamma column lane]
  exact mul_zero _

end Nightstream.Implementation.R1CS.PiCcsNc.MixedPolynomial
