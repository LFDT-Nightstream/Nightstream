import Nightstream.Implementation.R1CS.Canonical.KBooleanMle
import Nightstream.Implementation.R1CS.Canonical.KFrameAllocationCoverage
import Nightstream.Implementation.R1CS.Canonical.KPointEquality
import Nightstream.Implementation.R1CS.Canonical.KSparsePolynomial
import Nightstream.Implementation.R1CS.Canonical.KStrictNorm

/-!
Contract: exact allocation coverage for the composite `K` arithmetic gadgets
used by the selected Split-NC endpoint programs.

Every theorem follows the emitted row recursion and the canonical frame
allocator.  Row-count equalities are not used as substitutes for occurrence.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.KCompositeAllocationCoverage

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.AllocationCoverage
open Nightstream.Implementation.R1CS.Canonical.KFrameAllocationCoverage
open Nightstream.Implementation.R1CS.Canonical.KMul
open Nightstream.Implementation.R1CS.Canonical.KHorner
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

private theorem no_columns (rows : List Row) (base : Nat) :
    RowsCover rows (KFrames.frameColumns base 0) := by
  intro column member
  simp [KFrames.frameColumns] at member

private theorem locateFrame
    (base count column : Nat)
    (member : column ∈ KFrames.frameColumns base count) :
    ∃ step, step < count ∧
      column ∈ KFrames.frameColumns (base + 3 * step) 1 := by
  rw [KFrames.frameColumns_mem_iff] at member
  let offset := column - base
  let step := offset / 3
  have split := Nat.div_add_mod offset 3
  have remainder := Nat.mod_lt offset (by decide : 0 < 3)
  have columnEq : column = base + offset := by
    simp only [offset]
    omega
  refine ⟨step, ?_, ?_⟩
  · simp only [step] at *
    omega
  · rw [KFrames.frameColumns_mem_iff]
    simp only [step] at *
    omega

/-- The recursive Boolean-MLE program covers every one of its postorder
multiplication frames. -/
theorem booleanMle (base : Nat) :
    ∀ {variables : Nat}
      (table : BooleanTable Carried variables)
      (coordinates : List Carried) (step : Nat),
      RowsCover
        (KBooleanMle.rows (KFrames.frameAt base)
          table coordinates step)
        (KFrames.frameColumns (base + 3 * step)
          (KBooleanMle.frameCount variables))
  | 0, .leaf _, _, step => no_columns _ _
  | tailVariables + 1, .branch low high, coordinates, step => by
      intro column member
      rw [KFrames.frameColumns_mem_iff] at member
      let count := KBooleanMle.frameCount tailVariables
      by_cases inLow : column < base + 3 * (step + count)
      · have lowColumn :
            column ∈
              KFrames.frameColumns (base + 3 * step) count := by
          rw [KFrames.frameColumns_mem_iff]
          omega
        rcases booleanMle base low
            (KBooleanMle.tailCoordinates coordinates) step
            column lowColumn with ⟨row, rowMember, mentioned⟩
        exact
          ⟨row,
            List.mem_append_left _
              (List.mem_append_left _ rowMember),
            mentioned⟩
      · by_cases inHigh :
          column < base + 3 * (step + 2 * count)
        · have highColumn :
              column ∈
                KFrames.frameColumns (base + 3 * (step + count))
                  count := by
            rw [KFrames.frameColumns_mem_iff]
            omega
          rcases booleanMle base high
              (KBooleanMle.tailCoordinates coordinates)
              (step + count) column highColumn with
            ⟨row, rowMember, mentioned⟩
          exact
            ⟨row,
              List.mem_append_left _
                (List.mem_append_right _ rowMember),
              mentioned⟩
        · have rootColumn :
              column ∈
                KFrames.frameColumns
                  (base + 3 * (step + 2 * count)) 1 := by
            rw [KFrames.frameColumns_mem_iff]
            simp only [KBooleanMle.frameCount] at member
            omega
          rcases mul
              (KBooleanMle.headCoordinate coordinates)
              (KLinear.subCarried
                (KBooleanMle.carried (KFrames.frameAt base) high
                  (KBooleanMle.tailCoordinates coordinates)
                  (step + count))
                (KBooleanMle.carried (KFrames.frameAt base) low
                  (KBooleanMle.tailCoordinates coordinates) step))
              base (step + 2 * count) column rootColumn with
            ⟨row, rowMember, mentioned⟩
          exact
            ⟨row, List.mem_append_right _ rowMember, mentioned⟩

private theorem pointFactorRows
    {variables : Nat} (input : KPointEquality.Input variables) :
    RowsCover
      (KPointEquality.factorRows input)
      (KFrames.frameColumns input.frameBase variables) := by
  intro column member
  rcases locateFrame input.frameBase variables column member with
    ⟨step, stepLt, localColumn⟩
  let index : Fin variables := ⟨step, stepLt⟩
  rcases mul (input.left index) (KPointEquality.slope input index)
      input.frameBase step column localColumn with
    ⟨row, rowMember, mentioned⟩
  refine
    ⟨row, List.mem_flatMap.2 ⟨index, ?_, rowMember⟩, mentioned⟩
  rw [KPointEquality.indices, List.mem_ofFn]
  exact ⟨index, rfl⟩

private theorem pointProductRows
    {variables : Nat} (input : KPointEquality.Input variables) :
    RowsCover
      (KPointEquality.productRows input)
      (KFrames.frameColumns (KPointEquality.productBase input)
        (variables - 1)) := by
  unfold KPointEquality.productRows
  split
  next empty =>
    have sized := KPointEquality.factors_length input
    rw [empty] at sized
    simp only [List.length_nil] at sized
    have variablesZero : variables = 0 := sized.symm
    subst variables
    exact no_columns _ _
  next first rest equal =>
    have restLength : rest.length = variables - 1 := by
      have sized := KPointEquality.factors_length input
      rw [equal] at sized
      simp only [List.length_cons] at sized
      omega
    have covered :=
      mulChain (KPointEquality.productBase input)
        first rest 0
    simpa only [Nat.mul_zero, Nat.add_zero, restLength] using covered

/-- The point-equality program covers both factor and product intervals. -/
theorem pointEquality
    {variables : Nat} (input : KPointEquality.Input variables) :
    RowsCover (KPointEquality.rows input) (KPointEquality.columns input) := by
  unfold KPointEquality.rows KPointEquality.columns
  exact AllocationCoverage.append _ _ _ _
    (pointFactorRows input) (pointProductRows input)

private theorem sparseTerms
    {matrixCount : Nat}
    (point : Fin matrixCount → Carried) :
    ∀ (terms :
        List (CCSResidualTable.Monomial
          Nightstream.SuperNeo.Concrete.K matrixCount))
      (base offset : Nat),
      RowsCover
        (KSparsePolynomial.termsRows point terms base offset)
        (KFrames.frameColumns (base + 3 * offset)
          (KSparsePolynomial.totalDegreeSum terms))
  | [], base, offset => no_columns _ _
  | monomial :: rest, base, offset => by
      intro column member
      rw [KFrames.frameColumns_mem_iff] at member
      by_cases inHead :
          column <
            base + 3 * offset + 3 * monomial.totalDegree
      · have localColumn :
            column ∈
              KFrames.frameColumns (base + 3 * offset)
                monomial.totalDegree := by
          rw [KFrames.frameColumns_mem_iff]
          omega
        have factorsLength :=
          KSparsePolynomial.expandedFactors_length point monomial
        have covered :=
          mulChain (base + 3 * offset)
            (KLinear.constantCarried monomial.coefficient)
            (KSparsePolynomial.expandedFactors point monomial) 0
        rw [factorsLength] at covered
        rcases covered column localColumn with ⟨row, rowMember, mentioned⟩
        exact
          ⟨row, List.mem_append_left _ rowMember, mentioned⟩
      · have tail :
            column ∈
              KFrames.frameColumns
                (base + 3 * (offset + monomial.totalDegree))
                (KSparsePolynomial.totalDegreeSum rest) := by
          rw [KFrames.frameColumns_mem_iff]
          unfold KSparsePolynomial.totalDegreeSum at member ⊢
          simp only [List.map_cons, List.sum_cons] at member
          omega
        rcases sparseTerms point rest base
            (offset + monomial.totalDegree) column tail with
          ⟨row, rowMember, mentioned⟩
        exact
          ⟨row, List.mem_append_right _ rowMember, mentioned⟩

/-- The sparse polynomial program covers the frame interval derived from the
explicit monomial degrees. -/
theorem sparsePolynomial
    {matrixCount : Nat} (input : KSparsePolynomial.Input matrixCount) :
    RowsCover (KSparsePolynomial.rows input)
      (KSparsePolynomial.columns input) := by
  unfold KSparsePolynomial.rows KSparsePolynomial.columns
  simpa only [Nat.mul_zero, Nat.add_zero] using
    sparseTerms input.point input.polynomial.terms input.frameBase 0

/-- The strict-norm cubic covers both of its multiplication frames. -/
theorem strictNorm (input : KStrictNorm.Input) :
    RowsCover (KStrictNorm.rows input) (KStrictNorm.columns input) := by
  unfold KStrictNorm.rows KStrictNorm.columns
  intro column member
  rw [KFrames.frameColumns_mem_iff] at member
  by_cases first : column < input.frameBase + 3
  · have localColumn :
        column ∈ KFrames.frameColumns input.frameBase 1 := by
      rw [KFrames.frameColumns_mem_iff]
      omega
    rcases mul
        (KLinear.addCarried input.value KLinear.oneCarried)
        input.value input.frameBase 0 column localColumn with
      ⟨row, rowMember, mentioned⟩
    exact
      ⟨row, List.mem_append_left _ rowMember, mentioned⟩
  · have localColumn :
        column ∈ KFrames.frameColumns (input.frameBase + 3) 1 := by
      rw [KFrames.frameColumns_mem_iff]
      omega
    rcases mul
        (KStrictNorm.firstOutput input)
        (KLinear.subCarried input.value KLinear.oneCarried)
        input.frameBase 1 column (by
          simpa only [Nat.mul_one] using localColumn) with
      ⟨row, rowMember, mentioned⟩
    exact
      ⟨row, List.mem_append_right _ rowMember, mentioned⟩

end Nightstream.Implementation.R1CS.Canonical.KCompositeAllocationCoverage
