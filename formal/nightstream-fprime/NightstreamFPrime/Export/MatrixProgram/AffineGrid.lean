import NightstreamFPrime.Export.MatrixProgram

/-!
Owns a compact package language for affine sparse forms over a three-axis
row grid. Regions, retained-slot strides, coefficients, and rule order are
package data. The interpreter does not select phase dimensions or formulas.
-/

namespace NightstreamFPrime.Export.MatrixProgram.AffineGrid

open NightstreamFPrime.Export.Codec
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Spec

structure Coordinate where
  major : Nat
  middle : Nat
  minor : Nat
deriving Repr, DecidableEq

/-- One rectangular subregion of a three-axis row grid. -/
structure Region where
  majorStart : Nat
  majorCount : Nat
  middleStart : Nat
  middleCount : Nat
  minorStart : Nat
  minorCount : Nat
deriving Repr, DecidableEq

def Region.format : Format Region where
  encode := fun region => .array [
    .atom region.majorStart,
    .atom region.majorCount,
    .atom region.middleStart,
    .atom region.middleCount,
    .atom region.minorStart,
    .atom region.minorCount]
  decode
    | .array [.atom majorStart, .atom majorCount,
        .atom middleStart, .atom middleCount,
        .atom minorStart, .atom minorCount] =>
        .ok (Region.mk majorStart majorCount middleStart middleCount
          minorStart minorCount)
    | _ => .error "invalid affine-grid region"
  decode_encode := by
    intro region
    cases region
    rfl

def Region.offsets? (region : Region) (coordinate : Coordinate) :
    Option Coordinate :=
  if region.majorStart ≤ coordinate.major then
    let major := coordinate.major - region.majorStart
    if major < region.majorCount then
      if region.middleStart ≤ coordinate.middle then
        let middle := coordinate.middle - region.middleStart
        if middle < region.middleCount then
          if region.minorStart ≤ coordinate.minor then
            let minor := coordinate.minor - region.minorStart
            if minor < region.minorCount then
              some { major, middle, minor }
            else none
          else none
        else none
      else none
    else none
  else none

theorem Region.offsets?_of_offsets (region : Region)
    (major : Fin region.majorCount) (middle : Fin region.middleCount)
    (minor : Fin region.minorCount) :
    region.offsets? {
      major := region.majorStart + major.val
      middle := region.middleStart + middle.val
      minor := region.minorStart + minor.val } =
      some { major := major.val, middle := middle.val, minor := minor.val } := by
  unfold offsets?
  simp only
  rw [if_pos (by omega)]
  rw [show region.majorStart + major.val - region.majorStart = major.val by
    omega]
  rw [if_pos major.isLt, if_pos (by omega)]
  rw [show region.middleStart + middle.val - region.middleStart = middle.val by
    omega]
  rw [if_pos middle.isLt, if_pos (by omega)]
  rw [show region.minorStart + minor.val - region.minorStart = minor.val by
    omega]
  rw [if_pos minor.isLt]

theorem Region.offsets?_eq_none_of_outside (region : Region)
    (coordinate : Coordinate)
    (outside : ¬(
      region.majorStart ≤ coordinate.major ∧
      coordinate.major < region.majorStart + region.majorCount ∧
      region.middleStart ≤ coordinate.middle ∧
      coordinate.middle < region.middleStart + region.middleCount ∧
      region.minorStart ≤ coordinate.minor ∧
      coordinate.minor < region.minorStart + region.minorCount)) :
    region.offsets? coordinate = none := by
  unfold offsets?
  by_cases majorStart : region.majorStart ≤ coordinate.major
  · rw [if_pos majorStart]
    by_cases majorCount :
        coordinate.major - region.majorStart < region.majorCount
    · rw [if_pos majorCount]
      by_cases middleStart : region.middleStart ≤ coordinate.middle
      · rw [if_pos middleStart]
        by_cases middleCount :
            coordinate.middle - region.middleStart < region.middleCount
        · rw [if_pos middleCount]
          by_cases minorStart : region.minorStart ≤ coordinate.minor
          · rw [if_pos minorStart]
            by_cases minorCount :
                coordinate.minor - region.minorStart < region.minorCount
            · exfalso
              apply outside
              omega
            · rw [if_neg minorCount]
          · rw [if_neg minorStart]
        · rw [if_neg middleCount]
      · rw [if_neg middleStart]
    · rw [if_neg majorCount]
  · rw [if_neg majorStart]

/-- One affine summand. Retained terms carry a canonical field coefficient;
constant terms use the block's selector column. -/
inductive Term where
  | retained (block : RetainedBlock) (slotBase majorStride middleStride
      minorStride coefficient : Nat)
  | constant (coefficient : Nat)
deriving Repr, DecidableEq

/-- Apply a canonical coefficient without rewriting the sparse entry list
when the coefficient is one. This preserves literal matrix-row identity. -/
def applyCoefficient {logicalWidth : Nat} (coefficient : F)
    (form : SparseForm logicalWidth) : SparseForm logicalWidth :=
  if coefficient = 1 then form else SparseForm.scale coefficient form

def Term.format : Format Term where
  encode
    | .retained block slotBase majorStride middleStride minorStride
        coefficient => .array [
          .atom 0, RetainedBlock.format.encode block, .atom slotBase,
          .atom majorStride, .atom middleStride, .atom minorStride,
          .atom coefficient]
    | .constant coefficient => .array [.atom 1, .atom coefficient]
  decode
    | .array [.atom 0, block, .atom slotBase, .atom majorStride,
        .atom middleStride, .atom minorStride, .atom coefficient] => do
        pure (.retained (← RetainedBlock.format.decode block) slotBase
          majorStride middleStride minorStride coefficient)
    | .array [.atom 1, .atom coefficient] => .ok (.constant coefficient)
    | _ => .error "invalid affine-grid term"
  decode_encode := by
    intro term
    cases term <;> simp [RetainedBlock.format.decode_encode]

def Term.form? (term : Term) (logicalWidth oneColumn : Nat)
    (offsets : Coordinate) : Option (SparseForm logicalWidth) :=
  match term with
  | .retained block slotBase majorStride middleStride minorStride
      coefficient => do
      let form ← block.form? logicalWidth
        (slotBase + offsets.major * majorStride +
          offsets.middle * middleStride + offsets.minor * minorStride)
      if coefficientBound : coefficient < goldilocksModulus then
        let scalar : F := ⟨coefficient, coefficientBound⟩
        some (applyCoefficient scalar form)
      else none
  | .constant coefficient =>
      if oneBound : oneColumn < logicalWidth then
        if coefficientBound : coefficient < goldilocksModulus then
          some (SparseForm.singleton ⟨oneColumn, oneBound⟩
            ⟨coefficient, coefficientBound⟩)
        else none
      else none

theorem Term.retained_form?_ofSemantic
    {sourceWidth logicalWidth : Nat}
    (block : LowNormBlock.Block sourceWidth) (retainedStart : Nat)
    (fits : retainedStart + block.coordinateCount ≤ logicalWidth)
    (oneColumn slotBase majorStride middleStride minorStride : Nat)
    (coefficient : F) (offsets : Coordinate)
    (slotBound : slotBase + offsets.major * majorStride +
      offsets.middle * middleStride + offsets.minor * minorStride <
        block.slotCount) :
    (Term.retained (RetainedBlock.ofSemantic block retainedStart) slotBase
      majorStride middleStride minorStride coefficient.val).form?
        logicalWidth oneColumn offsets =
      some (applyCoefficient coefficient (block.form retainedStart fits
        ⟨_, slotBound⟩)) := by
  change (do
    let form ← (RetainedBlock.ofSemantic block retainedStart).form?
      logicalWidth (slotBase + offsets.major * majorStride +
        offsets.middle * middleStride + offsets.minor * minorStride)
    if coefficientBound : coefficient.val < goldilocksModulus then
      let scalar : F := ⟨coefficient.val, coefficientBound⟩
      some (applyCoefficient scalar form)
    else none) = _
  rw [RetainedBlock.form?_ofSemantic block retainedStart fits
    ⟨_, slotBound⟩]
  change (if coefficientBound : coefficient.val < goldilocksModulus then
      let scalar : F := ⟨coefficient.val, coefficientBound⟩
      some (applyCoefficient scalar
        (block.form retainedStart fits ⟨_, slotBound⟩))
    else none) = _
  rw [dif_pos coefficient.isLt]

theorem Term.constant_form? {logicalWidth : Nat}
    (oneColumn : Fin logicalWidth) (coefficient : F)
    (offsets : Coordinate) :
    (Term.constant coefficient.val).form? logicalWidth oneColumn.val offsets =
      some (SparseForm.singleton oneColumn coefficient) := by
  change (if oneBound : oneColumn.val < logicalWidth then
      if coefficientBound : coefficient.val < goldilocksModulus then
        some (SparseForm.singleton ⟨oneColumn.val, oneBound⟩
          ⟨coefficient.val, coefficientBound⟩)
      else none
    else none) = _
  rw [dif_pos oneColumn.isLt, dif_pos coefficient.isLt]

structure Rule where
  region : Region
  term : Term
deriving Repr, DecidableEq

def Rule.format : Format Rule where
  encode := fun rule => .array [
    Region.format.encode rule.region,
    Term.format.encode rule.term]
  decode
    | .array [region, term] => do
        pure ⟨← Region.format.decode region, ← Term.format.decode term⟩
    | _ => .error "invalid affine-grid rule"
  decode_encode := by
    rintro ⟨region, term⟩
    simp only
    rw [Region.format.decode_encode, Term.format.decode_encode]
    rfl

/-- `some none` means that this rule is outside its region. `none` means an
applicable rule was malformed. -/
def Rule.form? (rule : Rule) (logicalWidth oneColumn : Nat)
    (coordinate : Coordinate) : Option (Option (SparseForm logicalWidth)) :=
  match rule.region.offsets? coordinate with
  | none => some none
  | some offsets => do
      pure (some (← rule.term.form? logicalWidth oneColumn offsets))

theorem Rule.form?_eq_some_none (rule : Rule)
    (logicalWidth oneColumn : Nat) (coordinate : Coordinate)
    (outside : rule.region.offsets? coordinate = none) :
    rule.form? logicalWidth oneColumn coordinate = some none := by
  unfold form?
  rw [outside]

theorem Rule.retained_form?_ofSemantic
    {sourceWidth logicalWidth : Nat}
    (region : Region) (major : Fin region.majorCount)
    (middle : Fin region.middleCount) (minor : Fin region.minorCount)
    (block : LowNormBlock.Block sourceWidth) (retainedStart : Nat)
    (fits : retainedStart + block.coordinateCount ≤ logicalWidth)
    (oneColumn slotBase majorStride middleStride minorStride : Nat)
    (coefficient : F)
    (slotBound : slotBase + major.val * majorStride +
      middle.val * middleStride + minor.val * minorStride < block.slotCount) :
    (Rule.mk region (.retained (RetainedBlock.ofSemantic block retainedStart)
      slotBase majorStride middleStride minorStride coefficient.val)).form?
        logicalWidth oneColumn {
          major := region.majorStart + major.val
          middle := region.middleStart + middle.val
          minor := region.minorStart + minor.val } =
      some (some (applyCoefficient coefficient
        (block.form retainedStart fits
          ⟨slotBase + major.val * majorStride +
            middle.val * middleStride + minor.val * minorStride,
            slotBound⟩))) := by
  unfold form?
  rw [Region.offsets?_of_offsets region major middle minor]
  simp only
  rw [Term.retained_form?_ofSemantic block retainedStart fits oneColumn
    slotBase majorStride middleStride minorStride coefficient
    { major := major.val, middle := middle.val, minor := minor.val }
    (by simpa using slotBound)]
  rfl

theorem Rule.constant_form? {logicalWidth : Nat}
    (region : Region) (major : Fin region.majorCount)
    (middle : Fin region.middleCount) (minor : Fin region.minorCount)
    (oneColumn : Fin logicalWidth) (coefficient : F) :
    (Rule.mk region (.constant coefficient.val)).form? logicalWidth
        oneColumn.val {
          major := region.majorStart + major.val
          middle := region.middleStart + middle.val
          minor := region.minorStart + minor.val } =
      some (some (SparseForm.singleton oneColumn coefficient)) := by
  unfold form?
  rw [Region.offsets?_of_offsets region major middle minor]
  simp only
  rw [Term.constant_form? oneColumn coefficient]
  rfl

structure Program where
  rules : List Rule
deriving Repr, DecidableEq

def Program.format : Format Program where
  encode := fun program => (list Rule.format).encode program.rules
  decode := fun value => do
    pure ⟨← (list Rule.format).decode value⟩
  decode_encode := by
    intro program
    cases program
    simp [Format.decode_encode]

def addSelected {logicalWidth : Nat} (accumulated : SparseForm logicalWidth) :
    Option (SparseForm logicalWidth) → SparseForm logicalWidth
  | none => accumulated
  | some form => SparseForm.add accumulated form

def combine {logicalWidth : Nat}
    (results : List (Option (SparseForm logicalWidth))) :
    SparseForm logicalWidth :=
  results.foldl addSelected .empty

private def applyRules? {logicalWidth : Nat}
    (oneColumn : Nat) (coordinate : Coordinate) :
    List Rule → SparseForm logicalWidth → Option (SparseForm logicalWidth)
  | [], accumulated => some accumulated
  | rule :: rest, accumulated => do
      let selected ← rule.form? logicalWidth oneColumn coordinate
      applyRules? oneColumn coordinate rest (addSelected accumulated selected)

def Program.form? (program : Program) (logicalWidth oneColumn : Nat)
    (coordinate : Coordinate) : Option (SparseForm logicalWidth) :=
  applyRules? oneColumn coordinate program.rules .empty

/-- A singleton affine program returns its selected form exactly. -/
theorem Program.singleton_form?_of_selected {logicalWidth : Nat}
    (rule : Rule) (oneColumn : Nat) (coordinate : Coordinate)
    (form : SparseForm logicalWidth)
    (loaded : rule.form? logicalWidth oneColumn coordinate =
      some (some form)) :
    (Program.mk [rule]).form? logicalWidth oneColumn coordinate = some form := by
  unfold Program.form? applyRules?
  rw [loaded]
  rfl

theorem Program.two_form?_of_results {logicalWidth : Nat}
    (rule0 rule1 : Rule) (oneColumn : Nat) (coordinate : Coordinate)
    (result0 result1 : Option (SparseForm logicalWidth))
    (loaded0 : rule0.form? logicalWidth oneColumn coordinate = some result0)
    (loaded1 : rule1.form? logicalWidth oneColumn coordinate = some result1) :
    (Program.mk [rule0, rule1]).form? logicalWidth oneColumn coordinate =
      some (addSelected (addSelected SparseForm.empty result0) result1) := by
  simp only [Program.form?, applyRules?, loaded0, loaded1]
  rfl

theorem Program.three_form?_of_results {logicalWidth : Nat}
    (rule0 rule1 rule2 : Rule) (oneColumn : Nat) (coordinate : Coordinate)
    (result0 result1 result2 : Option (SparseForm logicalWidth))
    (loaded0 : rule0.form? logicalWidth oneColumn coordinate = some result0)
    (loaded1 : rule1.form? logicalWidth oneColumn coordinate = some result1)
    (loaded2 : rule2.form? logicalWidth oneColumn coordinate = some result2) :
    (Program.mk [rule0, rule1, rule2]).form? logicalWidth oneColumn coordinate =
      some (addSelected
        (addSelected (addSelected SparseForm.empty result0) result1)
        result2) := by
  simp only [Program.form?, applyRules?, loaded0, loaded1, loaded2]
  rfl

theorem Program.six_form?_of_results {logicalWidth : Nat}
    (rule0 rule1 rule2 rule3 rule4 rule5 : Rule)
    (oneColumn : Nat) (coordinate : Coordinate)
    (result0 result1 result2 result3 result4 result5 :
      Option (SparseForm logicalWidth))
    (loaded0 : rule0.form? logicalWidth oneColumn coordinate = some result0)
    (loaded1 : rule1.form? logicalWidth oneColumn coordinate = some result1)
    (loaded2 : rule2.form? logicalWidth oneColumn coordinate = some result2)
    (loaded3 : rule3.form? logicalWidth oneColumn coordinate = some result3)
    (loaded4 : rule4.form? logicalWidth oneColumn coordinate = some result4)
    (loaded5 : rule5.form? logicalWidth oneColumn coordinate = some result5) :
    (Program.mk [rule0, rule1, rule2, rule3, rule4, rule5]).form?
        logicalWidth oneColumn coordinate =
      some (addSelected (addSelected (addSelected
        (addSelected (addSelected (addSelected SparseForm.empty result0)
          result1) result2) result3) result4) result5) := by
  simp only [Program.form?, applyRules?, loaded0, loaded1, loaded2, loaded3,
    loaded4, loaded5]
  rfl

private theorem applyRules?_of_results {logicalWidth : Nat}
    (oneColumn : Nat) (coordinate : Coordinate) :
    ∀ (rules : List Rule)
      (results : List (Option (SparseForm logicalWidth)))
      (accumulated : SparseForm logicalWidth),
      List.Forall₂ (fun rule result =>
        rule.form? logicalWidth oneColumn coordinate = some result)
        rules results →
      applyRules? oneColumn coordinate rules accumulated =
        some (results.foldl addSelected accumulated)
  | [], [], accumulated, .nil => rfl
  | rule :: rules, result :: results, accumulated, .cons loaded rest => by
      unfold applyRules?
      rw [loaded]
      exact applyRules?_of_results oneColumn coordinate rules results
        (addSelected accumulated result) rest

/-- Exact per-rule results determine the affine sum in package rule order. -/
theorem Program.form?_of_results {logicalWidth : Nat}
    (program : Program) (oneColumn : Nat) (coordinate : Coordinate)
    (results : List (Option (SparseForm logicalWidth)))
    (loaded : List.Forall₂ (fun rule result =>
      rule.form? logicalWidth oneColumn coordinate = some result)
      program.rules results) :
    program.form? logicalWidth oneColumn coordinate = some (combine results) := by
  exact applyRules?_of_results oneColumn coordinate program.rules results
    .empty loaded

end NightstreamFPrime.Export.MatrixProgram.AffineGrid
