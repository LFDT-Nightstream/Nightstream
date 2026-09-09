import Mathlib.Tactic.FinCases
import NightstreamFPrime.Export.MatrixProgram
import NightstreamFPrime.Export.MatrixProgram.Affine
import NightstreamFPrime.Layout.ProductionRelation.PoseidonSboxPlan

/-!
Owns the package wire language for Poseidon2 input states. Rules add either a
retained low-norm form or a canonical constant on one rectangular invocation
and lane region. Ordered rules preserve sparse-entry order.

This module does not select transcript phases or challenge schedules.
-/

namespace NightstreamFPrime.Export.MatrixProgram.PoseidonInput

open NightstreamFPrime.Export.Codec
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

structure Region where
  invocationStart : Nat
  invocationCount : Nat
  laneStart : Nat
  laneCount : Nat
deriving Repr, DecidableEq

def Region.format : Format Region where
  encode := fun region => .array [
    .atom region.invocationStart,
    .atom region.invocationCount,
    .atom region.laneStart,
    .atom region.laneCount]
  decode
    | .array [.atom invocationStart, .atom invocationCount,
        .atom laneStart, .atom laneCount] =>
        .ok ⟨invocationStart, invocationCount, laneStart, laneCount⟩
    | _ => .error "invalid Poseidon2 input region"
  decode_encode := by
    intro region
    cases region
    rfl

def Region.offsets? (region : Region) (invocation lane : Nat) :
    Option (Nat × Nat) :=
  if region.invocationStart ≤ invocation then
    let invocationOffset := invocation - region.invocationStart
    if invocationOffset < region.invocationCount then
      if region.laneStart ≤ lane then
        let laneOffset := lane - region.laneStart
        if laneOffset < region.laneCount then
          some (invocationOffset, laneOffset)
        else none
      else none
    else none
  else none

theorem Region.offsets?_of_offsets (region : Region)
    (invocationOffset : Fin region.invocationCount)
    (laneOffset : Fin region.laneCount) :
    region.offsets?
        (region.invocationStart + invocationOffset.val)
        (region.laneStart + laneOffset.val) =
      some (invocationOffset.val, laneOffset.val) := by
  unfold Region.offsets?
  rw [if_pos (by omega)]
  rw [show region.invocationStart + invocationOffset.val -
      region.invocationStart = invocationOffset.val by omega]
  rw [if_pos invocationOffset.isLt, if_pos (by omega)]
  rw [show region.laneStart + laneOffset.val - region.laneStart =
      laneOffset.val by omega]
  rw [if_pos laneOffset.isLt]

inductive InvocationTag where
  | absorb
  | squeezeFirst
  | squeezeSecond
deriving Repr, DecidableEq

def InvocationTag.format : Format InvocationTag where
  encode
    | .absorb => .atom 0
    | .squeezeFirst => .atom 1
    | .squeezeSecond => .atom 2
  decode
    | .atom 0 => .ok .absorb
    | .atom 1 => .ok .squeezeFirst
    | .atom 2 => .ok .squeezeSecond
    | _ => .error "invalid Poseidon2 invocation tag"
  decode_encode := by
    intro tag
    cases tag <;> rfl

/-- Canonical random-access invocation tags. The wire is a plain ordered
array; the in-memory array gives constant-time lookup. -/
structure TagTable where
  tags : Array InvocationTag
deriving Repr, DecidableEq

def TagTable.format : Format TagTable where
  encode := fun table => (list InvocationTag.format).encode table.tags.toList
  decode := fun value => do
    pure ⟨(← (list InvocationTag.format).decode value).toArray⟩
  decode_encode := by
    intro table
    cases table with
    | mk tags =>
        simp [Format.decode_encode]

def TagTable.tag? (table : TagTable) (invocation : Nat) :
    Option InvocationTag :=
  table.tags[invocation]?

def TagTable.ofSemantic {invocationCount : Nat}
    (tag : Fin invocationCount → InvocationTag) : TagTable where
  tags := Array.ofFn tag

@[simp] theorem TagTable.tag?_ofSemantic {invocationCount : Nat}
    (tag : Fin invocationCount → InvocationTag)
    (invocation : Fin invocationCount) :
    (TagTable.ofSemantic tag).tag? invocation.val = some (tag invocation) := by
  simp [TagTable.tag?, TagTable.ofSemantic]

/-- Random-access optional constant words. An absent array cell is malformed;
a stored `none` is the canonical instruction to add no form. -/
structure OptionalConstantTable where
  values : Array (Option Nat)
deriving Repr, DecidableEq

def OptionalConstantTable.format : Format OptionalConstantTable where
  encode := fun table =>
    (list (option nat)).encode table.values.toList
  decode := fun value => do
    pure ⟨(← (list (option nat)).decode value).toArray⟩
  decode_encode := by
    intro table
    cases table with
    | mk values =>
        simp [Format.decode_encode]

def OptionalConstantTable.value? (table : OptionalConstantTable)
    (index : Nat) : Option (Option Nat) :=
  table.values[index]?

def OptionalConstantTable.ofSemantic {count : Nat}
    (value : Fin count → Option F) : OptionalConstantTable where
  values := Array.ofFn fun index =>
    (value index).map fun coefficient => coefficient.val

@[simp] theorem OptionalConstantTable.value?_ofSemantic {count : Nat}
    (value : Fin count → Option F) (index : Fin count) :
    (OptionalConstantTable.ofSemantic value).value? index.val =
      some ((value index).map fun coefficient => coefficient.val) := by
  simp [OptionalConstantTable.value?, OptionalConstantTable.ofSemantic]

inductive Term where
  | retained (block : RetainedBlock) (slotBase invocationStride laneStride : Nat)
  | constant (coefficient : Nat)
  | external (block : RetainedBlock) (slotBase invocationStride : Nat)
  | taggedRetained (block : RetainedBlock) (tags : TagTable)
      (required : InvocationTag)
      (slotBase invocationStride laneStride : Nat)
  | optionalConstant (values : OptionalConstantTable) (laneCount : Nat)
  | taggedAffine (values : Affine.Table) (substitution : SourceSubstitution)
      (tags : TagTable) (required : InvocationTag) (laneCount : Nat)
deriving Repr, DecidableEq

def Term.format : Format Term where
  encode
    | .retained block slotBase invocationStride laneStride => .array [
        .atom 0, RetainedBlock.format.encode block, .atom slotBase,
        .atom invocationStride, .atom laneStride]
    | .constant coefficient => .array [.atom 1, .atom coefficient]
    | .external block slotBase invocationStride => .array [
        .atom 2, RetainedBlock.format.encode block, .atom slotBase,
        .atom invocationStride]
    | .taggedRetained block tags required slotBase invocationStride
        laneStride => .array [
          .atom 3, RetainedBlock.format.encode block, TagTable.format.encode tags,
          InvocationTag.format.encode required, .atom slotBase,
          .atom invocationStride, .atom laneStride]
    | .optionalConstant values laneCount => .array [
        .atom 4, OptionalConstantTable.format.encode values, .atom laneCount]
    | .taggedAffine values substitution tags required laneCount => .array [
        .atom 5, Affine.Table.format.encode values,
        SourceSubstitution.format.encode substitution, TagTable.format.encode tags,
        InvocationTag.format.encode required, .atom laneCount]
  decode
    | .array [.atom 0, block, .atom slotBase, .atom invocationStride,
        .atom laneStride] => do
        pure (.retained (← RetainedBlock.format.decode block) slotBase
          invocationStride laneStride)
    | .array [.atom 1, .atom coefficient] => .ok (.constant coefficient)
    | .array [.atom 2, block, .atom slotBase, .atom invocationStride] => do
        pure (.external (← RetainedBlock.format.decode block) slotBase
          invocationStride)
    | .array [.atom 3, block, tags, required, .atom slotBase,
        .atom invocationStride, .atom laneStride] => do
        pure (.taggedRetained (← RetainedBlock.format.decode block)
          (← TagTable.format.decode tags) (← InvocationTag.format.decode required)
          slotBase invocationStride laneStride)
    | .array [.atom 4, values, .atom laneCount] => do
        pure (.optionalConstant (← OptionalConstantTable.format.decode values)
          laneCount)
    | .array [.atom 5, values, substitution, tags, required, .atom laneCount] => do
        pure (.taggedAffine (← Affine.Table.format.decode values)
          (← SourceSubstitution.format.decode substitution)
          (← TagTable.format.decode tags) (← InvocationTag.format.decode required)
          laneCount)
    | _ => .error "invalid Poseidon2 input term"
  decode_encode := by
    intro term
    cases term <;> simp [RetainedBlock.format.decode_encode,
      TagTable.format.decode_encode, InvocationTag.format.decode_encode,
      OptionalConstantTable.format.decode_encode, Affine.Table.format.decode_encode,
      SourceSubstitution.format.decode_encode] <;> rfl

def Term.form? (term : Term) (logicalWidth oneColumn : Nat)
    (invocationOffset laneOffset : Nat) : Option (SparseForm logicalWidth) :=
  match term with
  | .retained block slotBase invocationStride laneStride =>
      block.form? logicalWidth
        (slotBase + invocationOffset * invocationStride +
          laneOffset * laneStride)
  | .constant coefficient =>
      if oneBound : oneColumn < logicalWidth then
        if coefficientBound : coefficient < Spec.goldilocksModulus then
          some (SparseForm.singleton ⟨oneColumn, oneBound⟩
            ⟨coefficient, coefficientBound⟩)
        else none
      else none
  | .external block slotBase invocationStride => do
      block.externalForm? logicalWidth
        (slotBase + invocationOffset * invocationStride) laneOffset
  | .taggedRetained block tags required slotBase invocationStride laneStride => do
      let actual ← tags.tag? invocationOffset
      if actual = required then
        block.form? logicalWidth
          (slotBase + invocationOffset * invocationStride +
            laneOffset * laneStride)
      else
        some .empty
  | .optionalConstant values laneCount =>
      match values.value? (invocationOffset * laneCount + laneOffset) with
      | none => none
      | some none => some SparseForm.empty
      | some (some coefficient) =>
          if oneBound : oneColumn < logicalWidth then
            if coefficientBound : coefficient < Spec.goldilocksModulus then
              some (SparseForm.singleton ⟨oneColumn, oneBound⟩
                ⟨coefficient, coefficientBound⟩)
            else none
          else none
  | .taggedAffine values substitution tags required laneCount => do
      let actual ← tags.tag? invocationOffset
      if actual = required then
        values.compile? substitution logicalWidth oneColumn
          (invocationOffset * laneCount + laneOffset)
      else some .empty

theorem Term.retained_form?_ofSemantic
    {sourceWidth logicalWidth : Nat}
    (block : LowNormBlock.Block sourceWidth) (retainedStart : Nat)
    (fits : retainedStart + block.coordinateCount ≤ logicalWidth)
    (oneColumn slotBase invocationStride laneStride
      invocationOffset laneOffset : Nat)
    (slotBound : slotBase + invocationOffset * invocationStride +
      laneOffset * laneStride < block.slotCount) :
    (Term.retained (RetainedBlock.ofSemantic block retainedStart) slotBase
      invocationStride laneStride).form? logicalWidth oneColumn
        invocationOffset laneOffset =
      some (block.form retainedStart fits
        ⟨slotBase + invocationOffset * invocationStride +
          laneOffset * laneStride, slotBound⟩) := by
  change (RetainedBlock.ofSemantic block retainedStart).form? logicalWidth
      (slotBase + invocationOffset * invocationStride +
        laneOffset * laneStride) = _
  exact RetainedBlock.form?_ofSemantic block retainedStart fits
    ⟨slotBase + invocationOffset * invocationStride +
      laneOffset * laneStride, slotBound⟩

theorem Term.constant_form? {logicalWidth : Nat}
    (oneColumn : Fin logicalWidth) (coefficient : F)
    (invocationOffset laneOffset : Nat) :
    (Term.constant coefficient.val).form? logicalWidth oneColumn.val
        invocationOffset laneOffset =
      some (SparseForm.singleton oneColumn coefficient) := by
  change (if oneBound : oneColumn.val < logicalWidth then
      if coefficientBound : coefficient.val < goldilocksModulus then
        some (SparseForm.singleton ⟨oneColumn.val, oneBound⟩
          ⟨coefficient.val, coefficientBound⟩)
      else none
    else none) = _
  rw [dif_pos oneColumn.isLt, dif_pos coefficient.isLt]

/-- A selected affine word is compiled by the existing source substitution. -/
theorem Term.taggedAffine_form?_of_eq
    {logicalWidth invocationCount laneCount : Nat}
    (combinations : Fin (invocationCount * laneCount) → R1CS.LinearCombination)
    (substitution : SourceSubstitution) (oneColumn : Fin logicalWidth)
    (tag : Fin invocationCount → InvocationTag) (required : InvocationTag)
    (invocation : Fin invocationCount) (lane : Fin laneCount)
    (selected : tag invocation = required) :
    (Term.taggedAffine (Affine.Table.ofSemantic combinations) substitution
      (TagTable.ofSemantic tag) required laneCount).form? logicalWidth
        oneColumn.val invocation.val lane.val =
      Ordinary.compileCombination? substitution oneColumn
        (combinations (Fin.encodeProd (invocation, lane))) := by
  change (do
    let actual ← (TagTable.ofSemantic tag).tag? invocation.val
    if actual = required then
      (Affine.Table.ofSemantic combinations).compile? substitution logicalWidth
        oneColumn.val (invocation.val * laneCount + lane.val)
    else some .empty) = _
  rw [TagTable.tag?_ofSemantic]
  change (if tag invocation = required then
    (Affine.Table.ofSemantic combinations).compile? substitution logicalWidth
      oneColumn.val (invocation.val * laneCount + lane.val)
    else some .empty) = _
  rw [if_pos selected]
  have indexEq : invocation.val * laneCount + lane.val =
      (Fin.encodeProd (invocation, lane)).val := by
    change invocation.val * laneCount + lane.val = laneCount * invocation.val + lane.val
    rw [Nat.mul_comm invocation.val laneCount]
  rw [indexEq]
  exact Affine.Table.compile?_ofSemantic combinations substitution oneColumn
    (Fin.encodeProd (invocation, lane))

theorem Term.taggedAffine_form?_of_ne
    {logicalWidth invocationCount : Nat}
    (values : Affine.Table) (substitution : SourceSubstitution)
    (oneColumn laneCount laneOffset : Nat)
    (tag : Fin invocationCount → InvocationTag) (required : InvocationTag)
    (invocation : Fin invocationCount) (notSelected : tag invocation ≠ required) :
    (Term.taggedAffine values substitution (TagTable.ofSemantic tag) required laneCount).form?
        logicalWidth oneColumn invocation.val laneOffset = some .empty := by
  change (do
    let actual ← (TagTable.ofSemantic tag).tag? invocation.val
    if actual = required then
      values.compile? substitution logicalWidth oneColumn
        (invocation.val * laneCount + laneOffset)
    else some .empty) = _
  rw [TagTable.tag?_ofSemantic]
  change (if tag invocation = required then
    values.compile? substitution logicalWidth oneColumn
      (invocation.val * laneCount + laneOffset)
    else some .empty) = _
  rw [if_neg notSelected]

theorem Term.taggedRetained_form?_ofSemantic_of_eq
    {sourceWidth logicalWidth invocationCount : Nat}
    (block : LowNormBlock.Block sourceWidth) (retainedStart : Nat)
    (fits : retainedStart + block.coordinateCount ≤ logicalWidth)
    (oneColumn slotBase invocationStride laneStride : Nat)
    (tag : Fin invocationCount → InvocationTag) (required : InvocationTag)
    (invocation : Fin invocationCount) (laneOffset : Nat)
    (slotBound : slotBase + invocation.val * invocationStride +
      laneOffset * laneStride < block.slotCount)
    (selected : tag invocation = required) :
    (Term.taggedRetained (RetainedBlock.ofSemantic block retainedStart)
      (TagTable.ofSemantic tag) required slotBase invocationStride
      laneStride).form? logicalWidth oneColumn invocation.val laneOffset =
      some (block.form retainedStart fits
        ⟨slotBase + invocation.val * invocationStride +
          laneOffset * laneStride, slotBound⟩) := by
  change (do
    let actual ← (TagTable.ofSemantic tag).tag? invocation.val
    if actual = required then
      (RetainedBlock.ofSemantic block retainedStart).form? logicalWidth
        (slotBase + invocation.val * invocationStride +
          laneOffset * laneStride)
    else
      some .empty) = _
  rw [TagTable.tag?_ofSemantic]
  change (if tag invocation = required then
      (RetainedBlock.ofSemantic block retainedStart).form? logicalWidth
        (slotBase + invocation.val * invocationStride +
          laneOffset * laneStride)
    else some .empty) = _
  rw [if_pos selected]
  exact RetainedBlock.form?_ofSemantic block retainedStart fits
    ⟨slotBase + invocation.val * invocationStride +
      laneOffset * laneStride, slotBound⟩

theorem Term.taggedRetained_form?_ofSemantic_of_ne
    {sourceWidth logicalWidth invocationCount : Nat}
    (block : LowNormBlock.Block sourceWidth) (retainedStart : Nat)
    (oneColumn slotBase invocationStride laneStride : Nat)
    (tag : Fin invocationCount → InvocationTag) (required : InvocationTag)
    (invocation : Fin invocationCount) (laneOffset : Nat)
    (notSelected : tag invocation ≠ required) :
    (Term.taggedRetained (RetainedBlock.ofSemantic block retainedStart)
      (TagTable.ofSemantic tag) required slotBase invocationStride
      laneStride).form? logicalWidth oneColumn invocation.val laneOffset =
      some .empty := by
  change (do
    let actual ← (TagTable.ofSemantic tag).tag? invocation.val
    if actual = required then
      (RetainedBlock.ofSemantic block retainedStart).form? logicalWidth
        (slotBase + invocation.val * invocationStride +
          laneOffset * laneStride)
    else
      some .empty) = _
  rw [TagTable.tag?_ofSemantic]
  change (if tag invocation = required then
      (RetainedBlock.ofSemantic block retainedStart).form? logicalWidth
        (slotBase + invocation.val * invocationStride +
          laneOffset * laneStride)
    else some .empty) = _
  rw [if_neg notSelected]

theorem Term.optionalConstant_form?_ofSemantic_of_some
    {logicalWidth count : Nat} (oneColumn : Fin logicalWidth)
    (value : Fin count → Option F) (laneCount invocationOffset laneOffset : Nat)
    (index : Fin count)
    (indexEq : invocationOffset * laneCount + laneOffset = index.val)
    (coefficient : F) (found : value index = some coefficient) :
    (Term.optionalConstant (OptionalConstantTable.ofSemantic value)
      laneCount).form? logicalWidth oneColumn.val invocationOffset laneOffset =
      some (SparseForm.singleton oneColumn coefficient) := by
  change (match (OptionalConstantTable.ofSemantic value).value?
      (invocationOffset * laneCount + laneOffset) with
    | none => none
    | some none => some SparseForm.empty
    | some (some encoded) =>
        if oneBound : oneColumn.val < logicalWidth then
          if coefficientBound : encoded < goldilocksModulus then
            some (SparseForm.singleton ⟨oneColumn.val, oneBound⟩
              ⟨encoded, coefficientBound⟩)
          else none
        else none) = _
  rw [indexEq, OptionalConstantTable.value?_ofSemantic, found]
  simp [oneColumn.isLt, coefficient.isLt]

theorem Term.optionalConstant_form?_ofSemantic_of_none
    {logicalWidth count : Nat} (oneColumn : Nat)
    (value : Fin count → Option F) (laneCount invocationOffset laneOffset : Nat)
    (index : Fin count)
    (indexEq : invocationOffset * laneCount + laneOffset = index.val)
    (found : value index = none) :
    (Term.optionalConstant (OptionalConstantTable.ofSemantic value)
      laneCount).form? logicalWidth oneColumn invocationOffset laneOffset =
      some .empty := by
  change (match (OptionalConstantTable.ofSemantic value).value?
      (invocationOffset * laneCount + laneOffset) with
    | none => none
    | some none => some SparseForm.empty
    | some (some encoded) =>
        if oneBound : oneColumn < logicalWidth then
          if coefficientBound : encoded < goldilocksModulus then
            some (SparseForm.singleton ⟨oneColumn, oneBound⟩
              ⟨encoded, coefficientBound⟩)
          else none
        else none) = _
  rw [indexEq, OptionalConstantTable.value?_ofSemantic, found]
  simp

theorem Term.external_form?_ofSemantic
    {sourceWidth logicalWidth : Nat}
    (block : LowNormBlock.Block sourceWidth) (retainedStart : Nat)
    (fits : retainedStart + block.coordinateCount ≤ logicalWidth)
    (oneColumn slotBase invocationStride invocationOffset : Nat)
    (slotBound : ∀ lane : Fin 8,
      slotBase + invocationOffset * invocationStride + lane.val <
        block.slotCount)
    (lane : Fin 8) :
    (Term.external (RetainedBlock.ofSemantic block retainedStart) slotBase
      invocationStride).form? logicalWidth oneColumn invocationOffset lane.val =
      some (SparseLayer.external (fun selected : Fin 8 =>
        block.form retainedStart fits
          ⟨slotBase + invocationOffset * invocationStride + selected.val,
            slotBound selected⟩) lane) := by
  simpa [Nat.add_assoc] using
    RetainedBlock.externalForm?_ofSemantic block retainedStart fits
      (slotBase + invocationOffset * invocationStride)
      (fun selected => by simpa [Nat.add_assoc] using slotBound selected) lane

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
    | _ => .error "invalid Poseidon2 input rule"
  decode_encode := by
    intro rule
    cases rule
    simp [Region.format.decode_encode, Term.format.decode_encode]
    rfl

/-- `some none` means that the rule does not apply. `none` means that an
applicable rule was malformed. -/
def Rule.form? (rule : Rule) (logicalWidth oneColumn invocation lane : Nat) :
    Option (Option (SparseForm logicalWidth)) :=
  match rule.region.offsets? invocation lane with
  | none => some none
  | some offsets => do
      pure (some (← rule.term.form? logicalWidth oneColumn offsets.1 offsets.2))

theorem Rule.form?_eq_some_none (rule : Rule)
    (logicalWidth oneColumn invocation lane : Nat)
    (outside : rule.region.offsets? invocation lane = none) :
    rule.form? logicalWidth oneColumn invocation lane = some none := by
  unfold Rule.form?
  rw [outside]

theorem Rule.retained_form?_ofSemantic
    {sourceWidth logicalWidth : Nat}
    (region : Region) (invocationOffset : Fin region.invocationCount)
    (laneOffset : Fin region.laneCount)
    (block : LowNormBlock.Block sourceWidth) (retainedStart : Nat)
    (fits : retainedStart + block.coordinateCount ≤ logicalWidth)
    (oneColumn slotBase invocationStride laneStride : Nat)
    (slotBound : slotBase + invocationOffset.val * invocationStride +
      laneOffset.val * laneStride < block.slotCount) :
    (Rule.mk region (.retained (RetainedBlock.ofSemantic block retainedStart)
      slotBase invocationStride laneStride)).form? logicalWidth oneColumn
        (region.invocationStart + invocationOffset.val)
        (region.laneStart + laneOffset.val) =
      some (some (block.form retainedStart fits
        ⟨slotBase + invocationOffset.val * invocationStride +
          laneOffset.val * laneStride, slotBound⟩)) := by
  unfold Rule.form?
  rw [Region.offsets?_of_offsets region invocationOffset laneOffset]
  simp only
  rw [Term.retained_form?_ofSemantic block retainedStart fits oneColumn
    slotBase invocationStride laneStride invocationOffset.val laneOffset.val
    slotBound]
  rfl

theorem Rule.constant_form?
    {logicalWidth : Nat} (region : Region)
    (invocationOffset : Fin region.invocationCount)
    (laneOffset : Fin region.laneCount)
    (oneColumn : Fin logicalWidth) (coefficient : F) :
    (Rule.mk region (.constant coefficient.val)).form? logicalWidth
        oneColumn.val (region.invocationStart + invocationOffset.val)
        (region.laneStart + laneOffset.val) =
      some (some (SparseForm.singleton oneColumn coefficient)) := by
  unfold Rule.form?
  rw [Region.offsets?_of_offsets region invocationOffset laneOffset]
  simp only
  rw [Term.constant_form? oneColumn coefficient]
  rfl

theorem Rule.external_form?_ofSemantic
    {sourceWidth logicalWidth : Nat}
    (region : Region) (invocationOffset : Fin region.invocationCount)
    (laneOffset : Fin region.laneCount) (laneBound : laneOffset.val < 8)
    (block : LowNormBlock.Block sourceWidth) (retainedStart : Nat)
    (fits : retainedStart + block.coordinateCount ≤ logicalWidth)
    (oneColumn slotBase invocationStride : Nat)
    (slotBound : ∀ lane : Fin 8,
      slotBase + invocationOffset.val * invocationStride + lane.val <
        block.slotCount) :
    (Rule.mk region (.external (RetainedBlock.ofSemantic block retainedStart)
      slotBase invocationStride)).form? logicalWidth oneColumn
        (region.invocationStart + invocationOffset.val)
        (region.laneStart + laneOffset.val) =
      some (some (SparseLayer.external (fun selected : Fin 8 =>
        block.form retainedStart fits
          ⟨slotBase + invocationOffset.val * invocationStride + selected.val,
            slotBound selected⟩) ⟨laneOffset.val, laneBound⟩)) := by
  unfold Rule.form?
  rw [Region.offsets?_of_offsets region invocationOffset laneOffset]
  simp only
  rw [Term.external_form?_ofSemantic block retainedStart fits oneColumn
    slotBase invocationStride invocationOffset.val slotBound
    ⟨laneOffset.val, laneBound⟩]
  rfl

theorem Rule.taggedRetained_form?_ofSemantic_of_eq
    {sourceWidth logicalWidth : Nat}
    (region : Region) (invocationOffset : Fin region.invocationCount)
    (laneOffset : Fin region.laneCount)
    (block : LowNormBlock.Block sourceWidth) (retainedStart : Nat)
    (fits : retainedStart + block.coordinateCount ≤ logicalWidth)
    (oneColumn slotBase invocationStride laneStride : Nat)
    (tag : Fin region.invocationCount → InvocationTag)
    (required : InvocationTag)
    (slotBound : slotBase + invocationOffset.val * invocationStride +
      laneOffset.val * laneStride < block.slotCount)
    (selected : tag invocationOffset = required) :
    (Rule.mk region (.taggedRetained
      (RetainedBlock.ofSemantic block retainedStart)
      (TagTable.ofSemantic tag) required slotBase invocationStride
      laneStride)).form? logicalWidth oneColumn
        (region.invocationStart + invocationOffset.val)
        (region.laneStart + laneOffset.val) =
      some (some (block.form retainedStart fits
        ⟨slotBase + invocationOffset.val * invocationStride +
          laneOffset.val * laneStride, slotBound⟩)) := by
  unfold Rule.form?
  rw [Region.offsets?_of_offsets region invocationOffset laneOffset]
  simp only
  rw [Term.taggedRetained_form?_ofSemantic_of_eq block retainedStart fits
    oneColumn slotBase invocationStride laneStride tag required
    invocationOffset laneOffset.val slotBound selected]
  rfl

theorem Rule.taggedRetained_form?_ofSemantic_of_ne
    {sourceWidth logicalWidth : Nat}
    (region : Region) (invocationOffset : Fin region.invocationCount)
    (laneOffset : Fin region.laneCount)
    (block : LowNormBlock.Block sourceWidth) (retainedStart : Nat)
    (oneColumn slotBase invocationStride laneStride : Nat)
    (tag : Fin region.invocationCount → InvocationTag)
    (required : InvocationTag)
    (notSelected : tag invocationOffset ≠ required) :
    (Rule.mk region (.taggedRetained
      (RetainedBlock.ofSemantic block retainedStart)
      (TagTable.ofSemantic tag) required slotBase invocationStride
      laneStride)).form? logicalWidth oneColumn
        (region.invocationStart + invocationOffset.val)
        (region.laneStart + laneOffset.val) =
      some (some .empty) := by
  unfold Rule.form?
  rw [Region.offsets?_of_offsets region invocationOffset laneOffset]
  simp only
  rw [Term.taggedRetained_form?_ofSemantic_of_ne block retainedStart
    oneColumn slotBase invocationStride laneStride tag required
    invocationOffset laneOffset.val notSelected]
  rfl

theorem Rule.optionalConstant_form?_ofSemantic_of_some
    {logicalWidth count : Nat} (region : Region)
    (invocationOffset : Fin region.invocationCount)
    (laneOffset : Fin region.laneCount) (oneColumn : Fin logicalWidth)
    (value : Fin count → Option F) (laneCount : Nat) (index : Fin count)
    (indexEq : invocationOffset.val * laneCount + laneOffset.val = index.val)
    (coefficient : F) (found : value index = some coefficient) :
    (Rule.mk region (.optionalConstant
      (OptionalConstantTable.ofSemantic value) laneCount)).form?
        logicalWidth oneColumn.val
        (region.invocationStart + invocationOffset.val)
        (region.laneStart + laneOffset.val) =
      some (some (SparseForm.singleton oneColumn coefficient)) := by
  unfold Rule.form?
  rw [Region.offsets?_of_offsets region invocationOffset laneOffset]
  simp only
  rw [Term.optionalConstant_form?_ofSemantic_of_some oneColumn value
    laneCount invocationOffset.val laneOffset.val index indexEq coefficient found]
  rfl

theorem Rule.optionalConstant_form?_ofSemantic_of_none
    {logicalWidth count : Nat} (region : Region)
    (invocationOffset : Fin region.invocationCount)
    (laneOffset : Fin region.laneCount) (oneColumn : Nat)
    (value : Fin count → Option F) (laneCount : Nat) (index : Fin count)
    (indexEq : invocationOffset.val * laneCount + laneOffset.val = index.val)
    (found : value index = none) :
    (Rule.mk region (.optionalConstant
      (OptionalConstantTable.ofSemantic value) laneCount)).form?
        logicalWidth oneColumn
        (region.invocationStart + invocationOffset.val)
        (region.laneStart + laneOffset.val) =
      some (some .empty) := by
  unfold Rule.form?
  rw [Region.offsets?_of_offsets region invocationOffset laneOffset]
  simp only
  rw [Term.optionalConstant_form?_ofSemantic_of_none oneColumn value
    laneCount invocationOffset.val laneOffset.val index indexEq found]
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

private def applyRules? {logicalWidth : Nat} (oneColumn invocation lane : Nat) :
    List Rule → SparseForm logicalWidth → Option (SparseForm logicalWidth)
  | [], accumulated => some accumulated
  | rule :: rest, accumulated => do
      let selected ← rule.form? logicalWidth oneColumn invocation lane
      selected.elim
        (applyRules? oneColumn invocation lane rest accumulated)
        (fun value => applyRules? oneColumn invocation lane rest
          (SparseForm.add accumulated value))

/-- Decode one lane. An empty rule set produces the zero form. -/
def Program.form? (program : Program) (logicalWidth oneColumn invocation lane : Nat) :
    Option (SparseForm logicalWidth) :=
  applyRules? oneColumn invocation lane program.rules .empty

/-- A one-rule program returns the exact selected form. -/
theorem Program.singleton_form?_of_selected {logicalWidth : Nat}
    (rule : Rule) (oneColumn invocation lane : Nat)
    (value : SparseForm logicalWidth)
    (selected : rule.form? logicalWidth oneColumn invocation lane =
      some (some value)) :
    (Program.mk [rule]).form? logicalWidth oneColumn invocation lane =
      some value := by
  unfold Program.form? applyRules?
  rw [selected]
  rfl

/-- A two-rule program adds its selected forms in exact rule order. -/
theorem Program.two_form?_of_results {logicalWidth : Nat}
    (rule0 rule1 : Rule) (oneColumn invocation lane : Nat)
    (value0 value1 : Option (SparseForm logicalWidth))
    (result0 : rule0.form? logicalWidth oneColumn invocation lane = some value0)
    (result1 : rule1.form? logicalWidth oneColumn invocation lane = some value1) :
    (Program.mk [rule0, rule1]).form? logicalWidth oneColumn invocation lane =
      some (
        let accumulated0 :=
          value0.elim SparseForm.empty (SparseForm.add SparseForm.empty)
        value1.elim accumulated0 (SparseForm.add accumulated0)) := by
  cases value0 <;> cases value1 <;>
    simp [Program.form?, applyRules?, result0, result1]

/-- A three-rule program adds its selected forms in exact rule order. -/
theorem Program.three_form?_of_results {logicalWidth : Nat}
    (rule0 rule1 rule2 : Rule) (oneColumn invocation lane : Nat)
    (value0 value1 value2 : Option (SparseForm logicalWidth))
    (result0 : rule0.form? logicalWidth oneColumn invocation lane = some value0)
    (result1 : rule1.form? logicalWidth oneColumn invocation lane = some value1)
    (result2 : rule2.form? logicalWidth oneColumn invocation lane = some value2) :
    (Program.mk [rule0, rule1, rule2]).form? logicalWidth oneColumn
        invocation lane =
      some (
        let accumulated0 :=
          value0.elim SparseForm.empty (SparseForm.add SparseForm.empty)
        let accumulated1 :=
          value1.elim accumulated0 (SparseForm.add accumulated0)
        value2.elim accumulated1 (SparseForm.add accumulated1)) := by
  cases value0 <;> cases value1 <;> cases value2 <;>
    simp [Program.form?, applyRules?, result0, result1, result2]

/-- A four-rule program adds its selected forms in exact rule order. -/
theorem Program.four_form?_of_results {logicalWidth : Nat}
    (rule0 rule1 rule2 rule3 : Rule)
    (oneColumn invocation lane : Nat)
    (value0 value1 value2 value3 : Option (SparseForm logicalWidth))
    (result0 : rule0.form? logicalWidth oneColumn invocation lane = some value0)
    (result1 : rule1.form? logicalWidth oneColumn invocation lane = some value1)
    (result2 : rule2.form? logicalWidth oneColumn invocation lane = some value2)
    (result3 : rule3.form? logicalWidth oneColumn invocation lane = some value3) :
    (Program.mk [rule0, rule1, rule2, rule3]).form? logicalWidth oneColumn
        invocation lane =
      some (
        let accumulated0 :=
          value0.elim SparseForm.empty (SparseForm.add SparseForm.empty)
        let accumulated1 :=
          value1.elim accumulated0 (SparseForm.add accumulated0)
        let accumulated2 :=
          value2.elim accumulated1 (SparseForm.add accumulated1)
        value3.elim accumulated2 (SparseForm.add accumulated2)) := by
  cases value0 <;> cases value1 <;> cases value2 <;> cases value3 <;>
    simp [Program.form?, applyRules?, result0, result1, result2, result3]

private theorem applyRules?_of_allOutside {logicalWidth : Nat}
    (oneColumn invocation lane : Nat) (rules : List Rule)
    (accumulated : SparseForm logicalWidth)
    (outside : ∀ rule ∈ rules,
      rule.region.offsets? invocation lane = none) :
    applyRules? oneColumn invocation lane rules accumulated =
      some accumulated := by
  induction rules generalizing accumulated with
  | nil => rfl
  | cons rule rest inductionHypothesis =>
      have headOutside := outside rule (by simp)
      have tailOutside : ∀ candidate ∈ rest,
          candidate.region.offsets? invocation lane = none := by
        intro candidate member
        exact outside candidate (by simp [member])
      unfold applyRules?
      rw [Rule.form?_eq_some_none rule logicalWidth oneColumn invocation lane
        headOutside]
      exact inductionHypothesis accumulated tailOutside

/-- If no rule applies, lane decoding returns the exact zero form. -/
theorem Program.form?_eq_some_empty_of_allOutside
    (program : Program) (logicalWidth oneColumn invocation lane : Nat)
    (outside : ∀ rule ∈ program.rules,
      rule.region.offsets? invocation lane = none) :
    program.form? logicalWidth oneColumn invocation lane = some .empty := by
  exact applyRules?_of_allOutside oneColumn invocation lane program.rules
    .empty outside

/-- Decode all eight input lanes exactly once. -/
private def fixedState8 {Alpha : Type}
    (lane0 lane1 lane2 lane3 lane4 lane5 lane6 lane7 : Alpha) : Fin 8 → Alpha :=
  fun lane =>
    [lane0, lane1, lane2, lane3, lane4, lane5, lane6, lane7].get
      ⟨lane.val, by simpa using lane.isLt⟩

def Program.state? (program : Program) (logicalWidth oneColumn invocation : Nat) :
    Option (PoseidonSboxPlan.State logicalWidth) := do
  let lane0 ← program.form? logicalWidth oneColumn invocation 0
  let lane1 ← program.form? logicalWidth oneColumn invocation 1
  let lane2 ← program.form? logicalWidth oneColumn invocation 2
  let lane3 ← program.form? logicalWidth oneColumn invocation 3
  let lane4 ← program.form? logicalWidth oneColumn invocation 4
  let lane5 ← program.form? logicalWidth oneColumn invocation 5
  let lane6 ← program.form? logicalWidth oneColumn invocation 6
  let lane7 ← program.form? logicalWidth oneColumn invocation 7
  pure (fixedState8 lane0 lane1 lane2 lane3 lane4 lane5 lane6 lane7)

/-- Exact lane decoding gives the exact semantic input state. -/
theorem Program.state?_eq_some
    (program : Program) (logicalWidth oneColumn invocation : Nat)
    (state : PoseidonSboxPlan.State logicalWidth)
    (lane0 : program.form? logicalWidth oneColumn invocation 0 = some (state 0))
    (lane1 : program.form? logicalWidth oneColumn invocation 1 = some (state 1))
    (lane2 : program.form? logicalWidth oneColumn invocation 2 = some (state 2))
    (lane3 : program.form? logicalWidth oneColumn invocation 3 = some (state 3))
    (lane4 : program.form? logicalWidth oneColumn invocation 4 = some (state 4))
    (lane5 : program.form? logicalWidth oneColumn invocation 5 = some (state 5))
    (lane6 : program.form? logicalWidth oneColumn invocation 6 = some (state 6))
    (lane7 : program.form? logicalWidth oneColumn invocation 7 = some (state 7)) :
    program.state? logicalWidth oneColumn invocation = some state := by
  unfold Program.state?
  rw [lane0, lane1, lane2, lane3, lane4, lane5, lane6, lane7]
  apply congrArg some
  funext lane
  fin_cases lane <;> rfl

end NightstreamFPrime.Export.MatrixProgram.PoseidonInput
