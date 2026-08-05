import Nightstream.Implementation.Lowering.Goldilocks.CallRecipe

/-!
Contract: project authoritative base- and quadratic-extension values from
decoded fixed-width call operands.

The selected NIFS verifier consumes many `K`-valued subfields of its running,
fresh, and proof operands, as well as base-field coordinates inside their
typed public carriers.  `CallRecipe` decodes those operands as whole semantic
values, while the canonical arithmetic gadgets consume physical field
columns.  This module provides the narrow bridge between those layers:
a serialization profile names one or two coordinates of a codec, and
successful decoding proves that the physical coordinates equal the
corresponding semantic value.

The projection laws are serialization facts.  They do not carry verifier
acceptance, source authority, a quotient equation, or a named-event branch.

Emits constraints: none.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsCodecProjection

open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks

universe u v

abbrev K := Nightstream.SuperNeo.Concrete.K

private theorem k_eq
    (left right : K)
    (c0Equal : left.c0 = right.c0)
    (c1Equal : left.c1 = right.c1) :
    left = right := by
  cases left
  cases right
  simp only at c0Equal c1Equal
  cases c0Equal
  cases c1Equal
  rfl

/-- One exact codec coordinate representing a semantic base-field component. -/
structure FView
    {α : Type u}
    (codec : Codec α)
    (value : α → Field) where
  index : Fin codec.width
  encodeValue :
    ∀ input,
      (codec.encode input).getD index.val 0 = value input

/-- Two exact codec coordinates representing one semantic `K` component. -/
structure KView
    {α : Type u}
    (codec : Codec α)
    (value : α → K) where
  c0Index : Fin codec.width
  c1Index : Fin codec.width
  encodeC0 :
    ∀ input,
      (codec.encode input).getD c0Index.val 0 = (value input).c0
  encodeC1 :
    ∀ input,
      (codec.encode input).getD c1Index.val 0 = (value input).c1

private theorem getD_append_left
    {α : Type u}
    (left right : List α)
    (index : Nat)
    (default : α)
    (indexLt : index < left.length) :
    (left ++ right).getD index default = left.getD index default := by
  rw [List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD,
    List.getElem?_eq_getElem (by
      rw [List.length_append]
      omega),
    List.getElem?_eq_getElem indexLt,
    List.getElem_append_left]

private theorem getD_append_right
    {α : Type u}
    (left right : List α)
    (index : Nat)
    (default : α) :
    (left ++ right).getD (left.length + index) default =
      right.getD index default := by
  rw [List.getD_eq_getElem?_getD,
    List.getElem?_append_right (by omega)]
  simp only [Nat.add_sub_cancel_left]
  rfl

private theorem finBlock_index_lt
    {count width : Nat}
    (element : Fin count)
    (coordinate : Fin width) :
    element.val * width + coordinate.val < count * width := by
  have blockLe :
      (element.val + 1) * width <= count * width :=
    Nat.mul_le_mul_right width
      (Nat.succ_le_iff.mpr element.isLt)
  have localLt :
      element.val * width + coordinate.val <
        (element.val + 1) * width := by
    rw [Nat.add_mul, Nat.one_mul]
    omega
  exact Nat.lt_of_lt_of_le localLt blockLe

namespace FView

/-- Change only the semantic name of a view by a pointwise equality. -/
def congrValue
    {α : Type u}
    {codec : Codec α}
    {left right : α → Field}
    (view : FView codec left)
    (equal : ∀ input, left input = right input) :
    FView codec right where
  index := view.index
  encodeValue := by
    intro input
    rw [← equal input]
    exact view.encodeValue input

/-- Transport one view through a total pullback codec. -/
noncomputable def throughPullback
    {α : Type u}
    {β : Type v}
    {target : Codec β}
    {value : β → Field}
    (toTarget : α → β)
    (toInjective : Function.Injective toTarget)
    (view : FView target value) :
    FView (Codec.pullback target toTarget toInjective)
      (fun input => value (toTarget input)) where
  index := view.index
  encodeValue := fun input => view.encodeValue (toTarget input)

/-- Transport one view through a domain-restricted pullback codec. -/
noncomputable def throughPullbackOn
    {α : Type u}
    {β : Type v}
    {target : Codec β}
    {sourceAdmissible : α → Prop}
    {value : β → Field}
    (toTarget : α → β)
    (targetAdmissible :
      ∀ input, sourceAdmissible input →
        target.Admissible (toTarget input))
    (toInjective :
      ∀ {left right},
        sourceAdmissible left →
        sourceAdmissible right →
        toTarget left = toTarget right →
        left = right)
    (view : FView target value) :
    FView
      (Codec.pullbackOn target sourceAdmissible toTarget
        targetAdmissible toInjective)
      (fun input => value (toTarget input)) where
  index := view.index
  encodeValue := fun input => view.encodeValue (toTarget input)

/-- Lift a view through the left side of a product codec. -/
noncomputable def productLeft
    {α : Type u}
    {β : Type v}
    {left : Codec α}
    {right : Codec β}
    {value : α → Field}
    (view : FView left value) :
    FView (Codec.product left right)
      (fun input => value input.1) where
  index := ⟨view.index.val, by
    change view.index.val < left.width + right.width
    omega⟩
  encodeValue := by
    intro input
    change
      (left.encode input.1 ++ right.encode input.2).getD
          view.index.val 0 =
        value input.1
    rw [getD_append_left _ _ _ _
      (by
        rw [left.encode_length]
        exact view.index.isLt)]
    exact view.encodeValue input.1

/-- Lift a view through the right side of a product codec. -/
noncomputable def productRight
    {α : Type u}
    {β : Type v}
    {left : Codec α}
    {right : Codec β}
    {value : β → Field}
    (view : FView right value) :
    FView (Codec.product left right)
      (fun input => value input.2) where
  index := ⟨left.width + view.index.val, by
    change left.width + view.index.val < left.width + right.width
    omega⟩
  encodeValue := by
    intro input
    change
      (left.encode input.1 ++ right.encode input.2).getD
          (left.width + view.index.val) 0 =
        value input.2
    have indexShape :
        left.width + view.index.val =
          (left.encode input.1).length + view.index.val := by
      rw [left.encode_length]
    rw [indexShape, getD_append_right]
    exact view.encodeValue input.2

/-- Lift one element view through an index-major finite-function codec. -/
noncomputable def finElement
    {α : Type u}
    {codec : Codec α}
    {value : α → Field}
    (count : Nat)
    (element : Fin count)
    (view : FView codec value) :
    FView (Codec.finFunction count codec)
      (fun values => value (values element)) where
  index :=
    ⟨element.val * codec.width + view.index.val, by
      change
        element.val * codec.width + view.index.val <
          count * codec.width
      exact finBlock_index_lt element view.index⟩
  encodeValue := by
    intro values
    change
      (Codec.encodeFin codec count values).getD
          (element.val * codec.width + view.index.val) 0 =
        value (values element)
    rw [Codec.encodeFin_getD]
    exact view.encodeValue (values element)

/-- Lift one list element through an exact-length list codec. -/
noncomputable def fixedListElement
    {α : Type u}
    {codec : Codec α}
    {value : α → Field}
    (count : Nat)
    (default : α)
    (element : Fin count)
    (view : FView codec value) :
    FView (Codec.fixedList count default codec)
      (fun values => value (values.getD element.val default)) where
  index :=
    ⟨element.val * codec.width + view.index.val, by
      change
        element.val * codec.width + view.index.val <
          count * codec.width
      exact finBlock_index_lt element view.index⟩
  encodeValue := by
    intro values
    change
      (Codec.encodeFin codec count
        (fun index => values.getD index.val default)).getD
          (element.val * codec.width + view.index.val) 0 =
        value (values.getD element.val default)
    rw [Codec.encodeFin_getD]
    exact view.encodeValue _

/-- Lift one array element through an exact-size array codec. -/
noncomputable def fixedArrayElement
    {α : Type u}
    {codec : Codec α}
    {value : α → Field}
    (count : Nat)
    (default : α)
    (element : Fin count)
    (view : FView codec value) :
    FView (Codec.fixedArray count default codec)
      (fun values => value (values.getD element.val default)) where
  index :=
    ⟨element.val * codec.width + view.index.val, by
      change
        element.val * codec.width + view.index.val <
          count * codec.width
      exact finBlock_index_lt element view.index⟩
  encodeValue := by
    intro values
    change
      (Codec.encodeFin codec count
        (fun index => values.getD index.val default)).getD
          (element.val * codec.width + view.index.val) 0 =
        value (values.getD element.val default)
    rw [Codec.encodeFin_getD]
    exact view.encodeValue _

end FView

namespace KView

/-- Change only the semantic name of a quadratic-extension view. -/
def congrValue
    {α : Type u}
    {codec : Codec α}
    {left right : α → K}
    (view : KView codec left)
    (equal : ∀ input, left input = right input) :
    KView codec right where
  c0Index := view.c0Index
  c1Index := view.c1Index
  encodeC0 := by
    intro input
    rw [← equal input]
    exact view.encodeC0 input
  encodeC1 := by
    intro input
    rw [← equal input]
    exact view.encodeC1 input

/-- Transport one quadratic-extension view through a total pullback codec. -/
noncomputable def throughPullback
    {α : Type u}
    {β : Type v}
    {target : Codec β}
    {value : β → K}
    (toTarget : α → β)
    (toInjective : Function.Injective toTarget)
    (view : KView target value) :
    KView (Codec.pullback target toTarget toInjective)
      (fun input => value (toTarget input)) where
  c0Index := view.c0Index
  c1Index := view.c1Index
  encodeC0 := fun input => view.encodeC0 (toTarget input)
  encodeC1 := fun input => view.encodeC1 (toTarget input)

/-- Transport one quadratic-extension view through a restricted pullback. -/
noncomputable def throughPullbackOn
    {α : Type u}
    {β : Type v}
    {target : Codec β}
    {sourceAdmissible : α → Prop}
    {value : β → K}
    (toTarget : α → β)
    (targetAdmissible :
      ∀ input, sourceAdmissible input →
        target.Admissible (toTarget input))
    (toInjective :
      ∀ {left right},
        sourceAdmissible left →
        sourceAdmissible right →
        toTarget left = toTarget right →
        left = right)
    (view : KView target value) :
    KView
      (Codec.pullbackOn target sourceAdmissible toTarget
        targetAdmissible toInjective)
      (fun input => value (toTarget input)) where
  c0Index := view.c0Index
  c1Index := view.c1Index
  encodeC0 := fun input => view.encodeC0 (toTarget input)
  encodeC1 := fun input => view.encodeC1 (toTarget input)

/-- Lift a quadratic-extension view through the left product side. -/
noncomputable def productLeft
    {α : Type u}
    {β : Type v}
    {left : Codec α}
    {right : Codec β}
    {value : α → K}
    (view : KView left value) :
    KView (Codec.product left right)
      (fun input => value input.1) where
  c0Index := ⟨view.c0Index.val, by
    change view.c0Index.val < left.width + right.width
    omega⟩
  c1Index := ⟨view.c1Index.val, by
    change view.c1Index.val < left.width + right.width
    omega⟩
  encodeC0 := by
    intro input
    change
      (left.encode input.1 ++ right.encode input.2).getD
          view.c0Index.val 0 =
        (value input.1).c0
    rw [getD_append_left _ _ _ _
      (by
        rw [left.encode_length]
        exact view.c0Index.isLt)]
    exact view.encodeC0 input.1
  encodeC1 := by
    intro input
    change
      (left.encode input.1 ++ right.encode input.2).getD
          view.c1Index.val 0 =
        (value input.1).c1
    rw [getD_append_left _ _ _ _
      (by
        rw [left.encode_length]
        exact view.c1Index.isLt)]
    exact view.encodeC1 input.1

/-- Lift a quadratic-extension view through the right product side. -/
noncomputable def productRight
    {α : Type u}
    {β : Type v}
    {left : Codec α}
    {right : Codec β}
    {value : β → K}
    (view : KView right value) :
    KView (Codec.product left right)
      (fun input => value input.2) where
  c0Index := ⟨left.width + view.c0Index.val, by
    change left.width + view.c0Index.val < left.width + right.width
    omega⟩
  c1Index := ⟨left.width + view.c1Index.val, by
    change left.width + view.c1Index.val < left.width + right.width
    omega⟩
  encodeC0 := by
    intro input
    change
      (left.encode input.1 ++ right.encode input.2).getD
          (left.width + view.c0Index.val) 0 =
        (value input.2).c0
    have indexShape :
        left.width + view.c0Index.val =
          (left.encode input.1).length + view.c0Index.val := by
      rw [left.encode_length]
    rw [indexShape, getD_append_right]
    exact view.encodeC0 input.2
  encodeC1 := by
    intro input
    change
      (left.encode input.1 ++ right.encode input.2).getD
          (left.width + view.c1Index.val) 0 =
        (value input.2).c1
    have indexShape :
        left.width + view.c1Index.val =
          (left.encode input.1).length + view.c1Index.val := by
      rw [left.encode_length]
    rw [indexShape, getD_append_right]
    exact view.encodeC1 input.2

/-- Lift one quadratic-extension element through a finite-function codec. -/
noncomputable def finElement
    {α : Type u}
    {codec : Codec α}
    {value : α → K}
    (count : Nat)
    (element : Fin count)
    (view : KView codec value) :
    KView (Codec.finFunction count codec)
      (fun values => value (values element)) where
  c0Index :=
    ⟨element.val * codec.width + view.c0Index.val, by
      change
        element.val * codec.width + view.c0Index.val <
          count * codec.width
      exact finBlock_index_lt element view.c0Index⟩
  c1Index :=
    ⟨element.val * codec.width + view.c1Index.val, by
      change
        element.val * codec.width + view.c1Index.val <
          count * codec.width
      exact finBlock_index_lt element view.c1Index⟩
  encodeC0 := by
    intro values
    change
      (Codec.encodeFin codec count values).getD
          (element.val * codec.width + view.c0Index.val) 0 =
        (value (values element)).c0
    rw [Codec.encodeFin_getD]
    exact view.encodeC0 (values element)
  encodeC1 := by
    intro values
    change
      (Codec.encodeFin codec count values).getD
          (element.val * codec.width + view.c1Index.val) 0 =
        (value (values element)).c1
    rw [Codec.encodeFin_getD]
    exact view.encodeC1 (values element)

/-- Lift one quadratic-extension list element through an exact list codec. -/
noncomputable def fixedListElement
    {α : Type u}
    {codec : Codec α}
    {value : α → K}
    (count : Nat)
    (default : α)
    (element : Fin count)
    (view : KView codec value) :
    KView (Codec.fixedList count default codec)
      (fun values => value (values.getD element.val default)) where
  c0Index :=
    ⟨element.val * codec.width + view.c0Index.val, by
      change
        element.val * codec.width + view.c0Index.val <
          count * codec.width
      exact finBlock_index_lt element view.c0Index⟩
  c1Index :=
    ⟨element.val * codec.width + view.c1Index.val, by
      change
        element.val * codec.width + view.c1Index.val <
          count * codec.width
      exact finBlock_index_lt element view.c1Index⟩
  encodeC0 := by
    intro values
    change
      (Codec.encodeFin codec count
        (fun index => values.getD index.val default)).getD
          (element.val * codec.width + view.c0Index.val) 0 =
        (value (values.getD element.val default)).c0
    rw [Codec.encodeFin_getD]
    exact view.encodeC0 _
  encodeC1 := by
    intro values
    change
      (Codec.encodeFin codec count
        (fun index => values.getD index.val default)).getD
          (element.val * codec.width + view.c1Index.val) 0 =
        (value (values.getD element.val default)).c1
    rw [Codec.encodeFin_getD]
    exact view.encodeC1 _

/-- Lift one quadratic-extension array element through an exact array
codec. -/
noncomputable def fixedArrayElement
    {α : Type u}
    {codec : Codec α}
    {value : α → K}
    (count : Nat)
    (default : α)
    (element : Fin count)
    (view : KView codec value) :
    KView (Codec.fixedArray count default codec)
      (fun values => value (values.getD element.val default)) where
  c0Index :=
    ⟨element.val * codec.width + view.c0Index.val, by
      change
        element.val * codec.width + view.c0Index.val <
          count * codec.width
      exact finBlock_index_lt element view.c0Index⟩
  c1Index :=
    ⟨element.val * codec.width + view.c1Index.val, by
      change
        element.val * codec.width + view.c1Index.val <
          count * codec.width
      exact finBlock_index_lt element view.c1Index⟩
  encodeC0 := by
    intro values
    change
      (Codec.encodeFin codec count
        (fun index => values.getD index.val default)).getD
          (element.val * codec.width + view.c0Index.val) 0 =
        (value (values.getD element.val default)).c0
    rw [Codec.encodeFin_getD]
    exact view.encodeC0 _
  encodeC1 := by
    intro values
    change
      (Codec.encodeFin codec count
        (fun index => values.getD index.val default)).getD
          (element.val * codec.width + view.c1Index.val) 0 =
        (value (values.getD element.val default)).c1
    rw [Codec.encodeFin_getD]
    exact view.encodeC1 _

end KView

/-- Stable physical identities of one projected quadratic-extension value. -/
structure KColumnIds where
  c0 : ColumnId
  c1 : ColumnId
deriving DecidableEq, Repr

private theorem kColumnIds_eq
    (left right : KColumnIds)
    (c0Equal : left.c0 = right.c0)
    (c1Equal : left.c1 = right.c1) :
    left = right := by
  cases left
  cases right
  simp only at c0Equal c1Equal
  cases c0Equal
  cases c1Equal
  rfl

/-- Stable physical identity of one projected base-field value. -/
structure FColumnId where
  column : ColumnId
deriving DecidableEq, Repr

/-- Read one in-range coordinate from a physical bundle.  The width equation
comes from the selected call frame's `WidthsAgree` certificate. -/
def coordinateId
    {α : Type u}
    {layout : Layout}
    (codec : Codec α)
    (bundle : ColumnBundle layout)
    (widthsAgree : codec.width = layout.owners.length)
    (index : Fin codec.width) : ColumnId :=
  (bundle.columns.get
    ⟨index.val, by
      rw [bundle.length_eq, ← widthsAgree]
      exact index.isLt⟩).id

/-- Every projected coordinate is an existing coordinate of the decoded
bundle; projection never allocates a copy. -/
theorem coordinateId_mem
    {α : Type u}
    {layout : Layout}
    (codec : Codec α)
    (bundle : ColumnBundle layout)
    (widthsAgree : codec.width = layout.owners.length)
    (index : Fin codec.width) :
    coordinateId codec bundle widthsAgree index ∈ bundle.ids := by
  unfold coordinateId ColumnBundle.ids
  apply List.mem_map.mpr
  refine ⟨bundle.columns.get
      ⟨index.val, by
        rw [bundle.length_eq, ← widthsAgree]
        exact index.isLt⟩, ?_, rfl⟩
  exact List.get_mem _ _

/-- A projected coordinate is fixed by the ordered physical bundle and its
numeric codec index. Codec domain types and width proofs do not affect the
physical identity. -/
theorem coordinateId_eq_of_ids
    {α : Type u}
    {β : Type v}
    {leftCodec : Codec α}
    {rightCodec : Codec β}
    {leftLayout rightLayout : Layout}
    (leftBundle : ColumnBundle leftLayout)
    (rightBundle : ColumnBundle rightLayout)
    (leftWidthsAgree :
      leftCodec.width = leftLayout.owners.length)
    (rightWidthsAgree :
      rightCodec.width = rightLayout.owners.length)
    (leftIndex : Fin leftCodec.width)
    (rightIndex : Fin rightCodec.width)
    (idsEqual : leftBundle.ids = rightBundle.ids)
    (indexEqual : leftIndex.val = rightIndex.val) :
    coordinateId leftCodec leftBundle leftWidthsAgree leftIndex =
      coordinateId rightCodec rightBundle rightWidthsAgree rightIndex := by
  let fallback :=
    coordinateId leftCodec leftBundle leftWidthsAgree leftIndex
  have leftBound : leftIndex.val < leftBundle.ids.length := by
    simp only [ColumnBundle.ids, List.length_map, leftBundle.length_eq]
    rw [← leftWidthsAgree]
    exact leftIndex.isLt
  have rightBound : rightIndex.val < rightBundle.ids.length := by
    simp only [ColumnBundle.ids, List.length_map, rightBundle.length_eq]
    rw [← rightWidthsAgree]
    exact rightIndex.isLt
  calc
    coordinateId leftCodec leftBundle leftWidthsAgree leftIndex =
        leftBundle.ids.getD leftIndex.val fallback := by
      rw [List.getD_eq_getElem?_getD,
        List.getElem?_eq_getElem leftBound]
      simp only [Option.getD_some]
      unfold ColumnBundle.ids coordinateId
      rw [List.getElem_map]
      congr
    _ = rightBundle.ids.getD leftIndex.val fallback := by
      rw [idsEqual]
    _ = rightBundle.ids.getD rightIndex.val fallback := by
      rw [indexEqual]
    _ = coordinateId rightCodec rightBundle rightWidthsAgree rightIndex := by
      rw [List.getD_eq_getElem?_getD,
        List.getElem?_eq_getElem rightBound]
      simp only [Option.getD_some]
      unfold ColumnBundle.ids coordinateId
      rw [List.getElem_map]
      congr

def FView.column
    {α : Type u}
    {layout : Layout}
    {codec : Codec α}
    {value : α → Field}
    (view : FView codec value)
    (bundle : ColumnBundle layout)
    (widthsAgree : codec.width = layout.owners.length) : FColumnId where
  column := coordinateId codec bundle widthsAgree view.index

theorem FView.column_mem
    {α : Type u}
    {layout : Layout}
    {codec : Codec α}
    {value : α → Field}
    (view : FView codec value)
    (bundle : ColumnBundle layout)
    (widthsAgree : codec.width = layout.owners.length) :
    (view.column bundle widthsAgree).column ∈ bundle.ids :=
  coordinateId_mem codec bundle widthsAgree view.index

/-- Two base-field views select the same physical column when their bundles
have the same ordered identities and their numeric codec indices agree. -/
theorem FView.column_eq_of_ids
    {α : Type u}
    {β : Type v}
    {leftLayout rightLayout : Layout}
    {leftCodec : Codec α}
    {rightCodec : Codec β}
    {leftValue : α → Field}
    {rightValue : β → Field}
    (leftView : FView leftCodec leftValue)
    (rightView : FView rightCodec rightValue)
    (leftBundle : ColumnBundle leftLayout)
    (rightBundle : ColumnBundle rightLayout)
    (leftWidthsAgree :
      leftCodec.width = leftLayout.owners.length)
    (rightWidthsAgree :
      rightCodec.width = rightLayout.owners.length)
    (idsEqual : leftBundle.ids = rightBundle.ids)
    (indexEqual : leftView.index.val = rightView.index.val) :
    (leftView.column leftBundle leftWidthsAgree).column =
      (rightView.column rightBundle rightWidthsAgree).column := by
  exact coordinateId_eq_of_ids
    leftBundle rightBundle leftWidthsAgree rightWidthsAgree
    leftView.index rightView.index idsEqual indexEqual

def KView.columns
    {α : Type u}
    {layout : Layout}
    {codec : Codec α}
    {value : α → K}
    (view : KView codec value)
    (bundle : ColumnBundle layout)
    (widthsAgree : codec.width = layout.owners.length) : KColumnIds where
  c0 := coordinateId codec bundle widthsAgree view.c0Index
  c1 := coordinateId codec bundle widthsAgree view.c1Index

/-- Two quadratic-extension views select the same physical pair when their
bundles have the same ordered identities and both codec indices agree. -/
theorem KView.columns_eq_of_ids
    {α : Type u}
    {β : Type v}
    {leftLayout rightLayout : Layout}
    {leftCodec : Codec α}
    {rightCodec : Codec β}
    {leftValue : α → K}
    {rightValue : β → K}
    (leftView : KView leftCodec leftValue)
    (rightView : KView rightCodec rightValue)
    (leftBundle : ColumnBundle leftLayout)
    (rightBundle : ColumnBundle rightLayout)
    (leftWidthsAgree :
      leftCodec.width = leftLayout.owners.length)
    (rightWidthsAgree :
      rightCodec.width = rightLayout.owners.length)
    (idsEqual : leftBundle.ids = rightBundle.ids)
    (c0IndexEqual : leftView.c0Index.val = rightView.c0Index.val)
    (c1IndexEqual : leftView.c1Index.val = rightView.c1Index.val) :
    leftView.columns leftBundle leftWidthsAgree =
      rightView.columns rightBundle rightWidthsAgree := by
  let leftColumns := leftView.columns leftBundle leftWidthsAgree
  let rightColumns := rightView.columns rightBundle rightWidthsAgree
  change leftColumns = rightColumns
  have c0Equal : leftColumns.c0 = rightColumns.c0 :=
    coordinateId_eq_of_ids
      leftBundle rightBundle leftWidthsAgree rightWidthsAgree
      leftView.c0Index rightView.c0Index idsEqual c0IndexEqual
  have c1Equal : leftColumns.c1 = rightColumns.c1 :=
    coordinateId_eq_of_ids
      leftBundle rightBundle leftWidthsAgree rightWidthsAgree
      leftView.c1Index rightView.c1Index idsEqual c1IndexEqual
  exact kColumnIds_eq leftColumns rightColumns c0Equal c1Equal

theorem KView.c0_mem
    {α : Type u}
    {layout : Layout}
    {codec : Codec α}
    {value : α → K}
    (view : KView codec value)
    (bundle : ColumnBundle layout)
    (widthsAgree : codec.width = layout.owners.length) :
    (view.columns bundle widthsAgree).c0 ∈ bundle.ids :=
  coordinateId_mem codec bundle widthsAgree view.c0Index

theorem KView.c1_mem
    {α : Type u}
    {layout : Layout}
    {codec : Codec α}
    {value : α → K}
    (view : KView codec value)
    (bundle : ColumnBundle layout)
    (widthsAgree : codec.width = layout.owners.length) :
    (view.columns bundle widthsAgree).c1 ∈ bundle.ids :=
  coordinateId_mem codec bundle widthsAgree view.c1Index

def KColumnIds.value
    (columns : KColumnIds)
    (assignment : ColumnId → Field) : K where
  c0 := assignment columns.c0
  c1 := assignment columns.c1

def FColumnId.value
    (column : FColumnId)
    (assignment : ColumnId → Field) : Field :=
  assignment column.column

private theorem value_getD_eq_coordinate
    {α : Type u}
    {layout : Layout}
    (codec : Codec α)
    (bundle : ColumnBundle layout)
    (widthsAgree : codec.width = layout.owners.length)
    (assignment : ColumnId → Field)
    (index : Fin codec.width) :
    (bundle.values assignment).getD index.val 0 =
      assignment (coordinateId codec bundle widthsAgree index) := by
  have columnBound : index.val < bundle.columns.length := by
    rw [bundle.length_eq, ← widthsAgree]
    exact index.isLt
  rw [ColumnBundle.values, List.getD_eq_getElem?_getD,
    List.getElem?_map, List.getElem?_eq_getElem columnBound]
  rfl

/-- Reading a named view from the bundle's coordinate list is definitionally
the same physical assignment read used by row gadgets.  Unlike the decoding
lemmas below, this statement assumes no semantic output value. -/
theorem FView.bundle_getD_eq_value
    {α : Type u}
    {layout : Layout}
    {codec : Codec α}
    {value : α → Field}
    (view : FView codec value)
    (bundle : ColumnBundle layout)
    (widthsAgree : codec.width = layout.owners.length)
    (assignment : ColumnId → Field) :
    (bundle.values assignment).getD view.index.val 0 =
      (view.column bundle widthsAgree).value assignment :=
  value_getD_eq_coordinate codec bundle widthsAgree assignment view.index

/-- Successful decoding binds a projected physical pair to its exact semantic
`K` component.  There is no independently supplied physical value. -/
theorem KView.value_eq_of_decodes
    {α : Type u}
    {layout : Layout}
    {codec : Codec α}
    {value : α → K}
    (view : KView codec value)
    (bundle : ColumnBundle layout)
    (widthsAgree : codec.width = layout.owners.length)
    (assignment : ColumnId → Field)
    (input : α)
    (decoded : codec.decode (bundle.values assignment) = some input) :
    (view.columns bundle widthsAgree).value assignment = value input := by
  have coordinates :
      bundle.values assignment = codec.encode input :=
    (codec.encode_decode (bundle.values assignment) input decoded).2.symm
  have c0Equal := congrArg
    (fun values => values.getD view.c0Index.val 0) coordinates
  have c1Equal := congrArg
    (fun values => values.getD view.c1Index.val 0) coordinates
  dsimp only at c0Equal c1Equal
  rw [value_getD_eq_coordinate codec bundle widthsAgree assignment,
    view.encodeC0] at c0Equal
  rw [value_getD_eq_coordinate codec bundle widthsAgree assignment,
    view.encodeC1] at c1Equal
  exact k_eq _ _ c0Equal c1Equal

/-- Successful decoding binds a projected physical base-field coordinate to
its exact semantic component.  There is no independently supplied value. -/
theorem FView.value_eq_of_decodes
    {α : Type u}
    {layout : Layout}
    {codec : Codec α}
    {value : α → Field}
    (view : FView codec value)
    (bundle : ColumnBundle layout)
    (widthsAgree : codec.width = layout.owners.length)
    (assignment : ColumnId → Field)
    (input : α)
    (decoded : codec.decode (bundle.values assignment) = some input) :
    (view.column bundle widthsAgree).value assignment = value input := by
  have coordinates :
      bundle.values assignment = codec.encode input :=
    (codec.encode_decode (bundle.values assignment) input decoded).2.symm
  have selected := congrArg
    (fun values => values.getD view.index.val 0) coordinates
  dsimp only at selected
  rw [value_getD_eq_coordinate codec bundle widthsAgree assignment,
    view.encodeValue] at selected
  exact selected

/-- `ColumnBundle.Decodes` specialization used by call-frame proofs. -/
theorem KView.value_eq_of_bundle_decodes
    {types : TypeSystem.{u}}
    (family : Family types)
    (kind : types.Kind)
    {layout : Layout}
    {value : types.Value kind → K}
    (view : KView (family.codecFor kind) value)
    (bundle : ColumnBundle layout)
    (widthsAgree :
      (family.codecFor kind).width = layout.owners.length)
    (assignment : ColumnId → Field)
    (input : types.Value kind)
    (decoded : bundle.Decodes family kind assignment input) :
    (view.columns bundle widthsAgree).value assignment = value input := by
  exact view.value_eq_of_decodes bundle widthsAgree assignment input decoded

/-- Base-field `ColumnBundle.Decodes` specialization used by call-frame
proofs. -/
theorem FView.value_eq_of_bundle_decodes
    {types : TypeSystem.{u}}
    (family : Family types)
    (kind : types.Kind)
    {layout : Layout}
    {value : types.Value kind → Field}
    (view : FView (family.codecFor kind) value)
    (bundle : ColumnBundle layout)
    (widthsAgree :
      (family.codecFor kind).width = layout.owners.length)
    (assignment : ColumnId → Field)
    (input : types.Value kind)
    (decoded : bundle.Decodes family kind assignment input) :
    (view.column bundle widthsAgree).value assignment = value input := by
  exact view.value_eq_of_decodes bundle widthsAgree assignment input decoded

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsCodecProjection
