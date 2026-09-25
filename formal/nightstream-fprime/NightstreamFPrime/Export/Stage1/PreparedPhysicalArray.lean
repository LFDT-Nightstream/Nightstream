import Init.Data.Array.Lemmas

/-! Append mapped events or assertions directly
to their existing array. Mapped lists occur only in preservation proofs. -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PreparedPhysicalArray

variable {Alpha Beta : Type}

/-- Retain the existing prefix and append each mapped input in list order. -/
@[inline] def appendMap (initial : Array Beta) (items : List Alpha)
    (transform : Alpha → Beta) : Array Beta :=
  items.foldl (fun accumulated item => accumulated.push (transform item)) initial

/-- Reuse the standard fold/push law; no second traversal occurs at runtime. -/
theorem appendMap_toList (initial : Array Beta) (items : List Alpha)
    (transform : Alpha → Beta) :
    (appendMap initial items transform).toList = initial.toList ++ items.map transform := by
  exact Array.foldl_toList_eq_map (l := items) (acc := initial) (G := transform)

/-- Every result item comes from the original prefix or a mapped source item. -/
theorem mem_appendMap (initial : Array Beta) (items : List Alpha)
    (transform : Alpha → Beta) (value : Beta) :
    value ∈ appendMap initial items transform ↔
      value ∈ initial ∨ ∃ item ∈ items, transform item = value := by
  rw [← Array.mem_toList_iff, appendMap_toList, List.mem_append,
    Array.mem_toList_iff, List.mem_map]

end NightstreamFPrime.Export.Stage1.PreparedPhysicalArray
