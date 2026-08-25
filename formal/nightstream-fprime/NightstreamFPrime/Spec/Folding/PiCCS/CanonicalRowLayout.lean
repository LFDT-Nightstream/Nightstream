import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.NumericBooleanDomain
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.UnifiedSources

/-! Provenance: adapted from
`formal/nightstream-lean/Nightstream/SuperNeo/Folding/PiCCS/PaperJoint/PrefixLayout.lean`
at commit `fb7a8a99aefbb8ebb5474681ecf80f1b95a1b7a2`; the namespace and ownership
text now identify the SuperNeo v1.1 canonical row injection. The proof body is
unchanged. -/

/-!
Canonical prefix injection into a Boolean cube.

Protocol: shared row layout for the joint paper `Pi_CCS` model.
Phase: logical-index to Boolean-domain normalization.
Constraint family: none; this file emits no rows.

Owns: the little-endian prefix layout in which logical index `i` owns Boolean
vertex `NumericBooleanDomain.vertex variables i`; exact live-index
round-trips; and exact recognition of the padding suffix.

Does not own: selected production dimensions, matrix rows, assignment values,
CCS acceptance, SumCheck, Rust, R1CS, or constraint counts.

Emits constraints: no.

Authority boundary: the layout is computed from `variables`, `entries`, and a
proof that the cube covers the entries. A caller cannot select a second bit
order or a different placement map.

| Code owner | Protocol object | Mathematical obligation | Proven result |
|---|---|---|---|
| `layout` | logical-prefix row placement | inject live indices into the covered Boolean cube | exact little-endian map |
| `index_toVertex` | live row index | recover the source index from its vertex | exact round trip |
| `toColumn?_eq_some_iff` | live/padding classifier | decode exactly the logical prefix | exact iff |
| `toColumn?_eq_none_iff` | zero-row padding suffix | reject exactly indices outside the prefix | exact iff |
-/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Folding.PiCCS.CanonicalRowLayout

open PaperJoint
open PaperJoint.UnifiedSources

/-- The canonical zero-based little-endian prefix layout. -/
def layout
    (variables entries : Nat)
    (covered : entries <= 2 ^ variables) :
    ColumnLayout variables entries where
  columns_le := covered
  toVertex := fun entry =>
    NumericBooleanDomain.vertex variables
      ⟨entry.val, Nat.lt_of_lt_of_le entry.isLt covered⟩
  toColumn? := fun vertex =>
    if live : NumericBooleanDomain.index vertex < entries then
      some ⟨NumericBooleanDomain.index vertex, live⟩
    else
      none
  toColumn_toVertex := by
    intro entry
    simp [NumericBooleanDomain.index_vertex, entry.isLt]
  toVertex_toColumn := by
    intro vertex entry decoded
    change
      (if live : NumericBooleanDomain.index vertex < entries then
          some ⟨NumericBooleanDomain.index vertex, live⟩
        else none) = some entry at decoded
    split at decoded
    next live =>
      have indexEqual : NumericBooleanDomain.index vertex = entry.val :=
        congrArg Fin.val (Option.some.inj decoded)
      calc
        NumericBooleanDomain.vertex variables
            ⟨entry.val, Nat.lt_of_lt_of_le entry.isLt covered⟩ =
            NumericBooleanDomain.vertex variables
              ⟨NumericBooleanDomain.index vertex,
                NumericBooleanDomain.index_lt_twoPow vertex⟩ := by
              congr 1
              exact Fin.eq_of_val_eq indexEqual.symm
        _ = vertex := NumericBooleanDomain.vertex_index vertex
    next notLive =>
      contradiction

/-- The Boolean vertex assigned to one live entry has the same numeric index. -/
@[simp] theorem index_toVertex
    (variables entries : Nat)
    (covered : entries <= 2 ^ variables)
    (entry : Fin entries) :
    NumericBooleanDomain.index ((layout variables entries covered).toVertex entry) =
      entry.val := by
  exact NumericBooleanDomain.index_vertex variables
    ⟨entry.val, Nat.lt_of_lt_of_le entry.isLt covered⟩

/-- A vertex decodes to a live entry exactly when its numeric index is inside
the logical prefix. -/
theorem toColumn?_eq_some_iff
    (variables entries : Nat)
    (covered : entries <= 2 ^ variables)
    (vertex : BooleanVertex variables)
    (entry : Fin entries) :
    (layout variables entries covered).toColumn? vertex = some entry <->
      NumericBooleanDomain.index vertex = entry.val := by
  constructor
  · intro decoded
    simp only [layout] at decoded
    split at decoded
    next live =>
      exact congrArg Fin.val (Option.some.inj decoded)
    next notLive =>
      contradiction
  · intro indexEqual
    have live : NumericBooleanDomain.index vertex < entries := by
      rw [indexEqual]
      exact entry.isLt
    simp [layout, indexEqual]

/-- A vertex is padding exactly when its numeric index is outside the logical
prefix. -/
theorem toColumn?_eq_none_iff
    (variables entries : Nat)
    (covered : entries <= 2 ^ variables)
    (vertex : BooleanVertex variables) :
    (layout variables entries covered).toColumn? vertex = none <->
      entries <= NumericBooleanDomain.index vertex := by
  simp [layout, Nat.not_lt]

end NightstreamFPrime.Spec.Folding.PiCCS.CanonicalRowLayout
