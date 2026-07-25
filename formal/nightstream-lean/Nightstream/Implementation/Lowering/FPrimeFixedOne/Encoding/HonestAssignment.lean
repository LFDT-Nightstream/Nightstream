import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls.Common
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PrimitivePlan

/-!
Contract: artifact-independent construction of canonical visible assignments.

Owns:
- the exact flattened codec string for typed schema values;
- the corresponding explicit admissibility predicate;
- writing one schema's canonical coordinates into its exact physical bundles;
- proofs that the write encodes the requested values and changes no unrelated
  physical coordinate.

Does not own: primitive temporary witnesses, whole-program completion,
production codecs, Rust layouts, numeric R1CS matrices, or generated artifacts.

Emits constraints: no.
-/

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding

open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks

universe u

namespace HonestAssignment

/-- Every semantic value in an exact schema lies in its selected codec's
explicit domain. -/
def Admissible
    {types : TypeSystem.{u}}
    (family : Family types) :
    {schema : Schema types} ->
    Schema.Values types schema -> Prop
  | [], .nil => True
  | port :: _, .cons value values =>
      (family.codecFor port.kind).Admissible value ∧
        Admissible family values

/-- Admissibility of an appended value context restricts to its exact left
prefix. -/
theorem Admissible.left_of_append
    {types : TypeSystem.{u}}
    (family : Family types)
    {left right : Schema types}
    (leftValues : Schema.Values types left)
    (rightValues : Schema.Values types right)
    (admissible :
      Admissible family (leftValues.append rightValues)) :
    Admissible family leftValues := by
  induction leftValues with
  | nil =>
      trivial
  | cons value values inductionHypothesis =>
      exact ⟨admissible.1, inductionHypothesis admissible.2⟩

/-- Admissibility of an appended value context restricts to its exact right
suffix. -/
theorem Admissible.right_of_append
    {types : TypeSystem.{u}}
    (family : Family types)
    {left right : Schema types}
    (leftValues : Schema.Values types left)
    (rightValues : Schema.Values types right)
    (admissible :
      Admissible family (leftValues.append rightValues)) :
    Admissible family rightValues := by
  induction leftValues with
  | nil =>
      exact admissible
  | cons value values inductionHypothesis =>
      exact inductionHypothesis admissible.2

/-- Canonical physical coordinate string in exact schema and port order. -/
def coordinates
    {types : TypeSystem.{u}}
    (family : Family types) :
    {schema : Schema types} ->
    Schema.Values types schema -> List Field
  | [], .nil => []
  | port :: _, .cons value values =>
      (family.codecFor port.kind).encode value ++
        coordinates family values

theorem coordinates_length
    {types : TypeSystem.{u}}
    (family : Family types)
    {schema : Schema types}
    (columns : Columns schema)
    (values : Schema.Values types schema)
    (widths : SchemaWidthAgrees family schema) :
    (coordinates family values).length =
      columns.toSchemaBundles.ids.length := by
  induction columns with
  | nil =>
      cases values
      rfl
  | @cons port tail head rest inductionHypothesis =>
      cases values with
      | cons value values =>
          have headWidth :
              (family.codecFor port.kind).width =
                port.layout.owners.length :=
            widths port List.mem_cons_self
          have tailWidths : SchemaWidthAgrees family tail := by
            intro item member
            exact widths item (List.mem_cons_of_mem port member)
          have headIdsLength :
              head.toColumnBundle.ids.length =
                port.layout.owners.length := by
            rw [ColumnBundle.ids, List.length_map,
              head.toColumnBundle.length_eq]
          simp only [coordinates, Columns.toSchemaBundles,
            SchemaBundles.ids_cons, List.length_append,
            (family.codecFor port.kind).encode_length value,
            headWidth, headIdsLength,
            inductionHypothesis values tailWidths]

/-- Write one exact schema's canonical codec string.  The previous assignment
is retained outside the schema's physical identities. -/
def encodeInto
    {types : TypeSystem.{u}}
    (family : Family types)
    {schema : Schema types}
    (columns : Columns schema)
    (values : Schema.Values types schema)
    (assignment : ColumnId -> Field) : ColumnId -> Field :=
  DirectCalls.writeColumns assignment columns.toSchemaBundles.ids
    (coordinates family values)

theorem encodeInto_changesOnly
    {types : TypeSystem.{u}}
    (family : Family types)
    {schema : Schema types}
    (columns : Columns schema)
    (values : Schema.Values types schema)
    (assignment : ColumnId -> Field) :
    ChangesOnly columns.toSchemaBundles.ids assignment
      (encodeInto family columns values assignment) := by
  exact DirectCalls.writeColumns_changesOnly assignment
    columns.toSchemaBundles.ids (coordinates family values)

theorem encodeInto_values
    {types : TypeSystem.{u}}
    (family : Family types)
    {schema : Schema types}
    (columns : Columns schema)
    (values : Schema.Values types schema)
    (assignment : ColumnId -> Field)
    (widths : SchemaWidthAgrees family schema)
    (nodup : columns.toSchemaBundles.ids.Nodup) :
    columns.toSchemaBundles.columns.map
        (fun column => encodeInto family columns values assignment column.id) =
      coordinates family values := by
  have recovered :=
    DirectCalls.writeColumns_map_eq assignment
      columns.toSchemaBundles.ids (coordinates family values)
      (by
        rw [coordinates_length family columns values widths])
      nodup
  simpa [encodeInto, SchemaBundles.ids] using recovered

private theorem encodes_of_flattened_coordinates
    {types : TypeSystem.{u}}
    (family : Family types)
    {schema : Schema types}
    (bundles : SchemaBundles schema)
    (values : Schema.Values types schema)
    (assignment : ColumnId -> Field)
    (widths : SchemaWidthAgrees family schema)
    (admissible : Admissible family values)
    (exact :
      bundles.columns.map (fun column => assignment column.id) =
        coordinates family values) :
    bundles.Encodes family assignment values := by
  induction bundles generalizing assignment with
  | nil =>
      cases values
      trivial
  | @cons port tail head rest inductionHypothesis =>
      cases values with
      | cons value values =>
          have headWidth :
              (family.codecFor port.kind).width =
                port.layout.owners.length :=
            widths port List.mem_cons_self
          have tailWidths : SchemaWidthAgrees family tail := by
            intro item member
            exact widths item (List.mem_cons_of_mem port member)
          have splitExact :
              head.values assignment ++
                  rest.columns.map (fun column => assignment column.id) =
                (family.codecFor port.kind).encode value ++
                  coordinates family values := by
            simpa [SchemaBundles.columns, SchemaBundles.portColumns,
              ColumnBundle.values, coordinates, List.map_append] using exact
          have headLength :
              (head.values assignment).length =
                ((family.codecFor port.kind).encode value).length := by
            rw [ColumnBundle.values_length,
              (family.codecFor port.kind).encode_length value, headWidth]
          have split := List.append_inj splitExact headLength
          exact ⟨
            ⟨admissible.1, split.1⟩,
            inductionHypothesis values assignment tailWidths
              admissible.2 split.2⟩

/-- Writing canonical coordinates honestly encodes every typed value in the
schema. -/
theorem encodeInto_encodes
    {types : TypeSystem.{u}}
    (family : Family types)
    {schema : Schema types}
    (columns : Columns schema)
    (values : Schema.Values types schema)
    (assignment : ColumnId -> Field)
    (widths : SchemaWidthAgrees family schema)
    (admissible : Admissible family values)
    (nodup : columns.toSchemaBundles.ids.Nodup) :
    Columns.Encodes family columns
      (encodeInto family columns values assignment) values := by
  apply encodes_of_flattened_coordinates
    family columns.toSchemaBundles values
      (encodeInto family columns values assignment)
      widths admissible
  exact encodeInto_values family columns values assignment widths nodup

/-- A canonical assignment for one schema exists without assuming a
preexisting physical witness. -/
theorem exists_encodes
    {types : TypeSystem.{u}}
    (family : Family types)
    {schema : Schema types}
    (columns : Columns schema)
    (values : Schema.Values types schema)
    (widths : SchemaWidthAgrees family schema)
    (admissible : Admissible family values)
    (nodup : columns.toSchemaBundles.ids.Nodup) :
    ∃ assignment : ColumnId -> Field,
      Columns.Encodes family columns assignment values := by
  let assignment :=
    encodeInto family columns values (fun _ => 0)
  exact ⟨assignment,
    encodeInto_encodes family columns values (fun _ => 0)
      widths admissible nodup⟩

end HonestAssignment

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
