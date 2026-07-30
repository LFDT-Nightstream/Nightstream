import Nightstream.Implementation.Lowering.Goldilocks.BundleBridge
import Nightstream.Implementation.Lowering.Goldilocks.CodecRecovery

/-!
Contract: recover typed schema values from exact-width physical bundles when
every codec used by that schema is total at its declared width.

Owns: the schema induction and its column-plan specialization.

Does not own: proof-specific canonicality rows, application semantics,
protocol acceptance, Rust, or generated artifacts.

Emits constraints: none.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.Goldilocks

open Nightstream.Implementation.Lowering.Typed

universe u

/-- Every semantic codec used by one schema represents every coordinate list
of its declared width. -/
def SchemaExactWidthRecoverable
    {types : TypeSystem.{u}}
    (family : Family types)
    (schema : Schema types) : Prop :=
  ∀ port, port ∈ schema →
    (family.codecFor port.kind).ExactWidthRecoverable

namespace SchemaExactWidthRecoverable

theorem tail
    {types : TypeSystem.{u}}
    {family : Family types}
    {head : Port types}
    {tail : Schema types}
    (recoverable :
      SchemaExactWidthRecoverable family (head :: tail)) :
    SchemaExactWidthRecoverable family tail := by
  intro port member
  exact recoverable port (List.mem_cons_of_mem head member)

end SchemaExactWidthRecoverable

namespace SchemaBundles

/-- Exact widths plus codec totality construct one decoded value vector in
the same schema order. -/
theorem decode_exists
    {types : TypeSystem.{u}}
    {family : Family types}
    {schema : Schema types}
    (bundles : SchemaBundles schema)
    (assignment : ColumnId → Field)
    (widths : SchemaWidthAgrees family schema)
    (recoverable : SchemaExactWidthRecoverable family schema) :
    ∃ values : Schema.Values types schema,
      bundles.Decodes family assignment values := by
  induction bundles with
  | nil =>
      exact ⟨.nil, True.intro⟩
  | @cons port tail head rest inductionHypothesis =>
      have headLength :
          (head.values assignment).length =
            (family.codecFor port.kind).width := by
        rw [head.values_length]
        exact
          (widths port (List.mem_cons_self)).symm
      rcases
          Codec.decode_exists_of_exactWidthRecoverable
            (recoverable port List.mem_cons_self)
            (head.values assignment) headLength with
        ⟨headValue, headDecoded⟩
      have tailWidths :
          SchemaWidthAgrees family tail := by
        intro candidate member
        exact widths candidate (List.mem_cons_of_mem port member)
      rcases
          inductionHypothesis tailWidths recoverable.tail with
        ⟨tailValues, tailDecoded⟩
      exact
        ⟨.cons headValue tailValues, headDecoded, tailDecoded⟩

end SchemaBundles

namespace Columns

/-- Compiler columns inherit schema recovery without changing their physical
coordinate order. -/
theorem decode_exists
    {types : TypeSystem.{u}}
    {family : Family types}
    {schema : Schema types}
    (columns : Columns schema)
    (assignment : ColumnId → Field)
    (widths : SchemaWidthAgrees family schema)
    (recoverable : SchemaExactWidthRecoverable family schema) :
    ∃ values : Schema.Values types schema,
      columns.toSchemaBundles.Decodes family assignment values :=
  columns.toSchemaBundles.decode_exists assignment widths recoverable

end Columns

end Nightstream.Implementation.Lowering.Goldilocks
