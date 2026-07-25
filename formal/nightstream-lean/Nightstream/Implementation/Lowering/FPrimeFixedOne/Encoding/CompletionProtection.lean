import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PrimitiveRefinement

/-!
Contract: compositional protection of an earlier primitive result while an
SSA arm extends its typed context.

Owns:
- the exact inclusion of an earlier result context in a later context;
- disjointness of that later context from the earlier temporary witnesses;
- preservation of both facts across one distinct primitive result extension.

Does not own: concrete Step/Terminal occurrence order, semantic witnesses,
branch controls, assignment construction, or row satisfaction.

Emits constraints: no.
-/

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding

open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary

universe u

namespace PrimitivePlan

/-- A later typed context contains the complete result of an earlier
primitive and remains protected from that primitive's temporary witnesses. -/
structure ProtectedExtension
    {parameters : Parameters}
    {profile : Profile parameters}
    {input output : Schema (typeSystem parameters)}
    {primitive :
      Primitive (SelectedSignature parameters) input output}
    {path : OwnerPath}
    {inputColumns : Columns input}
    {one active : ColumnId}
    (plan :
      PrimitivePlan parameters profile primitive path
        inputColumns one active)
    {laterSchema : Schema (typeSystem parameters)}
    (laterColumns : Columns laterSchema) : Prop where
  resultIncluded :
    ∀ id, id ∈ plan.resultColumns.toSchemaBundles.ids ->
      id ∈ laterColumns.toSchemaBundles.ids
  temporariesDisjoint :
    IdsDisjoint plan.occurrence.temporaryIds
      laterColumns.toSchemaBundles.ids

/-- A primitive's own complete result context is the base protected
extension. -/
theorem protectsResult
    {parameters : Parameters}
    {profile : Profile parameters}
    {input output : Schema (typeSystem parameters)}
    {primitive :
      Primitive (SelectedSignature parameters) input output}
    {path : OwnerPath}
    {inputColumns : Columns input}
    {one active : ColumnId}
    (plan :
      PrimitivePlan parameters profile primitive path
        inputColumns one active) :
    plan.ProtectedExtension plan.resultColumns where
  resultIncluded := by
    intro id member
    exact member
  temporariesDisjoint :=
    plan.occurrenceTemporariesDisjointResultColumns

/-- One distinct SSA primitive extension preserves protection of every
earlier result and temporary set. -/
theorem ProtectedExtension.extend
    {parameters : Parameters}
    {profile : Profile parameters}
    {firstInput firstOutput nextInput nextOutput :
      Schema (typeSystem parameters)}
    {firstPrimitive :
      Primitive (SelectedSignature parameters)
        firstInput firstOutput}
    {nextPrimitive :
      Primitive (SelectedSignature parameters)
        nextInput nextOutput}
    {firstPath nextPath : OwnerPath}
    {firstInputColumns : Columns firstInput}
    {nextInputColumns : Columns nextInput}
    {one firstActive nextActive : ColumnId}
    {first :
      PrimitivePlan parameters profile firstPrimitive firstPath
        firstInputColumns one firstActive}
    (protection : first.ProtectedExtension nextInputColumns)
    (next :
      PrimitivePlan parameters profile nextPrimitive nextPath
        nextInputColumns one nextActive)
    (different : nextPath ≠ firstPath) :
    first.ProtectedExtension next.resultColumns where
  resultIncluded := by
    intro id member
    cases next with
    | invoke invokePlan =>
        change
          id ∈
            (Columns.toSchemaBundles
              (HVec.append
                (instructionColumns nextPath
                  ((SelectedSignature parameters).callOutputs _))
                nextInputColumns)).ids
        rw [Columns.append_ids]
        exact List.mem_append_right _
          (protection.resultIncluded id member)
    | literal literalPlan =>
        change
          id ∈
            (Columns.toSchemaBundles
              (HVec.append
                (instructionColumns nextPath [_])
                nextInputColumns)).ids
        rw [Columns.append_ids]
        exact List.mem_append_right _
          (protection.resultIncluded id member)
    | assertTrue assertPlan =>
        exact protection.resultIncluded id member
  temporariesDisjoint :=
    first.occurrenceTemporariesDisjointOtherResult
      next different protection.temporariesDisjoint

/-- Prepending a separately protected typed prefix preserves an earlier
result extension.  This is the branch-join analogue of `extend`. -/
theorem ProtectedExtension.prepend
    {parameters : Parameters}
    {profile : Profile parameters}
    {input output laterSchema prefixSchema :
      Schema (typeSystem parameters)}
    {primitive :
      Primitive (SelectedSignature parameters) input output}
    {path : OwnerPath}
    {inputColumns : Columns input}
    {one active : ColumnId}
    {plan :
      PrimitivePlan parameters profile primitive path
        inputColumns one active}
    {laterColumns : Columns laterSchema}
    (protection : plan.ProtectedExtension laterColumns)
    (prefixColumns : Columns prefixSchema)
    (prefixDisjoint :
      IdsDisjoint plan.occurrence.temporaryIds
        prefixColumns.toSchemaBundles.ids) :
    plan.ProtectedExtension (prefixColumns.append laterColumns) where
  resultIncluded := by
    intro id member
    rw [Columns.append_ids]
    exact List.mem_append_right _
      (protection.resultIncluded id member)
  temporariesDisjoint := by
    intro id temporaryMember combinedMember
    rw [Columns.append_ids] at combinedMember
    rcases List.mem_append.mp combinedMember with
      prefixMember | laterMember
    · exact prefixDisjoint id temporaryMember prefixMember
    · exact protection.temporariesDisjoint
        id temporaryMember laterMember

/-- A protected later input supplies the complete ordered pair certificate
needed by arm separation. -/
theorem ProtectedExtension.pairwiseSeparated
    {parameters : Parameters}
    {profile : Profile parameters}
    {firstInput firstOutput secondInput secondOutput :
      Schema (typeSystem parameters)}
    {firstPrimitive :
      Primitive (SelectedSignature parameters)
        firstInput firstOutput}
    {secondPrimitive :
      Primitive (SelectedSignature parameters)
        secondInput secondOutput}
    {firstPath secondPath : OwnerPath}
    {firstInputColumns : Columns firstInput}
    {secondInputColumns : Columns secondInput}
    {one active : ColumnId}
    {first :
      PrimitivePlan parameters profile firstPrimitive firstPath
        firstInputColumns one active}
    (protection : first.ProtectedExtension secondInputColumns)
    (second :
      PrimitivePlan parameters profile secondPrimitive secondPath
        secondInputColumns one active)
    (oneExcludes :
      one.owner ≠ .typed (.instruction firstPath))
    (activeExcludes :
      active.owner ≠ .typed (.instruction firstPath))
    (different : secondPath ≠ firstPath) :
    IdsDisjoint first.occurrence.temporaryIds
        second.occurrence.visibleIds ∧
      IdsDisjoint second.occurrence.temporaryIds
        first.occurrence.visibleIds ∧
      IdsDisjoint first.occurrence.temporaryIds
        second.occurrence.temporaryIds :=
  first.occurrencePairwiseSeparatedOfInput second
    oneExcludes activeExcludes different
    protection.temporariesDisjoint protection.resultIncluded

end PrimitivePlan

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
