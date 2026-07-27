import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.Poseidon23HashOccurrenceComplete

/-!
Contract: exact bundle views shared by the two fixed-23 binding-hash calls.

Owns: fixed-arity operand access, the nine-bundle hash footprint view, and
the occurrence constructor over those explicit bundles.

Does not own: call semantics, application serialization, a deployment
selection, Rust, generated rows, or collision resistance.
-/

set_option autoImplicit false
set_option maxRecDepth 32768

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding

open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls

namespace Poseidon23HashCallCommon

universe u

def firstOperand
    {types : TypeSystem.{u}}
    {context : Schema types}
    {firstKind secondKind thirdKind fourthKind : types.Kind}
    {first : Ref types context firstKind}
    {second : Ref types context secondKind}
    {third : Ref types context thirdKind}
    {fourth : Ref types context fourthKind}
    (bundles :
      RefBundles
        (Refs.cons first
          (Refs.cons second (Refs.cons third (Refs.cons fourth .nil))))) :
    ColumnBundle first.port.layout :=
  match bundles with
  | .cons firstBundle
      (.cons _ (.cons _ (.cons _ .nil))) => firstBundle

def secondOperand
    {types : TypeSystem.{u}}
    {context : Schema types}
    {firstKind secondKind thirdKind fourthKind : types.Kind}
    {first : Ref types context firstKind}
    {second : Ref types context secondKind}
    {third : Ref types context thirdKind}
    {fourth : Ref types context fourthKind}
    (bundles :
      RefBundles
        (Refs.cons first
          (Refs.cons second (Refs.cons third (Refs.cons fourth .nil))))) :
    ColumnBundle second.port.layout :=
  match bundles with
  | .cons _ (.cons secondBundle (.cons _ (.cons _ .nil))) =>
      secondBundle

def thirdOperand
    {types : TypeSystem.{u}}
    {context : Schema types}
    {firstKind secondKind thirdKind fourthKind : types.Kind}
    {first : Ref types context firstKind}
    {second : Ref types context secondKind}
    {third : Ref types context thirdKind}
    {fourth : Ref types context fourthKind}
    (bundles :
      RefBundles
        (Refs.cons first
          (Refs.cons second (Refs.cons third (Refs.cons fourth .nil))))) :
    ColumnBundle third.port.layout :=
  match bundles with
  | .cons _ (.cons _ (.cons thirdBundle (.cons _ .nil))) =>
      thirdBundle

def fourthOperand
    {types : TypeSystem.{u}}
    {context : Schema types}
    {firstKind secondKind thirdKind fourthKind : types.Kind}
    {first : Ref types context firstKind}
    {second : Ref types context secondKind}
    {third : Ref types context thirdKind}
    {fourth : Ref types context fourthKind}
    (bundles :
      RefBundles
        (Refs.cons first
          (Refs.cons second (Refs.cons third (Refs.cons fourth .nil))))) :
    ColumnBundle fourth.port.layout :=
  match bundles with
  | .cons _ (.cons _ (.cons _ (.cons fourthBundle .nil))) =>
      fourthBundle

@[simp] theorem operand_ids
    {types : TypeSystem.{u}}
    {context : Schema types}
    {firstKind secondKind thirdKind fourthKind : types.Kind}
    {first : Ref types context firstKind}
    {second : Ref types context secondKind}
    {third : Ref types context thirdKind}
    {fourth : Ref types context fourthKind}
    (bundles :
      RefBundles
        (Refs.cons first
          (Refs.cons second (Refs.cons third (Refs.cons fourth .nil))))) :
    (firstOperand bundles).ids ++
        ((secondOperand bundles).ids ++
          ((thirdOperand bundles).ids ++
            (fourthOperand bundles).ids)) =
      bundles.ids := by
  cases bundles with
  | cons firstBundle tail =>
      cases tail with
      | cons secondBundle tail =>
          cases tail with
          | cons thirdBundle tail =>
              cases tail with
              | cons fourthBundle tail =>
                  cases tail
                  change
                    firstBundle.ids ++
                        (secondBundle.ids ++
                          (thirdBundle.ids ++ fourthBundle.ids)) =
                      (RefBundles.cons firstBundle
                        (RefBundles.cons secondBundle
                          (RefBundles.cons thirdBundle
                            (RefBundles.cons fourthBundle
                              RefBundles.nil)))).ids
                  simp [RefBundles.ids, RefBundles.columns,
                    RefBundles.portColumns, ColumnBundle.ids,
                    List.append_assoc]

theorem fourOperand_decodes_iff
    {types : TypeSystem.{u}}
    (family : Family types)
    (assignment : ColumnId -> Field)
    {context : Schema types}
    {firstKind secondKind thirdKind fourthKind : types.Kind}
    {first : Ref types context firstKind}
    {second : Ref types context secondKind}
    {third : Ref types context thirdKind}
    {fourth : Ref types context fourthKind}
    (bundles :
      RefBundles
        (Refs.cons first
          (Refs.cons second (Refs.cons third (Refs.cons fourth .nil)))))
    (firstValue : types.Value firstKind)
    (secondValue : types.Value secondKind)
    (thirdValue : types.Value thirdKind)
    (fourthValue : types.Value fourthKind) :
    bundles.Decodes family assignment
        (.cons firstValue
          (.cons secondValue
            (.cons thirdValue (.cons fourthValue .nil)))) ↔
      (firstOperand bundles).Decodes
          family firstKind assignment firstValue ∧
        (secondOperand bundles).Decodes
          family secondKind assignment secondValue ∧
        (thirdOperand bundles).Decodes
          family thirdKind assignment thirdValue ∧
        (fourthOperand bundles).Decodes
          family fourthKind assignment fourthValue := by
  cases bundles with
  | cons firstBundle tail =>
      cases tail with
      | cons secondBundle tail =>
          cases tail with
          | cons thirdBundle tail =>
              cases tail with
              | cons fourthBundle tail =>
                  cases tail
                  simp [RefBundles.Decodes, firstOperand, secondOperand,
                    thirdOperand, fourthOperand]

theorem fourOperand_encodes_iff
    {types : TypeSystem.{u}}
    (family : Family types)
    (assignment : ColumnId -> Field)
    {context : Schema types}
    {firstKind secondKind thirdKind fourthKind : types.Kind}
    {first : Ref types context firstKind}
    {second : Ref types context secondKind}
    {third : Ref types context thirdKind}
    {fourth : Ref types context fourthKind}
    (bundles :
      RefBundles
        (Refs.cons first
          (Refs.cons second (Refs.cons third (Refs.cons fourth .nil)))))
    (firstValue : types.Value firstKind)
    (secondValue : types.Value secondKind)
    (thirdValue : types.Value thirdKind)
    (fourthValue : types.Value fourthKind) :
    bundles.Encodes family assignment
        (.cons firstValue
          (.cons secondValue
            (.cons thirdValue (.cons fourthValue .nil)))) ↔
      (firstOperand bundles).Encodes
          family firstKind assignment firstValue ∧
        (secondOperand bundles).Encodes
          family secondKind assignment secondValue ∧
        (thirdOperand bundles).Encodes
          family thirdKind assignment thirdValue ∧
        (fourthOperand bundles).Encodes
          family fourthKind assignment fourthValue := by
  cases bundles with
  | cons firstBundle tail =>
      cases tail with
      | cons secondBundle tail =>
          cases tail with
          | cons thirdBundle tail =>
              cases tail with
              | cons fourthBundle tail =>
                  cases tail
                  simp [RefBundles.Encodes, firstOperand, secondOperand,
                    thirdOperand, fourthOperand]

abbrev temporaryLayouts (alignmentWidth : Nat) : List Layout :=
  [ auxiliaryLayout 1
  , auxiliaryLayout 23
  , auxiliaryLayout alignmentWidth
  , auxiliaryLayout alignmentWidth
  , auxiliaryLayout alignmentWidth.pred
  , auxiliaryLayout 1
  , auxiliaryLayout 1
  , auxiliaryLayout 4
  , auxiliaryLayout 2464
  ]

structure TemporaryParts (alignmentWidth : Nat) where
  normalized : ColumnBundle (auxiliaryLayout 1)
  preimage : ColumnBundle (auxiliaryLayout 23)
  inverses : ColumnBundle (auxiliaryLayout alignmentWidth)
  equals : ColumnBundle (auxiliaryLayout alignmentWidth)
  products : ColumnBundle (auxiliaryLayout alignmentWidth.pred)
  equalityOutput : ColumnBundle (auxiliaryLayout 1)
  selected : ColumnBundle (auxiliaryLayout 1)
  coreOutput : ColumnBundle (auxiliaryLayout 4)
  coreTemporaries : ColumnBundle (auxiliaryLayout 2464)

def splitTemporaries
    {alignmentWidth : Nat}
    (bundles : LayoutBundles (temporaryLayouts alignmentWidth)) :
    TemporaryParts alignmentWidth :=
  match bundles with
  | .cons normalized
      (.cons preimage
        (.cons inverses
          (.cons equals
            (.cons products
              (.cons equalityOutput
                (.cons selected
                  (.cons coreOutput
                    (.cons coreTemporaries .nil)))))))) =>
      { normalized := normalized
        preimage := preimage
        inverses := inverses
        equals := equals
        products := products
        equalityOutput := equalityOutput
        selected := selected
        coreOutput := coreOutput
        coreTemporaries := coreTemporaries }

def TemporaryParts.ids
    {alignmentWidth : Nat}
    (parts : TemporaryParts alignmentWidth) : List ColumnId :=
  parts.normalized.ids ++
    (parts.preimage.ids ++
      (parts.inverses.ids ++
        (parts.equals.ids ++
          (parts.products.ids ++
            (parts.equalityOutput.ids ++
              (parts.selected.ids ++
                (parts.coreOutput.ids ++
                  parts.coreTemporaries.ids)))))))

def TemporaryParts.columns
    {alignmentWidth : Nat}
    (parts : TemporaryParts alignmentWidth) : List OwnedColumn :=
  parts.normalized.columns ++
    (parts.preimage.columns ++
      (parts.inverses.columns ++
        (parts.equals.columns ++
          (parts.products.columns ++
            (parts.equalityOutput.columns ++
              (parts.selected.columns ++
                (parts.coreOutput.columns ++
                  parts.coreTemporaries.columns)))))))

theorem splitTemporaries_ids
    {alignmentWidth : Nat}
    (bundles : LayoutBundles (temporaryLayouts alignmentWidth)) :
    (splitTemporaries bundles).ids = bundles.ids := by
  cases bundles with
  | cons normalized tail =>
      cases tail with
      | cons preimage tail =>
          cases tail with
          | cons inverses tail =>
              cases tail with
              | cons equals tail =>
                  cases tail with
                  | cons products tail =>
                      cases tail with
                      | cons equalityOutput tail =>
                          cases tail with
                          | cons selected tail =>
                              cases tail with
                              | cons coreOutput tail =>
                                  cases tail with
                                  | cons coreTemporaries tail =>
                                      cases tail
                                      change
                                        normalized.ids ++
                                            (preimage.ids ++
                                              (inverses.ids ++
                                                (equals.ids ++
                                                  (products.ids ++
                                                    (equalityOutput.ids ++
                                                      (selected.ids ++
                                                        (coreOutput.ids ++
                                                          coreTemporaries.ids))))))) =
                                          (LayoutBundles.cons normalized
                                            (LayoutBundles.cons preimage
                                              (LayoutBundles.cons inverses
                                                (LayoutBundles.cons equals
                                                  (LayoutBundles.cons products
                                                    (LayoutBundles.cons equalityOutput
                                                      (LayoutBundles.cons selected
                                                        (LayoutBundles.cons coreOutput
                                                          (LayoutBundles.cons
                                                            coreTemporaries
                                                            LayoutBundles.nil))))))))).ids
                                      simp [LayoutBundles.ids,
                                        LayoutBundles.columns,
                                        LayoutBundles.bundleColumns,
                                        ColumnBundle.ids,
                                        List.map_append,
                                        List.append_assoc]

theorem splitTemporaries_columns
    {alignmentWidth : Nat}
    (bundles : LayoutBundles (temporaryLayouts alignmentWidth)) :
    (splitTemporaries bundles).columns = bundles.columns := by
  cases bundles with
  | cons normalized tail =>
      cases tail with
      | cons preimage tail =>
          cases tail with
          | cons inverses tail =>
              cases tail with
              | cons equals tail =>
                  cases tail with
                  | cons products tail =>
                      cases tail with
                      | cons equalityOutput tail =>
                          cases tail with
                          | cons selected tail =>
                              cases tail with
                              | cons coreOutput tail =>
                                  cases tail with
                                  | cons coreTemporaries tail =>
                                      cases tail
                                      change
                                        normalized.columns ++
                                            (preimage.columns ++
                                              (inverses.columns ++
                                                (equals.columns ++
                                                  (products.columns ++
                                                    (equalityOutput.columns ++
                                                      (selected.columns ++
                                                        (coreOutput.columns ++
                                                          coreTemporaries.columns))))))) =
                                          (LayoutBundles.cons normalized
                                            (LayoutBundles.cons preimage
                                              (LayoutBundles.cons inverses
                                                (LayoutBundles.cons equals
                                                  (LayoutBundles.cons products
                                                    (LayoutBundles.cons equalityOutput
                                                      (LayoutBundles.cons selected
                                                        (LayoutBundles.cons coreOutput
                                                          (LayoutBundles.cons
                                                            coreTemporaries
                                                            LayoutBundles.nil))))))))).columns
                                      simp [LayoutBundles.columns,
                                        LayoutBundles.bundleColumns,
                                        List.append_assoc]

def occurrence
    {sourceWidth alignmentWidth : Nat}
    (owner : PhysicalOwner)
    (one active : ColumnId)
    (next : Bool)
    (iteration : OwnedColumn)
    (sourceTail : List OwnedColumn)
    (sourceTailLength : sourceTail.length + 1 = sourceWidth)
    (output : List OwnedColumn)
    (outputLength : output.length = 5)
    (parts : TemporaryParts alignmentWidth)
    (plan : Poseidon23Hash.CoordinatePlan sourceWidth alignmentWidth) :
    Poseidon23HashOccurrence.Frame sourceWidth alignmentWidth where
  owner := owner
  one := one
  active := active
  next := next
  iteration := iteration
  sourceTail := sourceTail
  sourceTailLength := sourceTailLength
  output := output
  outputLength := outputLength
  normalized := parts.normalized
  preimage := parts.preimage
  inverses := parts.inverses
  equals := parts.equals
  products := parts.products
  equalityOutput := parts.equalityOutput
  selected := parts.selected
  coreOutput := parts.coreOutput
  coreTemporaries := parts.coreTemporaries
  plan := plan

@[simp] theorem occurrence_temporaryIds
    {sourceWidth alignmentWidth : Nat}
    (owner : PhysicalOwner)
    (one active : ColumnId)
    (next : Bool)
    (iteration : OwnedColumn)
    (sourceTail : List OwnedColumn)
    (sourceTailLength : sourceTail.length + 1 = sourceWidth)
    (output : List OwnedColumn)
    (outputLength : output.length = 5)
    (parts : TemporaryParts alignmentWidth)
    (plan : Poseidon23Hash.CoordinatePlan sourceWidth alignmentWidth) :
    (occurrence owner one active next iteration sourceTail sourceTailLength
      output outputLength parts plan).temporaryIds = parts.ids := by
  simp [occurrence, Poseidon23HashOccurrence.Frame.temporaryIds,
    Poseidon23HashOccurrence.Frame.prefixTemporaryIds,
    TemporaryParts.ids, List.append_assoc]

/-- The call receipt's global freshness facts imply the smaller allocation
facts required by the embedded sponge core. -/
theorem coreAllocationFacts
    {sourceWidth alignmentWidth : Nat}
    (owner : PhysicalOwner)
    (one active : ColumnId)
    (next : Bool)
    (iteration : OwnedColumn)
    (sourceTail : List OwnedColumn)
    (sourceTailLength : sourceTail.length + 1 = sourceWidth)
    (output : List OwnedColumn)
    (outputLength : output.length = 5)
    (parts : TemporaryParts alignmentWidth)
    (plan : Poseidon23Hash.CoordinatePlan sourceWidth alignmentWidth)
    (temporaryNodup : parts.ids.Nodup)
    (temporariesDisjointVisible :
      IdsDisjoint parts.ids
        (occurrence owner one active next iteration sourceTail
          sourceTailLength output outputLength parts plan).visibleIds)
    (temporariesOwned :
      ∀ column, column ∈ parts.columns -> column.id.owner = owner) :
    Poseidon23HashOccurrence.CoreAllocationFacts
      (occurrence owner one active next iteration sourceTail sourceTailLength
        output outputLength parts plan) := by
  let selected :=
    occurrence owner one active next iteration sourceTail sourceTailLength
      output outputLength parts plan
  have afterNormalized :=
    (List.nodup_append.mp temporaryNodup).2.1
  have afterPreimage :=
    (List.nodup_append.mp afterNormalized).2.1
  have afterInverses :=
    (List.nodup_append.mp afterPreimage).2.1
  have afterEquals :=
    (List.nodup_append.mp afterInverses).2.1
  have afterProducts :=
    (List.nodup_append.mp afterEquals).2.1
  have afterEqualityOutput :=
    (List.nodup_append.mp afterProducts).2.1
  have afterSelected :=
    (List.nodup_append.mp afterEqualityOutput).2.1
  have coreSplit := List.nodup_append.mp afterSelected
  refine
    { allocationsNodup := ?_
      temporariesDisjointVisible := ?_
      outputsDisjointPreexisting := ?_
      allocationsOwned := ?_ }
  · exact afterSelected
  · intro id coreTemporary preexisting
    change id ∈ parts.coreTemporaries.ids at coreTemporary
    have coreTemporaryAll : id ∈ parts.ids := by
      simp [TemporaryParts.ids, coreTemporary]
    have coreTemporarySelectedTail :
        id ∈ parts.coreOutput.ids ++ parts.coreTemporaries.ids :=
      List.mem_append_right parts.coreOutput.ids coreTemporary
    have cases :
        id = one ∨ id = selected.selectedColumn.id ∨
          id ∈ selected.preimage.ids ∨ id ∈ selected.coreOutput.ids := by
      simpa [CanonicalPoseidon2Sponge23Recipe.Frame.visibleIds,
        Poseidon23HashOccurrence.core, selected, occurrence] using preexisting
    rcases cases with oneCase | selectedCase | preimageCase | outputCase
    · subst id
      exact temporariesDisjointVisible one coreTemporaryAll
        (by simp [Poseidon23HashOccurrence.Frame.visibleIds, selected,
          occurrence])
    · have selectedMember : id ∈ parts.selected.ids := by
        rw [selectedCase]
        unfold Poseidon23HashOccurrence.Frame.selectedColumn
        exact bundleColumn_id_mem parts.selected
          ⟨0, by simp [auxiliaryLayout, ownedLayout]⟩
      exact
        ((List.nodup_append.mp afterEqualityOutput).2.2
          id selectedMember id coreTemporarySelectedTail rfl)
    · have preimageMember : id ∈ parts.preimage.ids := by
        simpa [selected, occurrence] using preimageCase
      have suffixMember :
          id ∈ parts.inverses.ids ++
            (parts.equals.ids ++
              (parts.products.ids ++
                (parts.equalityOutput.ids ++
                  (parts.selected.ids ++
                    (parts.coreOutput.ids ++
                      parts.coreTemporaries.ids))))) := by
        simp [coreTemporary]
      exact
        ((List.nodup_append.mp afterNormalized).2.2
          id preimageMember id suffixMember rfl)
    · have outputMember : id ∈ parts.coreOutput.ids := by
        simpa [selected, occurrence] using outputCase
      exact coreSplit.2.2 id outputMember id coreTemporary rfl
  · intro id coreOutputMember preexisting
    change id ∈ parts.coreOutput.ids at coreOutputMember
    have coreOutputAll : id ∈ parts.ids := by
      simp [TemporaryParts.ids, coreOutputMember]
    have cases :
        id = one ∨ id = selected.selectedColumn.id ∨
          id ∈ selected.preimage.ids := by
      simpa [selected, occurrence] using preexisting
    rcases cases with oneCase | selectedCase | preimageCase
    · subst id
      exact temporariesDisjointVisible one coreOutputAll
        (by simp [Poseidon23HashOccurrence.Frame.visibleIds, selected,
          occurrence])
    · have selectedMember : id ∈ parts.selected.ids := by
        rw [selectedCase]
        unfold Poseidon23HashOccurrence.Frame.selectedColumn
        exact bundleColumn_id_mem parts.selected
          ⟨0, by simp [auxiliaryLayout, ownedLayout]⟩
      have coreTail :
          id ∈ parts.coreOutput.ids ++ parts.coreTemporaries.ids :=
        List.mem_append_left _ coreOutputMember
      exact
        ((List.nodup_append.mp afterEqualityOutput).2.2
          id selectedMember id coreTail rfl)
    · have preimageMember : id ∈ parts.preimage.ids := by
        simpa [selected, occurrence] using preimageCase
      have suffixMember :
          id ∈ parts.inverses.ids ++
            (parts.equals.ids ++
              (parts.products.ids ++
                (parts.equalityOutput.ids ++
                  (parts.selected.ids ++
                    (parts.coreOutput.ids ++
                      parts.coreTemporaries.ids))))) := by
        simp [coreOutputMember]
      exact
        ((List.nodup_append.mp afterNormalized).2.2
          id preimageMember id suffixMember rfl)
  · intro column member
    apply temporariesOwned column
    change
      column ∈ parts.coreOutput.columns ++
        parts.coreTemporaries.columns at member
    rcases List.mem_append.mp member with outputMember | temporaryMember
    · simp [TemporaryParts.columns, outputMember]
    · simp [TemporaryParts.columns, temporaryMember]

end Poseidon23HashCallCommon

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
