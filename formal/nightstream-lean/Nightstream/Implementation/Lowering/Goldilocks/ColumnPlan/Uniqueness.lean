import Nightstream.Implementation.Lowering.Goldilocks.CallRecipe
import Nightstream.Implementation.Lowering.Goldilocks.ColumnPlan

/-!
Contract: intrinsic identity uniqueness and separation for canonical physical
column allocations.

Owns:
- exact owner and half-open bundle-index bounds for `allocateSchemaFrom`;
- schema-allocation ID uniqueness derived from structural coordinates;
- joint uniqueness of instruction outputs followed by call temporaries;
- ID disjointness from unequal selected owners;
- separation of prelude, typed input/instruction/branch, and activation owner
  classes.

Does not own: caller metadata, semantic codecs, rows, Rust numeric columns,
generated artifacts, or fixed concrete widths.

Emits constraints: no.
-/

namespace Nightstream.Implementation.Lowering.Goldilocks

open Nightstream.Implementation.Lowering.Typed

universe u

namespace ColumnPlan

/-- Ordered physical identities of an exact dependent schema allocation. -/
def schemaColumnIds
    {types : TypeSystem.{u}}
    {schema : Schema types}
    (columns : Columns schema) : List ColumnId :=
  (schemaOwnedColumns columns).map (fun column => column.id)

/-- The bundle allocated at one absolute schema position. -/
private def allocatedBundle
    {types : TypeSystem.{u}}
    (ownerAt : Nat -> PhysicalOwner)
    (bundleIndex : Nat)
    (port : Port types) : Bundle port where
  column coordinate :=
    { owner := ownerAt bundleIndex
      bundleIndex := bundleIndex
      coordinateIndex := coordinate.val }

private theorem allocateSchemaFrom_cons
    {types : TypeSystem.{u}}
    (ownerAt : Nat -> PhysicalOwner)
    (bundleIndex : Nat)
    (port : Port types)
    (tail : Schema types) :
    allocateSchemaFrom ownerAt bundleIndex (port :: tail) =
      HVec.cons (allocatedBundle ownerAt bundleIndex port)
        (allocateSchemaFrom ownerAt (bundleIndex + 1) tail) :=
  rfl

private theorem schemaColumnIds_allocateSchemaFrom_cons
    {types : TypeSystem.{u}}
    (ownerAt : Nat -> PhysicalOwner)
    (bundleIndex : Nat)
    (port : Port types)
    (tail : Schema types) :
    schemaColumnIds
        (allocateSchemaFrom ownerAt bundleIndex (port :: tail)) =
      (bundleOwnedColumns port
          (allocatedBundle ownerAt bundleIndex port)).map
          (fun column => column.id) ++
        schemaColumnIds
          (allocateSchemaFrom ownerAt (bundleIndex + 1) tail) := by
  simp only [allocateSchemaFrom_cons, schemaColumnIds,
    schemaOwnedColumns, List.map_append]

private theorem nodup_ofFn_of_injective
    {alpha : Type} :
    ∀ {n : Nat}
      (function : Fin n -> alpha),
      Function.Injective function ->
      (List.ofFn function).Nodup
  | 0, function, injective => by
      simp
  | _ + 1, function, injective => by
      rw [List.ofFn_succ, List.nodup_cons]
      constructor
      · intro member
        rcases List.mem_ofFn.mp member with ⟨index, equal⟩
        exact Fin.succ_ne_zero index (injective equal)
      · exact nodup_ofFn_of_injective
          (fun index => function index.succ)
          (fun first second equal =>
            Fin.succ_inj.mp (injective equal))

private theorem allocatedBundle_ids_nodup
    {types : TypeSystem.{u}}
    (ownerAt : Nat -> PhysicalOwner)
    (bundleIndex : Nat)
    (port : Port types) :
    ((bundleOwnedColumns port
        (allocatedBundle ownerAt bundleIndex port)).map
      fun column => column.id).Nodup := by
  rw [bundleOwnedColumns, List.map_ofFn]
  apply nodup_ofFn_of_injective
  intro first second equal
  apply Fin.ext
  exact congrArg (fun id : ColumnId => id.coordinateIndex) equal

private theorem allocatedBundle_mem
    {types : TypeSystem.{u}}
    (ownerAt : Nat -> PhysicalOwner)
    (bundleIndex : Nat)
    (port : Port types)
    (id : ColumnId)
    (member :
      id ∈ (bundleOwnedColumns port
        (allocatedBundle ownerAt bundleIndex port)).map
          (fun column => column.id)) :
    id.owner = ownerAt bundleIndex ∧
      id.bundleIndex = bundleIndex := by
  rcases List.mem_map.mp member with ⟨column, columnMember, rfl⟩
  rw [bundleOwnedColumns, List.mem_ofFn] at columnMember
  rcases columnMember with ⟨coordinate, rfl⟩
  exact ⟨rfl, rfl⟩

/-- Every allocated identity records the owner selected at its own absolute
bundle index, and that index lies in this allocation's half-open range. -/
theorem mem_allocateSchemaFrom
    {types : TypeSystem.{u}}
    (ownerAt : Nat -> PhysicalOwner)
    (firstBundle : Nat)
    (schema : Schema types)
    (id : ColumnId)
    (member :
      id ∈ schemaColumnIds
        (allocateSchemaFrom ownerAt firstBundle schema)) :
    id.owner = ownerAt id.bundleIndex ∧
      firstBundle ≤ id.bundleIndex ∧
      id.bundleIndex < firstBundle + schema.length := by
  induction schema generalizing firstBundle with
  | nil =>
      simp [schemaColumnIds, schemaOwnedColumns,
        allocateSchemaFrom] at member
  | cons port tail inductionHypothesis =>
      rw [schemaColumnIds_allocateSchemaFrom_cons] at member
      rcases List.mem_append.mp member with headMember | tailMember
      · have headExact :=
          allocatedBundle_mem ownerAt firstBundle port id headMember
        refine ⟨headExact.2 ▸ headExact.1, ?_, ?_⟩
        · omega
        · simp only [List.length_cons]
          omega
      · have tailExact :=
          inductionHypothesis (firstBundle := firstBundle + 1) tailMember
        refine ⟨tailExact.1, ?_, ?_⟩
        · omega
        · simp only [List.length_cons]
          omega

/-- A schema allocation never repeats a physical column identity. -/
theorem allocateSchemaFrom_ids_nodup
    {types : TypeSystem.{u}}
    (ownerAt : Nat -> PhysicalOwner)
    (firstBundle : Nat)
    (schema : Schema types) :
    (schemaColumnIds
      (allocateSchemaFrom ownerAt firstBundle schema)).Nodup := by
  induction schema generalizing firstBundle with
  | nil =>
      exact List.nodup_nil
  | cons port tail inductionHypothesis =>
      rw [schemaColumnIds_allocateSchemaFrom_cons,
        List.nodup_append]
      refine ⟨
        allocatedBundle_ids_nodup ownerAt firstBundle port,
        inductionHypothesis (firstBundle := firstBundle + 1),
        ?_
      ⟩
      intro headId headMember tailId tailMember equal
      subst tailId
      have headExact :=
        allocatedBundle_mem ownerAt firstBundle port headId headMember
      have tailExact :=
        mem_allocateSchemaFrom ownerAt (firstBundle + 1) tail
          headId tailMember
      omega

/-- Nonoverlapping bundle-index intervals imply ID-disjoint schema
allocations, independently of the selected owners. -/
theorem allocateSchemaFrom_ids_disjoint_of_ranges
    {types : TypeSystem.{u}}
    (leftOwnerAt rightOwnerAt : Nat -> PhysicalOwner)
    (leftFirst rightFirst : Nat)
    (leftSchema rightSchema : Schema types)
    (before :
      leftFirst + leftSchema.length ≤ rightFirst) :
    IdsDisjoint
      (schemaColumnIds
        (allocateSchemaFrom leftOwnerAt leftFirst leftSchema))
      (schemaColumnIds
        (allocateSchemaFrom rightOwnerAt rightFirst rightSchema)) := by
  intro id leftMember rightMember
  have leftRange :=
    mem_allocateSchemaFrom leftOwnerAt leftFirst leftSchema id leftMember
  have rightRange :=
    mem_allocateSchemaFrom rightOwnerAt rightFirst rightSchema id rightMember
  omega

/-- Pointwise-incompatible selected owners imply ID-disjoint allocations,
even when their bundle-index intervals overlap. -/
theorem allocateSchemaFrom_ids_disjoint_of_owners
    {types : TypeSystem.{u}}
    (leftOwnerAt rightOwnerAt : Nat -> PhysicalOwner)
    (leftFirst rightFirst : Nat)
    (leftSchema rightSchema : Schema types)
    (ownersDifferent :
      ∀ leftIndex rightIndex,
        leftOwnerAt leftIndex ≠ rightOwnerAt rightIndex) :
    IdsDisjoint
      (schemaColumnIds
        (allocateSchemaFrom leftOwnerAt leftFirst leftSchema))
      (schemaColumnIds
        (allocateSchemaFrom rightOwnerAt rightFirst rightSchema)) := by
  intro id leftMember rightMember
  have leftExact :=
    mem_allocateSchemaFrom leftOwnerAt leftFirst leftSchema id leftMember
  have rightExact :=
    mem_allocateSchemaFrom rightOwnerAt rightFirst rightSchema id rightMember
  exact ownersDifferent id.bundleIndex id.bundleIndex
    (leftExact.1.symm.trans rightExact.1)

/-- Two constant, unequal physical owners cannot allocate the same identity. -/
theorem allocateSchemaFrom_ids_disjoint_of_owner_ne
    {types : TypeSystem.{u}}
    (leftOwner rightOwner : PhysicalOwner)
    (different : leftOwner ≠ rightOwner)
    (leftFirst rightFirst : Nat)
    (leftSchema rightSchema : Schema types) :
    IdsDisjoint
      (schemaColumnIds
        (allocateSchemaFrom (fun _ => leftOwner) leftFirst leftSchema))
      (schemaColumnIds
        (allocateSchemaFrom (fun _ => rightOwner) rightFirst rightSchema)) :=
  allocateSchemaFrom_ids_disjoint_of_owners
    (fun _ => leftOwner) (fun _ => rightOwner)
    leftFirst rightFirst leftSchema rightSchema
    (fun _ _ => different)

/-- Instruction outputs occupy bundles `0 .. outputSchema.length`; call
temporaries begin exactly at `outputSchema.length`, so their concatenated IDs
are unique. -/
theorem instructionOutputs_append_temporaryColumns_ids_nodup
    {types : TypeSystem.{u}}
    (path : OwnerPath)
    (outputSchema : Schema types)
    (layouts : List Layout) :
    (schemaColumnIds (instructionColumns path outputSchema) ++
      schemaColumnIds
        (temporaryColumns path outputSchema layouts)).Nodup := by
  rw [List.nodup_append]
  refine ⟨
    allocateSchemaFrom_ids_nodup
      (fun _ => .typed (.instruction path)) 0 outputSchema,
    allocateSchemaFrom_ids_nodup
      (fun _ => .typed (.instruction path))
      outputSchema.length
      (layouts.map fun layout =>
        { kind := (TypeSystem.Kind.field : types.Kind)
          layout := layout }),
    ?_
  ⟩
  exact allocateSchemaFrom_ids_disjoint_of_ranges
    (fun _ => .typed (.instruction path))
    (fun _ => .typed (.instruction path))
    0 outputSchema.length outputSchema
    (layouts.map fun layout =>
      { kind := (TypeSystem.Kind.field : types.Kind)
        layout := layout })
    (by omega)

/-! ## Owner-class separation -/

/-- The verifier prelude identity cannot collide with any typed allocation. -/
theorem prelude_typed_ids_disjoint
    {types : TypeSystem.{u}}
    (ownerAt : Nat -> Owner)
    (firstBundle : Nat)
    (schema : Schema types) :
    IdsDisjoint [oneColumn]
      (schemaColumnIds
        (allocateSchemaFrom
          (fun index => .typed (ownerAt index))
          firstBundle schema)) := by
  intro id member
  simp only [List.mem_singleton] at member
  subst id
  intro allocated
  have exactOwner :=
    mem_allocateSchemaFrom
      (fun index => PhysicalOwner.typed (ownerAt index))
      firstBundle schema oneColumn allocated
  exact PhysicalOwner.noConfusion exactOwner.1

/-- Input-owned and instruction-owned allocations cannot collide. -/
theorem input_instruction_ids_disjoint
    {types : TypeSystem.{u}}
    (path : OwnerPath)
    (inputSchema outputSchema : Schema types) :
    IdsDisjoint
      (schemaColumnIds (inputColumns inputSchema))
      (schemaColumnIds (instructionColumns path outputSchema)) := by
  exact allocateSchemaFrom_ids_disjoint_of_owners
    (fun slot => .typed (.input slot))
    (fun _ => .typed (.instruction path))
    0 0 inputSchema outputSchema (by
      intro leftIndex rightIndex equal
      exact Owner.noConfusion (PhysicalOwner.typed.inj equal))

/-- Input-owned and branch-join-owned allocations cannot collide. -/
theorem input_branch_ids_disjoint
    {types : TypeSystem.{u}}
    (path : OwnerPath)
    (inputSchema outputSchema : Schema types) :
    IdsDisjoint
      (schemaColumnIds (inputColumns inputSchema))
      (schemaColumnIds (branchJoinColumns path outputSchema)) := by
  exact allocateSchemaFrom_ids_disjoint_of_owners
    (fun slot => .typed (.input slot))
    (fun _ => .typed (.branch path))
    0 0 inputSchema outputSchema (by
      intro leftIndex rightIndex equal
      exact Owner.noConfusion (PhysicalOwner.typed.inj equal))

/-- Instruction-owned and branch-join-owned allocations cannot collide. -/
theorem instruction_branch_ids_disjoint
    {types : TypeSystem.{u}}
    (instructionPath branchPath : OwnerPath)
    (instructionSchema branchSchema : Schema types) :
    IdsDisjoint
      (schemaColumnIds
        (instructionColumns instructionPath instructionSchema))
      (schemaColumnIds
        (branchJoinColumns branchPath branchSchema)) := by
  exact allocateSchemaFrom_ids_disjoint_of_owners
    (fun _ => .typed (.instruction instructionPath))
    (fun _ => .typed (.branch branchPath))
    0 0 instructionSchema branchSchema (by
      intro leftIndex rightIndex equal
      exact Owner.noConfusion (PhysicalOwner.typed.inj equal))

/-- Any typed schema allocation is disjoint from both activation columns. -/
theorem typed_activation_ids_disjoint
    {types : TypeSystem.{u}}
    (ownerAt : Nat -> Owner)
    (firstBundle : Nat)
    (schema : Schema types)
    (path : OwnerPath) :
    IdsDisjoint
      (schemaColumnIds
        (allocateSchemaFrom
          (fun index => .typed (ownerAt index))
          firstBundle schema))
      ((activationColumns path).map fun column => column.id) := by
  intro id typedMember activationMember
  have typedExact :=
    mem_allocateSchemaFrom
      (fun index => PhysicalOwner.typed (ownerAt index))
      firstBundle schema id typedMember
  simp only [activationColumns, List.map_cons, List.map_nil,
    List.mem_cons, List.not_mem_nil, or_false] at activationMember
  rcases activationMember with equal | equal <;>
    subst id <;>
    exact PhysicalOwner.noConfusion typedExact.1

/-- The prelude identity is disjoint from both activation columns. -/
theorem prelude_activation_ids_disjoint
    (path : OwnerPath) :
    IdsDisjoint [oneColumn]
      ((activationColumns path).map fun column => column.id) := by
  intro id preludeMember activationMember
  simp only [List.mem_singleton] at preludeMember
  subst id
  simp [activationColumns, oneColumn, activationColumn] at activationMember

/-- The true/false activation bundle itself has unique physical IDs. -/
theorem activationColumns_ids_nodup
    (path : OwnerPath) :
    ((activationColumns path).map fun column => column.id).Nodup := by
  simp [activationColumns, activationColumn]

end ColumnPlan

end Nightstream.Implementation.Lowering.Goldilocks
