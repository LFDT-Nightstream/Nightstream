import Nightstream.Implementation.R1CS.Correspondence.Gadgets.PiDecStrictCanonicalX
import Nightstream.Implementation.R1CS.Correspondence.Gadgets.PiDecStrictReducedY

/-!
Exact model-level compiler for the production strict-`PiDEC` schedule.

Protocol: SuperNeo Section 7.5 at radix two and fourteen ordered children.
Phase: strict `PiDEC` inside the retained F-prime/NIFS verifier.
Constraint family: the complete strict source schedule after the two proved
reductions (semantic-prefix y recomposition and uniform-sign public digits).

Assurance tier: model-level.

Owns: the exact instruction order emitted by the reduced production Rust
gadget; explicit row-major `(sign, centered-product)` trace columns; soundness
to an independent endpoint consisting of the legacy strict semantics plus the
verifier-computed public split; same-assignment completeness with explicit
auxiliary-definition materialization; and generic source-row savings.

Does not own: a generated artifact, column lowering, selective-CCS rewriting,
Rust/audit identity, a concrete active layout, final constraint counts, or row
removal authorization. The legacy `PiDecStrictCompiler` remains unchanged for
the checked-in full-history artifact.

The schedule is, in order: commitment/adv, X recomposition, semantic-prefix y
recomposition, shape, r, s_col, inactive X, uniform-sign X canonicality, ct,
padding, and fold digest. The current production profile has no adv payload;
the empty adv slot is retained in the schedule and the generic theorems state
that host fact explicitly.
-/

namespace Nightstream.Implementation.R1CS.PiDecStrictProductionCompiler

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Radix
open Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Radix.UniformSignedDigits
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.CheckedProgram
open Nightstream.Implementation.R1CS.PiDecStrictCompiler

/-- The legacy base-wire layout plus the two explicit columns allocated for
each active parent-X coordinate. The child-count proof models the Rust host
check that occurs before any strict rows are emitted. -/
structure Layout where
  base : PiDecStrictCompiler.Layout
  xSignTraces : List (Nat × Nat)
  childCount : base.children.length = productionGlobalParams.k

/-- Number of row-major logical public coordinates receiving one shared sign. -/
def logicalXCount (layout : Layout) : Nat :=
  layout.base.parent.xRows * activeColumns layout.base

/-- Totalized row-major trace lookup. `ShapeValid.traceCount` proves every
production lookup is in range. -/
def traceAt (layout : Layout) (row column : Nat) : Nat × Nat :=
  layout.xSignTraces.getD
    (row * activeColumns layout.base + column) (0, 0)

/-- Ordered child lookup at the verifier-owned production arity. -/
def childLayout (layout : Layout) (child : ChildIndex) : ClaimLayout :=
  layout.base.children.get ⟨child.val, by
    rw [layout.childCount]
    exact child.isLt⟩

theorem childLayout_mem (layout : Layout) (child : ChildIndex) :
    childLayout layout child ∈ layout.base.children := by
  unfold childLayout
  exact List.get_mem _ _

/-- The indexed child view is exactly the source-order child list. -/
theorem childLayouts (layout : Layout) :
    List.ofFn (childLayout layout) = layout.base.children := by
  apply List.ext_get
  · simpa using layout.childCount.symm
  · intro index leftLt rightLt
    simp only [List.get_eq_getElem, List.getElem_ofFn]
    unfold childLayout
    rfl

/-- One coordinate's columns in the already-proved canonical-X compiler. -/
def coordinateLayout (layout : Layout) (row column : Nat) :
    PiDecStrictCanonicalX.Layout where
  parentColumn := xColumn layout.base layout.base.parent row column
  signColumn := (traceAt layout row column).1
  signOutputColumn := (traceAt layout row column).2
  digitColumns := fun child =>
    xColumn layout.base (childLayout layout child) row column

theorem coordinate_childColumns (layout : Layout) (row column : Nat) :
    PiDecStrictCanonicalX.childColumns (coordinateLayout layout row column) =
      layout.base.children.map fun child =>
        xColumn layout.base child row column := by
  let columnOf : ClaimLayout → Nat := fun child =>
    xColumn layout.base child row column
  change (List.ofFn fun child : ChildIndex =>
      columnOf (childLayout layout child)) =
    layout.base.children.map columnOf
  calc
    (List.ofFn fun child : ChildIndex => columnOf (childLayout layout child)) =
        (List.ofFn (childLayout layout)).map columnOf := by
      simpa only [Function.comp_apply] using
        (List.map_ofFn (f := childLayout layout) (g := columnOf)).symm
    _ = layout.base.children.map columnOf := by rw [childLayouts]

/-- The existing commitment group, including the position of the optional adv
subprogram. -/
def commitmentAdvInstructions (layout : Layout) (powers : List Nat) :
    List Instruction :=
  dataRecomposition layout.base.parent.commitment.dataCols
      (layout.base.children.map (·.commitment.dataCols)) powers ++
    advInstructions layout.base.parent.adv
      (layout.base.children.map (·.adv)) powers

/-- Sixteen canonicality rows for each active X coordinate, in the same
row-major order as the Rust trace vector. X recomposition remains in its
earlier schedule group. -/
def canonicalXInstructions (layout : Layout) : List Instruction :=
  (List.range layout.base.parent.xRows).flatMap fun row =>
    (List.range (activeColumns layout.base)).flatMap fun column =>
      PiDecStrictCanonicalX.canonicalityInstructions
        (coordinateLayout layout row column)

/-- Named-in-order production groups. Their flattening is the emitted program. -/
def groups (layout : Layout) : List (List Instruction) :=
  let powers := radixPowers layout.base.radix layout.base.children.length
  [commitmentAdvInstructions layout powers,
   xRecompositionInstructions layout.base powers,
   PiDecStrictReducedY.reducedYRecompositionInstructions layout.base powers,
   shapeInstructions layout.base,
   pairEqualityInstructions layout.base.parent.rCols
      (layout.base.children.map (·.rCols)),
   pairEqualityInstructions layout.base.parent.sColCols
      (layout.base.children.map (·.sColCols)),
   inactiveInstructions layout.base,
   canonicalXInstructions layout,
   ctInstructions layout.base,
   paddingInstructions layout.base,
   foldDigestInstructions layout.base]

def instructions (layout : Layout) : List Instruction :=
  (groups layout).flatten

def rows (layout : Layout) : List Row :=
  CheckedProgram.rows (instructions layout)

/-- Host-side facts required for exact production interpretation. In
particular, `semanticYFits` is the successful branch of Rust's fail-closed
semantic-prefix slice check. -/
structure ShapeValid (layout : Layout) : Prop where
  base : PiDecStrictCompiler.ShapeValid layout.base
  radixTwo : layout.base.radix = 2
  ringDimension : layout.base.ringDimension = 54
  extensionLimbs : layout.base.extensionLimbs = 2
  traceCount : layout.xSignTraces.length = logicalXCount layout
  semanticYFits : ∀ row, row < layout.base.parent.yRingCols.length →
    PiDecStrictReducedY.semanticYWidth layout.base ≤
      (layout.base.parent.yRingCols.getD row []).length

/-- Independent semantic meaning of all common-sign public-coordinate blocks. -/
def UniformXAccepted (layout : Layout) (assignment : Nat → Nat) : Prop :=
  ∀ row column,
    row < layout.base.parent.xRows →
    column < activeColumns layout.base →
    UniformSignedDigits.Accepted
      (PiDecStrictCanonicalX.decodedParent
        (coordinateLayout layout row column) assignment)
      (PiDecStrictCanonicalX.decodedSign
        (coordinateLayout layout row column) assignment)
      (PiDecStrictCanonicalX.decodedDigits
        (coordinateLayout layout row column) assignment)

/-- Independent endpoint for the reduced production compiler. The first field
preserves every legacy strict equation. The second strengthens public X from
arbitrary centered recomposition to the unique verifier-computed split. -/
structure Accepted (layout : Layout) (assignment : Nat → Nat) : Prop where
  legacy : PiDecStrictCompiler.Accepted layout.base assignment
  uniformX : UniformXAccepted layout assignment

/-- Every accepted child-X coordinate is exactly the verifier-computed signed
binary split of its parent coordinate. -/
theorem Accepted.childXExact
    {layout : Layout} {assignment : Nat → Nat}
    (accepted : Accepted layout assignment)
    (row column : Nat)
    (rowLt : row < layout.base.parent.xRows)
    (columnLt : column < activeColumns layout.base) :
    PiDecStrictCanonicalX.decodedDigits
        (coordinateLayout layout row column) assignment =
      splitScalar
        (PiDecStrictCanonicalX.decodedParent
          (coordinateLayout layout row column) assignment) := by
  exact (accepted.uniformX row column rowLt columnLt).digits_eq_splitScalar

/-! ## Model-level compiler soundness -/

private theorem group_satisfies
    {layout : Layout} {assignment : Nat → Nat}
    (satisfies : Satisfies (rows layout) assignment)
    {group : List Instruction} (member : group ∈ groups layout) :
    Satisfies (CheckedProgram.rows group) assignment := by
  intro row rowMember
  rcases List.mem_map.mp rowMember with ⟨instruction, instructionMember, rfl⟩
  apply satisfies instruction.row
  apply List.mem_map.mpr
  refine ⟨instruction, ?_, rfl⟩
  exact List.mem_flatten.mpr ⟨group, member, instructionMember⟩

private theorem satisfies_instruction_append_left
    {left right : List Instruction} {assignment : Nat → Nat}
    (satisfies : Satisfies
      (CheckedProgram.rows (left ++ right)) assignment) :
    Satisfies (CheckedProgram.rows left) assignment := by
  intro row member
  apply satisfies row
  simpa [CheckedProgram.rows] using
    List.mem_append_left (CheckedProgram.rows right) member

private theorem canonicality_satisfies_at
    {layout : Layout} {assignment : Nat → Nat}
    (satisfies : Satisfies
      (CheckedProgram.rows (canonicalXInstructions layout)) assignment)
    (row column : Nat)
    (rowLt : row < layout.base.parent.xRows)
    (columnLt : column < activeColumns layout.base) :
    Satisfies
      (CheckedProgram.rows
        (PiDecStrictCanonicalX.canonicalityInstructions
          (coordinateLayout layout row column))) assignment := by
  intro localRow localMember
  rcases List.mem_map.mp localMember with
    ⟨instruction, instructionMember, rfl⟩
  apply satisfies instruction.row
  apply List.mem_map.mpr
  refine ⟨instruction, ?_, rfl⟩
  apply List.mem_flatMap.mpr
  refine ⟨row, List.mem_range.mpr rowLt, ?_⟩
  apply List.mem_flatMap.mpr
  exact ⟨column, List.mem_range.mpr columnLt, instructionMember⟩

private theorem uniform_coordinate_sound
    (prime : EuclidPrime goldilocksP)
    {layout : Layout} (valid : ShapeValid layout)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (xSatisfies : Satisfies
      (CheckedProgram.rows (xRecompositionInstructions layout.base
        (radixPowers layout.base.radix
          layout.base.children.length))) assignment)
    (canonicalitySatisfies : Satisfies
      (CheckedProgram.rows (canonicalXInstructions layout)) assignment)
    (row column : Nat)
    (rowLt : row < layout.base.parent.xRows)
    (columnLt : column < activeColumns layout.base) :
    UniformSignedDigits.Accepted
      (PiDecStrictCanonicalX.decodedParent
        (coordinateLayout layout row column) assignment)
      (PiDecStrictCanonicalX.decodedSign
        (coordinateLayout layout row column) assignment)
      (PiDecStrictCanonicalX.decodedDigits
        (coordinateLayout layout row column) assignment) := by
  have constraint := PiDecStrictCanonicalX.canonicality_sound prime canonical
    one (canonicality_satisfies_at canonicalitySatisfies row column
      rowLt columnLt)
  have sourceRecomposes := PiDecStrictSound.xRecomposition_sound canonical one
    valid.base.powersCanonical xSatisfies row column rowLt columnLt
  have canonicalRecomposes :
      Recomposes assignment
        (coordinateLayout layout row column).parentColumn
        (PiDecStrictCanonicalX.childColumns
          (coordinateLayout layout row column))
        PiDecStrictCanonicalX.powers := by
    rw [coordinate_childColumns]
    simpa [coordinateLayout, PiDecStrictCanonicalX.powers,
      valid.radixTwo, layout.childCount] using sourceRecomposes
  exact {
    constraint := constraint
    recomposition :=
      PiDecStrictCanonicalX.decodedRecomposition_of_recomposes
        canonicalRecomposes
  }

private theorem fieldOfNat_injective_canonical
    {left right : Nat}
    (leftLt : left < goldilocksP) (rightLt : right < goldilocksP)
    (equal : fieldOfNat left = fieldOfNat right) : left = right := by
  have values := congrArg Fin.val equal
  change left % goldilocksP = right % goldilocksP at values
  rw [Nat.mod_eq_of_lt leftLt, Nat.mod_eq_of_lt rightLt] at values
  exact values

private theorem fieldOfNat_minus_one :
    fieldOfNat (goldilocksP - 1) = (-1 : F) := by
  decide

private theorem centered_of_decoded_sign
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (column : Nat)
    (allowed : SignAllowed (fieldOfNat (assignment column))) :
    CenteredUnit (assignment column) := by
  rcases allowed with zero | one | minusOne
  · left
    apply fieldOfNat_injective_canonical (canonical column) (by decide)
    simpa using zero
  · right; left
    apply fieldOfNat_injective_canonical (canonical column) (by decide)
    simpa using one
  · right; right
    apply fieldOfNat_injective_canonical (canonical column) (by decide)
    rw [fieldOfNat_minus_one]
    exact minusOne

private theorem selector_of_decoded_digit
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (digit sign : Nat)
    (accepted : fieldOfNat (assignment digit) = 0 ∨
      fieldOfNat (assignment digit) = fieldOfNat (assignment sign)) :
    assignment digit = 0 ∨ assignment digit = assignment sign := by
  rcases accepted with zero | signed
  · left
    apply fieldOfNat_injective_canonical (canonical digit) (by decide)
    simpa using zero
  · right
    exact fieldOfNat_injective_canonical
      (canonical digit) (canonical sign) signed

private theorem child_index_of_mem
    (layout : Layout) {child : ClaimLayout}
    (member : child ∈ layout.base.children) :
    ∃ index : ChildIndex, childLayout layout index = child := by
  rcases List.mem_iff_getElem.mp member with ⟨index, indexLt, childEq⟩
  let childIndex : ChildIndex := ⟨index, by
    rw [← layout.childCount]
    exact indexLt⟩
  refine ⟨childIndex, ?_⟩
  unfold childLayout
  simpa [childIndex, List.get_eq_getElem] using childEq

private theorem active_column_indices
    (base : PiDecStrictCompiler.Layout) (claim : ClaimLayout)
    {column : Nat} (member : column ∈ activeXColumns base claim) :
    ∃ row coordinate,
      row < claim.xRows ∧ coordinate < activeColumns base ∧
        column = xColumn base claim row coordinate := by
  rcases List.mem_flatMap.mp member with
    ⟨row, rowMember, columnMember⟩
  rcases List.mem_map.mp columnMember with
    ⟨coordinate, coordinateMember, columnEq⟩
  exact ⟨row, coordinate, List.mem_range.mp rowMember,
    List.mem_range.mp coordinateMember, columnEq.symm⟩

private theorem uniform_childCentered
    {layout : Layout} (valid : ShapeValid layout)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (uniform : UniformXAccepted layout assignment) :
    ∀ child ∈ layout.base.children,
      ∀ column ∈ activeXColumns layout.base child,
        CenteredUnit (assignment column) := by
  intro child childMember column columnMember
  rcases child_index_of_mem layout childMember with ⟨childIndex, childEq⟩
  rcases active_column_indices layout.base child columnMember with
    ⟨row, coordinate, rowLtChild, coordinateLt, columnEq⟩
  have rowLt : row < layout.base.parent.xRows := by
    have shape := valid.base.xShapes child childMember
    omega
  have coordinateAccepted := uniform row coordinate rowLt coordinateLt
  let coordinateLayout' := coordinateLayout layout row coordinate
  have signCentered : CenteredUnit (assignment coordinateLayout'.signColumn) :=
    centered_of_decoded_sign canonical coordinateLayout'.signColumn
      coordinateAccepted.constraint.1
  have selector :
      assignment (coordinateLayout'.digitColumns childIndex) = 0 ∨
        assignment (coordinateLayout'.digitColumns childIndex) =
          assignment coordinateLayout'.signColumn := by
    apply selector_of_decoded_digit canonical
    exact coordinateAccepted.constraint.2 childIndex
  change
    assignment
        (xColumn layout.base (childLayout layout childIndex) row coordinate) = 0 ∨
      assignment
          (xColumn layout.base (childLayout layout childIndex) row coordinate) =
        assignment (traceAt layout row coordinate).1 at selector
  rw [childEq] at selector
  rw [columnEq]
  rcases selector with zero | signed
  · exact Or.inl zero
  · rcases signCentered with signZero | signOne | signMinusOne
    · exact Or.inl (signed.trans signZero)
    · exact Or.inr (Or.inl (signed.trans signOne))
    · exact Or.inr (Or.inr (signed.trans signMinusOne))

/-- Generic no-adv model-level soundness for the exact reduced production
schedule. The conclusion is independent of row satisfaction and exposes the
deterministic child-X result separately. -/
theorem sound_noAdv
    (prime : EuclidPrime goldilocksP)
    {layout : Layout} (valid : ShapeValid layout)
    (parentNoAdv : layout.base.parent.adv = none)
    (childrenNoAdv : ∀ child ∈ layout.base.children, child.adv = none)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies (rows layout) assignment) :
    Accepted layout assignment := by
  let powers := radixPowers layout.base.radix layout.base.children.length
  have group0 : commitmentAdvInstructions layout powers ∈ groups layout := by
    simp [groups, powers]
  have group1 : xRecompositionInstructions layout.base powers ∈ groups layout := by
    simp [groups, powers]
  have group2 : PiDecStrictReducedY.reducedYRecompositionInstructions
      layout.base powers ∈ groups layout := by simp [groups, powers]
  have group3 : shapeInstructions layout.base ∈ groups layout := by simp [groups]
  have group4 : pairEqualityInstructions layout.base.parent.rCols
      (layout.base.children.map (·.rCols)) ∈ groups layout := by simp [groups]
  have group5 : pairEqualityInstructions layout.base.parent.sColCols
      (layout.base.children.map (·.sColCols)) ∈ groups layout := by simp [groups]
  have group6 : inactiveInstructions layout.base ∈ groups layout := by simp [groups]
  have group7 : canonicalXInstructions layout ∈ groups layout := by simp [groups]
  have group8 : ctInstructions layout.base ∈ groups layout := by simp [groups]
  have group9 : paddingInstructions layout.base ∈ groups layout := by simp [groups]
  have group10 : foldDigestInstructions layout.base ∈ groups layout := by simp [groups]
  have satisfies0 := group_satisfies satisfies group0
  have satisfies1 := group_satisfies satisfies group1
  have satisfies2 := group_satisfies satisfies group2
  have satisfies3 := group_satisfies satisfies group3
  have satisfies4 := group_satisfies satisfies group4
  have satisfies5 := group_satisfies satisfies group5
  have satisfies6 := group_satisfies satisfies group6
  have satisfies7 := group_satisfies satisfies group7
  have satisfies8 := group_satisfies satisfies group8
  have satisfies9 := group_satisfies satisfies group9
  have satisfies10 := group_satisfies satisfies group10
  have uniform : UniformXAccepted layout assignment := by
    intro row column rowLt columnLt
    exact uniform_coordinate_sound prime valid canonical one satisfies1
      satisfies7 row column rowLt columnLt
  have reducedY := PiDecStrictReducedY.reducedYRecomposition_sound canonical
    one valid.base.powersCanonical satisfies2
  have padding := PiDecStrictSound.paddingInstructions_sound canonical one
    satisfies9
  have fullY := PiDecStrictReducedY.fullY_of_reducedY_and_padding valid.base
    reducedY padding
  refine {
    legacy := {
      radixTwo := valid.radixTwo
      commitment := PiDecStrictSound.dataRecomposition_sound canonical one
        valid.base.powersCanonical
        (satisfies_instruction_append_left satisfies0)
      adv := ?_
      x := PiDecStrictSound.xRecomposition_sound canonical one
        valid.base.powersCanonical satisfies1
      y := fullY
      shape := PiDecStrictSound.shapeInstructions_sound canonical one satisfies3
      sameR := ?_
      sameSCol := ?_
      inactiveZero := PiDecStrictSound.inactiveInstructions_sound canonical one
        satisfies6
      childCentered := uniform_childCentered valid canonical uniform
      ct := PiDecStrictSound.ctInstructions_sound canonical one satisfies8
      paddingZero := padding
      foldDigest := PiDecStrictSound.foldDigestInstructions_sound canonical one
        satisfies10
    }
    uniformX := uniform
  }
  · simp [AdvAccepted, parentNoAdv]
    exact childrenNoAdv
  · intro child childMember
    apply PiDecStrictSound.pairEqualityInstructions_sound canonical one
      layout.base.parent.rCols (layout.base.children.map (·.rCols))
      satisfies4 child.rCols
    exact List.mem_map.mpr ⟨child, childMember, rfl⟩
  · intro child childMember
    apply PiDecStrictSound.pairEqualityInstructions_sound canonical one
      layout.base.parent.sColCols (layout.base.children.map (·.sColCols))
      satisfies5 child.sColCols
    exact List.mem_map.mpr ⟨child, childMember, rfl⟩

/-! ## Same-assignment completeness -/

/-- The only new deterministic auxiliaries are the centered-product outputs
paired with the explicit sign columns. A successful Rust builder execution
materializes exactly these equations. -/
def TraceDefinitions (layout : Layout) (assignment : Nat → Nat) : Prop :=
  ∀ row column,
    row < layout.base.parent.xRows →
    column < activeColumns layout.base →
    let coordinateLayout' := coordinateLayout layout row column
    Definition.Holds assignment {
      output := coordinateLayout'.signOutputColumn
      rhs := .product
        [(coordinateLayout'.signColumn, 1), (0, 1)]
        [(coordinateLayout'.signColumn, 1)]
    }

private theorem canonicality_complete_at
    {layout : Layout} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (accepted : UniformXAccepted layout assignment)
    (definitions : TraceDefinitions layout assignment)
    (row column : Nat)
    (rowLt : row < layout.base.parent.xRows)
    (columnLt : column < activeColumns layout.base) :
    Satisfies
      (CheckedProgram.rows
        (PiDecStrictCanonicalX.canonicalityInstructions
          (coordinateLayout layout row column))) assignment := by
  let coordinateLayout' := coordinateLayout layout row column
  have semantic := accepted row column rowLt columnLt
  have centered : CenteredUnit (assignment coordinateLayout'.signColumn) :=
    centered_of_decoded_sign canonical coordinateLayout'.signColumn
      semantic.constraint.1
  have selectors : ∀ child,
      assignment (coordinateLayout'.digitColumns child) = 0 ∨
        assignment (coordinateLayout'.digitColumns child) =
          assignment coordinateLayout'.signColumn := by
    intro child
    exact selector_of_decoded_digit canonical _ _
      (semantic.constraint.2 child)
  have definitionHolds : Definition.Holds assignment {
      output := coordinateLayout'.signOutputColumn
      rhs := .product
        [(coordinateLayout'.signColumn, 1), (0, 1)]
        [(coordinateLayout'.signColumn, 1)]
    } := by
    simpa [coordinateLayout'] using definitions row column rowLt columnLt
  have centeredSatisfies : Satisfies
      (CheckedProgram.rows (centeredUnitInstructions
        coordinateLayout'.signColumn coordinateLayout'.signOutputColumn))
      assignment := by
    intro localRow localMember
    rcases List.mem_map.mp localMember with
      ⟨instruction, instructionMember, rfl⟩
    simp only [centeredUnitInstructions, List.mem_cons,
      List.not_mem_nil, or_false] at instructionMember
    rcases instructionMember with rfl | rfl
    · exact builderDefinition_complete canonical one _ (by trivial)
        definitionHolds
    · exact PiDecStrictSound.centeredUnitCheck_complete one
        coordinateLayout'.signColumn coordinateLayout'.signOutputColumn
        definitionHolds centered
  have digitSatisfies : Satisfies
      (CheckedProgram.rows
        (PiDecStrictCanonicalX.digitInstructions coordinateLayout'))
      assignment := by
    intro localRow localMember
    rcases List.mem_map.mp localMember with
      ⟨instruction, instructionMember, rfl⟩
    rcases List.mem_ofFn.mp instructionMember with ⟨child, rfl⟩
    exact PiDecStrictCanonicalX.digitInstruction_complete child
      (selectors child)
  simpa [PiDecStrictCanonicalX.canonicalityInstructions,
    CheckedProgram.rows] using
      PiDecStrictSound.satisfies_append centeredSatisfies digitSatisfies

private theorem canonicalXInstructions_complete
    {layout : Layout} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (accepted : UniformXAccepted layout assignment)
    (definitions : TraceDefinitions layout assignment) :
    Satisfies
      (CheckedProgram.rows (canonicalXInstructions layout)) assignment := by
  intro row rowMember
  rcases List.mem_map.mp rowMember with
    ⟨instruction, instructionMember, rfl⟩
  rcases List.mem_flatMap.mp instructionMember with
    ⟨rowIndex, rowIndexMember, instructionMember⟩
  rcases List.mem_flatMap.mp instructionMember with
    ⟨columnIndex, columnIndexMember, instructionMember⟩
  apply canonicality_complete_at canonical one accepted definitions
      rowIndex columnIndex
      (List.mem_range.mp rowIndexMember)
      (List.mem_range.mp columnIndexMember)
      instruction.row
  exact List.mem_map.mpr ⟨instruction, instructionMember, rfl⟩

/-- Same-assignment completeness for the reduced production schedule. This
does not treat semantic acceptance as an execution oracle: the deterministic
product-wire equations are supplied separately by `TraceDefinitions`. -/
theorem complete_noAdv
    {layout : Layout} (valid : ShapeValid layout)
    (parentNoAdv : layout.base.parent.adv = none)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (accepted : Accepted layout assignment)
    (traceDefinitions : TraceDefinitions layout assignment) :
    Satisfies (rows layout) assignment := by
  let powers := radixPowers layout.base.radix layout.base.children.length
  have dataSatisfies := PiDecStrictSound.dataRecomposition_complete one
    valid.base.powersCanonical accepted.legacy.commitment
  have group0 : Satisfies
      (CheckedProgram.rows (commitmentAdvInstructions layout powers))
      assignment := by
    simpa [commitmentAdvInstructions, parentNoAdv, advInstructions,
      CheckedProgram.rows, powers] using dataSatisfies
  have group1 := PiDecStrictSound.xRecomposition_complete one
    valid.base.powersCanonical accepted.legacy.x
  have group2 := PiDecStrictReducedY.reducedYRecomposition_complete one
    valid.base.powersCanonical
    (PiDecStrictReducedY.reducedY_of_fullY accepted.legacy.y)
  have group3 := PiDecStrictSound.shapeInstructions_complete canonical one
    accepted.legacy.shape
  have group4 := PiDecStrictSound.pairEqualityInstructions_complete canonical
    one layout.base.parent.rCols (layout.base.children.map (·.rCols)) (by
      intro childPairs childPairsMember
      rcases List.mem_map.mp childPairsMember with ⟨child, childMember, rfl⟩
      exact accepted.legacy.sameR child childMember)
  have group5 := PiDecStrictSound.pairEqualityInstructions_complete canonical
    one layout.base.parent.sColCols
      (layout.base.children.map (·.sColCols)) (by
      intro childPairs childPairsMember
      rcases List.mem_map.mp childPairsMember with ⟨child, childMember, rfl⟩
      exact accepted.legacy.sameSCol child childMember)
  have group6 := PiDecStrictSound.inactiveInstructions_complete one
    accepted.legacy.inactiveZero
  have group7 := canonicalXInstructions_complete canonical one
    accepted.uniformX traceDefinitions
  have group8 := PiDecStrictSound.ctInstructions_complete canonical one
    accepted.legacy.ct
  have group9 := PiDecStrictSound.paddingInstructions_complete one
    accepted.legacy.paddingZero
  have group10 := PiDecStrictSound.foldDigestInstructions_complete canonical
    one accepted.legacy.foldDigest
  simpa [rows, instructions, groups, powers, CheckedProgram.rows] using
    PiDecStrictSound.satisfies_append group0
      (PiDecStrictSound.satisfies_append group1
        (PiDecStrictSound.satisfies_append group2
          (PiDecStrictSound.satisfies_append group3
            (PiDecStrictSound.satisfies_append group4
              (PiDecStrictSound.satisfies_append group5
                (PiDecStrictSound.satisfies_append group6
                  (PiDecStrictSound.satisfies_append group7
                    (PiDecStrictSound.satisfies_append group8
                      (PiDecStrictSound.satisfies_append group9
                        group10)))))))))

/-! ## Exact generic source-row census -/

private theorem length_flatMap_range_constant
    {Carrier : Type} (count width : Nat) (body : Nat → List Carrier)
    (bodyLength : ∀ index, index < count → (body index).length = width) :
    ((List.range count).flatMap body).length = count * width := by
  induction count with
  | zero => simp
  | succ count inductionHypothesis =>
      rw [List.range_succ, List.flatMap_append, List.length_append]
      simp only [List.flatMap_singleton]
      rw [inductionHypothesis (fun index indexLt =>
        bodyLength index (by omega))]
      rw [bodyLength count (by omega)]
      simp [Nat.succ_mul]

theorem canonicalXInstruction_count (layout : Layout) :
    (canonicalXInstructions layout).length =
      logicalXCount layout *
        PiDecStrictCanonicalX.canonicalityRowCount := by
  unfold canonicalXInstructions logicalXCount
  rw [length_flatMap_range_constant
    layout.base.parent.xRows
    (activeColumns layout.base *
      PiDecStrictCanonicalX.canonicalityRowCount)]
  · rw [Nat.mul_assoc]
  · intro row rowLt
    apply length_flatMap_range_constant
    intro column columnLt
    simpa [CheckedProgram.rows] using
      PiDecStrictCanonicalX.canonicality_rows_exact
        (coordinateLayout layout row column)

private theorem alphabetFrom_length (columns : List Nat) (output : Nat) :
    (alphabetFrom output columns).length = 2 * columns.length := by
  induction columns generalizing output with
  | nil => simp [alphabetFrom]
  | cons head tail inductionHypothesis =>
      simp [alphabetFrom, centeredUnitInstructions, inductionHypothesis,
        Nat.mul_add]

private theorem length_flatMap_constant_list
    {Input Output : Type} (values : List Input) (width : Nat)
    (body : Input → List Output)
    (bodyLength : ∀ value ∈ values, (body value).length = width) :
    (values.flatMap body).length = values.length * width := by
  induction values with
  | nil => simp
  | cons head tail inductionHypothesis =>
      rw [List.flatMap_cons, List.length_append]
      rw [bodyLength head (by simp)]
      rw [inductionHypothesis (fun value member =>
        bodyLength value (by simp [member]))]
      simp only [List.length_cons, Nat.succ_mul]
      exact Nat.add_comm _ _

private theorem activeXColumns_length
    (base : PiDecStrictCompiler.Layout) (claim : ClaimLayout) :
    (activeXColumns base claim).length =
      claim.xRows * activeColumns base := by
  unfold activeXColumns
  apply length_flatMap_range_constant
  intro row rowLt
  simp

private theorem flatMap_activeX_length
    {layout : Layout} (valid : ShapeValid layout) :
    (layout.base.children.flatMap (activeXColumns layout.base)).length =
      layout.base.children.length * logicalXCount layout := by
  apply length_flatMap_constant_list
  intro child childMember
  rw [activeXColumns_length]
  rw [(valid.base.xShapes child childMember).1]
  rfl

theorem legacyAlphabetInstruction_count
    {layout : Layout} (valid : ShapeValid layout) :
    (alphabetInstructions layout.base).length =
      logicalXCount layout *
        PiDecStrictCanonicalX.currentIndependentAlphabetRowCount := by
  rw [alphabetInstructions, alphabetFrom_length,
    flatMap_activeX_length valid, layout.childCount]
  simp [PiDecStrictCanonicalX.currentIndependentAlphabetRowCount,
    productionGlobalParams]
  omega

/-- The common-sign family saves exactly twelve source R1CS rows per logical
public coordinate relative to the legacy independent-centered family. -/
theorem canonicalX_saving
    {layout : Layout} (valid : ShapeValid layout) :
    (alphabetInstructions layout.base).length =
      (canonicalXInstructions layout).length +
        logicalXCount layout *
          PiDecStrictCanonicalX.rowsSavedPerCoordinate := by
  rw [legacyAlphabetInstruction_count valid,
    canonicalXInstruction_count]
  simp [PiDecStrictCanonicalX.currentIndependentAlphabetRowCount,
    PiDecStrictCanonicalX.canonicalityRowCount,
    PiDecStrictCanonicalX.rowsSavedPerCoordinate]
  omega

/-- Rows unchanged by either reduction. This is a source-schedule count, not a
selective-CCS count. -/
def retainedInstructionCount (layout : Layout) : Nat :=
  let powers := radixPowers layout.base.radix layout.base.children.length
  (commitmentAdvInstructions layout powers).length +
  (xRecompositionInstructions layout.base powers).length +
  (shapeInstructions layout.base).length +
  (pairEqualityInstructions layout.base.parent.rCols
    (layout.base.children.map (·.rCols))).length +
  (pairEqualityInstructions layout.base.parent.sColCols
    (layout.base.children.map (·.sColCols))).length +
  (inactiveInstructions layout.base).length +
  (ctInstructions layout.base).length +
  (paddingInstructions layout.base).length +
  (foldDigestInstructions layout.base).length

theorem production_rows_count (layout : Layout) :
    (rows layout).length =
      retainedInstructionCount layout +
        (PiDecStrictReducedY.reducedYRecompositionInstructions layout.base
          (radixPowers layout.base.radix
            layout.base.children.length)).length +
        (canonicalXInstructions layout).length := by
  simp only [rows, instructions, CheckedProgram.rows, List.length_map,
    List.length_flatten, groups, List.map_cons, List.map_nil, List.sum_cons,
    List.sum_nil, Nat.add_zero, retainedInstructionCount]
  omega

theorem legacy_rows_count (layout : Layout) :
    (PiDecStrictCompiler.rows layout.base).length =
      retainedInstructionCount layout +
        (yRecompositionInstructions layout.base
          (radixPowers layout.base.radix
            layout.base.children.length)).length +
        (alphabetInstructions layout.base).length := by
  simp only [PiDecStrictCompiler.rows, PiDecStrictCompiler.instructions,
    CheckedProgram.rows, List.length_map, List.length_flatten,
    PiDecStrictCompiler.groups, List.map_cons, List.map_nil, List.sum_cons,
    List.sum_nil, Nat.add_zero, retainedInstructionCount,
    commitmentAdvInstructions]
  omega

private theorem yInstruction_saving
    {layout : Layout} {width : Nat}
    (shape : PiDecStrictReducedY.UniformParentYWidth layout.base width) :
    (yRecompositionInstructions layout.base
      (radixPowers layout.base.radix layout.base.children.length)).length =
      (PiDecStrictReducedY.reducedYRecompositionInstructions layout.base
        (radixPowers layout.base.radix layout.base.children.length)).length +
        layout.base.parent.yRingCols.length *
          (width - PiDecStrictReducedY.semanticYWidth layout.base) := by
  rw [PiDecStrictReducedY.fullYRecompositionInstruction_count shape,
    PiDecStrictReducedY.reducedYRecompositionInstruction_count shape]
  have semanticFits := shape.semanticFits
  have widthDecomposition : width =
      PiDecStrictReducedY.semanticYWidth layout.base +
        (width - PiDecStrictReducedY.semanticYWidth layout.base) := by
    omega
  calc
    layout.base.parent.yRingCols.length * width =
        layout.base.parent.yRingCols.length *
          (PiDecStrictReducedY.semanticYWidth layout.base +
            (width - PiDecStrictReducedY.semanticYWidth layout.base)) :=
      congrArg (fun value =>
        layout.base.parent.yRingCols.length * value) widthDecomposition
    _ = _ := Nat.mul_add _ _ _

/-- Exact combined saving over the legacy source compiler: twelve rows per
active public coordinate plus every padded y-recomposition row. All retained
families, including padding-zero checks, cancel syntactically. -/
theorem combined_source_saving
    {layout : Layout} (valid : ShapeValid layout) {width : Nat}
    (yShape : PiDecStrictReducedY.UniformParentYWidth layout.base width) :
    (PiDecStrictCompiler.rows layout.base).length =
      (rows layout).length +
        logicalXCount layout *
          PiDecStrictCanonicalX.rowsSavedPerCoordinate +
        layout.base.parent.yRingCols.length *
          (width - PiDecStrictReducedY.semanticYWidth layout.base) := by
  rw [legacy_rows_count, production_rows_count,
    canonicalX_saving valid, yInstruction_saving yShape]
  omega

end Nightstream.Implementation.R1CS.PiDecStrictProductionCompiler
