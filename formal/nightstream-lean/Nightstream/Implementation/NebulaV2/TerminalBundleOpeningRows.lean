import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.TerminalR1cs.Ajtai
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.TerminalR1cs.Norm
import Nightstream.Implementation.NebulaV2.CommitmentBundleFieldRows
import Nightstream.Implementation.NebulaV2.SeedSchedule
import Nightstream.Implementation.R1CS.Canonical.GoldilocksField
import Nightstream.Protocol.NebulaV2.LaneLayout
import Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Commitment

/-!
Contract: exact typed terminal rows that open the mandatory V2 commitment
bundle against one bounded complete SuperNeo assignment.

Assurance tier: implementation-to-protocol bridge.

Owns the one shared full-witness column family, whole-ring operations and
snapshot projections, the verifier-key-selected full/operations/snapshot
Ajtai keys, four complete commitment openings, and the strict `b = 2` norm
check on the one full assignment. Initial and final snapshots use the same
selected snapshot key.

Does not own the numeric-to-typed generated-row bridge, CCS/CE terminal
acceptance, NIFS verification, public-result checks, Module-SIS security,
Rust, or a deployed Spartan verifier.

Emits constraints: `2 * fullShape.carrierWidth + 4 * 18 * 54` typed
Goldilocks R1CS rows.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace Nightstream.Implementation.NebulaV2.TerminalBundleOpeningRows

open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.TerminalR1cs
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.Typed
open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.CommitmentBundle
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

abbrev Rank := MemoryWireGeometry.commitmentRank
abbrev BundleValue := Component → PiRLCAlgebra.Commitment.Value Rank

/-- Flatten one typed commitment row/lane pair into the canonical public
bundle coordinate order. -/
def bundleCoordinate (row : Fin Rank) (lane : Fin ringDegree) :
    CommitmentBundleCodec.Coordinate :=
  ⟨row.val * ringDegree + lane.val, by
    have rowLt := row.isLt
    have laneLt := lane.isLt
    change row.val < 18 at rowLt
    change lane.val < 54 at laneLt
    change row.val * 54 + lane.val < 972
    omega⟩

/-- Cast one setup-selected block coordinate to the exact complete carrier
block domain of a typed SuperNeo shape. -/
def castBlock
    {shape : Phi81Relation.Shape} {columns : Nat}
    (blocks : Phi81ColumnLayout.blockCount shape.carrierWidth = columns)
    (block : Fin (Phi81ColumnLayout.blockCount shape.carrierWidth)) :
    Fin columns :=
  Fin.cast blocks block

/-- Cast one V2 rank coordinate to the selected setup role rank. -/
def castRank
    {roleRows : Nat} (rows : Rank = roleRows) (row : Fin Rank) :
    Fin roleRows :=
  Fin.cast rows row

/-- The exact three shape-to-setup width equalities. They prevent a selected
matrix from being truncated, padded, or reused at another lane width. -/
structure ShapeAgreement
    (manifest : SeedSchedule.Manifest)
    (fullShape operationsShape snapshotShape : Phi81Relation.Shape) : Prop where
  fullBlocks :
    Phi81ColumnLayout.blockCount fullShape.carrierWidth =
      manifest.geometry.fullAssignmentRingColumns
  operationsBlocks :
    Phi81ColumnLayout.blockCount operationsShape.carrierWidth =
      manifest.geometry.operationsRingColumns
  snapshotBlocks :
    Phi81ColumnLayout.blockCount snapshotShape.carrierWidth =
      manifest.geometry.snapshotRingColumns

namespace ShapeAgreement

variable {manifest : SeedSchedule.Manifest}
variable {fullShape operationsShape snapshotShape : Phi81Relation.Shape}

def fullKey
    (agreement : ShapeAgreement manifest fullShape operationsShape
      snapshotShape) :
    PiRLCAlgebra.Commitment.Key fullShape Rank :=
  fun row block =>
    (manifest.setup .bundleFull).verifierKey
      (castRank (by rfl) row)
      (castBlock agreement.fullBlocks block)

def operationsKey
    (agreement : ShapeAgreement manifest fullShape operationsShape
      snapshotShape) :
    PiRLCAlgebra.Commitment.Key operationsShape Rank :=
  fun row block =>
    (manifest.setup .bundleOperations).verifierKey
      (castRank (by rfl) row)
      (castBlock agreement.operationsBlocks block)

/-- This is the sole selected snapshot key. Both snapshot openings below use
this exact value. -/
def snapshotKey
    (agreement : ShapeAgreement manifest fullShape operationsShape
      snapshotShape) :
    PiRLCAlgebra.Commitment.Key snapshotShape Rank :=
  fun row block =>
    (manifest.setup .bundleSnapshot).verifierKey
      (castRank (by rfl) row)
      (castBlock agreement.snapshotBlocks block)

end ShapeAgreement

/-- Exact physical placement. The operations and snapshot frames do not own
independent witnesses: their witness maps are definitions from `fullWitness`
and the whole-ring lane layout. -/
structure Layout
    (manifest : SeedSchedule.Manifest)
    (fullShape operationsShape snapshotShape : Phi81Relation.Shape) where
  agreement : ShapeAgreement manifest fullShape operationsShape snapshotShape
  lanes : LaneLayout.Layout fullShape.carrierWidth
    operationsShape.carrierWidth snapshotShape.carrierWidth
  one : ColumnId
  fullWitness : Fin fullShape.carrierWidth → ColumnId
  fullSquare : Fin fullShape.carrierWidth → ColumnId
  bundleFields : CommitmentBundleFieldRows.Layout
  normOwner : PhysicalOwner
  fullOwner : PhysicalOwner
  operationsOwner : PhysicalOwner
  initialSnapshotOwner : PhysicalOwner
  finalSnapshotOwner : PhysicalOwner
  normFirstOrdinal : Nat
  fullFirstOrdinal : Nat
  operationsFirstOrdinal : Nat
  initialSnapshotFirstOrdinal : Nat
  finalSnapshotFirstOrdinal : Nat

namespace Layout

variable {manifest : SeedSchedule.Manifest}
variable {fullShape operationsShape snapshotShape : Phi81Relation.Shape}

/-- The one structural embedding used by the combined numeric/typed
Goldilocks compiler. The coordinate index makes this map injective; a caller
cannot collapse or overlap numeric columns by selecting another map. -/
def numericColumn (column : Nat) : ColumnId where
  owner := .prelude
  bundleIndex := 1
  coordinateIndex := column

theorem numericColumn_injective : Function.Injective numericColumn := by
  intro left right equal
  exact congrArg ColumnId.coordinateIndex equal

def commitmentColumn
    (layout : Layout manifest fullShape operationsShape snapshotShape)
    (component : Component) (row : Fin Rank) (lane : Fin ringDegree) :
    ColumnId :=
  numericColumn
    (layout.bundleFields.fieldColumn component (bundleCoordinate row lane))

def fullAssignment
    (layout : Layout manifest fullShape operationsShape snapshotShape)
    (assignment : ColumnId → F) : Assignment fullShape :=
  fun coordinate => assignment (layout.fullWitness coordinate)

def operationsAssignment
    (layout : Layout manifest fullShape operationsShape snapshotShape)
    (assignment : ColumnId → F) : Assignment operationsShape :=
  fun coordinate =>
    assignment (layout.fullWitness (layout.lanes.operationsIndex coordinate))

def initialSnapshotAssignment
    (layout : Layout manifest fullShape operationsShape snapshotShape)
    (assignment : ColumnId → F) : Assignment snapshotShape :=
  fun coordinate =>
    assignment
      (layout.fullWitness (layout.lanes.initialSnapshotIndex coordinate))

def finalSnapshotAssignment
    (layout : Layout manifest fullShape operationsShape snapshotShape)
    (assignment : ColumnId → F) : Assignment snapshotShape :=
  fun coordinate =>
    assignment
      (layout.fullWitness (layout.lanes.finalSnapshotIndex coordinate))

/-- The public bundle read in the same row/lane order as the exact Ajtai
commitment carrier. -/
def publicBundle
    (layout : Layout manifest fullShape operationsShape snapshotShape)
    (assignment : ColumnId → F) : BundleValue :=
  fun component row lane =>
    assignment (layout.commitmentColumn component row lane)

/-- Canonical conversion from the protocol codec residue to the concrete
Goldilocks carrier used by the typed terminal compiler. -/
def codecField
    (value : ShiftedTernary41V1.CanonicalGoldilocks) : F :=
  ⟨value.val, by
    simpa [ShiftedTernary41V1.modulus, goldilocksModulus] using value.property⟩

/-- The exact row/lane view of one codec bundle. -/
def codecBundle (bundle : CommitmentBundleCodec.Value) : BundleValue :=
  fun component row lane =>
    codecField (bundle component (bundleCoordinate row lane))

/-- Exact assignment agreement required from the combined numeric/typed
compiler. It uses the fixed injective `numericColumn` map. -/
def NumericAgreement
    (numericAssignment : Nat → Nat)
    (typedAssignment : ColumnId → F) : Prop :=
  ∀ column,
    (typedAssignment (numericColumn column)).val = numericAssignment column

/-- Canonical numeric bundle fields and exact combined-assignment agreement
identify the typed public commitment columns with the same folded bundle. -/
theorem publicBundle_eq_codecBundle
    (layout : Layout manifest fullShape operationsShape snapshotShape)
    (numericAssignment : Nat → Nat) (typedAssignment : ColumnId → F)
    (bundle : CommitmentBundleCodec.Value)
    (agreement : NumericAgreement numericAssignment typedAssignment)
    (numericFields : ∀ component coordinate,
      numericAssignment
          (layout.bundleFields.fieldColumn component coordinate) =
        (bundle component coordinate).val) :
    layout.publicBundle typedAssignment = codecBundle bundle := by
  funext component row lane
  apply Fin.ext
  change
    (typedAssignment
      (numericColumn
        (layout.bundleFields.fieldColumn component
          (bundleCoordinate row lane)))).val = _
  rw [agreement, numericFields]
  rfl

def normFrame
    (layout : Layout manifest fullShape operationsShape snapshotShape) :
    Norm.Frame fullShape where
  owner := layout.normOwner
  firstOrdinal := layout.normFirstOrdinal
  witness := layout.fullWitness
  square := layout.fullSquare

def fullFrame
    (layout : Layout manifest fullShape operationsShape snapshotShape) :
    Ajtai.Frame fullShape Rank where
  owner := layout.fullOwner
  firstOrdinal := layout.fullFirstOrdinal
  one := layout.one
  key := layout.agreement.fullKey
  witness := layout.fullWitness
  commitment := layout.commitmentColumn .full

def operationsFrame
    (layout : Layout manifest fullShape operationsShape snapshotShape) :
    Ajtai.Frame operationsShape Rank where
  owner := layout.operationsOwner
  firstOrdinal := layout.operationsFirstOrdinal
  one := layout.one
  key := layout.agreement.operationsKey
  witness := fun coordinate =>
    layout.fullWitness (layout.lanes.operationsIndex coordinate)
  commitment := layout.commitmentColumn .operations

def initialSnapshotFrame
    (layout : Layout manifest fullShape operationsShape snapshotShape) :
    Ajtai.Frame snapshotShape Rank where
  owner := layout.initialSnapshotOwner
  firstOrdinal := layout.initialSnapshotFirstOrdinal
  one := layout.one
  key := layout.agreement.snapshotKey
  witness := fun coordinate =>
    layout.fullWitness (layout.lanes.initialSnapshotIndex coordinate)
  commitment := layout.commitmentColumn .initialSnapshot

def finalSnapshotFrame
    (layout : Layout manifest fullShape operationsShape snapshotShape) :
    Ajtai.Frame snapshotShape Rank where
  owner := layout.finalSnapshotOwner
  firstOrdinal := layout.finalSnapshotFirstOrdinal
  one := layout.one
  key := layout.agreement.snapshotKey
  witness := fun coordinate =>
    layout.fullWitness (layout.lanes.finalSnapshotIndex coordinate)
  commitment := layout.commitmentColumn .finalSnapshot

end Layout

/-- The concrete four-component Ajtai map. This is one function of one full
assignment. The operations and snapshot inputs are exact projections of that
assignment. -/
def exactBundle
    {manifest : SeedSchedule.Manifest}
    {fullShape operationsShape snapshotShape : Phi81Relation.Shape}
    (layout : Layout manifest fullShape operationsShape snapshotShape)
    (assignment : ColumnId → F) : BundleValue
  | .full => PiRLCAlgebra.Commitment.commit layout.agreement.fullKey
      (layout.fullAssignment assignment)
  | .operations =>
      PiRLCAlgebra.Commitment.commit layout.agreement.operationsKey
        (layout.operationsAssignment assignment)
  | .initialSnapshot =>
      PiRLCAlgebra.Commitment.commit layout.agreement.snapshotKey
        (layout.initialSnapshotAssignment assignment)
  | .finalSnapshot =>
      PiRLCAlgebra.Commitment.commit layout.agreement.snapshotKey
        (layout.finalSnapshotAssignment assignment)

def rows
    {manifest : SeedSchedule.Manifest}
    {fullShape operationsShape snapshotShape : Phi81Relation.Shape}
    (layout : Layout manifest fullShape operationsShape snapshotShape) :
    List OwnedRow :=
  Norm.rows layout.normFrame ++
    Ajtai.rows layout.fullFrame ++
    Ajtai.rows layout.operationsFrame ++
    Ajtai.rows layout.initialSnapshotFrame ++
    Ajtai.rows layout.finalSnapshotFrame

/-- Semantic result of the exact row family. The opening equality is one
product-map equation from one assignment. It is not four existential
openings. -/
structure Sound
    {manifest : SeedSchedule.Manifest}
    {fullShape operationsShape snapshotShape : Phi81Relation.Shape}
    (layout : Layout manifest fullShape operationsShape snapshotShape)
    (assignment : ColumnId → F) : Prop where
  bounded : Phi81Relation.assignmentNormBounded 2
    (layout.fullAssignment assignment)
  opensAll : exactBundle layout assignment = layout.publicBundle assignment

private theorem split_rows
    {manifest : SeedSchedule.Manifest}
    {fullShape operationsShape snapshotShape : Phi81Relation.Shape}
    (layout : Layout manifest fullShape operationsShape snapshotShape)
    (assignment : ColumnId → F)
    (satisfied : Satisfies (rows layout) assignment) :
    Satisfies (Norm.rows layout.normFrame) assignment ∧
      Satisfies (Ajtai.rows layout.fullFrame) assignment ∧
      Satisfies (Ajtai.rows layout.operationsFrame) assignment ∧
      Satisfies (Ajtai.rows layout.initialSnapshotFrame) assignment ∧
      Satisfies (Ajtai.rows layout.finalSnapshotFrame) assignment := by
  rw [rows, satisfies_append_iff, satisfies_append_iff,
    satisfies_append_iff, satisfies_append_iff] at satisfied
  exact ⟨satisfied.1.1.1.1, satisfied.1.1.1.2,
    satisfied.1.1.2, satisfied.1.2, satisfied.2⟩

/-- Row satisfaction derives one bounded common-witness opening. No opening
equation is an input premise. -/
theorem sound
    {manifest : SeedSchedule.Manifest}
    {fullShape operationsShape snapshotShape : Phi81Relation.Shape}
    (layout : Layout manifest fullShape operationsShape snapshotShape)
    (assignment : ColumnId → F)
    (constantOne : assignment layout.one = 1)
    (satisfied : Satisfies (rows layout) assignment) :
    Sound layout assignment := by
  rcases split_rows layout assignment satisfied with
    ⟨normRows, fullRows, operationsRows, initialRows, finalRows⟩
  refine
    { bounded := Norm.rows_sound
        (NormRange.baseFieldNoZeroDivisors_of_modulusEuclid
          Nightstream.Implementation.R1CS.Canonical.GoldilocksField.goldilocks_euclidPrime)
        layout.normFrame assignment normRows
      opensAll := ?_ }
  change exactBundle layout assignment = layout.publicBundle assignment
  funext component row lane
  cases component with
  | full =>
      have exact := Ajtai.rows_sound layout.fullFrame assignment
        constantOne fullRows
      exact (congrFun (congrFun exact row) lane).symm
  | operations =>
      have exact := Ajtai.rows_sound layout.operationsFrame assignment
        constantOne operationsRows
      exact (congrFun (congrFun exact row) lane).symm
  | initialSnapshot =>
      have exact := Ajtai.rows_sound layout.initialSnapshotFrame assignment
        constantOne initialRows
      exact (congrFun (congrFun exact row) lane).symm
  | finalSnapshot =>
      have exact := Ajtai.rows_sound layout.finalSnapshotFrame assignment
        constantOne finalRows
      exact (congrFun (congrFun exact row) lane).symm

/-- End-to-end local bridge for the final folded bundle. Numeric decoder rows
and typed opening rows derive one bounded common-witness opening of the exact
codec bundle. The only cross-program premise is `NumericAgreement`, which a
combined generated compiler must prove from its one physical assignment. -/
theorem sound_opens_codec_bundle
    {manifest : SeedSchedule.Manifest}
    {fullShape operationsShape snapshotShape : Phi81Relation.Shape}
    (layout : Layout manifest fullShape operationsShape snapshotShape)
    (numericAssignment : Nat → Nat) (typedAssignment : ColumnId → F)
    (bundle : CommitmentBundleCodec.Value)
    (numericCanonical : ∀ column, numericAssignment column <
      Nightstream.Implementation.R1CS.goldilocksP)
    (numericOne : numericAssignment 0 = 1)
    (bundleBits : CommitmentBundleFieldRows.BitsPlaced
      layout.bundleFields numericAssignment bundle)
    (numericRows : Nightstream.Implementation.R1CS.Satisfies
      (CommitmentBundleFieldRows.rows layout.bundleFields) numericAssignment)
    (assignmentAgreement : Layout.NumericAgreement numericAssignment
      typedAssignment)
    (typedOne : typedAssignment layout.one = 1)
    (typedRows : Satisfies (rows layout) typedAssignment) :
    Phi81Relation.assignmentNormBounded 2
        (layout.fullAssignment typedAssignment) ∧
      exactBundle layout typedAssignment = Layout.codecBundle bundle := by
  have opened := sound layout typedAssignment typedOne typedRows
  have numericFields :=
    CommitmentBundleFieldRows.typed_columns_of_bits_and_rows
      numericCanonical numericOne numericRows bundleBits
  exact ⟨opened.bounded, opened.opensAll.trans
    (Layout.publicBundle_eq_codecBundle layout numericAssignment
      typedAssignment bundle assignmentAgreement numericFields)⟩

theorem rows_length
    {manifest : SeedSchedule.Manifest}
    {fullShape operationsShape snapshotShape : Phi81Relation.Shape}
    (layout : Layout manifest fullShape operationsShape snapshotShape) :
    (rows layout).length = 2 * fullShape.carrierWidth + 3888 := by
  simp [rows, MemoryWireGeometry.commitmentRank,
    MemoryWireGeometry.commitmentFieldCount, ringDegree]

end Nightstream.Implementation.NebulaV2.TerminalBundleOpeningRows
