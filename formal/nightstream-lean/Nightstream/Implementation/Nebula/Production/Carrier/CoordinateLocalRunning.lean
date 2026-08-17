import Nightstream.Implementation.Nebula.NIFS.Running.RunningCoordinatesFor

/-!
Contract: coordinate-local view of the complete production running claim.

Assurance tier: model-level carrier isomorphism and exact field geometry.

Owns a grouped view that places the same coordinate from all sixteen running
claims together, exact conversion to and from the typed paper running claim,
injectivity, and exact field counts at a verifier-selected exponent.

Does not own transcript order, generated rows, a streamed PiCCS, PiRLC, or
PiDEC verifier, Rust refinement, candidate selection, or a row and column
reduction claim.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace Nightstream.Implementation.Nebula.ProductionCoordinateLocalRunning

open Nightstream.Implementation.Nebula
open Nightstream.Implementation.Nebula.ProductNifsCodec
open Nightstream.Implementation.Nebula.ProductNifsRunningCoordinatesFor
open Nightstream.Implementation.Encoding.NifsCanonicalCodec
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Protocol.Nebula.CommitmentBundle
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

abbrev Source := Fin 16
abbrev Matrix := Fin 14

/-- Same typed data as `RunningFor`, grouped by arithmetic coordinate before
source claim. -/
structure View (rowVariables : Nat) (fullShape : Phi81Relation.Shape) where
  point : Fin rowVariables → K
  commitment : Component → Fin ProductCommitmentAlgebra.Rank →
    Fin ringDegree → Source → F
  publicInput : Fin fullShape.publicWidth → Source → F
  evaluation : Matrix → Fin ringDegree → Source → K

/-- Read one dimension-checked point coordinate without a default branch. -/
def pointCoordinate {rowVariables : Nat}
    (point : CubePoint K rowVariables) (index : Fin rowVariables) : K :=
  point.coordinates.get
    ⟨index.val, by rw [point.dimension]; exact index.isLt⟩

noncomputable def ofRunning
    {rowVariables : Nat} {fullShape : Phi81Relation.Shape}
    (running : RunningFor rowVariables fullShape) :
    View rowVariables fullShape where
  point := pointCoordinate running.point
  commitment := fun component row lane source =>
    running.commitments source component row lane
  publicInput := fun column source => running.publicInputs source column
  evaluation := fun matrix lane source =>
    running.evaluations source matrix lane

noncomputable def toRunning
    {rowVariables : Nat} {fullShape : Phi81Relation.Shape}
    (view : View rowVariables fullShape) :
    RunningFor rowVariables fullShape where
  point := {
    coordinates := List.ofFn view.point
    dimension := by simp
  }
  commitments := fun source component row lane =>
    view.commitment component row lane source
  publicInputs := fun source column => view.publicInput column source
  evaluations := fun source matrix lane => view.evaluation matrix lane source

private theorem point_coordinates_ofFn
    {rowVariables : Nat} (point : CubePoint K rowVariables) :
    List.ofFn (pointCoordinate point) = point.coordinates := by
  apply List.ext_get
  · simp [point.dimension]
  · intro index leftBound rightBound
    rw [List.get_ofFn]
    rfl

private theorem cubePoint_eq_of_coordinates
    {rowVariables : Nat} {left right : CubePoint K rowVariables}
    (coordinates : left.coordinates = right.coordinates) : left = right := by
  cases left
  cases right
  simp_all

/-- Grouping and then ungrouping preserves every typed field exactly. -/
theorem toRunning_ofRunning
    {rowVariables : Nat} {fullShape : Phi81Relation.Shape}
    (running : RunningFor rowVariables fullShape) :
    toRunning (ofRunning running) = running := by
  cases running with
  | mk point commitments publicInputs evaluations =>
      have pointExact :
          ({ coordinates := List.ofFn (pointCoordinate point)
             dimension := by simp } : CubePoint K rowVariables) = point := by
        apply cubePoint_eq_of_coordinates
        exact point_coordinates_ofFn point
      unfold toRunning ofRunning
      congr 1

/-- The coordinate-local view loses no running-claim authority. -/
theorem ofRunning_injective
    {rowVariables : Nat} {fullShape : Phi81Relation.Shape} :
    Function.Injective
      (ofRunning : RunningFor rowVariables fullShape →
        View rowVariables fullShape) := by
  intro left right equal
  calc
    left = toRunning (ofRunning left) := (toRunning_ofRunning left).symm
    _ = toRunning (ofRunning right) := congrArg toRunning equal
    _ = right := toRunning_ofRunning right

/-! ## Canonical coordinate-major field codec -/

/-- Exact semantic product serialized by the coordinate-major codec. The
finite-function order is point, then commitment component/row/lane/source,
then public column/source, then evaluation matrix/lane/source. -/
abbrev Data (rowVariables : Nat) (fullShape : Phi81Relation.Shape) :=
  (Fin rowVariables → K) ×
    ((Fin 4 → Fin ProductCommitmentAlgebra.Rank → Fin ringDegree → Source → F) ×
      ((Fin fullShape.publicWidth → Source → F) ×
        (Matrix → Fin ringDegree → Source → K)))

def viewData
    {rowVariables : Nat} {fullShape : Phi81Relation.Shape}
    (view : View rowVariables fullShape) :
    Data rowVariables fullShape :=
  (view.point,
    (fun component row lane source =>
      view.commitment (componentAt component) row lane source,
      (view.publicInput, view.evaluation)))

private theorem viewData_injective
    {rowVariables : Nat} {fullShape : Phi81Relation.Shape} :
    Function.Injective
      (viewData : View rowVariables fullShape →
        Data rowVariables fullShape) := by
  intro left right equal
  have pointEqual : left.point = right.point :=
    congrArg Prod.fst equal
  have tailEqual :
      (viewData left).2 = (viewData right).2 :=
    congrArg Prod.snd equal
  have commitmentEqual :
      (fun component row lane source =>
          left.commitment (componentAt component) row lane source) =
        (fun component row lane source =>
          right.commitment (componentAt component) row lane source) :=
    congrArg Prod.fst tailEqual
  have finalEqual :
      (left.publicInput, left.evaluation) =
        (right.publicInput, right.evaluation) :=
    congrArg Prod.snd tailEqual
  have publicInputEqual : left.publicInput = right.publicInput :=
    congrArg Prod.fst finalEqual
  have evaluationEqual : left.evaluation = right.evaluation :=
    congrArg Prod.snd finalEqual
  have commitmentAllEqual : left.commitment = right.commitment := by
    funext component row lane source
    cases component with
    | full =>
        simpa [componentAt] using
          congrFun (congrFun (congrFun
            (congrFun commitmentEqual (0 : Fin 4)) row) lane) source
    | operations =>
        simpa [componentAt] using
          congrFun (congrFun (congrFun
            (congrFun commitmentEqual (1 : Fin 4)) row) lane) source
    | initialSnapshot =>
        simpa [componentAt] using
          congrFun (congrFun (congrFun
            (congrFun commitmentEqual (2 : Fin 4)) row) lane) source
    | finalSnapshot =>
        simpa [componentAt] using
          congrFun (congrFun (congrFun
            (congrFun commitmentEqual (3 : Fin 4)) row) lane) source
  cases left
  cases right
  simp_all

private theorem codec_pullback_width
    {Alpha Beta : Type} (target : Codec Beta)
    (toTarget : Alpha → Beta) (injective : Function.Injective toTarget) :
    (Codec.pullback target toTarget injective).width = target.width :=
  rfl

private theorem codec_product_width
    {Alpha Beta : Type} (left : Codec Alpha) (right : Codec Beta) :
    (Codec.product left right).width = left.width + right.width :=
  rfl

private theorem codec_finFunction_width
    {Alpha : Type} (count : Nat) (codec : Codec Alpha) :
    (Codec.finFunction count codec).width = count * codec.width :=
  rfl

noncomputable def dataCodec
    (rowVariables : Nat) (fullShape : Phi81Relation.Shape) :
    Codec (Data rowVariables fullShape) :=
  Codec.product
    (Codec.finFunction rowVariables kCodec)
    (Codec.product
      (Codec.finFunction 4
        (Codec.finFunction ProductCommitmentAlgebra.Rank
          (Codec.finFunction ringDegree
            (Codec.finFunction 16 fieldCodec))))
      (Codec.product
        (Codec.finFunction fullShape.publicWidth
          (Codec.finFunction 16 fieldCodec))
        (Codec.finFunction 14
          (Codec.finFunction ringDegree
            (Codec.finFunction 16 kCodec)))))

/-- Fixed-width field codec for the same typed running claim, with sources
adjacent inside each arithmetic coordinate. -/
noncomputable def coordinateLocalCodec
    (rowVariables : Nat) (fullShape : Phi81Relation.Shape) :
    Codec (View rowVariables fullShape) :=
  Codec.pullback (dataCodec rowVariables fullShape)
    viewData viewData_injective

theorem coordinateLocalCodec_admissible
    {rowVariables : Nat} {fullShape : Phi81Relation.Shape}
    (view : View rowVariables fullShape) :
    (coordinateLocalCodec rowVariables fullShape).Admissible view := by
  exact
    ⟨(fun coordinate => kCodec_admissible (view.point coordinate)),
      ⟨(fun _component _row _lane _source => True.intro),
        ⟨(fun _column _source => True.intro),
          (fun matrix lane source =>
            kCodec_admissible (view.evaluation matrix lane source))⟩⟩⟩

def pointFieldCount (rowVariables : Nat) : Nat := rowVariables * 2
def commitmentFieldCount : Nat :=
  4 * ProductCommitmentAlgebra.Rank * ringDegree * 16
def publicInputFieldCount (fullShape : Phi81Relation.Shape) : Nat :=
  fullShape.publicWidth * 16
def evaluationFieldCount : Nat := 14 * ringDegree * 16 * 2

def totalFieldCount
    (rowVariables : Nat) (fullShape : Phi81Relation.Shape) : Nat :=
  pointFieldCount rowVariables + commitmentFieldCount +
    publicInputFieldCount fullShape + evaluationFieldCount

/-- The explicit coordinate-major codec has exactly the semantic field count.
It is a permutation of authority, not a compression claim. -/
theorem coordinateLocalCodec_width
    (rowVariables : Nat) (fullShape : Phi81Relation.Shape) :
    (coordinateLocalCodec rowVariables fullShape).width =
      totalFieldCount rowVariables fullShape := by
  simp [coordinateLocalCodec, codec_pullback_width, dataCodec,
    codec_product_width, codec_finFunction_width, fieldCodec,
    totalFieldCount, pointFieldCount,
    commitmentFieldCount, publicInputFieldCount, evaluationFieldCount,
    ProductCommitmentAlgebra.Rank,
    Nightstream.Protocol.Nebula.MemoryWireGeometry.commitmentRank,
    ringDegree]
  omega

noncomputable def encodeRunning
    {rowVariables : Nat} {fullShape : Phi81Relation.Shape}
    (running : RunningFor rowVariables fullShape) : List F :=
  (coordinateLocalCodec rowVariables fullShape).encode (ofRunning running)

noncomputable def decodeRunning
    {rowVariables : Nat} {fullShape : Phi81Relation.Shape}
    (fields : List F) : Option (RunningFor rowVariables fullShape) :=
  Option.map toRunning
    ((coordinateLocalCodec rowVariables fullShape).decode fields)

@[simp] theorem encodeRunning_length
    {rowVariables : Nat} {fullShape : Phi81Relation.Shape}
    (running : RunningFor rowVariables fullShape) :
    (encodeRunning running).length =
      totalFieldCount rowVariables fullShape := by
  rw [encodeRunning,
    (coordinateLocalCodec rowVariables fullShape).encode_length,
    coordinateLocalCodec_width]

/-- The field layout reconstructs the exact typed running claim. -/
theorem decodeRunning_encodeRunning
    {rowVariables : Nat} {fullShape : Phi81Relation.Shape}
    (running : RunningFor rowVariables fullShape) :
    decodeRunning (encodeRunning running) = some running := by
  rw [decodeRunning, encodeRunning,
    (coordinateLocalCodec rowVariables fullShape).decode_encode
      (ofRunning running) (coordinateLocalCodec_admissible (ofRunning running))]
  simp [toRunning_ofRunning]

/-- Equal coordinate-major field strings imply equal authoritative running
claims. -/
theorem encodeRunning_injective
    {rowVariables : Nat} {fullShape : Phi81Relation.Shape} :
    Function.Injective
      (encodeRunning : RunningFor rowVariables fullShape → List F) := by
  intro left right encodedEqual
  apply ofRunning_injective
  exact (coordinateLocalCodec rowVariables fullShape).encode_injective_of_admissible
    (coordinateLocalCodec_admissible (ofRunning left))
    (coordinateLocalCodec_admissible (ofRunning right)) encodedEqual

/-- Reordering changes locality, not the number of authoritative fields. -/
theorem totalFieldCount_eq_runningFieldCountFor
    {rowVariables : Nat} {fullShape : Phi81Relation.Shape}
    (contract : FullShapeContractFor rowVariables fullShape) :
    totalFieldCount rowVariables fullShape =
      runningFieldCountFor rowVariables := by
  unfold totalFieldCount publicInputFieldCount
  rw [contract.publicWidth]
  simp [pointFieldCount, commitmentFieldCount, evaluationFieldCount,
    ProductCommitmentAlgebra.Rank,
    Nightstream.Protocol.Nebula.MemoryWireGeometry.commitmentRank,
    MemoryBoundCcsPublic.coordinateCount, ringDegree, runningFieldCountFor]
  omega

theorem totalFieldCount_r26
    {fullShape : Phi81Relation.Shape}
    (contract : FullShapeContractFor 26 fullShape) :
    totalFieldCount 26 fullShape = 95092 := by
  rw [totalFieldCount_eq_runningFieldCountFor contract]
  decide

/-- One coordinate-local ring action consumes sixteen running ring values and
one fresh ring value. Each ring value has 54 base-field lanes. -/
def fullSourceRingWindowFieldCount : Nat := 17 * ringDegree

theorem fullSourceRingWindowFieldCount_eq :
    fullSourceRingWindowFieldCount = 918 := by
  decide

end Nightstream.Implementation.Nebula.ProductionCoordinateLocalRunning
