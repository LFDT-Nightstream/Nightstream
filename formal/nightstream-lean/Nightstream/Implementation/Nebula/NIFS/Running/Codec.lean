import Nightstream.Implementation.Encoding.NifsCanonicalCodec
import Nightstream.Implementation.Nebula.Commitment.Core.Algebra
import Nightstream.Implementation.Nebula.Commitment.Bundle.Codec
import Nightstream.Implementation.Nebula.Core.FixedBits
import Nightstream.Implementation.Nebula.Memory.Claim.BoundCcsPublic
import Nightstream.Implementation.Nebula.Application.Wasm.ResultCodec
import Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.Types

/-!
Contract: exact canonical carrier codec for the V2 product-commitment NIFS
public input.

Assurance tier: implementation model.

Owns the fixed 25-variable paper shape, the complete four-component running
claim codec, exact field and bit widths, an injective canonical bit image, a
fail-closed semantic inverse, and the fresh claim derived from the exact CCS
public bits and mandatory bundle.

Does not own an executable generated parser, NIFS verifier rows, the final
assignment width, relation matrices, or cryptographic soundness.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 10000

namespace Nightstream.Implementation.Nebula.ProductNifsCodec

open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Encoding.NifsCanonicalCodec
open Nightstream.Implementation.Nebula
open Nightstream.Protocol.Nebula
open Nightstream.Protocol.Nebula.CommitmentBundle
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongReduction

abbrev JointShape :=
  Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Shape

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

/-- Paper arity at one verifier-key-selected relation exponent. The exponent
is part of the relation and changes the point codec, SumCheck schedule,
transcript, security census, and verifier key. -/
def shapeFor (rowVariables : Nat) : JointShape where
  cubeVariables := rowVariables
  freshCount := 1
  runningCount := 14
  matrixCount := 14
  coefficientCount := ringDegree

@[simp] theorem shapeFor_cubeVariables (rowVariables : Nat) :
    (shapeFor rowVariables).cubeVariables = rowVariables := rfl

/-- The original bit-serial V2 reference shape. Field-native production
profiles must not reuse this value unless their complete generated augmented
relation proves that it fits. -/
def shape : JointShape where
  cubeVariables := 25
  freshCount := 1
  runningCount := 14
  matrixCount := 14
  coefficientCount := ringDegree

theorem shape_eq_shapeFor_25 : shape = shapeFor 25 := rfl

theorem shape_exact :
    shape.cubeVariables = 25 /\
      shape.freshCount = 1 /\
      shape.runningCount = 14 /\
      shape.matrixCount = 14 /\
      shape.coefficientCount = 54 := by
  decide

/-- The final generated full relation can vary only in its exact logical
width. All public and folding dimensions are fixed before key generation. -/
structure FullShapeContract (fullShape : Phi81Relation.Shape) : Prop where
  rowVariables : fullShape.rowVariables = 25
  matrixCount : fullShape.matrixCount = 14
  publicRingColumns : fullShape.publicRingColumns = 10

/-- Shape contract for a verifier-key-selected augmented-relation exponent.
The field-native production path must use this contract instead of the
fixed-25 reference contract. -/
structure FullShapeContractFor
    (rowVariables : Nat) (fullShape : Phi81Relation.Shape) : Prop where
  rowVariablesExact : fullShape.rowVariables = rowVariables
  matrixCount : fullShape.matrixCount = 14
  publicRingColumns : fullShape.publicRingColumns = 10

namespace FullShapeContract

theorem publicWidth
    {fullShape : Phi81Relation.Shape}
    (contract : FullShapeContract fullShape) :
    fullShape.publicWidth = MemoryBoundCcsPublic.coordinateCount := by
  simp [Phi81Relation.Shape.publicWidth, contract.publicRingColumns, ringDegree]
  rfl

end FullShapeContract

namespace FullShapeContractFor

theorem publicWidth
    {rowVariables : Nat} {fullShape : Phi81Relation.Shape}
    (contract : FullShapeContractFor rowVariables fullShape) :
    fullShape.publicWidth = MemoryBoundCcsPublic.coordinateCount := by
  simp [Phi81Relation.Shape.publicWidth, contract.publicRingColumns, ringDegree]
  rfl

/-- Reindex a generated-exponent contract by the exponent stored in the
shape. This changes no structural fact. -/
def toShape
    {rowVariables : Nat} {fullShape : Phi81Relation.Shape}
    (contract : FullShapeContractFor rowVariables fullShape) :
    FullShapeContractFor fullShape.rowVariables fullShape where
  rowVariablesExact := rfl
  matrixCount := contract.matrixCount
  publicRingColumns := contract.publicRingColumns

end FullShapeContractFor

/-- Convert the fixed reference contract to the shape-indexed contract. The
target exponent is read from the shape itself. -/
def FullShapeContract.toSelected
    {fullShape : Phi81Relation.Shape}
    (contract : FullShapeContract fullShape) :
    FullShapeContractFor fullShape.rowVariables fullShape where
  rowVariablesExact := rfl
  matrixCount := contract.matrixCount
  publicRingColumns := contract.publicRingColumns

instance {fullShape : Phi81Relation.Shape} :
    Coe (FullShapeContract fullShape)
      (FullShapeContractFor fullShape.rowVariables fullShape) :=
  ⟨FullShapeContract.toSelected⟩

abbrev Running (fullShape : Phi81Relation.Shape) :=
  Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.Running
    K ProductCommitmentAlgebra.BundleValue (PublicInput fullShape) shape

abbrev Fresh (fullShape : Phi81Relation.Shape) :=
  Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.Fresh
    ProductCommitmentAlgebra.BundleValue (PublicInput fullShape) shape

/-- Running paper claim indexed by the exact augmented-relation exponent. -/
abbrev RunningFor (rowVariables : Nat) (fullShape : Phi81Relation.Shape) :=
  Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.Running
    K ProductCommitmentAlgebra.BundleValue (PublicInput fullShape)
      (shapeFor rowVariables)

/-- Fresh paper claim indexed by the exact augmented-relation exponent. -/
abbrev FreshFor (rowVariables : Nat) (fullShape : Phi81Relation.Shape) :=
  Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.Fresh
    ProductCommitmentAlgebra.BundleValue (PublicInput fullShape)
      (shapeFor rowVariables)

abbrev ComponentCommitment :=
  Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.Commitment.Value
    ProductCommitmentAlgebra.Rank

/-! ## Product commitment codec -/

def bundleData (value : ProductCommitmentAlgebra.BundleValue) :=
  (value .full,
    (value .operations, (value .initialSnapshot, value .finalSnapshot)))

theorem bundleData_injective : Function.Injective bundleData := by
  intro left right equal
  funext component
  simp only [bundleData, Prod.mk.injEq] at equal
  cases component <;> simp_all

noncomputable def bundleCodec :
    Codec ProductCommitmentAlgebra.BundleValue :=
  Codec.pullback
    (Codec.product (commitmentCodec ProductCommitmentAlgebra.Rank)
      (Codec.product (commitmentCodec ProductCommitmentAlgebra.Rank)
        (Codec.product (commitmentCodec ProductCommitmentAlgebra.Rank)
          (commitmentCodec ProductCommitmentAlgebra.Rank))))
    bundleData bundleData_injective

@[simp] theorem bundleCodec_width : bundleCodec.width = 3888 := by
  rfl

theorem bundleCodec_admissible
    (value : ProductCommitmentAlgebra.BundleValue) :
    bundleCodec.Admissible value := by
  exact
    ⟨commitmentCodec_admissible (value .full),
      commitmentCodec_admissible (value .operations),
      commitmentCodec_admissible (value .initialSnapshot),
      commitmentCodec_admissible (value .finalSnapshot)⟩

/-! ## Evaluation and running codecs -/

abbrev Evaluation := EvaluationFamily K shape

noncomputable def evaluationCodec : Codec Evaluation :=
  Codec.finFunction shape.matrixCount
    (Codec.finFunction shape.coefficientCount kCodec)

@[simp] theorem evaluationCodec_width : evaluationCodec.width = 1512 := by
  rfl

theorem evaluationCodec_admissible (value : Evaluation) :
    evaluationCodec.Admissible value := by
  intro matrix coefficient
  exact kCodec_admissible (value matrix coefficient)

def runningData {fullShape : Phi81Relation.Shape}
    (value : Running fullShape) :=
  (value.point,
    (value.commitments, (value.publicInputs, value.evaluations)))

theorem runningData_injective {fullShape : Phi81Relation.Shape} :
    Function.Injective (runningData (fullShape := fullShape)) := by
  intro left right equal
  cases left
  cases right
  cases equal
  rfl

noncomputable def runningCodec (fullShape : Phi81Relation.Shape) :
    Codec (Running fullShape) :=
  Codec.pullback
    (Codec.product (pointCodec shape.cubeVariables)
      (Codec.product
        (Codec.finFunction shape.runningCount bundleCodec)
        (Codec.product
          (Codec.finFunction shape.runningCount
            (publicInputCodec fullShape.publicWidth))
          (Codec.finFunction shape.runningCount evaluationCodec))))
    runningData runningData_injective

theorem runningCodec_width
    {fullShape : Phi81Relation.Shape}
    (contract : FullShapeContract fullShape) :
    (runningCodec fullShape).width = 83210 := by
  rw [runningCodec, codec_pullback_width, codec_product_width,
    pointCodec_width, codec_product_width, codec_finFunction_width,
    bundleCodec_width, codec_product_width, codec_finFunction_width,
    publicInputCodec_width, codec_finFunction_width, evaluationCodec_width]
  rw [contract.publicWidth]
  decide

theorem runningCodec_admissible
    {fullShape : Phi81Relation.Shape} (value : Running fullShape) :
    (runningCodec fullShape).Admissible value := by
  exact
    ⟨pointCodec_admissible value.point,
      (fun index => bundleCodec_admissible (value.commitments index)),
      (fun index => publicInputCodec_admissible (value.publicInputs index)),
      (fun index => evaluationCodec_admissible (value.evaluations index))⟩

/-! ## Verifier-key-selected running codec -/

abbrev EvaluationFor (rowVariables : Nat) :=
  EvaluationFamily K (shapeFor rowVariables)

noncomputable def evaluationCodecFor (rowVariables : Nat) :
    Codec (EvaluationFor rowVariables) :=
  Codec.finFunction (shapeFor rowVariables).matrixCount
    (Codec.finFunction (shapeFor rowVariables).coefficientCount kCodec)

@[simp] theorem evaluationCodecFor_width (rowVariables : Nat) :
    (evaluationCodecFor rowVariables).width = 1512 := by
  rfl

theorem evaluationCodecFor_admissible
    {rowVariables : Nat} (value : EvaluationFor rowVariables) :
    (evaluationCodecFor rowVariables).Admissible value := by
  intro matrix coefficient
  exact kCodec_admissible (value matrix coefficient)

def runningDataFor
    {rowVariables : Nat} {fullShape : Phi81Relation.Shape}
    (value : RunningFor rowVariables fullShape) :=
  (value.point,
    (value.commitments, (value.publicInputs, value.evaluations)))

theorem runningDataFor_injective
    {rowVariables : Nat} {fullShape : Phi81Relation.Shape} :
    Function.Injective
      (runningDataFor (rowVariables := rowVariables)
        (fullShape := fullShape)) := by
  intro left right equal
  cases left
  cases right
  cases equal
  rfl

noncomputable def runningCodecFor
    (rowVariables : Nat) (fullShape : Phi81Relation.Shape) :
    Codec (RunningFor rowVariables fullShape) :=
  Codec.pullback
    (Codec.product (pointCodec (shapeFor rowVariables).cubeVariables)
      (Codec.product
        (Codec.finFunction (shapeFor rowVariables).runningCount bundleCodec)
        (Codec.product
          (Codec.finFunction (shapeFor rowVariables).runningCount
            (publicInputCodec fullShape.publicWidth))
          (Codec.finFunction (shapeFor rowVariables).runningCount
            (evaluationCodecFor rowVariables)))))
    runningDataFor runningDataFor_injective

/-- Exact number of base-field coordinates in the complete running claim.
Only the two coordinates of each additional extension-field point coordinate
depend on the augmented-relation exponent. -/
def runningFieldCountFor (rowVariables : Nat) : Nat :=
  83160 + 2 * rowVariables

theorem runningCodecFor_width
    {rowVariables : Nat} {fullShape : Phi81Relation.Shape}
    (contract : FullShapeContractFor rowVariables fullShape) :
    (runningCodecFor rowVariables fullShape).width =
      runningFieldCountFor rowVariables := by
  rw [runningCodecFor, codec_pullback_width, codec_product_width,
    pointCodec_width, codec_product_width, codec_finFunction_width,
    bundleCodec_width, codec_product_width, codec_finFunction_width,
    publicInputCodec_width, codec_finFunction_width,
    evaluationCodecFor_width]
  rw [contract.publicWidth]
  change
    rowVariables * 2 +
        (14 * 3888 + (14 * 540 + 14 * 1512)) =
      83160 + 2 * rowVariables
  omega

theorem runningCodecFor_admissible
    {rowVariables : Nat} {fullShape : Phi81Relation.Shape}
    (value : RunningFor rowVariables fullShape) :
    (runningCodecFor rowVariables fullShape).Admissible value := by
  exact
    ⟨pointCodec_admissible value.point,
      (fun index => bundleCodec_admissible (value.commitments index)),
      (fun index => publicInputCodec_admissible (value.publicInputs index)),
      (fun index =>
        evaluationCodecFor_admissible (value.evaluations index))⟩

theorem runningFieldCountFor_25 : runningFieldCountFor 25 = 83210 := by
  decide

/-! ## Canonical bit image -/

def fieldBitWidth : Nat := 64
def runningFieldCount : Nat := 83210
def runningBitCount : Nat := runningFieldCount * fieldBitWidth

theorem runningBitCount_exact : runningBitCount = 5325440 := by decide

def fieldBits (value : F) : List Nat :=
  WasmStateCodec.encodeWord fieldBitWidth value.val

theorem fieldBits_length (value : F) :
    (fieldBits value).length = fieldBitWidth :=
  WasmStateCodec.encodeWord_length _ _

theorem fieldBits_binary (value : F) (digit : Nat)
    (member : digit ∈ fieldBits value) : digit < 2 :=
  WasmStateCodec.encodeWord_binary _ _ _ member

theorem fieldBits_injective : Function.Injective fieldBits := by
  intro left right equal
  apply Fin.ext
  apply WasmStateCodec.encodeWord_injective_of_bound
    (width := fieldBitWidth)
    (Nat.lt_trans left.isLt (by decide))
    (Nat.lt_trans right.isLt (by decide))
  exact equal

def fieldBlocks (fields : List F) : List (List Nat) :=
  fields.map fieldBits

def encodeFieldBits (fields : List F) : List Nat :=
  (fieldBlocks fields).flatten

theorem fieldBlocks_lengths (fields : List F) :
    (fieldBlocks fields).map List.length =
      List.replicate fields.length fieldBitWidth := by
  induction fields with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      change
        (fieldBits head).length :: (fieldBlocks tail).map List.length =
          List.replicate (head :: tail).length fieldBitWidth
      rw [fieldBits_length, List.length_cons]
      rw [inductionHypothesis]
      exact List.replicate_succ.symm

theorem encodeFieldBits_length (fields : List F) :
    (encodeFieldBits fields).length = fields.length * fieldBitWidth := by
  rw [encodeFieldBits, List.length_flatten, fieldBlocks_lengths]
  simp [fieldBitWidth]

theorem encodeFieldBits_binary (fields : List F) (digit : Nat)
    (member : digit ∈ encodeFieldBits fields) : digit < 2 := by
  rcases List.mem_flatten.mp member with ⟨block, blockMember, digitMember⟩
  rcases List.mem_map.mp blockMember with ⟨field, _fieldMember, rfl⟩
  exact fieldBits_binary field digit digitMember

theorem encodeFieldBits_injective : Function.Injective encodeFieldBits := by
  intro left right equal
  have lengths : left.length = right.length := by
    have equalLengths := congrArg List.length equal
    rw [encodeFieldBits_length, encodeFieldBits_length] at equalLengths
    exact Nat.eq_of_mul_eq_mul_right (by decide : 0 < fieldBitWidth)
      equalLengths
  have blocksEqual : fieldBlocks left = fieldBlocks right := by
    apply WasmResultCodec.flatten_injective_of_lengths
      (fieldBlocks_lengths left)
      (by simpa [lengths] using fieldBlocks_lengths right)
      equal
  exact (List.map_inj_right fieldBits_injective).mp blocksEqual

noncomputable def runningBits {fullShape : Phi81Relation.Shape}
    (value : Running fullShape) : List Nat :=
  encodeFieldBits ((runningCodec fullShape).encode value)

theorem runningBits_length
    {fullShape : Phi81Relation.Shape}
    (contract : FullShapeContract fullShape)
    (value : Running fullShape) :
    (runningBits value).length = runningBitCount := by
  rw [runningBits, encodeFieldBits_length,
    (runningCodec fullShape).encode_length, runningCodec_width contract]
  rfl

theorem runningBits_binary
    {fullShape : Phi81Relation.Shape} (value : Running fullShape)
    (digit : Nat) (member : digit ∈ runningBits value) : digit < 2 :=
  encodeFieldBits_binary _ _ member

theorem runningBits_injective {fullShape : Phi81Relation.Shape} :
    Function.Injective
      (runningBits : Running fullShape → List Nat) := by
  intro left right equal
  apply (runningCodec fullShape).encode_injective_of_admissible
    (runningCodec_admissible left) (runningCodec_admissible right)
  exact encodeFieldBits_injective equal

noncomputable def blockOfRunning
    {fullShape : Phi81Relation.Shape}
    (contract : FullShapeContract fullShape)
    (value : Running fullShape) : FixedBits.Word runningBitCount :=
  ⟨runningBits value, runningBits_length contract value,
    fun digit member => runningBits_binary value digit member⟩

theorem blockOfRunning_value
    {fullShape : Phi81Relation.Shape}
    (contract : FullShapeContract fullShape)
    (value : Running fullShape) :
    (blockOfRunning contract value).val =
      encodeFieldBits ((runningCodec fullShape).encode value) :=
  rfl

/-- Fail-closed semantic inverse of the exact canonical image. This definition
is suitable for the specification boundary. A release still needs a generated
executable parser that refines it. -/
noncomputable def decodeRunning
    {fullShape : Phi81Relation.Shape}
    (_contract : FullShapeContract fullShape)
    (block : FixedBits.Word runningBitCount) : Option (Running fullShape) :=
  letI : Decidable
      (∃ value : Running fullShape, runningBits value = block.val) :=
    Classical.propDecidable _
  if existsValue : ∃ value : Running fullShape, runningBits value = block.val then
    some (Classical.choose existsValue)
  else
    none

theorem decodeRunning_blockOfRunning
    {fullShape : Phi81Relation.Shape}
    (contract : FullShapeContract fullShape)
    (value : Running fullShape) :
    decodeRunning contract (blockOfRunning contract value) = some value := by
  let existsValue : ∃ candidate : Running fullShape,
      runningBits candidate = (blockOfRunning contract value).val :=
    ⟨value, rfl⟩
  rw [decodeRunning, dif_pos existsValue]
  apply congrArg some
  exact runningBits_injective (Classical.choose_spec existsValue)

theorem decodeRunning_success_reencodes
    {fullShape : Phi81Relation.Shape}
    (contract : FullShapeContract fullShape)
    {block : FixedBits.Word runningBitCount}
    {value : Running fullShape}
    (decoded : decodeRunning contract block = some value) :
    runningBits value = block.val := by
  unfold decodeRunning at decoded
  split at decoded
  next existsValue =>
    have chosen : Classical.choose existsValue = value :=
      Option.some.inj decoded
    rw [← chosen]
    exact Classical.choose_spec existsValue
  next noValue => contradiction

/-! ## Fresh claim from the exact envelope sections -/

def codecField
    (value : ShiftedTernary41V1.CanonicalGoldilocks) : F :=
  ⟨value.val, by
    simpa [ShiftedTernary41V1.modulus, goldilocksModulus] using value.property⟩

theorem codecField_injective : Function.Injective codecField := by
  intro left right equal
  apply Subtype.ext
  exact congrArg Fin.val equal

def codecBundle (bundle : CommitmentBundleCodec.Value) :
    ProductCommitmentAlgebra.BundleValue :=
  fun component row lane =>
    codecField (bundle component
      ⟨row.val * ringDegree + lane.val, by
        have rowLt := row.isLt
        have laneLt := lane.isLt
        change row.val < 18 at rowLt
        change lane.val < 54 at laneLt
        change row.val * 54 + lane.val < 972
        omega⟩)

theorem codecBundle_injective : Function.Injective codecBundle := by
  intro left right equal
  funext component coordinate
  let row : Fin ProductCommitmentAlgebra.Rank :=
    ⟨coordinate.val / ringDegree, by
      have coordinateLt := coordinate.isLt
      change coordinate.val < 972 at coordinateLt
      change coordinate.val / 54 < 18
      omega⟩
  let lane : Fin ringDegree :=
    ⟨coordinate.val % ringDegree, Nat.mod_lt _ (by decide)⟩
  have coordinateExact :
      row.val * ringDegree + lane.val = coordinate.val := by
    simpa [row, lane, Nat.mul_comm] using
      Nat.div_add_mod coordinate.val ringDegree
  have mappedCoordinate :
      (⟨row.val * ringDegree + lane.val, by
        change row.val * 54 + lane.val < 972
        have rowLt := row.isLt
        have laneLt := lane.isLt
        change row.val < 18 at rowLt
        change lane.val < 54 at laneLt
        omega⟩ : CommitmentBundleCodec.Coordinate) =
        coordinate := by
    apply Fin.ext
    exact coordinateExact
  have selected := congrFun (congrFun (congrFun equal component) row) lane
  change
    codecField (left component
        ⟨row.val * ringDegree + lane.val, by
          change row.val * 54 + lane.val < 972
          have rowLt := row.isLt
          have laneLt := lane.isLt
          change row.val < 18 at rowLt
          change lane.val < 54 at laneLt
          omega⟩) =
      codecField (right component
        ⟨row.val * ringDegree + lane.val, by
          change row.val * 54 + lane.val < 972
          have rowLt := row.isLt
          have laneLt := lane.isLt
          change row.val < 18 at rowLt
          change lane.val < 54 at laneLt
          omega⟩) at selected
  rw [mappedCoordinate] at selected
  exact codecField_injective selected

/-- Canonical protocol representation of one concrete algebra bundle. This
is the completeness direction of `codecBundle`: fresh-claim construction
must not choose an unrelated codec value for a computed commitment. -/
def protocolBundleOf
    (bundle : ProductCommitmentAlgebra.BundleValue) :
    CommitmentBundleCodec.Value :=
  fun component coordinate =>
    let row : Fin ProductCommitmentAlgebra.Rank :=
      ⟨coordinate.val / ringDegree, by
        have coordinateLt := coordinate.isLt
        change coordinate.val < 972 at coordinateLt
        change coordinate.val / 54 < 18
        omega⟩
    let lane : Fin ringDegree :=
      ⟨coordinate.val % ringDegree, Nat.mod_lt _ (by decide)⟩
    ⟨(bundle component row lane).val, by
      simpa [ShiftedTernary41V1.modulus, goldilocksModulus] using
        (bundle component row lane).isLt⟩

/-- Converting a concrete commitment to its protocol representation and
back recovers every component, row, and ring lane exactly. -/
theorem codecBundle_protocolBundleOf
    (bundle : ProductCommitmentAlgebra.BundleValue) :
    codecBundle (protocolBundleOf bundle) = bundle := by
  funext component row lane
  apply Fin.ext
  change
    (bundle component
      ⟨(row.val * ringDegree + lane.val) / ringDegree, _⟩
      ⟨(row.val * ringDegree + lane.val) % ringDegree, _⟩).val =
      (bundle component row lane).val
  have laneLt := lane.isLt
  have rowExact :
      (row.val * ringDegree + lane.val) / ringDegree = row.val := by
    norm_num [ringDegree] at laneLt ⊢
    omega
  have laneExact :
      (row.val * ringDegree + lane.val) % ringDegree = lane.val := by
    norm_num [ringDegree] at laneLt ⊢
    omega
  have rowFinExact :
      (⟨(row.val * ringDegree + lane.val) / ringDegree, by
        change (row.val * ringDegree + lane.val) / ringDegree <
          ProductCommitmentAlgebra.Rank
        rw [rowExact]
        exact row.isLt⟩ : Fin ProductCommitmentAlgebra.Rank) = row :=
    Fin.ext rowExact
  have laneFinExact :
      (⟨(row.val * ringDegree + lane.val) % ringDegree, by
        exact Nat.mod_lt _ (by decide)⟩ : Fin ringDegree) = lane :=
    Fin.ext laneExact
  rw [rowFinExact, laneFinExact]

/-- The protocol representation is also a left inverse on canonical codec
bundles. -/
theorem protocolBundleOf_codecBundle
    (bundle : CommitmentBundleCodec.Value) :
    protocolBundleOf (codecBundle bundle) = bundle := by
  apply codecBundle_injective
  exact codecBundle_protocolBundleOf _

def fieldOfBit (digit : Nat) : F :=
  if digit = 0 then 0 else 1

theorem fieldOfBit_injective_on_binary
    {left right : Nat} (leftBinary : left < 2) (rightBinary : right < 2)
    (equal : fieldOfBit left = fieldOfBit right) : left = right := by
  have valueEqual := congrArg Fin.val equal
  interval_cases left <;> interval_cases right <;>
    simp_all [fieldOfBit, goldilocksModulus]

def publicInputOf
    {fullShape : Phi81Relation.Shape}
    (contract : FullShapeContract fullShape)
    (word : FixedBits.Word MemoryBoundCcsPublic.coordinateCount) :
    PublicInput fullShape :=
  fun coordinate =>
    let index : Fin MemoryBoundCcsPublic.coordinateCount :=
      Fin.cast contract.publicWidth coordinate
    fieldOfBit (word.val.getD index.val 0)

theorem publicInputOf_injective
    {fullShape : Phi81Relation.Shape}
    (contract : FullShapeContract fullShape) :
    Function.Injective (publicInputOf contract) := by
  intro left right equal
  apply Subtype.ext
  apply List.ext_get
  · exact left.property.1.trans right.property.1.symm
  · intro index leftBound rightBound
    let wordIndex : Fin MemoryBoundCcsPublic.coordinateCount := ⟨index, by
      rw [← left.property.1]
      exact leftBound⟩
    let coordinate : Fin fullShape.publicWidth :=
      Fin.cast contract.publicWidth.symm wordIndex
    have selected := congrFun equal coordinate
    change
      fieldOfBit (left.val.getD wordIndex.val 0) =
        fieldOfBit (right.val.getD wordIndex.val 0) at selected
    have leftGetD : left.val.getD index 0 = left.val[index]'leftBound := by
      simp [List.getD_eq_getElem?_getD, leftBound]
    have rightGetD : right.val.getD index 0 = right.val[index]'rightBound := by
      simp [List.getD_eq_getElem?_getD, rightBound]
    rw [leftGetD, rightGetD] at selected
    exact fieldOfBit_injective_on_binary
      (left.property.2 _ (List.getElem_mem leftBound))
      (right.property.2 _ (List.getElem_mem rightBound)) selected

/-- Public input for a verifier-key-selected augmented-relation exponent. -/
def publicInputOfFor
    {rowVariables : Nat} {fullShape : Phi81Relation.Shape}
    (contract : FullShapeContractFor rowVariables fullShape)
    (word : FixedBits.Word MemoryBoundCcsPublic.coordinateCount) :
    PublicInput fullShape :=
  fun coordinate =>
    let index : Fin MemoryBoundCcsPublic.coordinateCount :=
      Fin.cast contract.publicWidth coordinate
    fieldOfBit (word.val.getD index.val 0)

theorem publicInputOfFor_injective
    {rowVariables : Nat} {fullShape : Phi81Relation.Shape}
    (contract : FullShapeContractFor rowVariables fullShape) :
    Function.Injective (publicInputOfFor contract) := by
  intro left right equal
  apply Subtype.ext
  apply List.ext_get
  · exact left.property.1.trans right.property.1.symm
  · intro index leftBound rightBound
    let wordIndex : Fin MemoryBoundCcsPublic.coordinateCount := ⟨index, by
      rw [← left.property.1]
      exact leftBound⟩
    let coordinate : Fin fullShape.publicWidth :=
      Fin.cast contract.publicWidth.symm wordIndex
    have selected := congrFun equal coordinate
    change
      fieldOfBit (left.val.getD wordIndex.val 0) =
        fieldOfBit (right.val.getD wordIndex.val 0) at selected
    have leftGetD : left.val.getD index 0 = left.val[index]'leftBound := by
      simp [List.getD_eq_getElem?_getD, leftBound]
    have rightGetD : right.val.getD index 0 = right.val[index]'rightBound := by
      simp [List.getD_eq_getElem?_getD, rightBound]
    rw [leftGetD, rightGetD] at selected
    exact fieldOfBit_injective_on_binary
      (left.property.2 _ (List.getElem_mem leftBound))
      (right.property.2 _ (List.getElem_mem rightBound)) selected

def freshOf
    {fullShape : Phi81Relation.Shape}
    (contract : FullShapeContract fullShape)
    (bundle : CommitmentBundleCodec.Value)
    (ccsPublic : FixedBits.Word MemoryBoundCcsPublic.coordinateCount) :
    Fresh fullShape where
  commitments := fun _ => codecBundle bundle
  publicInputs := fun _ => publicInputOf contract ccsPublic

@[simp] theorem freshOf_commitment
    {fullShape : Phi81Relation.Shape}
    (contract : FullShapeContract fullShape)
    (bundle : CommitmentBundleCodec.Value)
    (ccsPublic : FixedBits.Word MemoryBoundCcsPublic.coordinateCount) :
    (freshOf contract bundle ccsPublic).commitments
        ⟨0, by decide⟩ = codecBundle bundle :=
  rfl

@[simp] theorem freshOf_publicInput
    {fullShape : Phi81Relation.Shape}
    (contract : FullShapeContract fullShape)
    (bundle : CommitmentBundleCodec.Value)
    (ccsPublic : FixedBits.Word MemoryBoundCcsPublic.coordinateCount) :
    (freshOf contract bundle ccsPublic).publicInputs ⟨0, by decide⟩ =
      publicInputOf contract ccsPublic :=
  rfl

/-- The exact fresh paper claim retains both authority-bearing envelope
sections. No bundle or CCS-public alias can select the same fresh claim. -/
theorem freshOf_pair_injective
    {fullShape : Phi81Relation.Shape}
    (contract : FullShapeContract fullShape) :
    Function.Injective
      (fun value : CommitmentBundleCodec.Value ×
          FixedBits.Word MemoryBoundCcsPublic.coordinateCount =>
        freshOf contract value.1 value.2) := by
  intro left right equal
  apply Prod.ext
  · apply codecBundle_injective
    have selected := congrFun
      (congrArg (fun value : Fresh fullShape => value.commitments) equal)
      ⟨0, by decide⟩
    simpa using selected
  · apply publicInputOf_injective contract
    have selected := congrFun
      (congrArg (fun value : Fresh fullShape => value.publicInputs) equal)
      ⟨0, by decide⟩
    simpa using selected

/-- Fresh claim for a verifier-key-selected augmented-relation exponent. -/
def freshOfFor
    {rowVariables : Nat} {fullShape : Phi81Relation.Shape}
    (contract : FullShapeContractFor rowVariables fullShape)
    (bundle : CommitmentBundleCodec.Value)
    (ccsPublic : FixedBits.Word MemoryBoundCcsPublic.coordinateCount) :
    FreshFor rowVariables fullShape where
  commitments := fun _ => codecBundle bundle
  publicInputs := fun _ => publicInputOfFor contract ccsPublic

@[simp] theorem freshOfFor_commitment
    {rowVariables : Nat} {fullShape : Phi81Relation.Shape}
    (contract : FullShapeContractFor rowVariables fullShape)
    (bundle : CommitmentBundleCodec.Value)
    (ccsPublic : FixedBits.Word MemoryBoundCcsPublic.coordinateCount) :
    (freshOfFor contract bundle ccsPublic).commitments
        ⟨0, by simp [shapeFor]⟩ = codecBundle bundle :=
  rfl

@[simp] theorem freshOfFor_publicInput
    {rowVariables : Nat} {fullShape : Phi81Relation.Shape}
    (contract : FullShapeContractFor rowVariables fullShape)
    (bundle : CommitmentBundleCodec.Value)
    (ccsPublic : FixedBits.Word MemoryBoundCcsPublic.coordinateCount) :
    (freshOfFor contract bundle ccsPublic).publicInputs
        ⟨0, by simp [shapeFor]⟩ =
      publicInputOfFor contract ccsPublic :=
  rfl

theorem freshOfFor_pair_injective
    {rowVariables : Nat} {fullShape : Phi81Relation.Shape}
    (contract : FullShapeContractFor rowVariables fullShape) :
    Function.Injective
      (fun value : CommitmentBundleCodec.Value ×
          FixedBits.Word MemoryBoundCcsPublic.coordinateCount =>
        freshOfFor contract value.1 value.2) := by
  intro left right equal
  apply Prod.ext
  · apply codecBundle_injective
    have selected := congrFun
      (congrArg
        (fun value : FreshFor rowVariables fullShape => value.commitments)
        equal)
      ⟨0, by simp [shapeFor]⟩
    simpa using selected
  · apply publicInputOfFor_injective contract
    have selected := congrFun
      (congrArg
        (fun value : FreshFor rowVariables fullShape => value.publicInputs)
        equal)
      ⟨0, by simp [shapeFor]⟩
    simpa using selected

end Nightstream.Implementation.Nebula.ProductNifsCodec
