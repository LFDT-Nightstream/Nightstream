import Nightstream.Protocol.FPrime.ConcretePhi81.AccumulatorBinding.DelayedPending
import Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.Commitment

/-!
Exact typed field carrier for the fixed-point pending accumulator family.

Assurance tier: model-level representation refinement.

Owns: the exact domain/config header and field order for one verifier-fixed
Phi81 relation, one shared row point, one shared production column point, one
shared `mIn`, one shared four-lane fold digest, fourteen ordered typed child
payloads, and the optional delayed packed `yZcol` value. It proves fixed-width
encoding, injectivity, projection to `PendingFamilyPayload`, and the exact
fixed-profile field-count formulas for both the bounded κ=4 fixture and the
production κ=18 commitment width.

Does not own: the concrete Rust serializer, domain-prefix bytes, SIS or
Poseidon2, Ajtai binding, production wire provenance, R1CS rows, measured row
savings, or permission to remove the current per-child digests.

Emits constraints: no.

Authority boundary: relation structure is a verifier-owned argument to
`toPendingFamilyPayload`; it is not replaced by a digest. The retained child
carrier contains every commitment coefficient, all 270 public-input fields,
and all 13×54 extension-field evaluation coefficients. The shared column point
and pending state are retained explicitly even though they are outside the
paper CE payload. A concrete implementation may use this codec only after it
proves its children have exactly this typed shape and derives omitted cached or
canonical fields from accepted production rows.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `fprime.accumulator.pending_family.point` | retain the shared 23-coordinate row point | direct dataflow | `Carrier.point` |
| `fprime.accumulator.pending_family.column_point` | retain the shared 19-coordinate block point | direct dataflow | `Carrier.columnPoint` |
| `fprime.accumulator.pending_family.m_in` | retain the shared input width until accepted production rows derive 270 | direct dataflow | `Carrier.mIn` |
| `fprime.accumulator.pending_family.fold_digest` | retain the shared four-lane transcript cache until continuity derives it | direct dataflow | `Carrier.foldDigest` |
| `fprime.accumulator.pending_family.children` | retain 14 ordered commitment/public-input/evaluation payloads | authoritative typed payload | `Carrier.children`, `encodeChildren_injective` |
| `fprime.accumulator.pending_family.pending` | retain absence or the exact 19-coordinate old block and 54-lane parent vector | direct dataflow | `Carrier.pending`, `encodePending_injective` |
| `fprime.accumulator.pending_family.codec` | serialize the complete carrier without aliases | derived | `encodeCarrier_injective` |
| `fprime.accumulator.pending_family.count` | pin the κ=4/κ=18 field counts before physical lowering | computed | `bounded_field_count`, `production_field_count` |
-/

namespace Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.Accumulator.PendingFamilyCodec

open Nightstream.Protocol.FPrime.ConcretePhi81.AccumulatorBinding.DelayedPending
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.TranscriptAuthority.BlockLane

abbrev TypedCubePoint (variables : Nat) :=
  Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.CubePoint K variables

private theorem ofFn_injective
    {Item : Type}
    {count : Nat} :
    Function.Injective (List.ofFn : (Fin count -> Item) -> List Item) := by
  intro left right same
  funext index
  have reads := congrArg
    (fun values => values.getD index.val (left index)) same
  simpa [List.getD_eq_getElem?_getD, index.isLt] using reads

private def encodeBlocks {Item : Type}
    (encode : Item -> List F) : List Item -> List F
  | [] => []
  | item :: items => encode item ++ encodeBlocks encode items

private theorem encodeBlocks_length
    {Item : Type}
    (encode : Item -> List F)
    (blockWidth : Nat)
    (blockLength : forall item, (encode item).length = blockWidth)
    (items : List Item) :
    (encodeBlocks encode items).length = items.length * blockWidth := by
  induction items with
  | nil => simp [encodeBlocks]
  | cons item items inductionHypothesis =>
      simp [encodeBlocks, blockLength, inductionHypothesis, Nat.succ_mul,
        Nat.add_comm]

private theorem encodeBlocks_injective
    {Item : Type}
    (encode : Item -> List F)
    (blockWidth : Nat)
    (blockWidthPositive : 0 < blockWidth)
    (blockLength : forall item, (encode item).length = blockWidth)
    (blockInjective : Function.Injective encode) :
    Function.Injective (encodeBlocks encode) := by
  intro left
  induction left with
  | nil =>
      intro right same
      cases right with
      | nil => rfl
      | cons item items =>
          have lengths := congrArg List.length same
          rw [encodeBlocks_length encode blockWidth blockLength []] at lengths
          rw [encodeBlocks_length encode blockWidth blockLength
            (item :: items)] at lengths
          simp only [List.length_nil, Nat.zero_mul, List.length_cons,
            Nat.succ_mul] at lengths
          omega
  | cons leftHead leftTail inductionHypothesis =>
      intro right same
      cases right with
      | nil =>
          have lengths := congrArg List.length same
          rw [encodeBlocks_length encode blockWidth blockLength
            (leftHead :: leftTail)] at lengths
          rw [encodeBlocks_length encode blockWidth blockLength []] at lengths
          simp only [List.length_cons, Nat.succ_mul, List.length_nil,
            Nat.zero_mul] at lengths
          omega
      | cons rightHead rightTail =>
          have heads := congrArg (List.take blockWidth) same
          have tails := congrArg (List.drop blockWidth) same
          simp only [encodeBlocks] at heads tails
          have headFields : encode leftHead = encode rightHead := by
            simpa [blockLength] using heads
          have tailFields :
              encodeBlocks encode leftTail =
                encodeBlocks encode rightTail := by
            simpa [blockLength] using tails
          cases blockInjective headFields
          cases inductionHypothesis tailFields
          rfl

def encodeK (value : K) : List F :=
  [value.c0, value.c1]

@[simp] theorem encodeK_length (value : K) :
    (encodeK value).length = 2 := by
  rfl

theorem encodeK_injective : Function.Injective encodeK := by
  intro left right same
  cases left with
  | mk leftC0 leftC1 =>
      cases right with
      | mk rightC0 rightC1 =>
          have fields : leftC0 = rightC0 /\ leftC1 = rightC1 := by
            simpa [encodeK] using same
          cases fields.1
          cases fields.2
          rfl

def encodePoint {variables : Nat}
    (point : TypedCubePoint variables) : List F :=
  encodeBlocks encodeK point.coordinates

@[simp] theorem encodePoint_length {variables : Nat}
    (point : TypedCubePoint variables) :
    (encodePoint point).length = 2 * variables := by
  rw [encodePoint, encodeBlocks_length encodeK 2 encodeK_length,
    point.dimension]
  omega

theorem encodePoint_injective {variables : Nat} :
    Function.Injective
      (encodePoint : TypedCubePoint variables -> List F) := by
  intro left right same
  have coordinates : left.coordinates = right.coordinates :=
    encodeBlocks_injective encodeK 2 (by decide) encodeK_length
      encodeK_injective same
  cases left
  cases right
  cases coordinates
  rfl

def encodeRingF (value : RingF) : List F :=
  List.ofFn value

@[simp] theorem encodeRingF_length (value : RingF) :
    (encodeRingF value).length = ringDegree := by
  simp [encodeRingF]

theorem encodeRingF_injective : Function.Injective encodeRingF := by
  exact ofFn_injective

def encodeRingK (value : RingK) : List F :=
  encodeBlocks encodeK (List.ofFn value)

@[simp] theorem encodeRingK_length (value : RingK) :
    (encodeRingK value).length = 2 * ringDegree := by
  rw [encodeRingK, encodeBlocks_length encodeK 2 encodeK_length]
  simp
  omega

theorem encodeRingK_injective : Function.Injective encodeRingK := by
  intro left right same
  apply ofFn_injective
  exact encodeBlocks_injective encodeK 2 (by decide) encodeK_length
    encodeK_injective same

def encodeCommitment {verifierRows : Nat}
    (commitment : Commitment.Value verifierRows) : List F :=
  encodeBlocks encodeRingF (List.ofFn commitment)

def commitmentFieldCount (verifierRows : Nat) : Nat :=
  verifierRows * ringDegree

@[simp] theorem encodeCommitment_length {verifierRows : Nat}
    (commitment : Commitment.Value verifierRows) :
    (encodeCommitment commitment).length =
      commitmentFieldCount verifierRows := by
  rw [encodeCommitment,
    encodeBlocks_length encodeRingF ringDegree encodeRingF_length]
  simp [commitmentFieldCount]

theorem encodeCommitment_injective {verifierRows : Nat} :
    Function.Injective
      (encodeCommitment : Commitment.Value verifierRows -> List F) := by
  cases verifierRows with
  | zero =>
      intro left right _same
      funext row
      exact Fin.elim0 row
  | succ rows =>
      intro left right same
      apply ofFn_injective
      exact encodeBlocks_injective encodeRingF ringDegree (by decide)
        encodeRingF_length encodeRingF_injective same

def encodePublicInput {shape : Shape}
    (input : PublicInput shape) : List F :=
  List.ofFn input

@[simp] theorem encodePublicInput_length {shape : Shape}
    (input : PublicInput shape) :
    (encodePublicInput input).length = shape.publicWidth := by
  simp [encodePublicInput]

theorem encodePublicInput_injective {shape : Shape} :
    Function.Injective
      (encodePublicInput : PublicInput shape -> List F) := by
  exact ofFn_injective

def encodeEvaluations {shape : Shape}
    (evaluations : Fin shape.matrixCount -> RingK) : List F :=
  encodeBlocks encodeRingK (List.ofFn evaluations)

def evaluationFieldCount (shape : Shape) : Nat :=
  shape.matrixCount * (2 * ringDegree)

@[simp] theorem encodeEvaluations_length {shape : Shape}
    (evaluations : Fin shape.matrixCount -> RingK) :
    (encodeEvaluations evaluations).length = evaluationFieldCount shape := by
  rw [encodeEvaluations,
    encodeBlocks_length encodeRingK (2 * ringDegree) encodeRingK_length]
  simp [evaluationFieldCount]

theorem encodeEvaluations_injective {shape : Shape} :
    Function.Injective
      (encodeEvaluations :
        (Fin shape.matrixCount -> RingK) -> List F) := by
  intro left right same
  apply ofFn_injective
  exact encodeBlocks_injective encodeRingK (2 * ringDegree) (by
      simp [ringDegree]) encodeRingK_length encodeRingK_injective same

/-- Exact typed child payload retained by the production family serializer.
Array lengths are eliminated at this layer by giving matrix order a type. -/
structure Child (shape : Shape) (verifierRows : Nat) where
  commitment : Commitment.Value verifierRows
  publicInput : PublicInput shape
  evaluations : Fin shape.matrixCount -> RingK

namespace Child

def toPiDecPayload {shape : Shape} {verifierRows : Nat}
    (child : Child shape verifierRows) :
    PiDecChildPayload shape (Commitment.Value verifierRows) where
  commitment := child.commitment
  publicInput := child.publicInput
  evaluations := Array.ofFn child.evaluations

end Child

def encodeChild {shape : Shape} {verifierRows : Nat}
    (child : Child shape verifierRows) : List F :=
  encodeCommitment child.commitment ++
    encodePublicInput child.publicInput ++
    encodeEvaluations child.evaluations

def childFieldCount (shape : Shape) (verifierRows : Nat) : Nat :=
  commitmentFieldCount verifierRows + shape.publicWidth +
    evaluationFieldCount shape

@[simp] theorem encodeChild_length {shape : Shape} {verifierRows : Nat}
    (child : Child shape verifierRows) :
    (encodeChild child).length = childFieldCount shape verifierRows := by
  simp [encodeChild, childFieldCount]
  omega

theorem encodeChild_injective {shape : Shape} {verifierRows : Nat} :
    Function.Injective
      (encodeChild : Child shape verifierRows -> List F) := by
  intro left right same
  let commitmentWidth := commitmentFieldCount verifierRows
  let publicWidth := shape.publicWidth
  have commitmentFields := congrArg (List.take commitmentWidth) same
  have afterCommitment := congrArg (List.drop commitmentWidth) same
  have publicFields := congrArg (List.take publicWidth) afterCommitment
  have evaluationFields := congrArg (List.drop publicWidth) afterCommitment
  have commitmentEq : left.commitment = right.commitment := by
    apply encodeCommitment_injective
    simpa [encodeChild, commitmentWidth] using commitmentFields
  have publicEq : left.publicInput = right.publicInput := by
    apply encodePublicInput_injective
    simpa [encodeChild, commitmentWidth, publicWidth] using publicFields
  have evaluationsEq : left.evaluations = right.evaluations := by
    apply encodeEvaluations_injective
    simpa [encodeChild, commitmentWidth, publicWidth] using evaluationFields
  cases left
  cases right
  cases commitmentEq
  cases publicEq
  cases evaluationsEq
  rfl

def encodeChildren {shape : Shape} {verifierRows count : Nat}
    (children : Fin count -> Child shape verifierRows) : List F :=
  encodeBlocks encodeChild (List.ofFn children)

@[simp] theorem encodeChildren_length
    {shape : Shape} {verifierRows count : Nat}
    (children : Fin count -> Child shape verifierRows) :
    (encodeChildren children).length =
      count * childFieldCount shape verifierRows := by
  rw [encodeChildren,
    encodeBlocks_length encodeChild (childFieldCount shape verifierRows)
      encodeChild_length]
  simp

theorem encodeChildren_injective
    {shape : Shape} {verifierRows count : Nat}
    (publicPositive : 0 < shape.publicWidth) :
    Function.Injective
      (encodeChildren :
        (Fin count -> Child shape verifierRows) -> List F) := by
  cases count with
  | zero =>
      intro left right _same
      funext child
      exact Fin.elim0 child
  | succ children =>
      intro left right same
      apply ofFn_injective
      exact encodeBlocks_injective encodeChild
        (childFieldCount shape verifierRows) (by
          simp [childFieldCount, evaluationFieldCount, ringDegree]
          omega)
        encodeChild_length encodeChild_injective same

def pendingPayloadFieldCount : Nat :=
  2 * PiCcsDomains.fixedPointProduction.nc.blockVariables + 2 * ringDegree

def encodePendingPayload
    (pending : ProductionDelayedBlockLane) : List F :=
  encodePoint pending.oldBlock ++ encodeRingK pending.parentYZcol

@[simp] theorem encodePendingPayload_length
    (pending : ProductionDelayedBlockLane) :
    (encodePendingPayload pending).length = pendingPayloadFieldCount := by
  simp [encodePendingPayload, pendingPayloadFieldCount]

theorem encodePendingPayload_injective :
    Function.Injective encodePendingPayload := by
  intro left right same
  let pointWidth := 2 * PiCcsDomains.fixedPointProduction.nc.blockVariables
  have pointFields := congrArg (List.take pointWidth) same
  have ringFields := congrArg (List.drop pointWidth) same
  have pointEq : left.oldBlock = right.oldBlock := by
    apply encodePoint_injective
    simpa [encodePendingPayload, pointWidth] using pointFields
  have ringEq : left.parentYZcol = right.parentYZcol := by
    apply encodeRingK_injective
    simpa [encodePendingPayload, pointWidth] using ringFields
  exact ProductionDelayedBlockLane.ext left right pointEq ringEq

/-- Fixed-width option encoding used by the recursive circuit. `none` keeps
the same physical slots as `some`; the leading discriminator prevents an
all-zero payload from aliasing absence. -/
def encodePending : Option ProductionDelayedBlockLane -> List F
  | none => (0 : F) :: List.replicate pendingPayloadFieldCount (0 : F)
  | some pending => (1 : F) :: encodePendingPayload pending

def pendingFieldCount : Nat := 1 + pendingPayloadFieldCount

@[simp] theorem encodePending_length
    (pending : Option ProductionDelayedBlockLane) :
    (encodePending pending).length = pendingFieldCount := by
  cases pending <;> simp [encodePending, pendingFieldCount]
  all_goals omega

theorem encodePending_injective : Function.Injective encodePending := by
  intro left right same
  cases left with
  | none =>
      cases right with
      | none => rfl
      | some right =>
          have heads := congrArg List.head? same
          simp [encodePending] at heads
          have modulusNeOne : goldilocksModulus ≠ 1 := by decide
          exact False.elim (modulusNeOne heads)
  | some left =>
      cases right with
      | none =>
          have heads := congrArg List.head? same
          simp [encodePending] at heads
          have modulusNeOne : goldilocksModulus ≠ 1 := by decide
          exact False.elim (modulusNeOne heads)
      | some right =>
          have tails := congrArg (List.drop 1) same
          have payloadEq :
              encodePendingPayload left = encodePendingPayload right := by
            simpa [encodePending] using tails
          cases encodePendingPayload_injective payloadEq
          rfl

/-- Rust-shaped direct carrier. The verifier-owned relation structure is an
argument to the semantic projection and never becomes digest authority. -/
structure Carrier (shape : Shape) (verifierRows count : Nat) where
  point : Point shape
  columnPoint : TypedCubePoint
    PiCcsDomains.fixedPointProduction.nc.blockVariables
  mIn : F
  foldDigest : Fin 4 -> F
  children : Fin count -> Child shape verifierRows
  pending : Option ProductionDelayedBlockLane

namespace Carrier

def toPendingFamilyPayload
    {shape : Shape} {verifierRows count : Nat}
    (system : Structure shape)
    (carrier : Carrier shape verifierRows count) :
    PendingFamilyPayload shape (Commitment.Value verifierRows) where
  family := {
    constraintSystem := system
    point := carrier.point
    children := List.ofFn fun child =>
      (carrier.children child).toPiDecPayload
  }
  pending := carrier.pending

end Carrier

def encodeCarrier
    {shape : Shape} {verifierRows count : Nat}
    (carrier : Carrier shape verifierRows count) : List F :=
  encodePoint carrier.point ++
    encodePoint carrier.columnPoint ++
    [carrier.mIn] ++
    List.ofFn carrier.foldDigest ++
    encodeChildren carrier.children ++
    encodePending carrier.pending

def carrierFieldCount
    (shape : Shape) (verifierRows count : Nat) : Nat :=
  2 * shape.rowVariables +
    2 * PiCcsDomains.fixedPointProduction.nc.blockVariables +
    1 +
    4 +
    count * childFieldCount shape verifierRows + pendingFieldCount

@[simp] theorem encodeCarrier_length
    {shape : Shape} {verifierRows count : Nat}
    (carrier : Carrier shape verifierRows count) :
    (encodeCarrier carrier).length =
      carrierFieldCount shape verifierRows count := by
  simp [encodeCarrier, carrierFieldCount]
  omega

theorem encodeCarrier_injective
    {shape : Shape} {verifierRows count : Nat}
    (publicPositive : 0 < shape.publicWidth) :
    Function.Injective
      (encodeCarrier : Carrier shape verifierRows count -> List F) := by
  intro left right same
  let pointWidth := 2 * shape.rowVariables
  let columnWidth := 2 * PiCcsDomains.fixedPointProduction.nc.blockVariables
  let mInWidth := 1
  let foldWidth := 4
  let childrenWidth := count * childFieldCount shape verifierRows
  have pointFields := congrArg (List.take pointWidth) same
  have afterPoint := congrArg (List.drop pointWidth) same
  have columnFields := congrArg (List.take columnWidth) afterPoint
  have afterColumn := congrArg (List.drop columnWidth) afterPoint
  have mInFields := congrArg (List.take mInWidth) afterColumn
  have afterMIn := congrArg (List.drop mInWidth) afterColumn
  have foldFields := congrArg (List.take foldWidth) afterMIn
  have afterFold := congrArg (List.drop foldWidth) afterMIn
  have childrenFields := congrArg (List.take childrenWidth) afterFold
  have pendingFields := congrArg (List.drop childrenWidth) afterFold
  have pointEq : left.point = right.point := by
    apply encodePoint_injective
    simpa [encodeCarrier, pointWidth] using pointFields
  have columnEq : left.columnPoint = right.columnPoint := by
    apply encodePoint_injective
    simpa [encodeCarrier, pointWidth, columnWidth] using columnFields
  have mInEq : left.mIn = right.mIn := by
    simpa [encodeCarrier, pointWidth, columnWidth, mInWidth] using mInFields
  have foldEq : left.foldDigest = right.foldDigest := by
    apply ofFn_injective
    simpa [encodeCarrier, pointWidth, columnWidth, mInWidth, foldWidth] using
      foldFields
  have childrenEq : left.children = right.children := by
    apply encodeChildren_injective publicPositive
    simpa [encodeCarrier, pointWidth, columnWidth, mInWidth, foldWidth,
      childrenWidth] using childrenFields
  have pendingEq : left.pending = right.pending := by
    apply encodePending_injective
    simpa [encodeCarrier, pointWidth, columnWidth, mInWidth, foldWidth,
      childrenWidth] using pendingFields
  cases left
  cases right
  cases pointEq
  cases columnEq
  cases mInEq
  cases foldEq
  cases childrenEq
  cases pendingEq
  rfl

/-- Equal field encodings recover both the existing exact semantic payload and
the production-only shared column point. -/
theorem semanticPayload_columnPoint_eq_of_encode_eq
    {shape : Shape} {verifierRows count : Nat}
    (system : Structure shape)
    (left right : Carrier shape verifierRows count)
    (publicPositive : 0 < shape.publicWidth)
    (same : encodeCarrier left = encodeCarrier right) :
    left.toPendingFamilyPayload system =
        right.toPendingFamilyPayload system /\
      left.columnPoint = right.columnPoint /\
      left.mIn = right.mIn /\
      left.foldDigest = right.foldDigest := by
  cases encodeCarrier_injective publicPositive same
  exact ⟨rfl, rfl, rfl, rfl⟩

/-- A caller-owned, domain-separated prefix may be prepended without changing
the exact carrier or creating a serialization alias. -/
def encodeWithPrefix
    {shape : Shape} {verifierRows count : Nat}
    (header : List F)
    (carrier : Carrier shape verifierRows count) : List F :=
  header ++ encodeCarrier carrier

@[simp] theorem encodeWithPrefix_length
    {shape : Shape} {verifierRows count : Nat}
    (header : List F)
    (carrier : Carrier shape verifierRows count) :
    (encodeWithPrefix header carrier).length =
      header.length + carrierFieldCount shape verifierRows count := by
  simp [encodeWithPrefix]

theorem encodeWithPrefix_injective
    {shape : Shape} {verifierRows count : Nat}
    (header : List F)
    (publicPositive : 0 < shape.publicWidth) :
    Function.Injective
      (encodeWithPrefix header :
        Carrier shape verifierRows count -> List F) := by
  intro left right same
  apply encodeCarrier_injective publicPositive
  have tails := congrArg (List.drop header.length) same
  simpa [encodeWithPrefix] using tails

/-! Fixed production-profile header and arithmetic. The byte-domain encoding
matches Rust's `pack_bytes_as_fields`: byte length first, then little-endian
seven-byte limbs. The remaining fields pin the child count and the two shared
extension-point lengths. Physical Rust/R1CS refinement remains outside this
model-level leaf. -/

def productionDomainTagFields : List F :=
  [59,
    30521782141150574,
    31069335676202596,
    30796712693949999,
    30809842190987109,
    13355146924878965,
    29113321536775536,
    34177672244455007,
    32777976662287455,
    3241519]

def productionHeader : List F :=
  productionDomainTagFields ++ [14, 23, 19]

@[simp] theorem productionDomainTagFields_length :
    productionDomainTagFields.length = 10 := by
  rfl

@[simp] theorem productionHeader_length :
    productionHeader.length = 13 := by
  rfl

def encodeProductionCarrier
    {shape : Shape} {verifierRows : Nat}
    (carrier : Carrier shape verifierRows 14) : List F :=
  productionHeader ++ encodeCarrier carrier

@[simp] theorem encodeProductionCarrier_length
    {shape : Shape} {verifierRows : Nat}
    (carrier : Carrier shape verifierRows 14) :
    (encodeProductionCarrier carrier).length =
      productionHeader.length + carrierFieldCount shape verifierRows 14 := by
  simp [encodeProductionCarrier]

theorem encodeProductionCarrier_injective
    {shape : Shape} {verifierRows : Nat}
    (publicPositive : 0 < shape.publicWidth) :
    Function.Injective
      (encodeProductionCarrier :
        Carrier shape verifierRows 14 -> List F) := by
  simpa [encodeProductionCarrier] using
    (encodeWithPrefix_injective productionHeader publicPositive)

theorem fixed_child_field_count
    (shape : Shape)
    (publicWidth : shape.publicWidth = 270)
    (matrixCount : shape.matrixCount = 13)
    (verifierRows : Nat) :
    childFieldCount shape verifierRows =
      54 * verifierRows + 1674 := by
  simp [childFieldCount, commitmentFieldCount, evaluationFieldCount,
    ringDegree, publicWidth, matrixCount]
  omega

theorem fixed_carrier_field_count
    (shape : Shape)
    (rowVariables : shape.rowVariables = 23)
    (publicWidth : shape.publicWidth = 270)
    (matrixCount : shape.matrixCount = 13)
    (verifierRows : Nat) :
    productionHeader.length +
        carrierFieldCount shape verifierRows 14 =
      14 * (54 * verifierRows) + 23685 := by
  simp [carrierFieldCount, pendingFieldCount,
    pendingPayloadFieldCount, childFieldCount, commitmentFieldCount,
    evaluationFieldCount, ringDegree, rowVariables, publicWidth, matrixCount,
    PiCcsDomains.fixedPointProduction, Domains.nc]
  omega

theorem bounded_field_count
    (shape : Shape)
    (rowVariables : shape.rowVariables = 23)
    (publicWidth : shape.publicWidth = 270)
    (matrixCount : shape.matrixCount = 13) :
    productionHeader.length + carrierFieldCount shape 4 14 = 26709 := by
  rw [fixed_carrier_field_count shape rowVariables publicWidth matrixCount 4]

theorem production_field_count
    (shape : Shape)
    (rowVariables : shape.rowVariables = 23)
    (publicWidth : shape.publicWidth = 270)
    (matrixCount : shape.matrixCount = 13) :
    productionHeader.length + carrierFieldCount shape 18 14 = 37293 := by
  rw [fixed_carrier_field_count shape rowVariables publicWidth matrixCount 18]

/-- Current conservative two-level encoding count, before physical lowering.
This is deliberately only a comparison formula until a generated Rust census
proves that the production serializer has exactly this preimage. -/
def conservativeFamilyFieldCount (verifierRows : Nat) : Nat :=
  14 * (54 * verifierRows) + 29436

theorem pendingFamily_field_saving (verifierRows : Nat) :
    conservativeFamilyFieldCount verifierRows -
        (14 * (54 * verifierRows) + 23685) = 5751 := by
  simp [conservativeFamilyFieldCount]

end Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.Accumulator.PendingFamilyCodec
