import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.PiRlc.RingTransport
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.PiDec.PublicInputBridge
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.RelabeledCarrier
import Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.Assignment
import Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.PublicInput

/-!
Typed 270-coordinate public-input refinement for production `Pi_RLC`.

Assurance tier: model-level. The theorems connect independent Lean carriers;
they do not establish Rust-conformant row emission.

Protocol: SuperNeo `Pi_RLC` inside the fixed F' NIFS.
Phase: five-block projection decoding and public-input ring action.
Constraint family: the five active `X` rings only; this file emits no rows.

Owns: direct decoding of five 54-lane projection rings into the independent
typed Phi81 public carrier; exact agreement with production's lane-major
packed columns; equality between the production `phi81Combine` operation and
the typed public-input action; the typed public-input equation implied by the
exact projection-reduction artifact; and identification of that output with
the typed `Pi_DEC` parent.

Does not own: Fiat--Shamir challenge derivation, strong-set security,
production R1CS-row soundness, fresh zero-tail authority, private CE openings,
commitments, evaluations, costs, or row removal.

Emits constraints: no. It interprets existing projection values.

Authority boundary: the typed equation is derived from the exact per-leaf
projection identity and executable Phi81 multiplication. It does not consume
the generic `AlgebraRefinement.x` field and does not define correctness as
whatever a caller-supplied codec accepts.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_rlc.verify.identities.x` | five 54-lane blocks cover the typed 270-coordinate carrier | computed | `decodeXRings` |
| `nifs.pi_rlc.verify.identities.x` | one decoded block is exactly `ringOfList` | derived | `publicBlock_decodeXRings` |
| `nifs.pi_rlc.verify.identities.x` | lane-major packed columns decode to the same 270 logical values | derived | `packXRings`, `decode_packXRings`, `decode_assembledX`, `decode_codec_x` |
| `nifs.pi_rlc.verify.identities.x` | `phi81Combine` equals the typed finite public-input action | derived | `decodeXRings_phi81Combine` |
| `nifs.pi_rlc.verify.identities.x` | exact projection identities imply typed output combination | derived | `typedPublicInputEquation_of_refinement` |
| `nifs.pi_rlc.verify.fold_wires.x` | active-X widths and X-parent column equality identify the typed parent without other carrier facts | derived | `typedOutput_eq_parent_of_wiring` |
| `nifs.pi_rlc.verify.fold_wires.x` | local-layout `Pi_RLC` combination equals typed `Pi_DEC` child recomposition | derived | `typedPiRlcPiDecPublicInputComposition` |
| `nifs.pi_rlc.verify.fold_wires.x` | globally mapped `Pi_RLC` parent equals locally decoded strict-`Pi_DEC` parent | derived | `typedPiRlcPiDecPublicInputComposition_relabel` |
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.PublicInputBridge

set_option maxRecDepth 4096

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270
open Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra
open Nightstream.Implementation.R1CS.ProjectionProgram
open Nightstream.Implementation.R1CS.FPrimeFullHistoryPiDec
open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper
open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc
open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.RingTransport
open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.RelabeledCarrier

namespace PiDecBridge

export Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiDec.PublicInputBridge
  (decode decode_injective_of_length decodedPublicInput
    strictAccepted_typedPublicInputEquation)

end PiDecBridge

/-- Convert a typed public block index to the fixed five-block production
domain. -/
def productionBlock {dimensions : Dimensions}
    (block : Fin dimensions.shape.publicRingColumns) : Fin 5 :=
  ⟨block.val, by simpa using block.isLt⟩

/-- Read the five production `X` rings in logical block-major order. -/
def decodeXRings {dimensions : Dimensions}
    (rings : XRings) : Phi81Relation.PublicInput dimensions.shape :=
  fun column =>
    (rings (productionBlock
      (PublicInput.publicBlockIndex dimensions.shape column))).getD
        (PublicInput.publicLaneIndex column).val 0

@[simp] theorem decodeXRings_apply {dimensions : Dimensions}
    (rings : XRings) (column : Fin dimensions.shape.publicWidth) :
    decodeXRings rings column =
      (rings (productionBlock
        (PublicInput.publicBlockIndex dimensions.shape column))).getD
          (PublicInput.publicLaneIndex column).val 0 := by
  rfl

/-! ## Lane-major production packing -/

private theorem getD_ofFn {Carrier : Type} {size : Nat}
    (items : Fin size -> Carrier) (index : Fin size) (default : Carrier) :
    (List.ofFn items).getD index.val default = items index := by
  rw [List.getD_eq_getElem?_getD,
    List.getElem?_eq_getElem (by simp), List.getElem_ofFn]
  simp

private theorem getD_flatten_ofFn_ofFn
    {Carrier : Type} {outer inner : Nat}
    (items : Fin outer -> Fin inner -> Carrier)
    (outerIndex : Fin outer) (innerIndex : Fin inner) (default : Carrier) :
    ((List.ofFn fun outerPosition =>
        List.ofFn fun innerPosition =>
          items outerPosition innerPosition).flatten).getD
      (outerIndex.val * inner + innerIndex.val) default =
        items outerIndex innerIndex := by
  induction outer with
  | zero => exact Fin.elim0 outerIndex
  | succ outer inductionHypothesis =>
      refine Fin.cases ?_ (fun index => ?_) outerIndex
      · simp only [List.ofFn_succ, List.flatten_cons, Fin.val_zero,
          Nat.zero_mul, Nat.zero_add]
        simp only [List.getD_eq_getElem?_getD]
        rw [List.getElem?_append_left (by simp)]
        change (List.ofFn (items 0)).getD innerIndex.val default = _
        exact getD_ofFn _ innerIndex default
      · simp only [List.ofFn_succ, List.flatten_cons, Fin.val_succ]
        simp only [List.getD_eq_getElem?_getD]
        rw [List.getElem?_append_right (by
          simp only [List.length_ofFn]
          rw [Nat.add_mul, Nat.one_mul]
          omega)]
        simp only [List.length_ofFn]
        change ((List.ofFn fun outerPosition =>
          List.ofFn fun innerPosition =>
            items outerPosition.succ innerPosition).flatten).getD
          ((index.val + 1) * inner + innerIndex.val - inner) default = _
        have indexArithmetic :
            (index.val + 1) * inner + innerIndex.val - inner =
              index.val * inner + innerIndex.val := by
          rw [Nat.add_mul, Nat.one_mul]
          omega
        rw [indexArithmetic]
        exact inductionHypothesis
          (fun outerPosition innerPosition =>
            items outerPosition.succ innerPosition)
          index

theorem assembleXColumns_length {matrixCount : Nat}
    (columns : ProjectionColumns matrixCount) :
    (assembleXColumns columns).length = alignedPublicWidth := by
  simp [assembleXColumns, alignedPublicWidth, ringDegree,
    publicRingColumns]

/-- Exact coordinate read from the lane-major production `X` layout. -/
theorem assembleXColumns_getD
    {matrixCount : Nat} (columns : ProjectionColumns matrixCount)
    (lane : Fin ringDegree) (block : Fin 5) :
    (assembleXColumns columns).getD
        (lane.val * 5 + block.val) 0 =
      (columns.x block).getD lane.val 0 := by
  exact getD_flatten_ofFn_ofFn
    (fun lane block => (columns.x block).getD lane.val 0)
    lane block 0

private theorem values_assembleXColumns_getD
    {matrixCount : Nat}
    (assignment : Nat -> Nat) (columns : ProjectionColumns matrixCount)
    (lane : Fin ringDegree) (block : Fin 5) :
    (values assignment (assembleXColumns columns)).getD
        (lane.val * 5 + block.val) 0 =
      residue (assignment ((columns.x block).getD lane.val 0)) := by
  have slotLt : lane.val * 5 + block.val <
      (assembleXColumns columns).length := by
    rw [assembleXColumns_length]
    have laneLt := lane.isLt
    have blockLt := block.isLt
    simp only [alignedPublicWidth, ringDegree, publicRingColumns] at laneLt ⊢
    omega
  unfold values
  rw [List.getD_eq_getElem?_getD, List.getElem?_map,
    List.getElem?_eq_getElem slotLt]
  simp only [Option.map_some, Option.getD_some]
  apply congrArg (fun column => residue (assignment column))
  rw [List.getElem_eq_getD]
  exact assembleXColumns_getD columns lane block

theorem values_getD_of_length
    (assignment : Nat -> Nat) (columns : List Nat)
    (width : columns.length = ringDegree) (lane : Fin ringDegree) :
    (values assignment columns).getD lane.val 0 =
      residue (assignment (columns.getD lane.val 0)) := by
  have laneLt : lane.val < columns.length := by
    rw [width]
    exact lane.isLt
  unfold values
  rw [List.getD_eq_getElem?_getD, List.getElem?_map,
    List.getElem?_eq_getElem laneLt]
  simp only [Option.map_some, Option.getD_some]
  apply congrArg (fun column => residue (assignment column))
  rw [List.getElem_eq_getD]

/-- Canonical lane-major serialization of five value-level production rings. -/
def packXRings (rings : XRings) : PackedPublicInput :=
  PackedPublicInput.mk
    ((List.ofFn fun lane : Fin ringDegree =>
      List.ofFn fun block : Fin 5 =>
        (rings block).getD lane.val 0).flatten)

@[simp] theorem packXRings_length (rings : XRings) :
    (packXRings rings).data.length = alignedPublicWidth := by
  simp [packXRings, alignedPublicWidth, ringDegree, publicRingColumns]

/-- The canonical lane-major serializer and direct five-ring typed decoder
are exact inverses at the typed observation boundary. -/
theorem decode_packXRings {dimensions : Dimensions} (rings : XRings) :
    PiDecBridge.decode (dimensions := dimensions) (packXRings rings) =
      decodeXRings (dimensions := dimensions) rings := by
  funext column
  let block := productionBlock
    (PublicInput.publicBlockIndex dimensions.shape column)
  let lane := PublicInput.publicLaneIndex column
  change ((List.ofFn fun lane : Fin ringDegree =>
      List.ofFn fun block : Fin 5 =>
        (rings block).getD lane.val 0).flatten).getD
        ((column.val % ringDegree) * 5 + column.val / ringDegree) 0 =
    (rings block).getD lane.val 0
  have slotShape :
      (column.val % ringDegree) * 5 + column.val / ringDegree =
        lane.val * 5 + block.val := by
    rfl
  rw [slotShape]
  exact getD_flatten_ofFn_ofFn
    (fun lane block => (rings block).getD lane.val 0) lane block 0

/-- The direct five-ring decoder agrees exactly with the lane-major packed
production layout. The width premise excludes every default read from an
accepted artifact. -/
theorem decode_assembledX {dimensions : Dimensions}
    {matrixCount : Nat}
    (assignment : Nat -> Nat) (columns : ProjectionColumns matrixCount)
    (width : forall block, (columns.x block).length = ringDegree) :
    PiDecBridge.decode (dimensions := dimensions)
        (PackedPublicInput.mk (values assignment (assembleXColumns columns))) =
      decodeXRings (dimensions := dimensions)
        (decodeOpening assignment columns).x := by
  funext column
  let block := productionBlock
    (PublicInput.publicBlockIndex dimensions.shape column)
  let lane := PublicInput.publicLaneIndex column
  change (values assignment (assembleXColumns columns)).getD
      ((column.val % ringDegree) * 5 + column.val / ringDegree) 0 =
    (values assignment (columns.x block)).getD lane.val 0
  have slotShape :
      (column.val % ringDegree) * 5 + column.val / ringDegree =
        lane.val * 5 + block.val := by
    rfl
  rw [slotShape, values_assembleXColumns_getD,
    values_getD_of_length assignment (columns.x block) (width block) lane]

/-- The deterministic production codec and the direct typed decoder agree.
Only the codec's concrete layout theorem and exact block widths are used. -/
theorem decode_codec_x {dimensions : Dimensions}
    {matrixCount : Nat}
    (codec : CarrierCodec matrixCount) (assignment : Nat -> Nat)
    (columns : ProjectionColumns matrixCount)
    (codecLayout : CodecArtifact codec)
    (width : forall block, (columns.x block).length = ringDegree) :
    PiDecBridge.decode (dimensions := dimensions)
        (codec.x.encode (decodeOpening assignment columns).x) =
      decodeXRings (dimensions := dimensions)
        (decodeOpening assignment columns).x := by
  rw [codecLayout.x assignment columns width]
  exact decode_assembledX assignment columns width

/-- A decoded typed block is exactly the production list interpreted as one
Phi81 ring. No width premise is needed because both sides use the same total
out-of-range-zero interpretation. -/
theorem publicBlock_decodeXRings {dimensions : Dimensions}
    (rings : XRings) (block : Fin dimensions.shape.publicRingColumns) :
    PublicInput.publicBlock (decodeXRings rings) block =
      ringOfList (rings (productionBlock block)) := by
  funext lane
  unfold PublicInput.publicBlock decodeXRings ringOfList
  congr 1
  · apply congrArg rings
    apply Fin.ext
    simp only [productionBlock, PublicInput.publicBlockIndex,
      Dimensions.shape, ringDegree, publicRingColumns]
    omega
  · congr
    apply Fin.ext
    change (block.val * ringDegree + lane.val) % ringDegree = lane.val
    rw [Nat.mul_add_mod_self_right, Nat.mod_eq_of_lt lane.isLt]

/-- The list-level production combination and the typed public-input action
are the same function on all 270 logical coordinates. -/
theorem decodeXRings_phi81Combine
    {dimensions : Dimensions} {count : Nat}
    (challenges : Fin count -> Ring)
    (inputs : Fin count -> XRings) :
    decodeXRings (dimensions := dimensions)
        (fun block => phi81Combine challenges (fun index => inputs index block)) =
      PublicInput.combinePublicInputs
        (fun index => ringOfList (challenges index))
        (fun index => decodeXRings (dimensions := dimensions) (inputs index)) := by
  induction count with
  | zero =>
      funext column
      simp [decodeXRings, phi81Combine,
        PublicInput.combinePublicInputs, PublicInput.publicZero]
  | succ count inductionHypothesis =>
      funext column
      let block := PublicInput.publicBlockIndex dimensions.shape column
      let lane := PublicInput.publicLaneIndex column
      change ringOfList
          (phi81Combine challenges fun index =>
            inputs index
              (productionBlock
                (PublicInput.publicBlockIndex dimensions.shape column)))
          (PublicInput.publicLaneIndex column) = _
      rw [phi81Combine_coefficient]
      simp only [List.ofFn_succ, scalarSum,
        PublicInput.combinePublicInputs, PublicInput.publicAdd,
        PublicInput.publicAct]
      rw [publicBlock_decodeXRings]
      have tail := congrFun
        (inductionHypothesis
          (fun index => challenges index.succ)
          (fun index => inputs index.succ)) column
      change ringOfList
          (phi81Combine
            (fun index => challenges index.succ)
            (fun index =>
              inputs index.succ
                (productionBlock
                  (PublicInput.publicBlockIndex dimensions.shape column))))
          (PublicInput.publicLaneIndex column) = _ at tail
      rw [phi81Combine_coefficient] at tail
      exact congrArg
        (fun value =>
          ringFMul (ringOfList (challenges 0))
            (ringOfList
              (inputs 0
                (productionBlock
                  (PublicInput.publicBlockIndex dimensions.shape column))))
            (PublicInput.publicLaneIndex column) + value)
        tail

/-- Exact production projection identities imply the independent typed
public-input combination equation. Only challenge wiring, equation carrier
wiring, and reduction are consumed. -/
theorem typedPublicInputEquation_of_refinement
    {matrixCount : Nat}
    {params : GlobalParams} {arity : BatchArity params}
    (assignment : Nat -> Nat)
    (columns : BatchColumns params arity matrixCount)
    (tree : TraceTree arity matrixCount)
    (refinement : EquationRefinement assignment columns tree)
    (dimensions : Dimensions) :
    decodeXRings (dimensions := dimensions)
        (decodeOpening assignment columns.output).x =
      PublicInput.combinePublicInputs
        (fun index => ringOfList (values assignment (columns.challenges index)))
        (fun index =>
          decodeXRings (dimensions := dimensions)
            (decodeOpening assignment (columns.inputs index)).x) := by
  have outputEquation :
      (decodeOpening assignment columns.output).x =
        fun block => phi81Combine
          (fun index => values assignment (columns.challenges index))
          (fun index =>
            (decodeOpening assignment (columns.inputs index)).x block) := by
    funext block
    exact phi81_reduction_at refinement.challengeWiring refinement.wiring
      refinement.reduction (.x block)
  rw [outputEquation]
  exact decodeXRings_phi81Combine _ _

/-- Exact active-X widths and the single X-parent column equality identify the
typed PiRLC output with the named parent. Codec, commitment, evaluation,
point, equation, reduction, input-wiring, and trace-census facts are unused. -/
theorem typedOutput_eq_parent_of_wiring
    {matrixCount : Nat}
    {params : GlobalParams} {arity : BatchArity params}
    (assignment : Nat -> Nat)
    (columns : BatchColumns params arity matrixCount)
    (outputWidth : forall block,
      (columns.output.x block).length = ringDegree)
    (parentX : assembleXColumns columns.output =
      columns.parentClaim.xActiveCols)
    (dimensions : Dimensions) :
    decodeXRings (dimensions := dimensions)
        (decodeOpening assignment columns.output).x =
      PiDecBridge.decode (dimensions := dimensions)
        (decodedPackedInput assignment columns.parentClaim) := by
  calc
    decodeXRings (dimensions := dimensions)
        (decodeOpening assignment columns.output).x =
        PiDecBridge.decode (dimensions := dimensions)
          (PackedPublicInput.mk
            (values assignment (assembleXColumns columns.output))) :=
      (decode_assembledX assignment columns.output outputWidth).symm
    _ = PiDecBridge.decode (dimensions := dimensions)
          (decodedPackedInput assignment columns.parentClaim) :=
      congrArg (fun packed => PiDecBridge.decode
        (dimensions := dimensions) packed) (by
          rw [parentX]
          rfl)

/-- Local strict-PiDEC layout specialization. Production profiles generally
need the relabeled specialization below instead. -/
theorem typedOutput_eq_piDecParent_of_wiring
    {matrixCount : Nat}
    {params : GlobalParams} {arity : BatchArity params}
    (assignment : Nat -> Nat)
    (columns : BatchColumns params arity matrixCount)
    (outputWidth : forall block,
      (columns.output.x block).length = ringDegree)
    (parentX : assembleXColumns columns.output =
      columns.parentClaim.xActiveCols)
    (parentClaim : columns.parentClaim = layout.parent)
    (dimensions : Dimensions) :
    decodeXRings (dimensions := dimensions)
        (decodeOpening assignment columns.output).x =
      PiDecBridge.decodedPublicInput dimensions assignment layout.parent := by
  simpa [PiDecBridge.decodedPublicInput, parentClaim] using
    typedOutput_eq_parent_of_wiring
      assignment columns outputWidth parentX dimensions

/-- Complete typed public-input flow for the production-shaped reduction:
the strict `Pi_DEC` parent is the canonical Phi81 combination of the decoded
`Pi_RLC` inputs under the decoded batch challenge columns. This theorem does
not establish their source-bound sampler authority. -/
theorem typedPiRlcToPiDecParentEquation_of_refinement
    {matrixCount : Nat}
    {params : GlobalParams} {arity : BatchArity params}
    (assignment : Nat -> Nat)
    (columns : BatchColumns params arity matrixCount)
    (tree : TraceTree arity matrixCount)
    (refinement : EquationRefinement assignment columns tree)
    (outputWidth : forall block,
      (columns.output.x block).length = ringDegree)
    (parentX : assembleXColumns columns.output =
      columns.parentClaim.xActiveCols)
    (parentClaim : columns.parentClaim = layout.parent)
    (dimensions : Dimensions) :
    PiDecBridge.decodedPublicInput dimensions assignment layout.parent =
      PublicInput.combinePublicInputs
        (fun index => ringOfList (values assignment (columns.challenges index)))
        (fun index =>
          decodeXRings (dimensions := dimensions)
            (decodeOpening assignment (columns.inputs index)).x) := by
  rw [← typedOutput_eq_piDecParent_of_wiring
    assignment columns outputWidth parentX parentClaim dimensions]
  exact typedPublicInputEquation_of_refinement
    assignment columns tree refinement dimensions

/-- Public-input composition across the concrete reduction boundary. Under
the exact `Pi_RLC` artifacts and strict semantic `Pi_DEC` acceptance, the
column-computed Phi81 parent equals the radix recomposition of the typed
children. Source-bound challenge authority and R1CS satisfaction must still
be proved to imply the premises. -/
theorem typedPiRlcPiDecPublicInputComposition
    {matrixCount : Nat}
    {params : GlobalParams} {arity : BatchArity params}
    (assignment : Nat -> Nat)
    (columns : BatchColumns params arity matrixCount)
    (tree : TraceTree arity matrixCount)
    (refinement : EquationRefinement assignment columns tree)
    (outputWidth : forall block,
      (columns.output.x block).length = ringDegree)
    (parentX : assembleXColumns columns.output =
      columns.parentClaim.xActiveCols)
    (parentClaim : columns.parentClaim = layout.parent)
    (dimensions : Dimensions)
    (piDecAccepted : PiDecStrictCompiler.Accepted layout assignment) :
    PublicInput.combinePublicInputs
        (fun index => ringOfList (values assignment (columns.challenges index)))
        (fun index =>
          decodeXRings (dimensions := dimensions)
            (decodeOpening assignment (columns.inputs index)).x) =
      PiDECAlgebra.PublicInput.recomposePublicInput fun index =>
        PiDecBridge.decodedPublicInput dimensions assignment (childLayout index) := by
  calc
    PublicInput.combinePublicInputs
        (fun index => ringOfList (values assignment (columns.challenges index)))
        (fun index =>
          decodeXRings (dimensions := dimensions)
            (decodeOpening assignment (columns.inputs index)).x) =
        PiDecBridge.decodedPublicInput dimensions assignment layout.parent :=
      (typedPiRlcToPiDecParentEquation_of_refinement
        assignment columns tree refinement outputWidth parentX parentClaim
        dimensions).symm
    _ = PiDECAlgebra.PublicInput.recomposePublicInput fun index =>
          PiDecBridge.decodedPublicInput dimensions assignment (childLayout index) :=
      PiDecBridge.strictAccepted_typedPublicInputEquation
        dimensions assignment piDecAccepted

/-- Production-column specialization. The `Pi_RLC` projection tree is decoded
against the global assignment, while strict `Pi_DEC` is decoded against its
local assignment through the explicit checked column map. -/
theorem typedPiRlcPiDecPublicInputComposition_relabel
    {matrixCount : Nat}
    {params : GlobalParams} {arity : BatchArity params}
    (assignment : Nat -> Nat)
    (columns : BatchColumns params arity matrixCount)
    (tree : TraceTree arity matrixCount)
    (refinement : EquationRefinement assignment columns tree)
    (outputWidth : forall block,
      (columns.output.x block).length = ringDegree)
    (parentX : assembleXColumns columns.output =
      columns.parentClaim.xActiveCols)
    (columnMap : List Nat)
    (parentClaim : columns.parentClaim =
      relabelClaim columnMap layout.parent)
    (dimensions : Dimensions)
    (piDecAccepted : PiDecStrictCompiler.Accepted layout
      (Relabel.assignment columnMap assignment)) :
    PublicInput.combinePublicInputs
        (fun index => ringOfList (values assignment (columns.challenges index)))
        (fun index =>
          decodeXRings (dimensions := dimensions)
            (decodeOpening assignment (columns.inputs index)).x) =
      PiDECAlgebra.PublicInput.recomposePublicInput fun index =>
        PiDecBridge.decodedPublicInput dimensions
          (Relabel.assignment columnMap assignment) (childLayout index) := by
  calc
    PublicInput.combinePublicInputs
        (fun index => ringOfList (values assignment (columns.challenges index)))
        (fun index =>
          decodeXRings (dimensions := dimensions)
            (decodeOpening assignment (columns.inputs index)).x) =
        PiDecBridge.decode (dimensions := dimensions)
          (decodedPackedInput assignment columns.parentClaim) := by
      exact
        (typedPublicInputEquation_of_refinement
          assignment columns tree refinement dimensions).symm.trans
        (typedOutput_eq_parent_of_wiring
          assignment columns outputWidth parentX dimensions)
    _ = PiDecBridge.decodedPublicInput dimensions
          (Relabel.assignment columnMap assignment) layout.parent := by
      rw [parentClaim, decodedPackedInput_relabel]
      rfl
    _ = PiDECAlgebra.PublicInput.recomposePublicInput fun index =>
          PiDecBridge.decodedPublicInput dimensions
            (Relabel.assignment columnMap assignment) (childLayout index) :=
      PiDecBridge.strictAccepted_typedPublicInputEquation
        dimensions (Relabel.assignment columnMap assignment) piDecAccepted

end Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.PublicInputBridge
