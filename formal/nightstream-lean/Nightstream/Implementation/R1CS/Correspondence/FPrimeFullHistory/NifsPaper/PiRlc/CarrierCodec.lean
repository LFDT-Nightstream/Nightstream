import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.PiRlc.PublicInputBridge
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.PiRlc.EvaluationBridge

/-!
Canonical matrix-indexed `Pi_RLC` public-carrier codec.

Assurance tier: model-level. These are exact Lean representation theorems,
not a Rust-conformant claim about emitted columns.

Owns: deterministic commitment, five-ring public-input, and shape-indexed
evaluation encoders; and their exact agreement with the column layouts under
the widths checked by `CarrierArtifact`.

Does not own: a matrix count, projection identities, transcript challenges,
R1CS-row soundness, private CE openings, commitment binding, evaluation
padding, costs, row removal, or a generated production profile.

Emits constraints: no.

Authority boundary: the caller must choose the matrix count through a semantic
shape or an explicitly labelled fixture. The codec never derives it from a
generated header. Full-width `X` equality uses the independently proved
270-coordinate decoder permutation, so no packed coordinate can disappear
behind a default read.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_rlc.verify.identities.commitment` | flatten eighteen commitment rings without reordering | derived | `encodeCommitment_decodeOpening` |
| `nifs.pi_rlc.verify.identities.x` | serialize five rings in lane-major order | derived | `encodeX_decodeOpening` |
| `nifs.pi_rlc.verify.identities.y_ring` | pair two active limbs for every matrix-indexed evaluation | derived | `encodeYRing_decodeOpening` |
| `nifs.pi_rlc.verify.identities.public` | one canonical codec satisfies every generic public layout law | derived | `canonical_artifact` |
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.CarrierCodec

set_option maxRecDepth 4096

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270
open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper
open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc
open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.EvaluationBridge
open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.PublicInputBridge

/-- Commitment rings use the existing ring-major order. -/
def encodeCommitment (rings : CommitmentRings) : PackedCommitment :=
  PackedCommitment.mk (List.ofFn rings).flatten

/-- Public-input rings use the lane-major transpose. -/
def encodeX (rings : XRings) : PackedPublicInput :=
  packXRings rings

/-- Pair the two base-field limbs of every matrix-indexed evaluation. -/
def encodeYRing {matrixCount : Nat}
    (rings : YRingRings matrixCount) : Array Evaluation :=
  decodeYRingRings rings

/-- The canonical value-level codec at an explicit matrix count. -/
def canonical (matrixCount : Nat) : PiRlc.CarrierCodec matrixCount where
  commitment := ⟨encodeCommitment⟩
  x := ⟨encodeX⟩
  yRing := ⟨encodeYRing⟩

@[simp] theorem canonical_yRing_encode {matrixCount : Nat}
    (rings : YRingRings matrixCount) :
    (canonical matrixCount).yRing.encode rings = decodeYRingRings rings := by
  rfl

theorem encodeCommitment_decodeOpening
    {matrixCount : Nat}
    (assignment : Nat -> Nat) (columns : ProjectionColumns matrixCount) :
    encodeCommitment (decodeOpening assignment columns).commitment =
      PackedCommitment.mk
        (values assignment (assembleCommitmentColumns columns)) := by
  apply PackedCommitment.eq_of_data_eq
  simp [encodeCommitment, decodeOpening, assembleCommitmentColumns, values]

/-! The dimensions below are only a type witness for the fixed 270-coordinate
decoder permutation. Its row, logical-private, and matrix dimensions are not
read by either side of `encodeX_decodeOpening`. -/

private def codecDimensions : Dimensions where
  rowVariables := 0
  legacyLogicalWidth := legacyPublicWidth
  matrixCount := 0
  legacyPublicFits := Nat.le_refl _

theorem encodeX_decodeOpening
    {matrixCount : Nat}
    (assignment : Nat -> Nat) (columns : ProjectionColumns matrixCount)
    (width : forall block, (columns.x block).length = ringDegree) :
    encodeX (decodeOpening assignment columns).x =
      PackedPublicInput.mk
        (values assignment (assembleXColumns columns)) := by
  apply PiDecBridge.decode_injective_of_length
    (dimensions := codecDimensions)
  · exact packXRings_length _
  · simp [values, assembleXColumns_length]
  · exact (decode_packXRings
      (dimensions := codecDimensions)
      (decodeOpening assignment columns).x).trans
        (decode_assembledX
          (dimensions := codecDimensions)
          assignment columns width).symm

theorem encodeYRing_decodeOpening
    {matrixCount : Nat}
    (assignment : Nat -> Nat) (columns : ProjectionColumns matrixCount)
    (width : forall row limb,
      (columns.yRing row limb).length = ringDegree) :
    encodeYRing (decodeOpening assignment columns).yRing =
      decodeYRingActive assignment columns := by
  apply Array.ext
  · simp [encodeYRing, decodeYRingActive]
  · intro index leftLt _rightLt
    have indexLt : index < matrixCount := by
      simpa [encodeYRing] using leftLt
    let row : Fin matrixCount := ⟨index, indexLt⟩
    simp only [encodeYRing, decodeYRingRings, pairRings, decodeOpening,
      Array.getElem_ofFn, decodeYRingActive]
    funext coefficient
    apply k_eq_of_coeffs
    · exact values_getD_of_length assignment
        (columns.yRing row ⟨0, by decide⟩)
        (width row ⟨0, by decide⟩) coefficient
    · exact values_getD_of_length assignment
        (columns.yRing row ⟨1, by decide⟩)
        (width row ⟨1, by decide⟩) coefficient

/-- The deterministic generic codec discharges the complete layout artifact
at any explicit matrix count. -/
theorem canonical_artifact (matrixCount : Nat) :
    CodecArtifact (canonical matrixCount) := by
  constructor
  · exact encodeCommitment_decodeOpening
  · exact encodeX_decodeOpening
  · exact encodeYRing_decodeOpening

end Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.CarrierCodec
