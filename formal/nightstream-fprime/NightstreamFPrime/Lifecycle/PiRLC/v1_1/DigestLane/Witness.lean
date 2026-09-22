import NightstreamFPrime.Gadgets.Range.CanonicalU64.Witness
import NightstreamFPrime.Gadgets.Sampling.Candidate16Five.Witness
import NightstreamFPrime.Lifecycle.PiRLC.v1_1.DigestLane

/-!
Child-owned witness IR contract for the opaque PiRLC digest-lane circuit.

This companion module inspects only its own three-child composition. It uses
the two gadget witness contracts and does not unfold either gadget circuit.
-/

namespace NightstreamFPrime.Lifecycle.PiRLC.v1_1.DigestLane

open NightstreamFPrime.Circuit
open NightstreamFPrime.Gadgets.Range
open NightstreamFPrime.Gadgets.Sampling

def witnessBatches (interface : Interface) (offset : Nat) : List WitnessBatch :=
  CanonicalU64.witnessBatches (canonicalInterface interface offset)
      (canonicalOffset offset) ++
    Candidate16Five.witnessBatches (decoderInterface offset lowPart)
      (decoderOffset offset lowPart) ++
    Candidate16Five.witnessBatches (decoderInterface offset highPart)
      (decoderOffset offset highPart)

def witnessBatchesForSource (source : Expr) (offset : Nat) : List WitnessBatch :=
  witnessBatches { source := fun _ => source } offset

@[simp] theorem witnessBatchesForSource_eq
    (interface : Interface) (offset : Nat) :
    witnessBatchesForSource (interface.source offset) offset =
      witnessBatches interface offset := by
  rfl

@[simp] theorem witnesses_main (interface : Interface) (offset : Nat) :
    witnesses (Circuit.ops (main interface) offset) =
      witnessBatches interface offset := by
  have canonicalWitnesses :
      Op.witnesses (canonicalOp interface offset) =
        CanonicalU64.witnessBatches (canonicalInterface interface offset)
          (canonicalOffset offset) := by
    change witnesses (Circuit.ops
      (CanonicalU64.main (canonicalInterface interface offset))
      (canonicalOffset offset)) = _
    exact CanonicalU64.witnesses_main _ _
  have lowWitnesses :
      Op.witnesses (lowOp offset) =
        Candidate16Five.witnessBatches (decoderInterface offset lowPart)
          (decoderOffset offset lowPart) := by
    change witnesses (Circuit.ops
      (Candidate16Five.main (decoderInterface offset lowPart))
      (decoderOffset offset lowPart)) = _
    exact Candidate16Five.witnesses_main _ _
  have highWitnesses :
      Op.witnesses (highOp offset) =
        Candidate16Five.witnessBatches (decoderInterface offset highPart)
          (decoderOffset offset highPart) := by
    change witnesses (Circuit.ops
      (Candidate16Five.main (decoderInterface offset highPart))
      (decoderOffset offset highPart)) = _
    exact Candidate16Five.witnesses_main _ _
  change witnesses [canonicalOp interface offset, lowOp offset,
    highOp offset] = _
  simp only [witnesses, List.flatMap_cons, List.flatMap_nil,
    List.append_nil, canonicalWitnesses, lowWitnesses, highWitnesses,
    witnessBatches, List.append_assoc]

@[simp] theorem witnesses_circuit_main (interface : Interface) (offset : Nat) :
    witnesses ((circuit interface).main.ops offset) =
      witnessBatches interface offset := by
  change witnesses (Circuit.ops (main interface) offset) = _
  exact witnesses_main interface offset

/-- The nine actual batches read only the transcript lane or this child's
100 allocated cells. Child operations remain behind their witness contracts. -/
theorem witnessBatches_readsSatisfy (interface : Interface) (offset : Nat)
    (allowed : Nat → Prop)
    (sourceSupported : (interface.source offset).VarsSatisfy allowed)
    (localSupported : ∀ index, index < logicalPrivateCount → allowed (offset + index)) :
    ∀ batch ∈ witnessBatches interface offset, batch.ReadsSatisfy allowed := by
  have canonical := CanonicalU64.witnessBatches_readsSatisfy
    (canonicalInterface interface offset) (canonicalOffset offset) allowed
    sourceSupported (by
      intro index bounded
      exact localSupported index (by
        change index < 66 at bounded
        change index < 100
        omega))
  have decoder (part : Fin 2) := Candidate16Five.witnessBatches_readsSatisfy
    (decoderInterface offset part) (decoderOffset offset part) allowed
    (decoderCandidate_varsSatisfy offset part allowed localSupported)
    (decoderBits_varsSatisfy offset part allowed localSupported) (by
      intro index bounded
      have partBound := part.isLt
      unfold decoderOffset
      simpa [Nat.add_assoc] using localSupported
        (CanonicalU64.auxiliaryCount + part.val * Candidate16Five.auxiliaryCount + index) (by
          change index < 17 at bounded
          change 66 + part.val * 17 + index < 100
          omega))
  intro batch member
  simp only [witnessBatches, List.mem_append] at member
  rcases member with (member | member) | member
  · exact canonical batch member
  · exact decoder lowPart batch member
  · exact decoder highPart batch member

end NightstreamFPrime.Lifecycle.PiRLC.v1_1.DigestLane
