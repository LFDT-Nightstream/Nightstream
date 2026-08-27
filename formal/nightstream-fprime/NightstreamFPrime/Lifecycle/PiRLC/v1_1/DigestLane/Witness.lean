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

end NightstreamFPrime.Lifecycle.PiRLC.v1_1.DigestLane
