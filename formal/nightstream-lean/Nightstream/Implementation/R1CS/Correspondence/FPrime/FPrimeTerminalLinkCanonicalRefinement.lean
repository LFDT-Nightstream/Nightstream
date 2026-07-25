import Nightstream.Implementation.R1CS.Correspondence.FPrime.FPrimeTerminalLinkSound
import Nightstream.Implementation.R1CS.Correspondence.FPrime.FPrimeEncodingCanonicalBits

/-!
Contract: artifact-checked refinement of the complete terminal latest-link
rows to the frozen logical public-input equality.

Owns:
- the exact assignment view of the fresh 270-coordinate plain carrier;
- an explicit producer-alignment proposition relating the last step's 256
  output-bit columns to the independently defined canonical digest encoder;
- soundness and completeness of the 270 emitted rows with respect to the
  typed carrier checker;
- reduction of row satisfaction to the paper-owned 257-coordinate equality,
  with all thirteen physical padding coordinates discharged.

Does not own: the host nonempty/length checks, output-encoding rows that
establish `ProducerAligned`, the Rust-to-artifact drift gate, an optional
application suffix, or the surrounding terminal NIFS verifier.

Emits constraints: no; it interprets the exact ownership artifact.
-/

namespace Nightstream.Implementation.R1CS.FPrimeTerminalLinkCanonicalRefinement

open Nightstream.Implementation.Encoding.FPrime
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.FPrimeTerminalLink
open Nightstream.Implementation.Rust.CanonicalConformance.NativeStep

set_option maxRecDepth 32768

/-- Typed view of the fresh claim columns after the host has checked the
plain carrier profile and exact public-input length. -/
def claimOfAssignment (z : Nat → Nat) :
    CanonicalPlainCarrierLink.Claim where
  mIn := CanonicalPlainCarrierLink.carrierWidth
  x :=
    { one := z freshOneCol
      body := fun lane bit =>
        z (freshBitCol (lane.val * 64 + bit.val))
      padding := fun padding =>
        z (freshPaddingCol padding.val) }

/-- Required coordinate alignment with the producer-side canonical encoder.
This is deliberately a concrete proposition, not a generic refinement-failure
escape: the output-encoding owner must prove each exact lane/bit equality. -/
def ProducerAligned (digest : Digest) (z : Nat → Nat) : Prop :=
  ∀ lane bit,
    z (lastXOutBitCol (lane.val * 64 + bit.val)) =
      CanonicalPlainCarrierLink.encodedBit digest lane bit

/-- Exact placement contract between an encoding-owner assignment and the
terminal-link owner's producer columns. A surrounding artifact must prove
this map from its concrete column schedule. -/
def ProducerColumnsAligned
    (producer terminal : Nat → Nat) : Prop :=
  ∀ (lane : Fin 4) (bit : Fin 64),
    terminal
        (lastXOutBitCol (lane.val * 64 + bit.val)) =
      producer
        (FPrimeEncoding.publicBitCol lane.val bit.val)

/-- Exact encoding rows discharge `ProducerAligned`; the only remaining
surrounding-artifact premise is the concrete column placement map. -/
theorem producerAligned_of_encodingRows
    (goldilocksPrime : EuclidPrime goldilocksP)
    {producer terminal : Nat → Nat}
    (producerCanonical :
      ∀ column, producer column < goldilocksP)
    (producerOne : producer 0 = 1)
    (encodingSatisfies :
      Satisfies FPrimeEncoding.rows producer)
    (columnsAligned :
      ProducerColumnsAligned producer terminal) :
    ProducerAligned
      (FPrimeEncodingCanonicalBits.digestOfAssignment
        producer producerCanonical)
      terminal := by
  have encodingHolds :=
    FPrimeEncodingSound.fPrimeEncoding_sound
      goldilocksPrime producerCanonical producerOne encodingSatisfies
  intro lane bit
  exact
    (columnsAligned lane bit).trans
      (FPrimeEncodingCanonicalBits.publicBit_eq_encodedBit
        producerCanonical encodingHolds lane bit)

/-- Satisfied rows plus exact producer alignment make the executable typed
plain-carrier checker accept. -/
theorem check_of_satisfies
    (digest : Digest) {z : Nat → Nat}
    (canonical : ∀ column, z column < goldilocksP)
    (one : z 0 = 1)
    (satisfies : Satisfies rows z)
    (producerAligned : ProducerAligned digest z) :
    CanonicalPlainCarrierLink.check digest (claimOfAssignment z) = true := by
  have holds :=
    FPrimeTerminalLinkSound.fPrimeTerminalLink_sound
      canonical one satisfies
  apply
    (CanonicalPlainCarrierLink.check_eq_true_iff
      digest (claimOfAssignment z)).2
  apply CanonicalPlainCarrierLink.Claim.eq_of_fields
  · rfl
  · apply CanonicalPlainCarrierLink.Carrier.eq_of_fields
    · exact holds.affineOne
    · funext lane bit
      have flatLt : lane.val * 64 + bit.val < 256 := by
        have laneLt := lane.isLt
        have bitLt := bit.isLt
        omega
      exact
        (holds.linked (lane.val * 64 + bit.val) flatLt).trans
          (producerAligned lane bit)
    · funext padding
      apply holds.paddingZero padding.val
      have paddingLt := padding.isLt
      simp only [
        CanonicalPlainCarrierLink.paddingWidth,
        CanonicalPlainCarrierLink.carrierWidth,
        CanonicalPlainCarrierLink.logicalWidth
      ] at paddingLt
      exact paddingLt

/-- Conversely, typed acceptance and the same producer alignment construct
semantic validity for every exact row and therefore a satisfying assignment
for this ownership block. -/
theorem satisfies_of_check
    (digest : Digest) {z : Nat → Nat}
    (canonical : ∀ column, z column < goldilocksP)
    (one : z 0 = 1)
    (accepted :
      CanonicalPlainCarrierLink.check digest (claimOfAssignment z) = true)
    (producerAligned : ProducerAligned digest z) :
    Satisfies rows z := by
  have claimEqual :
      claimOfAssignment z =
        CanonicalPlainCarrierLink.encodeClaim digest :=
    (CanonicalPlainCarrierLink.check_eq_true_iff
      digest (claimOfAssignment z)).1 accepted
  apply FPrimeTerminalLinkSound.fPrimeTerminalLink_complete canonical one
  refine {
    affineOne := ?_
    linked := ?_
    paddingZero := ?_
  }
  · simpa [
      claimOfAssignment,
      CanonicalPlainCarrierLink.encodeClaim,
      CanonicalPlainCarrierLink.encodeCarrier
    ] using
      congrArg
        (fun claim : CanonicalPlainCarrierLink.Claim => claim.x.one)
        claimEqual
  · intro bit bitLt
    let lane : Fin 4 := ⟨bit / 64, by omega⟩
    let offset : Fin 64 := ⟨bit % 64, Nat.mod_lt bit (by omega)⟩
    have flatEqual : lane.val * 64 + offset.val = bit := by
      simp only [lane, offset]
      omega
    have freshEqual :
        z (freshBitCol (lane.val * 64 + offset.val)) =
          CanonicalPlainCarrierLink.encodedBit digest lane offset := by
      simpa [
        claimOfAssignment,
        CanonicalPlainCarrierLink.encodeClaim,
        CanonicalPlainCarrierLink.encodeCarrier
      ] using
        congrArg
          (fun claim : CanonicalPlainCarrierLink.Claim =>
            claim.x.body lane offset)
          claimEqual
    rw [flatEqual] at freshEqual
    have aligned := producerAligned lane offset
    rw [flatEqual] at aligned
    exact freshEqual.trans aligned.symm
  · intro padding paddingLt
    let paddingFin : Fin CanonicalPlainCarrierLink.paddingWidth :=
      ⟨padding, by
        simpa [
          CanonicalPlainCarrierLink.paddingWidth,
          CanonicalPlainCarrierLink.carrierWidth,
          CanonicalPlainCarrierLink.logicalWidth
        ] using paddingLt⟩
    simpa [
      claimOfAssignment,
      CanonicalPlainCarrierLink.encodeClaim,
      CanonicalPlainCarrierLink.encodeCarrier,
      paddingFin
    ] using
      congrArg
        (fun claim : CanonicalPlainCarrierLink.Claim =>
          claim.x.padding paddingFin)
        claimEqual

/-- Exact artifact-level bridge: under explicit producer alignment, the
complete 270-row ownership block is satisfied exactly when its typed carrier
is the zero completion of a logical input accepted by the frozen paper-link
checker. -/
theorem satisfies_iff_logicalPaperLink
    (digest : Digest) {z : Nat → Nat}
    (canonical : ∀ column, z column < goldilocksP)
    (one : z 0 = 1)
    (producerAligned : ProducerAligned digest z) :
    Satisfies rows z ↔
      ∃ logical,
        CanonicalPublicInputLink.check digest logical = true /\
          claimOfAssignment z =
            CanonicalPlainCarrierLink.completeClaim logical := by
  constructor
  · intro satisfies
    exact
      (CanonicalPlainCarrierLink.check_reduces_to_logicalPaperLink
        digest (claimOfAssignment z)).1
        (check_of_satisfies
          digest canonical one satisfies producerAligned)
  · intro reduced
    apply satisfies_of_check digest canonical one
    · exact
        (CanonicalPlainCarrierLink.check_reduces_to_logicalPaperLink
          digest (claimOfAssignment z)).2 reduced
    · exact producerAligned

/-- Composed artifact theorem: satisfying the exact 532-row output encoder
and proving its concrete public-bit column placement removes the abstract
producer-alignment premise from the complete 270-row terminal-to-paper
equivalence. -/
theorem satisfies_iff_logicalPaperLink_of_encodingRows
    (goldilocksPrime : EuclidPrime goldilocksP)
    {producer terminal : Nat → Nat}
    (producerCanonical :
      ∀ column, producer column < goldilocksP)
    (producerOne : producer 0 = 1)
    (encodingSatisfies :
      Satisfies FPrimeEncoding.rows producer)
    (terminalCanonical :
      ∀ column, terminal column < goldilocksP)
    (terminalOne : terminal 0 = 1)
    (columnsAligned :
      ProducerColumnsAligned producer terminal) :
    Satisfies rows terminal ↔
      ∃ logical,
        CanonicalPublicInputLink.check
          (FPrimeEncodingCanonicalBits.digestOfAssignment
            producer producerCanonical)
          logical = true /\
        claimOfAssignment terminal =
          CanonicalPlainCarrierLink.completeClaim logical := by
  apply satisfies_iff_logicalPaperLink
  · exact terminalCanonical
  · exact terminalOne
  · exact producerAligned_of_encodingRows
      goldilocksPrime producerCanonical producerOne
      encodingSatisfies columnsAligned

end Nightstream.Implementation.R1CS.FPrimeTerminalLinkCanonicalRefinement
