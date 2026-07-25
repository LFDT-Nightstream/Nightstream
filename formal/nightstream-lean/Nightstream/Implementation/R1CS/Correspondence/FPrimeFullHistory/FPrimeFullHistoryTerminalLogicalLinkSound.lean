import Nightstream.Implementation.R1CS.Correspondence.FPrime.FPrimeEncodingCanonicalBits
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.FPrimeFullHistoryOutputEncodingSound

/-!
Contract: logical-prefix refinement of the captured full-history terminal-link
snapshot.

Owns:
- reconstruction of the terminal fresh claim's logical 257-coordinate prefix
  from its affine-one and 256 physical bit columns;
- exact use of the generated output-encoding column map and generated terminal
  equality pairs;
- acceptance of the paper-owned logical public-input equality from the
  captured 532-row producer encoding and 257-row terminal-link snapshot.

Does not own: the thirteen plain-carrier zero-padding rows emitted by current
production, terminal host shape/nonempty checks, the rest of the terminal
NIFS rows, Rust-source refinement, or Poseidon2 digest semantics. In
particular, this theorem does not establish current full-history artifact
conformance; `FPrimeFullHistoryTerminalLinkDrift` records the 257-versus-270
row obstruction.

Emits constraints: no; it composes two exact generated ownership blocks.
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalLogicalLinkSound

open Nightstream.Implementation.Encoding.FPrime
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.Rust.CanonicalConformance.NativeStep

set_option maxRecDepth 32768

abbrev Pulled (assignment : Nat → Nat) : Nat → Nat :=
  FPrimeFullHistoryOutputEncodingSound.Pulled assignment

def outputDigest
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP) :
    Digest :=
  FPrimeEncodingCanonicalBits.digestOfAssignment
    (Pulled assignment) (Relabel.canonical canonical)

def terminalFreshLane
    (assignment : Nat → Nat) (lane : Fin 4) : BitVec 64 :=
  FPrimeEncodingCanonicalBits.bitVectorPrefix
    (fun bit =>
      assignment
        (FPrimeFullHistoryTerminalLinkSound.freshBitCol
          (lane.val * 64 + bit)))
    64

def terminalLogicalPublic
    (assignment : Nat → Nat) : PublicInput where
  one :=
    assignment FPrimeFullHistoryTerminalLinkSound.freshOneCol
  body := terminalFreshLane assignment

private theorem terminalFreshLane_getLsbD
    (assignment : Nat → Nat)
    (lane : Fin 4) (bit : Fin 64) :
    (terminalFreshLane assignment lane).getLsbD bit.val =
      decide
        (assignment
          (FPrimeFullHistoryTerminalLinkSound.freshBitCol
            (lane.val * 64 + bit.val)) = 1) := by
  exact
    FPrimeEncodingCanonicalBits.bitVectorPrefix_getLsbD
      _ 64 bit.val bit.isLt

private theorem decide_eq_bool_of_nat_eq
    (value : Nat) (bit : Bool)
    (equal : value = if bit then 1 else 0) :
    decide (value = 1) = bit := by
  cases bit <;> simp_all

/-- Semantic composition of the two generated owners: their decoded terminal
logical prefix equals the independently defined canonical public encoder. -/
theorem logicalCheck_of_holds
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (encoding :
      FPrimeEncodingSound.Holds (Pulled assignment))
    (terminalLink :
      FPrimeFullHistoryTerminalLinkSound.Holds assignment) :
    CanonicalPublicInputLink.check
      (outputDigest assignment canonical)
      (terminalLogicalPublic assignment) = true := by
  apply
    (CanonicalPublicInputLink.check_eq_true_iff
      (outputDigest assignment canonical)
      (terminalLogicalPublic assignment)).2
  change
    PublicInput.mk
        (assignment
          FPrimeFullHistoryTerminalLinkSound.freshOneCol)
        (terminalFreshLane assignment) =
      PublicInput.mk 1
        (encodeEncInst (outputDigest assignment canonical))
  rw [terminalLink.affineOne]
  congr 1
  funext lane
  apply BitVec.eq_of_getLsbD_eq
  intro bit bitLt
  let bitFin : Fin 64 := ⟨bit, bitLt⟩
  have flatLt : lane.val * 64 + bit < 256 := by
    have laneLt := lane.isLt
    omega
  have linked :=
    terminalLink.linked
      (lane.val * 64 + bit) flatLt
  have encoded :=
    FPrimeEncodingCanonicalBits.publicBit_eq_encodedBit
      (Relabel.canonical canonical) encoding lane bitFin
  change
    assignment
        (Relabel.column
          FPrimeFullHistoryOutputEncoding.columnMap
          (FPrimeEncoding.publicBitCol lane.val bit)) =
      CanonicalPlainCarrierLink.encodedBit
        (outputDigest assignment canonical) lane bitFin at encoded
  rw [
    FPrimeFullHistoryOutputEncodingSound.publicBitColumnMap
      lane bitFin
  ] at encoded
  have freshEncoded :
      assignment
          (FPrimeFullHistoryTerminalLinkSound.freshBitCol
            (lane.val * 64 + bit)) =
        CanonicalPlainCarrierLink.encodedBit
          (outputDigest assignment canonical) lane bitFin :=
    linked.trans encoded
  rw [terminalFreshLane_getLsbD assignment lane bitFin]
  unfold CanonicalPlainCarrierLink.encodedBit at freshEncoded
  change
    assignment
        (FPrimeFullHistoryTerminalLinkSound.freshBitCol
          (lane.val * 64 + bit)) =
      if
        (encodeEncInst
          (outputDigest assignment canonical) lane).getLsbD bit
      then 1 else 0 at freshEncoded
  exact decide_eq_bool_of_nat_eq _ _ freshEncoded

/-- Exact captured-row theorem for the full-history logical prefix. The stale
257-row snapshot omits the current plain carrier's thirteen zero pins. -/
theorem logicalCheck_of_rows
    (goldilocksPrime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (encodingSatisfies :
      Satisfies FPrimeFullHistoryOutputEncoding.rows assignment)
    (terminalLinkSatisfies :
      Satisfies FPrimeFullHistoryTerminalLink.rows assignment) :
    CanonicalPublicInputLink.check
      (outputDigest assignment canonical)
      (terminalLogicalPublic assignment) = true := by
  apply logicalCheck_of_holds canonical
  · exact
      FPrimeFullHistoryOutputEncodingSound.sound
        goldilocksPrime canonical one encodingSatisfies
  · exact
      FPrimeFullHistoryTerminalLinkSound.sound
        canonical one terminalLinkSatisfies

end Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalLogicalLinkSound
