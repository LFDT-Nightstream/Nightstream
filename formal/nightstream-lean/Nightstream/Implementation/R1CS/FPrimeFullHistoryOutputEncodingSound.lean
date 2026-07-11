import Nightstream.Implementation.R1CS.FPrimeEncodingSound
import Nightstream.Implementation.R1CS.FPrimeFullHistoryOutputEncodingArtifact
import Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveOutputArtifact
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalLinkSound
import Nightstream.Implementation.R1CS.FPrimeFullHistoryBaseStepSound

/-!
Contract: universal semantics of the exact recursive-output
`enc_inst(x_out)` suffix and its terminal consumer.

The Rust drift gate proves that the relabeled 532-row program is the literal
suffix of the generated recursive owner. Lean derives canonical digest lanes
from those rows, then uses the exact terminal link rows to reconstruct the
same four-lane digest from the terminal fresh public bits.
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryOutputEncodingSound

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.FPrimeFullHistoryOutputEncoding

set_option maxRecDepth 65536
set_option maxHeartbeats 4000000

abbrev Pulled (assignment : Nat → Nat) : Nat → Nat :=
  Relabel.assignment columnMap assignment

theorem mapsOne : Relabel.column columnMap 0 = 0 := by
  native_decide

/-- The exact generated suffix forces the same canonical four-lane encoding
contract as the production helper in isolation. -/
theorem sound (goldilocksPrime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies rows assignment) :
    FPrimeEncodingSound.Holds (Pulled assignment) := by
  apply FPrimeEncodingSound.fPrimeEncoding_sound goldilocksPrime
    (Relabel.canonical canonical)
    (Relabel.constantOne mapsOne one)
  intro source sourceMember
  apply (Relabel.rowHolds_iff columnMap assignment source).mp
  apply satisfies
  exact List.mem_map.mpr ⟨source, sourceMember, rfl⟩

theorem digestColumnMap :
    ∀ lane : Fin 4,
      Relabel.column columnMap (FPrimeEncoding.digestCol lane.val) =
        FPrimeFullHistoryRecursiveOutput.xOutColumns.getD lane.val 0 := by
  native_decide

theorem publicBitColumnMap :
    ∀ lane : Fin 4, ∀ bit : Fin 64,
      Relabel.column columnMap
          (FPrimeEncoding.publicBitCol lane.val bit.val) =
        FPrimeFullHistoryTerminalLinkSound.lastXOutBitCol
          (lane.val * 64 + bit.val) := by
  native_decide

def outputDigest (assignment : Nat → Nat) : List Nat :=
  (List.range 4).map fun lane =>
    assignment
      (Relabel.column columnMap (FPrimeEncoding.digestCol lane))

def terminalFreshLaneBitsValue (assignment : Nat → Nat) (lane : Nat) : Nat :=
  (List.range 64).foldl
    (fun total bit => total + 2 ^ bit *
      assignment (FPrimeFullHistoryTerminalLinkSound.freshBitCol
        (lane * 64 + bit))) 0

def terminalFreshDigest (assignment : Nat → Nat) : List Nat :=
  (List.range 4).map (terminalFreshLaneBitsValue assignment)

def decodedTerminalFresh (assignment : Nat → Nat) :
    FPrimeFullHistoryBaseStepSound.Fresh :=
  { publicXOut := terminalFreshDigest assignment }

private theorem foldl_bits_congr (xs : List Nat) (left right : Nat → Nat)
    (equal : ∀ bit ∈ xs, left bit = right bit) (initial : Nat) :
    xs.foldl (fun total bit => total + 2 ^ bit * left bit) initial =
      xs.foldl (fun total bit => total + 2 ^ bit * right bit) initial := by
  induction xs generalizing initial with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp only [List.foldl_cons]
      rw [equal head (by simp)]
      apply inductionHypothesis
      intro bit member
      exact equal bit (by simp [member])

theorem terminalFreshLane_eq_outputLane
    {assignment : Nat → Nat}
    (encoding : FPrimeEncodingSound.Holds (Pulled assignment))
    (terminalLink : FPrimeFullHistoryTerminalLinkSound.Holds assignment)
    (lane : Fin 4) :
    terminalFreshLaneBitsValue assignment lane.val =
      assignment
        (Relabel.column columnMap
          (FPrimeEncoding.digestCol lane.val)) := by
  change terminalFreshLaneBitsValue assignment lane.val =
    Pulled assignment (FPrimeEncoding.digestCol lane.val)
  rw [(encoding.laneCanonical lane.val lane.isLt).1]
  unfold terminalFreshLaneBitsValue FPrimeEncodingSound.laneBitsValue
    Pulled Relabel.assignment
  apply foldl_bits_congr
  intro bit member
  let bitFin : Fin 64 := ⟨bit, List.mem_range.mp member⟩
  have bitLt : bit < 64 := List.mem_range.mp member
  have flatLt : lane.val * 64 + bit < 256 := by
    have laneLt := lane.isLt
    omega
  have linked := terminalLink.linked (lane.val * 64 + bit) flatLt
  rw [publicBitColumnMap lane bitFin]
  exact linked

private theorem rangeFour : List.range 4 = [0, 1, 2, 3] := by decide

/-- The terminal fresh input decodes to the recursive producer's actual
canonical `x_out` lanes. -/
theorem terminalFreshDigest_eq_outputDigest
    {assignment : Nat → Nat}
    (encoding : FPrimeEncodingSound.Holds (Pulled assignment))
    (terminalLink : FPrimeFullHistoryTerminalLinkSound.Holds assignment) :
    terminalFreshDigest assignment = outputDigest assignment := by
  have lane0 := terminalFreshLane_eq_outputLane encoding terminalLink (0 : Fin 4)
  have lane1 := terminalFreshLane_eq_outputLane encoding terminalLink (1 : Fin 4)
  have lane2 := terminalFreshLane_eq_outputLane encoding terminalLink (2 : Fin 4)
  have lane3 := terminalFreshLane_eq_outputLane encoding terminalLink (3 : Fin 4)
  simpa [terminalFreshDigest, outputDigest, rangeFour] using
    And.intro lane0 (And.intro lane1 (And.intro lane2 lane3))

end Nightstream.Implementation.R1CS.FPrimeFullHistoryOutputEncodingSound
