import Mathlib.Data.List.OfFn
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.TranscriptMachine
import Nightstream.Implementation.R1CS.Correspondence.Poseidon2.ExtractedReference

/-!
Contract: byte framing and bounded checkpoint model for the streaming
claim-state transcript domain.

Owns the exact application-domain bytes, seven-byte little-endian packing,
the 18-word framing, four fixed four-word permutation blocks, the two-word
remainder, and compact state views used by the checkpoint certificates.

Does not own any generated column, state preimage, public word, or recursive
lifecycle statement.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayDigestDomain

open Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachine

/-- Exact bytes of `neo-transcript`'s Poseidon2 application domain. -/
def transcriptApplicationDomain : List Nat :=
  [110, 101, 111, 47, 116, 114, 97, 110, 115, 99, 114, 105, 112, 116,
   47, 118, 49, 124, 112, 111, 115, 101, 105, 100, 111, 110, 50, 45,
   103, 111, 108, 100, 105, 108, 111, 99, 107, 115, 45, 119, 56, 45,
   114, 52]

/-- Exact bytes of the claim-state digest application label. -/
def stateDigestDomain : List Nat :=
  [110, 101, 111, 46, 102, 111, 108, 100, 46, 99, 108, 101, 97, 110,
   47, 110, 101, 98, 117, 108, 97, 47, 102, 45, 112, 114, 105, 109,
   101, 47, 115, 116, 114, 101, 97, 109, 105, 110, 103, 45, 99, 108,
   97, 105, 109, 45, 115, 116, 97, 116, 101, 47, 118, 49]

/-- Exact bytes of the labelled field list. -/
def stateFieldsLabel : List Nat := [115, 116, 97, 116, 101]

/-- One seven-byte little-endian transcript limb. Missing bytes in the final
limb are zero. -/
def packSevenAt (bytes : List Nat) (chunk : Nat) : Nat :=
  (List.range 7).foldl (fun value offset =>
    value + bytes.getD (chunk * 7 + offset) 0 * 2 ^ (8 * offset)) 0

/-- Rust's injective packed-byte encoding, including the byte length. -/
def packedBytesWithLen (bytes : List Nat) : List Nat :=
  bytes.length ::
    (List.range ((bytes.length + 6) / 7)).map (packSevenAt bytes)

def emptyState : State where
  lanes := fun _ => wordField 0
  absorbed := ⟨0, by decide⟩

def absorbWords (initial : State) (words : List Nat) : State :=
  words.foldl (fun state word => absorbElem state (wordField word)) initial

theorem absorbWords_append
    (initial : State) (first second : List Nat) :
    absorbWords initial (first ++ second) =
      absorbWords (absorbWords initial first) second := by
  simp [absorbWords, List.foldl_append]

/-- A nonempty absorption from a full cursor first applies exactly one
permutation, as the native transcript does. -/
theorem absorbWords_full
    (initial : State) (words : List Nat)
    (full : initial.absorbed.val = rate)
    (nonempty : words ≠ []) :
    absorbWords initial words = absorbWords (permute initial) words := by
  cases words with
  | nil => exact False.elim (nonempty rfl)
  | cons head tail =>
      have noRoom : ¬initial.absorbed.val < rate := by
        simp [full]
      have room : (permute initial).absorbed.val < rate := by
        simp [permute, rate]
      have firstEqual :
          absorbElem initial (wordField head) =
            absorbElem (permute initial) (wordField head) := by
        unfold absorbElem
        rw [dif_neg noRoom, dif_pos room]
        simp [permute]
      simp only [absorbWords, List.foldl_cons]
      rw [firstEqual]

/-- Exact `append_message(label, message)` field schedule. -/
def appendMessage (initial : State) (label message : List Nat) : State :=
  absorbWords initial
    ([1] ++ packedBytesWithLen label ++ packedBytesWithLen message)

def domainBlock1 : List Nat :=
  [1, 44, 27428916078536046, 32774695491433326]

def domainBlock2 : List Nat :=
  [32492151232362031, 12721823848622437,
   31362922327076711, 12728458466782051]

def domainBlock3 : List Nat :=
  [13426, 54, 30521782141150574, 31069335676202596]

def domainBlock4 : List Nat :=
  [27422324158721583, 30796712690673199,
   27414614995316581, 30508344144718189]

def domainRemainder : List Nat :=
  [27431110773139809, 212436215156]

def domainCertificateIndices : List Nat :=
  List.range 4 ++
    (List.range 4).map (4 + ·) ++
    (List.range 4).map (8 + ·) ++
    (List.range 4).map (12 + ·) ++
    (List.range 2).map (16 + ·)

/-- The native checkpoint certificates cover exactly 18 framed words, in
four non-overlapping four-word blocks and one exact two-word remainder. -/
theorem domain_certificate_partition_exact :
    domainBlock1.length = 4 ∧
      domainBlock2.length = 4 ∧
      domainBlock3.length = 4 ∧
      domainBlock4.length = 4 ∧
      domainRemainder.length = 2 ∧
      domainCertificateIndices = List.range 18 ∧
      domainCertificateIndices.Nodup := by
  native_decide

/-- The byte-level framing is exactly the bounded checkpoint partition. -/
theorem domain_framing_words_exact :
    [1] ++ packedBytesWithLen transcriptApplicationDomain ++
        packedBytesWithLen stateDigestDomain =
      domainBlock1 ++ domainBlock2 ++ domainBlock3 ++ domainBlock4 ++
        domainRemainder := by
  native_decide

def checkpoint1Values : List Nat :=
  [15335109097073140235, 17619563798142362813,
   11645649210628966215, 4436364367798357556,
   14285079556494616407, 10815961559651698320,
   18260705026218357875, 8424582804285107233]

def checkpoint2Values : List Nat :=
  [12824850162859434436, 16884249588232806831,
   1238414030862021266, 16194180760988878864,
   2250958222046521952, 15440103732085447511,
   6547395164335706312, 10272340994444577429]

def checkpoint3Values : List Nat :=
  [6958520756929626742, 16947407995347177160,
   4955651861384673240, 11357146294475889773,
   5510511107940736673, 9313957564688670493,
   1768918467504634943, 14404461253576933351]

def checkpoint4Values : List Nat :=
  [16534366849561726655, 6810547603550404849,
   7420078321807019432, 14323236552110360532,
   1298986797814860681, 17392165756113845022,
   8388603933087874784, 14187929483296301137]

def stateFromValues
    (values : List Nat) (absorbed : Fin (rate + 1)) : State where
  lanes := fun lane => fieldValue (values.getD lane.val 0)
  absorbed

def checkpointState (values : List Nat) : State :=
  stateFromValues values ⟨0, by decide⟩

structure StateView where
  lanes : List Nat
  absorbed : Nat
deriving DecidableEq

@[ext] theorem StateView.ext
    {left right : StateView}
    (lanes : left.lanes = right.lanes)
    (absorbed : left.absorbed = right.absorbed) :
    left = right := by
  cases left
  cases right
  simp_all

def stateView (state : State) : StateView where
  lanes := List.ofFn fun lane => (state.lanes lane).val
  absorbed := state.absorbed.val

/-- Total canonical input view used by the exact 600-definition SSA
interpreter. -/
def stateLaneValues (state : State) : Nat → Nat := fun lane =>
  if bounded : lane < width then state.lanes ⟨lane, bounded⟩ |>.val else 0

theorem stateLaneValues_canonical (state : State) :
    ∀ lane, lane < width → stateLaneValues state lane < goldilocksP := by
  intro lane bounded
  simp [stateLaneValues, bounded]

/-- Eight output values computed by the exact Rust-emitted 600-definition
SSA interpreter. -/
def ssaPermutationValues (state : State) : List Nat :=
  List.ofFn fun lane : Fin width =>
    Nightstream.Implementation.R1CS.Poseidon2ExtractedReference.execution
      (stateLaneValues state)
      (Nightstream.Implementation.R1CS.Poseidon2CompactTraceRefinement.traceOutputColumn
        lane)

/-- Generic bridge from the compact SSA output list to the canonical
transcript-machine permutation. -/
theorem permute_lanes_eq_ssa (state : State) :
    List.ofFn (fun lane : Fin width => (permute state).lanes lane |>.val) =
      ssaPermutationValues state := by
  unfold ssaPermutationValues
  apply congrArg List.ofFn
  funext lane
  change
    Nightstream.Implementation.R1CS.Poseidon2PermutationSound.permute
          (laneNat state) lane.val % goldilocksP =
      Nightstream.Implementation.R1CS.Poseidon2ExtractedReference.execution
        (stateLaneValues state)
        (Nightstream.Implementation.R1CS.Poseidon2CompactTraceRefinement.traceOutputColumn
          lane)
  have inputFunctionsEqual : laneNat state = stateLaneValues state := by
    funext inputLane
    rfl
  rw [inputFunctionsEqual,
    Nat.mod_eq_of_lt
      (Nightstream.Implementation.R1CS.Poseidon2PermutationSound.permute_lt
        (stateLaneValues_canonical state) lane.val)]
  exact
    (Nightstream.Implementation.R1CS.Poseidon2ExtractedReference.permute_eq_reference
      (stateLaneValues_canonical state) lane).trans
    (Nightstream.Implementation.R1CS.Poseidon2ExtractedReference.execution_computes_reference
      (stateLaneValues_canonical state) lane).symm

theorem stateView_injective : Function.Injective stateView := by
  intro left right equal
  cases left with
  | mk leftLanes leftAbsorbed =>
      cases right with
      | mk rightLanes rightAbsorbed =>
          have lanesView := congrArg StateView.lanes equal
          change
            List.ofFn (fun lane => (leftLanes lane).val) =
              List.ofFn (fun lane => (rightLanes lane).val) at lanesView
          have laneValuesEqual :
              (fun lane => (leftLanes lane).val) =
                fun lane => (rightLanes lane).val :=
            List.ofFn_injective lanesView
          have lanesEqual : leftLanes = rightLanes := by
            funext lane
            apply Fin.ext
            exact congrFun laneValuesEqual lane
          have absorbedView := congrArg StateView.absorbed equal
          change leftAbsorbed.val = rightAbsorbed.val at absorbedView
          have absorbedEqual : leftAbsorbed = rightAbsorbed :=
            Fin.ext absorbedView
          subst rightLanes
          subst rightAbsorbed
          rfl

/-- Extensional equality for transcript states without unfolding a
permutation while the caller proves the equality. -/
theorem state_ext
    {left right : State}
    (lanes : ∀ lane, (left.lanes lane).val = (right.lanes lane).val)
    (absorbed : left.absorbed.val = right.absorbed.val) :
    left = right := by
  cases left with
  | mk leftLanes leftAbsorbed =>
      cases right with
      | mk rightLanes rightAbsorbed =>
          congr
          · funext lane
            apply Fin.ext
            exact lanes lane
          · apply Fin.ext
            exact absorbed

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayDigestDomain
