import Nightstream.Implementation.R1CS.Core.ConstantPins
import Nightstream.Implementation.R1CS.Core.EqualityPins
import Nightstream.Implementation.R1CS.Ownership.FPrimeFullHistory.FPrimeFullHistoryRecursivePreludeArtifact
import Nightstream.Implementation.R1CS.Core.Poseidon2Sponge

/-! Exact fixed-profile chunk-shape digest certificate for the recursive prelude. -/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursivePreludeHashes

open Nightstream.Implementation.R1CS

set_option maxRecDepth 1048576

def startColumn : Nat := 10843
def nextChunkDigestColumns : List Nat := [10864, 10865, 10866, 10867]
def constantPins : List (Nat × Nat) := [(10833, 1), (10868, 44), (10869, 30521782141150574), (10870, 31069335676202596), (10871, 30796712693949999), (10872, 30239273049612133), (10873, 26860422160999263), (10874, 13357362876737892), (10875, 12662), (10876, 54), (10877, 18), (10878, 257), (10879, 0), (13292, 45), (13293, 30521782141150574), (13294, 31069335676202596), (13295, 30796712693949999), (13296, 30239273049612133), (13297, 27981936923603039), (13298, 32777976662287455), (13299, 3241519), (13300, 1), (13301, 0)]
def chunkDigestPairs : List (Nat × Nat) := [(10864, 16309), (10865, 16310), (10866, 16311), (10867, 16312)]

def claimTrace : Poseidon2Sponge.Trace :=
  { inputColumns := [10868, 10869, 10870, 10871, 10872, 10873, 10874, 10875, 10876, 10877, 10878], zeroColumn := 10879, zeroRow := 12, rounds := [
      { kind := .absorb [10868, 10869, 10870, 10871], stateBeforeColumns := [10879, 10879, 10879, 10879, 10879, 10879, 10879, 10879], permutationInputColumns := [10880, 10881, 10882, 10883, 10879, 10879, 10879, 10879], permutationOutputColumns := [11476, 11477, 11478, 11479, 11480, 11481, 11482, 11483], definingRows := [13, 14, 15, 16], call := { rowStart := 17, rowEnd := 617, inputColumns := [10880, 10881, 10882, 10883, 10879, 10879, 10879, 10879], firstAllocatedColumn := 10884 } }
    , { kind := .absorb [10872, 10873, 10874, 10875], stateBeforeColumns := [11476, 11477, 11478, 11479, 11480, 11481, 11482, 11483], permutationInputColumns := [11484, 11485, 11486, 11487, 11480, 11481, 11482, 11483], permutationOutputColumns := [12080, 12081, 12082, 12083, 12084, 12085, 12086, 12087], definingRows := [617, 618, 619, 620], call := { rowStart := 621, rowEnd := 1221, inputColumns := [11484, 11485, 11486, 11487, 11480, 11481, 11482, 11483], firstAllocatedColumn := 11488 } }
    , { kind := .absorb [10876, 10877, 10878], stateBeforeColumns := [12080, 12081, 12082, 12083, 12084, 12085, 12086, 12087], permutationInputColumns := [12088, 12089, 12090, 12083, 12084, 12085, 12086, 12087], permutationOutputColumns := [12683, 12684, 12685, 12686, 12687, 12688, 12689, 12690], definingRows := [1221, 1222, 1223], call := { rowStart := 1224, rowEnd := 1824, inputColumns := [12088, 12089, 12090, 12083, 12084, 12085, 12086, 12087], firstAllocatedColumn := 12091 } }
    , { kind := .pad, stateBeforeColumns := [12683, 12684, 12685, 12686, 12687, 12688, 12689, 12690], permutationInputColumns := [12691, 12684, 12685, 12686, 12687, 12688, 12689, 12690], permutationOutputColumns := [13284, 13285, 13286, 13287, 13288, 13289, 13290, 13291], definingRows := [1824], call := { rowStart := 1825, rowEnd := 2425, inputColumns := [12691, 12684, 12685, 12686, 12687, 12688, 12689, 12690], firstAllocatedColumn := 12692 } }
    ], outputColumns := [13284, 13285, 13286, 13287] }

def chunkTrace : Poseidon2Sponge.Trace :=
  { inputColumns := [13292, 13293, 13294, 13295, 13296, 13297, 13298, 13299, 10843, 13300, 13284, 13285, 13286, 13287], zeroColumn := 13301, zeroRow := 2434, rounds := [
      { kind := .absorb [13292, 13293, 13294, 13295], stateBeforeColumns := [13301, 13301, 13301, 13301, 13301, 13301, 13301, 13301], permutationInputColumns := [13302, 13303, 13304, 13305, 13301, 13301, 13301, 13301], permutationOutputColumns := [13898, 13899, 13900, 13901, 13902, 13903, 13904, 13905], definingRows := [2435, 2436, 2437, 2438], call := { rowStart := 2439, rowEnd := 3039, inputColumns := [13302, 13303, 13304, 13305, 13301, 13301, 13301, 13301], firstAllocatedColumn := 13306 } }
    , { kind := .absorb [13296, 13297, 13298, 13299], stateBeforeColumns := [13898, 13899, 13900, 13901, 13902, 13903, 13904, 13905], permutationInputColumns := [13906, 13907, 13908, 13909, 13902, 13903, 13904, 13905], permutationOutputColumns := [14502, 14503, 14504, 14505, 14506, 14507, 14508, 14509], definingRows := [3039, 3040, 3041, 3042], call := { rowStart := 3043, rowEnd := 3643, inputColumns := [13906, 13907, 13908, 13909, 13902, 13903, 13904, 13905], firstAllocatedColumn := 13910 } }
    , { kind := .absorb [10843, 13300, 13284, 13285], stateBeforeColumns := [14502, 14503, 14504, 14505, 14506, 14507, 14508, 14509], permutationInputColumns := [14510, 14511, 14512, 14513, 14506, 14507, 14508, 14509], permutationOutputColumns := [15106, 15107, 15108, 15109, 15110, 15111, 15112, 15113], definingRows := [3643, 3644, 3645, 3646], call := { rowStart := 3647, rowEnd := 4247, inputColumns := [14510, 14511, 14512, 14513, 14506, 14507, 14508, 14509], firstAllocatedColumn := 14514 } }
    , { kind := .absorb [13286, 13287], stateBeforeColumns := [15106, 15107, 15108, 15109, 15110, 15111, 15112, 15113], permutationInputColumns := [15114, 15115, 15108, 15109, 15110, 15111, 15112, 15113], permutationOutputColumns := [15708, 15709, 15710, 15711, 15712, 15713, 15714, 15715], definingRows := [4247, 4248], call := { rowStart := 4249, rowEnd := 4849, inputColumns := [15114, 15115, 15108, 15109, 15110, 15111, 15112, 15113], firstAllocatedColumn := 15116 } }
    , { kind := .pad, stateBeforeColumns := [15708, 15709, 15710, 15711, 15712, 15713, 15714, 15715], permutationInputColumns := [15716, 15709, 15710, 15711, 15712, 15713, 15714, 15715], permutationOutputColumns := [16309, 16310, 16311, 16312, 16313, 16314, 16315, 16316], definingRows := [4849], call := { rowStart := 4850, rowEnd := 5450, inputColumns := [15716, 15709, 15710, 15711, 15712, 15713, 15714, 15715], firstAllocatedColumn := 15717 } }
    ], outputColumns := [16309, 16310, 16311, 16312] }

theorem constantPins_canonical : ConstantPins.ValuesCanonical constantPins := by native_decide
theorem constantRows_included :
  rowsIncluded (ConstantPins.rows constantPins)
    FPrimeFullHistoryRecursivePrelude.rows = true := by native_decide

theorem chunkDigestRows_included :
  rowsIncluded (EqualityPins.rows chunkDigestPairs)
    FPrimeFullHistoryRecursivePrelude.rows = true := by native_decide

theorem claimTrace_valid :
  claimTrace.Valid FPrimeFullHistoryRecursivePrelude.rows := by native_decide

theorem chunkTrace_valid :
  chunkTrace.Valid FPrimeFullHistoryRecursivePrelude.rows := by native_decide

def claimInputValues : List Nat :=
  claimTrace.inputColumns.map (ConstantPins.lookup constantPins)

def traceOutputPins (trace : Poseidon2Sponge.Trace)
    (inputValues : List Nat) : List (Nat × Nat) :=
  (List.range 4).map fun lane =>
    (trace.outputColumns.getD lane 0,
     Poseidon2Sponge.runValueRounds trace.rounds inputValues (fun _ => 0) lane)

def traceOutputKeys (trace : Poseidon2Sponge.Trace) : List Nat :=
  (List.range 4).map fun lane => trace.outputColumns.getD lane 0

theorem traceOutputPins_keys (trace : Poseidon2Sponge.Trace)
    (inputValues : List Nat) :
    ConstantPins.keys (traceOutputPins trace inputValues) =
      traceOutputKeys trace := by
  simp [ConstantPins.keys, traceOutputPins, traceOutputKeys, List.map_map,
    Function.comp_def]

def claimOutputPins : List (Nat × Nat) :=
  traceOutputPins claimTrace claimInputValues

def fixedInputPins : List (Nat × Nat) :=
  (startColumn, 1) :: constantPins ++ claimOutputPins

def fixedInputKeys : List Nat :=
  startColumn :: ConstantPins.keys constantPins ++ traceOutputKeys claimTrace

def chunkInputValues : List Nat :=
  chunkTrace.inputColumns.map (ConstantPins.lookup fixedInputPins)

def chunkDigestValue : List Nat :=
  (List.range 4).map fun lane =>
    Poseidon2Sponge.runValueRounds chunkTrace.rounds
      chunkInputValues (fun _ => 0) lane

theorem claimInputs_covered :
  ConstantPins.Covers claimTrace.inputColumns constantPins := by native_decide

theorem fixedInputPins_keys :
    ConstantPins.keys fixedInputPins = fixedInputKeys := by
  simp only [fixedInputPins, fixedInputKeys, ConstantPins.keys, List.map_cons,
    List.map_append]
  rw [show List.map Prod.fst claimOutputPins = traceOutputKeys claimTrace by
    simpa [ConstantPins.keys, claimOutputPins] using
      traceOutputPins_keys claimTrace claimInputValues]

theorem chunkInputKeys_covered :
    ConstantPins.KeysCover chunkTrace.inputColumns fixedInputKeys := by
  native_decide

theorem chunkInputs_covered :
    ConstantPins.Covers chunkTrace.inputColumns fixedInputPins := by
  rw [ConstantPins.covers_iff_keys, fixedInputPins_keys]
  exact chunkInputKeys_covered

theorem next_chunk_digest_fixed
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (start : assignment startColumn = 1)
    (satisfies : Satisfies FPrimeFullHistoryRecursivePrelude.rows assignment) :
    ∀ lane, lane < 4 →
      assignment (nextChunkDigestColumns.getD lane 0) =
        chunkDigestValue.getD lane 0 := by
  have constants := ConstantPins.sound constantPins_canonical
    constantRows_included canonical one satisfies
  have equalities := EqualityPins.sound chunkDigestRows_included
    canonical one satisfies
  have claimInputEq :
      claimTrace.inputColumns.map assignment = claimInputValues :=
    ConstantPins.map_assignment_eq_lookup constants claimInputs_covered
  have claimOutputs : ∀ pin ∈ claimOutputPins, assignment pin.1 = pin.2 := by
    intro pin member
    rcases List.mem_map.mp member with ⟨lane, laneMember, rfl⟩
    have laneLt := List.mem_range.mp laneMember
    simpa [traceOutputPins, claimInputEq] using
      Poseidon2Sponge.trace_values_sound claimTrace_valid canonical one
        satisfies lane laneLt
  have fixedFacts : ∀ pin ∈ fixedInputPins, assignment pin.1 = pin.2 := by
    intro pin member
    simp only [fixedInputPins, List.mem_cons, List.mem_append] at member
    rcases member with (startPin | constantPin) | claimPin
    · subst pin
      exact start
    · exact constants pin constantPin
    · exact claimOutputs pin claimPin
  have chunkInputEq :
      chunkTrace.inputColumns.map assignment = chunkInputValues :=
    ConstantPins.map_assignment_eq_lookup fixedFacts chunkInputs_covered
  have chunkOutputs := Poseidon2Sponge.trace_values_sound
    chunkTrace_valid canonical one satisfies
  intro lane laneLt
  have pairMember :
      (nextChunkDigestColumns.getD lane 0,
        chunkTrace.outputColumns.getD lane 0) ∈ chunkDigestPairs := by
    have cases : lane = 0 ∨ lane = 1 ∨ lane = 2 ∨ lane = 3 := by omega
    rcases cases with rfl | rfl | rfl | rfl <;> native_decide
  rw [equalities _ pairMember, chunkOutputs lane laneLt, chunkInputEq]
  simp [chunkDigestValue, laneLt]

end Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursivePreludeHashes
