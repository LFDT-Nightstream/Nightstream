import Mathlib.Tactic
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyReplay

/-!
Contract: structural validity certificate for the Rust-emitted PiRLC family
replay artifact.

Assurance tier: Rust-to-Lean artifact geometry certificate.

Owns bounded call-geometry leaves, bounded call-chain leaves, exact column
layouts, and their composition into validity for both replay arms.

Does not own replay semantics, PiRLC algebra, or lifecycle integration.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 524288

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyReplay

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyReplay.Artifact

private theorem valid_of_take_drop
    {α : Type} {property : α → Prop} {items : List α} {count : Nat}
    (head : ∀ item ∈ items.take count, property item)
    (tail : ∀ item ∈ items.drop count, property item) :
    ∀ item ∈ items, property item := by
  intro item member
  rw [← List.take_append_drop count items] at member
  rcases List.mem_append.mp member with member | member
  · exact head item member
  · exact tail item member

private theorem length_of_take_drop
    {α : Type} {items : List α} {count headLength tailLength : Nat}
    (head : (items.take count).length = headLength)
    (tail : (items.drop count).length = tailLength) :
    items.length = headLength + tailLength := by
  have split := congrArg List.length (List.take_append_drop count items)
  simpa only [List.length_append, head, tail] using split.symm

private theorem exactCallChainFrom_append
    (row column : Nat) (left right : List Poseidon2Call.Call)
    (leftValid : exactCallChainFrom row column left = true)
    (rightValid :
      exactCallChainFrom (row + left.length * 600)
        (column + left.length * 600) right = true) :
    exactCallChainFrom row column (left ++ right) = true := by
  induction left generalizing row column with
  | nil =>
      simpa [exactCallChainFrom] using rightValid
  | cons call rest inductionHypothesis =>
      simp only [List.length_cons] at rightValid
      simp only [List.cons_append, exactCallChainFrom, Bool.and_eq_true] at leftValid ⊢
      constructor
      · exact leftValid.1
      · apply inductionHypothesis (row := row + 600) (column := column + 600)
        · exact leftValid.2
        · simpa [Nat.succ_mul, Nat.add_assoc, Nat.add_comm,
            Nat.add_left_comm] using rightValid

def evenTail0 := evenArm.poseidon2Calls
def evenChunk0 := evenTail0.take 64
def evenTail1 := evenTail0.drop 64
def evenChunk1 := evenTail1.take 64
def evenTail2 := evenTail1.drop 64
def evenChunk2 := evenTail2.take 64
def evenTail3 := evenTail2.drop 64

theorem evenChunk0_length : evenChunk0.length = 64 := by rfl
theorem evenChunk1_length : evenChunk1.length = 64 := by rfl
theorem evenChunk2_length : evenChunk2.length = 64 := by rfl
theorem evenTail3_length : evenTail3.length = 50 := by rfl

theorem evenArm_poseidon2Calls_length :
    evenArm.poseidon2Calls.length = 242 := by
  have tail2 : evenTail2.length = 114 :=
    length_of_take_drop (items := evenTail2) (count := 64)
      evenChunk2_length evenTail3_length
  have tail1 : evenTail1.length = 178 :=
    length_of_take_drop (items := evenTail1) (count := 64)
      evenChunk1_length tail2
  exact length_of_take_drop (items := evenTail0) (count := 64)
    evenChunk0_length tail1

theorem evenChunk0_valid :
    ∀ call ∈ evenChunk0, PoseidonCallValid 310880 call := by
  norm_num [evenChunk0, evenTail0, PoseidonCallValid, evenArm]

theorem evenChunk1_valid :
    ∀ call ∈ evenChunk1, PoseidonCallValid 310880 call := by
  norm_num [evenChunk1, evenTail1, evenTail0, PoseidonCallValid, evenArm]

theorem evenChunk2_valid :
    ∀ call ∈ evenChunk2, PoseidonCallValid 310880 call := by
  norm_num [evenChunk2, evenTail2, evenTail1, evenTail0,
    PoseidonCallValid, evenArm]

theorem evenTail3_valid :
    ∀ call ∈ evenTail3, PoseidonCallValid 310880 call := by
  norm_num [evenTail3, evenTail2, evenTail1, evenTail0,
    PoseidonCallValid, evenArm]

theorem evenArm_poseidon2Calls_valid :
    ∀ call ∈ evenArm.poseidon2Calls,
      PoseidonCallValid evenArm.columnCount call := by
  change ∀ call ∈ evenTail0, PoseidonCallValid 310880 call
  exact valid_of_take_drop evenChunk0_valid
    (valid_of_take_drop evenChunk1_valid
      (valid_of_take_drop evenChunk2_valid evenTail3_valid))

theorem evenChunk0_chain :
    exactCallChainFrom 0 165680 evenChunk0 = true := by rfl

theorem evenChunk1_chain :
    exactCallChainFrom 38400 204080 evenChunk1 = true := by rfl

theorem evenChunk2_chain :
    exactCallChainFrom 76800 242480 evenChunk2 = true := by rfl

theorem evenTail3_chain :
    exactCallChainFrom 115200 280880 evenTail3 = true := by rfl

theorem evenTail2_chain :
    exactCallChainFrom 76800 242480 evenTail2 = true := by
  rw [← List.take_append_drop 64 evenTail2]
  apply exactCallChainFrom_append
  · exact evenChunk2_chain
  · simpa [evenChunk2_length] using evenTail3_chain

theorem evenTail1_chain :
    exactCallChainFrom 38400 204080 evenTail1 = true := by
  rw [← List.take_append_drop 64 evenTail1]
  apply exactCallChainFrom_append
  · exact evenChunk1_chain
  · simpa [evenChunk1_length] using evenTail2_chain

theorem evenArm_call_chain :
    exactCallChainFrom 0 165680 evenArm.poseidon2Calls = true := by
  change exactCallChainFrom 0 165680 evenTail0 = true
  rw [← List.take_append_drop 64 evenTail0]
  apply exactCallChainFrom_append
  · exact evenChunk0_chain
  · simpa [evenChunk0_length] using evenTail1_chain

def oddTail0 := oddArm.poseidon2Calls
def oddChunk0 := oddTail0.take 64
def oddTail1 := oddTail0.drop 64
def oddChunk1 := oddTail1.take 64
def oddTail2 := oddTail1.drop 64
def oddChunk2 := oddTail2.take 64
def oddTail3 := oddTail2.drop 64

theorem oddChunk0_length : oddChunk0.length = 64 := by rfl
theorem oddChunk1_length : oddChunk1.length = 64 := by rfl
theorem oddChunk2_length : oddChunk2.length = 64 := by rfl
theorem oddTail3_length : oddTail3.length = 52 := by rfl

theorem oddArm_poseidon2Calls_length :
    oddArm.poseidon2Calls.length = 244 := by
  have tail2 : oddTail2.length = 116 :=
    length_of_take_drop (items := oddTail2) (count := 64)
      oddChunk2_length oddTail3_length
  have tail1 : oddTail1.length = 180 :=
    length_of_take_drop (items := oddTail1) (count := 64)
      oddChunk1_length tail2
  exact length_of_take_drop (items := oddTail0) (count := 64)
    oddChunk0_length tail1

theorem oddChunk0_valid :
    ∀ call ∈ oddChunk0, PoseidonCallValid 312080 call := by
  norm_num [oddChunk0, oddTail0, PoseidonCallValid, oddArm]

theorem oddChunk1_valid :
    ∀ call ∈ oddChunk1, PoseidonCallValid 312080 call := by
  norm_num [oddChunk1, oddTail1, oddTail0, PoseidonCallValid, oddArm]

theorem oddChunk2_valid :
    ∀ call ∈ oddChunk2, PoseidonCallValid 312080 call := by
  norm_num [oddChunk2, oddTail2, oddTail1, oddTail0,
    PoseidonCallValid, oddArm]

theorem oddTail3_valid :
    ∀ call ∈ oddTail3, PoseidonCallValid 312080 call := by
  norm_num [oddTail3, oddTail2, oddTail1, oddTail0,
    PoseidonCallValid, oddArm]

theorem oddArm_poseidon2Calls_valid :
    ∀ call ∈ oddArm.poseidon2Calls,
      PoseidonCallValid oddArm.columnCount call := by
  change ∀ call ∈ oddTail0, PoseidonCallValid 312080 call
  exact valid_of_take_drop oddChunk0_valid
    (valid_of_take_drop oddChunk1_valid
      (valid_of_take_drop oddChunk2_valid oddTail3_valid))

theorem oddChunk0_chain :
    exactCallChainFrom 0 165680 oddChunk0 = true := by rfl

theorem oddChunk1_chain :
    exactCallChainFrom 38400 204080 oddChunk1 = true := by rfl

theorem oddChunk2_chain :
    exactCallChainFrom 76800 242480 oddChunk2 = true := by rfl

theorem oddTail3_chain :
    exactCallChainFrom 115200 280880 oddTail3 = true := by rfl

theorem oddTail2_chain :
    exactCallChainFrom 76800 242480 oddTail2 = true := by
  rw [← List.take_append_drop 64 oddTail2]
  apply exactCallChainFrom_append
  · exact oddChunk2_chain
  · simpa [oddChunk2_length] using oddTail3_chain

theorem oddTail1_chain :
    exactCallChainFrom 38400 204080 oddTail1 = true := by
  rw [← List.take_append_drop 64 oddTail1]
  apply exactCallChainFrom_append
  · exact oddChunk1_chain
  · simpa [oddChunk1_length] using oddTail2_chain

theorem oddArm_call_chain :
    exactCallChainFrom 0 165680 oddArm.poseidon2Calls = true := by
  change exactCallChainFrom 0 165680 oddTail0 = true
  rw [← List.take_append_drop 64 oddTail0]
  apply exactCallChainFrom_append
  · exact oddChunk0_chain
  · simpa [oddChunk0_length] using oddTail1_chain

private theorem evenArm_inputAfterColumns_valid :
    columnsValid evenArm.columnCount 8 evenArm.inputAfterColumns := by
  norm_num [columnsValid, evenArm]

private theorem evenArm_outputAfterColumns_valid :
    columnsValid evenArm.columnCount 8 evenArm.outputAfterColumns := by
  norm_num [columnsValid, evenArm]

private theorem oddArm_inputAfterColumns_valid :
    columnsValid oddArm.columnCount 8 oddArm.inputAfterColumns := by
  norm_num [columnsValid, oddArm]

private theorem oddArm_outputAfterColumns_valid :
    columnsValid oddArm.columnCount 8 oddArm.outputAfterColumns := by
  norm_num [columnsValid, oddArm]

theorem evenArm_valid : evenArm.Valid 0 2 229 13 := by
  refine ⟨by norm_num [evenArm], by norm_num [evenArm], rfl, rfl, rfl, rfl, ?_,
    rfl, rfl, rfl, rfl, evenArm_inputAfterColumns_valid,
    evenArm_outputAfterColumns_valid, evenArm_poseidon2Calls_valid, ?_⟩
  · rw [evenArm_poseidon2Calls_length]
  · exact ⟨evenArm_call_chain,
      by rw [evenArm_poseidon2Calls_length]; norm_num [evenArm],
      by norm_num [evenArm]⟩

theorem oddArm_valid : oddArm.Valid 2 0 230 14 := by
  refine ⟨by norm_num [oddArm], by norm_num [oddArm], rfl, rfl, rfl, rfl, ?_,
    rfl, rfl, rfl, rfl, oddArm_inputAfterColumns_valid,
    oddArm_outputAfterColumns_valid, oddArm_poseidon2Calls_valid, ?_⟩
  · rw [oddArm_poseidon2Calls_length]
  · exact ⟨oddArm_call_chain,
      by rw [oddArm_poseidon2Calls_length]; norm_num [oddArm],
      by norm_num [oddArm]⟩

/-- The even replay reads the canonical contiguous PiRLC input slice. -/
theorem evenArm_inputColumns_exact :
    evenArm.inputColumns = List.range' 919 918 :=
  evenArm_valid.2.2.2.2.2.2.2.1

/-- The odd replay reads the same canonical contiguous PiRLC input slice. -/
theorem oddArm_inputColumns_exact :
    oddArm.inputColumns = List.range' 919 918 :=
  oddArm_valid.2.2.2.2.2.2.2.1

/-- The even replay reads the canonical contiguous PiRLC output slice. -/
theorem evenArm_outputColumns_exact :
    evenArm.outputColumns = List.range' 1837 54 :=
  evenArm_valid.2.2.2.2.2.2.2.2.1

/-- The odd replay reads the same canonical contiguous PiRLC output slice. -/
theorem oddArm_outputColumns_exact :
    oddArm.outputColumns = List.range' 1837 54 :=
  oddArm_valid.2.2.2.2.2.2.2.2.1

theorem rawArtifact_valid : rawArtifact.Valid :=
  ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, evenArm_valid, oddArm_valid⟩

end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyReplay
