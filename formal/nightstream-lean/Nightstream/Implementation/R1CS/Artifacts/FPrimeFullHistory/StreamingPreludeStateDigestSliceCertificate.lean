import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPreludeSource

/-!
Leaf certificate for the exact Rust rows that own the Prelude state digest.

Owns four Poseidon2 calls and the residual constants, pad, and digest aliases
needed by the typed Prelude relation. It does not evaluate the complete source
artifact or authorize removal of any other row.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPreludeStateDigestSliceCertificate

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplay.Artifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPreludeSource

def calls : List Poseidon2Call.Call :=
  poseidonCallsPart0.take 4

def call0 : Poseidon2Call.Call :=
  { rowStart := 26, rowEnd := 626,
    inputColumns := [23, 24, 25, 26, 19, 20, 21, 22],
    firstAllocatedColumn := 27 }

def call1 : Poseidon2Call.Call :=
  { rowStart := 626, rowEnd := 1226,
    inputColumns := [1, 2, 3, 4, 623, 624, 625, 626],
    firstAllocatedColumn := 627 }

def call2 : Poseidon2Call.Call :=
  { rowStart := 1226, rowEnd := 1826,
    inputColumns := [5, 6, 7, 8, 1223, 1224, 1225, 1226],
    firstAllocatedColumn := 1227 }

def call3 : Poseidon2Call.Call :=
  { rowStart := 1827, rowEnd := 2427,
    inputColumns := [9, 10, 1827, 1822, 1823, 1824, 1825, 1826],
    firstAllocatedColumn := 1828 }

theorem calls_exact :
    calls =
      [call0, call1, call2, call3] := by
  rfl

theorem calls_subset :
    ∀ call ∈ calls, call ∈ artifact.poseidon2Calls := by
  intro call member
  exact poseidonCallsPart0_subset call (List.mem_of_mem_take member)

theorem call0_member : call0 ∈ artifact.poseidon2Calls :=
  calls_subset call0 (by rw [calls_exact]; simp)

theorem call1_member : call1 ∈ artifact.poseidon2Calls :=
  calls_subset call1 (by rw [calls_exact]; simp)

theorem call2_member : call2 ∈ artifact.poseidon2Calls :=
  calls_subset call2 (by rw [calls_exact]; simp)

theorem call3_member : call3 ∈ artifact.poseidon2Calls :=
  calls_subset call3 (by rw [calls_exact]; simp)

def prefixAndBeforeRows : List IndexedRow :=
  residualRows0Part0.take 27 ++ (residualRows0Part0.drop 29).take 4

theorem prefixAndBeforeRows_exact :
    prefixAndBeforeRows =
      [{ index := 0, row := ⟨[(1, 1)], [(0, 1)], []⟩ },
       { index := 1, row := ⟨[(2, 1)], [(0, 1)], []⟩ },
       { index := 2, row := ⟨[(3, 1)], [(0, 1)], []⟩ },
       { index := 3, row := ⟨[(4, 1)], [(0, 1)], []⟩ },
       { index := 4, row := ⟨[(5, 1)], [(0, 1)], []⟩ },
       { index := 5, row := ⟨[(6, 1)], [(0, 1)], []⟩ },
       { index := 6, row := ⟨[(7, 1)], [(0, 1)], []⟩ },
       { index := 7, row := ⟨[(8, 1)], [(0, 1)], []⟩ },
       { index := 8, row := ⟨[(9, 1)], [(0, 1)], []⟩ },
       { index := 9, row := ⟨[(10, 1)], [(0, 1)], []⟩ },
       { index := 10, row := ⟨[(11, 1)], [(0, 1)], []⟩ },
       { index := 11, row := ⟨[(12, 1)], [(0, 1)], []⟩ },
       { index := 12, row := ⟨[(13, 1)], [(0, 1)], []⟩ },
       { index := 13, row := ⟨[(14, 1)], [(0, 1)], []⟩ },
       { index := 14, row := ⟨[(0, 15802041653089849246), (15, 1)], [(0, 1)], []⟩ },
       { index := 15, row := ⟨[(0, 9594157335387961847), (16, 1)], [(0, 1)], []⟩ },
       { index := 16, row := ⟨[(0, 7485132455936495468), (17, 1)], [(0, 1)], []⟩ },
       { index := 17, row := ⟨[(0, 14687844690344412664), (18, 1)], [(0, 1)], []⟩ },
       { index := 18, row := ⟨[(0, 9361295336785638312), (19, 1)], [(0, 1)], []⟩ },
       { index := 19, row := ⟨[(0, 4766135141031501574), (20, 1)], [(0, 1)], []⟩ },
       { index := 20, row := ⟨[(0, 16455650279185320667), (21, 1)], [(0, 1)], []⟩ },
       { index := 21, row := ⟨[(0, 11540510937154493680), (22, 1)], [(0, 1)], []⟩ },
       { index := 22, row := ⟨[(0, 18446744069414584319), (23, 1)], [(0, 1)], []⟩ },
       { index := 23, row := ⟨[(0, 18446744069414584316), (24, 1)], [(0, 1)], []⟩ },
       { index := 24, row := ⟨[(0, 18446743633670343566), (25, 1)], [(0, 1)], []⟩ },
       { index := 25, row := ⟨[(0, 18446744069414584311), (26, 1)], [(0, 1)], []⟩ },
       { index := 1826, row := ⟨[(0, 18446744069414584320), (1827, 1)], [(0, 1)], []⟩ },
       { index := 2429, row := ⟨[(11, 18446744069414584320), (2430, 1)], [(0, 1)], []⟩ },
       { index := 2430, row := ⟨[(12, 18446744069414584320), (2431, 1)], [(0, 1)], []⟩ },
       { index := 2431, row := ⟨[(13, 18446744069414584320), (2432, 1)], [(0, 1)], []⟩ },
       { index := 2432, row := ⟨[(14, 18446744069414584320), (2433, 1)], [(0, 1)], []⟩ }] := by
  rfl

theorem prefixAndBeforeRows_subset :
    ∀ indexed ∈ prefixAndBeforeRows, indexed ∈ artifact.residualRows := by
  intro indexed member
  unfold prefixAndBeforeRows at member
  rw [List.mem_append] at member
  rcases member with prefixMember | beforeMember
  · exact residualRows0Part0_subset indexed
      (List.mem_of_mem_take prefixMember)
  · exact residualRows0Part0_subset indexed
      (List.mem_of_mem_drop (List.mem_of_mem_take beforeMember))

def afterRows : List IndexedRow :=
  (residualRows1Part3.drop 34).take 4

theorem afterRows_exact :
    afterRows =
      [{ index := 4602, row := ⟨[(2420, 18446744069414584320), (4603, 1)], [(0, 1)], []⟩ },
       { index := 4603, row := ⟨[(2421, 18446744069414584320), (4604, 1)], [(0, 1)], []⟩ },
       { index := 4604, row := ⟨[(2422, 18446744069414584320), (4605, 1)], [(0, 1)], []⟩ },
       { index := 4605, row := ⟨[(2423, 18446744069414584320), (4606, 1)], [(0, 1)], []⟩ }] := by
  rfl

theorem afterRows_subset :
    ∀ indexed ∈ afterRows, indexed ∈ artifact.residualRows := by
  intro indexed member
  unfold afterRows at member
  exact residualRows1Part3_subset indexed
    (List.mem_of_mem_drop (List.mem_of_mem_take member))

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPreludeStateDigestSliceCertificate
