import Nightstream.Implementation.R1CS.Canonical.Poseidon2HonestFrom
import Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge

/-!
Contract: the exact fixed-23, rate-four, seven-permutation canonical sponge
core used by the two F′ binding hashes.

Owns: the closed chunk schedule, authoritative 23-coordinate input order,
padding on the constant-one wire, a concrete disjoint layout, and soundness of
the normalized 2,464-row core against `Poseidon2Sponge.digest`.

Does not own: typed call activation/output copies, XOut serialization,
Poseidon2 constants, or collision resistance.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Core
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Program
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Schedule
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Layout
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge

abbrev Preimage := Fin sponge23Fields → Nat

def calls : Nat := 7
def dataCalls : Nat := 6
def callStride : Nat := canonicalColumnTotal
def inputBase : Nat := calls * callStride

theorem callStride_eq : callStride = 361 := canonicalColumnTotal_eq
theorem inputBase_eq : inputBase = 2527 := by decide

/-- The 23 authoritative inputs occupy one contiguous visible block after all
seven permutation column spaces. -/
def inputColumn (index : Nat) : Nat := inputBase + index

/-- Five full chunks, one three-coordinate chunk, and the final `[1]` padding
chunk. -/
def chunkLength : Nat → Nat
  | 0 | 1 | 2 | 3 | 4 => 4
  | 5 => 3
  | 6 => 1
  | _ => 0

theorem chunkLength_bounded (call : Nat) : chunkLength call ≤ rate := by
  unfold chunkLength rate
  split <;> omega

def chunkValue (input : Preimage) (call lane : Nat) : Nat :=
  if inInput : call < dataCalls ∧ call * rate + lane < sponge23Fields
  then input ⟨call * rate + lane, inInput.2⟩
  else 1

/-- The chunk at a call is generated at its exact statically selected length.
For call six this is definitionally the singleton padding value. -/
def chunkAt (input : Preimage) (call : Nat) : List Nat :=
  List.ofFn (fun lane : Fin (chunkLength call) =>
    chunkValue input call lane.val)

theorem chunkAt_length (input : Preimage) (call : Nat) :
    (chunkAt input call).length = chunkLength call := by
  simp [chunkAt]

theorem chunkAt_bounded (input : Preimage) (call : Nat) :
    (chunkAt input call).length ≤ rate := by
  rw [chunkAt_length]
  exact chunkLength_bounded call

theorem chunkAt_padding (input : Preimage) :
    chunkAt input 6 = [1] := by
  simp [chunkAt, chunkLength, chunkValue, dataCalls]

/-- Exact six data chunks consumed by the value-level sponge. -/
def dataChunks (input : Preimage) : List RateChunk :=
  chunkList (chunkAt input) (chunkAt_bounded input) dataCalls

theorem dataChunks_length (input : Preimage) :
    (dataChunks input).length = dataCalls := by
  simp [dataChunks, chunkList]

/-- Seven calls use consecutive full permutation spaces.  Data chunk
coordinates point into the authoritative input block; padding points to the
constant-one wire. -/
def layout : SpongeLayout where
  call := fun call => shiftedLayout (call * callStride)
  chunkColumn := fun call lane =>
    if call < dataCalls then inputColumn (call * rate + lane.val) else 0

theorem layout_wellFormed :
    SpongeLayout.WellFormed layout callStride where
  perCall := fun call => shiftedLayout_wellFormed _
  strideClears := Nat.le_refl _
  callAtShift := fun _ => rfl

theorem chunkColumn_data
    (call : Nat) (lane : Fin width) (isData : call < dataCalls) :
    layout.chunkColumn call lane = inputColumn (call * rate + lane.val) := by
  simp [layout, isData]

theorem chunkColumn_padding (lane : Fin width) :
    layout.chunkColumn 6 lane = 0 := by
  simp [layout, dataCalls]

def program (constants : Constants) : List Row :=
  spongeProgram layout constants chunkLength calls

theorem program_length (constants : Constants) :
    (program constants).length = 2464 := by
  unfold program calls
  exact sponge23Program_length layout constants chunkLength

def InputsAgree (z : Nat → Nat) (input : Preimage) : Prop :=
  ∀ index : Fin sponge23Fields, z (inputColumn index.val) = input index

theorem chunkAgrees
    (z : Nat → Nat) (input : Preimage)
    (constantWire : z 0 = 1)
    (inputsAgree : InputsAgree z input)
    (call : Nat) (lane : Fin width) (value : Nat)
    (covering : (chunkAt input call)[lane.val]? = some value) :
    z (layout.chunkColumn call lane) = value := by
  have laneBelow : lane.val < chunkLength call := by
    by_cases below : lane.val < chunkLength call
    · exact below
    · have beyond : chunkLength call ≤ lane.val := Nat.le_of_not_gt below
      have isNone : (chunkAt input call)[lane.val]? = none :=
        List.getElem?_eq_none (by
          rw [chunkAt_length]
          exact beyond)
      rw [isNone] at covering
      simp at covering
  have valueEq : chunkValue input call lane.val = value := by
    rw [List.getElem?_eq_getElem (by simpa [chunkAt_length] using laneBelow)]
      at covering
    simpa [chunkAt] using covering
  by_cases isData : call < dataCalls
  · have indexBelow : call * rate + lane.val < sponge23Fields := by
      simp only [dataCalls, rate, sponge23Fields] at isData laneBelow ⊢
      unfold chunkLength at laneBelow
      split at laneBelow <;> omega
    rw [chunkColumn_data call lane isData,
      inputsAgree ⟨call * rate + lane.val, indexBelow⟩]
    simpa [chunkValue, isData, indexBelow] using valueEq
  · have callEq : call = 6 := by
      simp only [dataCalls] at isData
      unfold chunkLength at laneBelow
      split at laneBelow <;> omega
    subst call
    rw [chunkColumn_padding, constantWire]
    simpa [chunkValue, dataCalls] using valueEq

theorem chunkList_seven (input : Preimage) :
    chunkList (chunkAt input) (chunkAt_bounded input) calls
      = dataChunks input ++ [paddingChunk] := by
  unfold calls dataChunks dataCalls
  rw [chunkList_succ]
  congr 2

/-- **Fixed-23 core soundness.**  Every satisfying assignment with
authoritative inputs and constant wire computes the selected value-level
sponge digest on final call lanes zero through three. -/
theorem program_computes_digest
    (constants : Constants) (z : Nat → Nat) (input : Preimage)
    (residues : ∀ column, z column < goldilocksP)
    (constantWire : z 0 = 1)
    (inputsAgree : InputsAgree z input)
    (satisfied : Satisfies (program constants) z)
    (lane : Fin digestLength) :
    z ((layout.call 6).outputPort
        ⟨lane.val, by
          have laneLt := lane.isLt
          simp only [digestLength, width] at laneLt ⊢
          omega⟩)
      = digest constants (dataChunks input) lane := by
  let outputLane : Fin width :=
    ⟨lane.val, by
      have laneLt := lane.isLt
      simp only [digestLength, width] at laneLt ⊢
      omega⟩
  have result :=
    spongeProgram_computes_digest layout constants chunkLength calls z
      (chunkAt input) (chunkAt_bounded input) residues constantWire satisfied
      (fun call => (chunkAt_length input call).symm)
      (chunkAgrees z input constantWire inputsAgree)
      6 (by decide) outputLane
  change z ((layout.call 6).outputPort outputLane) =
    absorb constants
      (chunkList (chunkAt input) (chunkAt_bounded input) calls)
      initialSpongeState outputLane at result
  rw [chunkList_seven, ← spongeFinal_eq_absorb_padding] at result
  change z ((layout.call 6).outputPort outputLane)
    = digest constants (dataChunks input) lane
  exact result

end Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23
