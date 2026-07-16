import Nightstream.Implementation.R1CS.Core.ChaCha8
import Nightstream.SuperNeo.Concrete.Algebra

/-!
Canonical rejection sampler for seeded `Phi_81` coefficient vectors.

Assurance tier: executable primitive semantics. The algorithm is parameterized
by a finite `u64` stream so its sampling and cursor rules have one owner. The
inductive `FirstAccepted` relation gives the unbounded mathematical meaning;
the fuel-bounded executable is proved sound for every successful result.

Owns: 54-candidate vector width; initial-vector and replacement cursor rules;
chunk/message traversal; bounded execution; and unbounded first-acceptance
semantics.

Does not own: a ChaCha8 implementation; Rust `rand_chacha` conformance;
verifier-owned seed derivation; Phi81 rotation; SIS security; R1CS rows;
Poseidon2; transcript authority; row removal; or cost totals.

Emits constraints: no.

Authority boundary: a `WordStream` is an explicit parameter. A later theorem
must identify the selected stream with verifier-owned pure ChaCha8 semantics.
Fuel is an execution bound, never part of the unbounded acceptance relation.

| Protocol | Phase | Mathematical branch | Definition/theorem | Exact guarantee |
|---|---|---|---|---|
| seeded SIS | coefficient sampling | first accepted field word | `FirstAccepted` | rejected words are skipped until the first canonical Goldilocks word |
| seeded SIS | coefficient sampling | bounded execution | `nextAccepted_sound` | every successful bounded result satisfies unbounded first-acceptance |
| seeded SIS | coefficient sampling | liveness witness | `FirstAccepted.exists_fuel` | every finite unbounded derivation succeeds for some fuel |
| seeded SIS | coefficient sampling | vector cursor | `sampleVector` | 54 initial words followed by sequential replacements |
| seeded SIS | coefficient sampling | chunk traversal | `Schedule.baseRotations` | output/chunk/message traversal has one canonical owner |
-/

namespace Nightstream.Implementation.R1CS.SeededPhi81Sampler

def dimension : Nat := 54

def modulus : Nat := Nightstream.SuperNeo.Concrete.goldilocksModulus

/-- Finite little-endian `u64` slice from a deterministic stream. -/
abbrev WordStream := List Nat -> Nat -> Nat -> List Nat

def pureStream : WordStream := ChaCha8.u64s

def candidateAt (stream : WordStream) (seed : List Nat)
    (wordPosition : Nat) : Nat :=
  (stream seed wordPosition 1).getD 0 0

/-- Unbounded mathematical statement that `value` is the first canonical
field word at or after `wordPosition`, with two `u32` words consumed per
candidate. -/
inductive FirstAccepted (stream : WordStream) (seed : List Nat) :
    Nat -> Nat -> Nat -> Prop
  | here (wordPosition : Nat)
      (accepted : candidateAt stream seed wordPosition < modulus) :
      FirstAccepted stream seed wordPosition
        (candidateAt stream seed wordPosition) (wordPosition + 2)
  | later (wordPosition value nextPosition : Nat)
      (rejected : modulus <= candidateAt stream seed wordPosition)
      (tail : FirstAccepted stream seed (wordPosition + 2) value nextPosition) :
      FirstAccepted stream seed wordPosition value nextPosition

def nextAccepted (stream : WordStream) (seed : List Nat) :
    Nat -> Nat -> Option (Nat × Nat)
  | _, 0 => none
  | wordPosition, fuel + 1 =>
      let candidate := candidateAt stream seed wordPosition
      if candidate < modulus then some (candidate, wordPosition + 2)
      else nextAccepted stream seed (wordPosition + 2) fuel

theorem nextAccepted_sound
    {stream : WordStream} {seed : List Nat}
    {wordPosition fuel value nextPosition : Nat}
    (success : nextAccepted stream seed wordPosition fuel =
      some (value, nextPosition)) :
    FirstAccepted stream seed wordPosition value nextPosition := by
  induction fuel generalizing wordPosition with
  | zero => simp [nextAccepted] at success
  | succ fuel ih =>
      by_cases accepted : candidateAt stream seed wordPosition < modulus
      · have success' :
            some (candidateAt stream seed wordPosition, wordPosition + 2) =
              some (value, nextPosition) := by
          simpa [nextAccepted, accepted] using success
        have pairEq := Option.some.inj success'
        cases pairEq
        exact .here wordPosition accepted
      · simp [nextAccepted, accepted] at success
        exact .later wordPosition value nextPosition
          (Nat.le_of_not_gt accepted) (ih success)

theorem FirstAccepted.unique
    {stream : WordStream} {seed : List Nat}
    {wordPosition leftValue leftNext rightValue rightNext : Nat}
    (left : FirstAccepted stream seed wordPosition leftValue leftNext)
    (right : FirstAccepted stream seed wordPosition rightValue rightNext) :
    leftValue = rightValue /\ leftNext = rightNext := by
  induction left with
  | here wordPosition accepted =>
      cases right with
      | here => exact ⟨rfl, rfl⟩
      | later _ _ _ rejected _ => omega
  | later wordPosition value nextPosition rejected tail ih =>
      cases right with
      | here _ accepted => omega
      | later _ _ _ _ rightTail => exact ih rightTail

theorem FirstAccepted.exists_fuel
    {stream : WordStream} {seed : List Nat}
    {wordPosition value nextPosition : Nat}
    (accepted : FirstAccepted stream seed wordPosition value nextPosition) :
    exists fuel,
      nextAccepted stream seed wordPosition fuel = some (value, nextPosition) := by
  induction accepted with
  | here wordPosition accepted =>
      exact ⟨1, by simp [nextAccepted, accepted]⟩
  | later wordPosition value nextPosition rejected _ ih =>
      rcases ih with ⟨fuel, success⟩
      exact ⟨fuel + 1, by
        simp [nextAccepted, Nat.not_lt.mpr rejected, success]⟩

def repairRejected (stream : WordStream) (seed : List Nat) (fuel : Nat) :
    List Nat -> Nat -> Option (List Nat × Nat)
  | [], wordPosition => some ([], wordPosition)
  | candidate :: tail, wordPosition =>
      let accepted :=
        if candidate < modulus then some (candidate, wordPosition)
        else nextAccepted stream seed wordPosition fuel
      match accepted with
      | none => none
      | some (value, nextPosition) =>
          match repairRejected stream seed fuel tail nextPosition with
          | none => none
          | some (values, finalPosition) =>
              some (value :: values, finalPosition)

def sampleVector (stream : WordStream) (seed : List Nat)
    (fuel wordPosition : Nat) : Option (List Nat × Nat) :=
  let raw := stream seed wordPosition dimension
  repairRejected stream seed fuel raw (wordPosition + 2 * dimension)

def sampleVectors.go (stream : WordStream) (seed : List Nat) (fuel : Nat) :
    Nat -> Nat -> List (List Nat) -> Option (List (List Nat))
  | 0, _, reversed => some reversed.reverse
  | count + 1, wordPosition, reversed =>
      match sampleVector stream seed fuel wordPosition with
      | none => none
      | some (vector, nextPosition) =>
          sampleVectors.go stream seed fuel count nextPosition
            (vector :: reversed)

def sampleVectors (stream : WordStream) (seed : List Nat) (fuel : Nat)
    (count wordPosition : Nat) : Option (List (List Nat)) :=
  sampleVectors.go stream seed fuel count wordPosition []

def chunkMessageCount (messageCols chunkSize chunkIndex : Nat) : Nat :=
  let start := chunkIndex * chunkSize
  if start < messageCols then Nat.min chunkSize (messageCols - start) else 0

def sampleOutput (stream : WordStream) (messageCols chunkSize fuel : Nat) :
    Nat -> List (List Nat) -> Option (List (List Nat))
  | _, [] => some []
  | chunkIndex, seed :: tail =>
      match sampleVectors stream seed fuel
          (chunkMessageCount messageCols chunkSize chunkIndex) 0 with
      | none => none
      | some vectors =>
          match sampleOutput stream messageCols chunkSize fuel
              (chunkIndex + 1) tail with
          | none => none
          | some rest => some (vectors ++ rest)

structure Schedule where
  chunkSize : Nat
  seedsByOutput : List (List (List Nat))
  rejectionFuel : Nat
deriving DecidableEq, Repr

def sampleScheduleOutputs (stream : WordStream) (messageCols chunkSize fuel : Nat) :
    List (List (List Nat)) -> Option (List (List (List Nat)))
  | [] => some []
  | seeds :: tail =>
      match sampleOutput stream messageCols chunkSize fuel 0 seeds with
      | none => none
      | some rotations =>
          match sampleScheduleOutputs stream messageCols chunkSize fuel tail with
          | none => none
          | some rest => some (rotations :: rest)

def Schedule.baseRotations (schedule : Schedule) (stream : WordStream)
    (messageCols : Nat) : Option (List (List (List Nat))) :=
  sampleScheduleOutputs stream messageCols schedule.chunkSize
    schedule.rejectionFuel schedule.seedsByOutput

theorem Schedule.baseRotations_congr
    (schedule : Schedule) (left right : WordStream) (messageCols : Nat)
    (streamsEqual : forall seed wordStart count,
      left seed wordStart count = right seed wordStart count) :
    schedule.baseRotations left messageCols =
      schedule.baseRotations right messageCols := by
  have streamEquality : left = right := by
    funext seed wordStart count
    exact streamsEqual seed wordStart count
  subst right
  rfl

end Nightstream.Implementation.R1CS.SeededPhi81Sampler
