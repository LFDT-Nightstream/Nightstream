import NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcSampler.Definition
import NightstreamFPrime.Spec.Phi81StrongSet
import NightstreamFPrime.Spec.Poseidon2

/-! PiRLC transcript schedule: enter the scalar domain, read one joint
four-field block, and advance once. This deterministic specification makes
no randomness claim about Poseidon2. -/

namespace NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcSampler.Transcript

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Phi81StrongSet

def enter (state : Poseidon2.State) (coordinate : Nat) : Poseidon2.State :=
  Poseidon2.absorbBlock state [Poseidon2.ofNat 4, Poseidon2.ofNat coordinate]

def block (state : Poseidon2.State) : Draw :=
  fun lane => state.getD lane.val 0

def next (state : Poseidon2.State) (coordinate : Nat) : Poseidon2.State :=
  Poseidon2.permute (enter state coordinate)

def stateAt (initial : Poseidon2.State) : Nat → Poseidon2.State
  | 0 => initial
  | coordinate + 1 => next (stateAt initial coordinate) coordinate

def drawAt (initial : Poseidon2.State) (coordinate : Nat) : Draw :=
  block (enter (stateAt initial coordinate) coordinate)

def scalarAt (initial : Poseidon2.State) (coordinate : Nat) : Scalar :=
  sample (drawAt initial coordinate)

def challengeAt (initial : Poseidon2.State) (coordinate : Nat) : RingF :=
  Phi81StrongSet.embedScalar (scalarAt initial coordinate)

theorem challengeAt_member (initial : Poseidon2.State) (coordinate : Nat) :
    Phi81StrongSet.ProductionMember (challengeAt initial coordinate) :=
  ⟨scalarAt initial coordinate, rfl⟩

@[simp] theorem stateAt_zero (initial : Poseidon2.State) : stateAt initial 0 = initial := rfl

@[simp] theorem stateAt_succ (initial : Poseidon2.State) (coordinate : Nat) :
    stateAt initial (coordinate + 1) =
      Poseidon2.permute (enter (stateAt initial coordinate) coordinate) := rfl

theorem scalarAt_digit (initial : Poseidon2.State) (coordinate : Nat)
    (position : Fin coefficientCount) :
    (scalarAt initial coordinate position).val =
      (drawIndex (drawAt initial coordinate)).val % scalarCount /
        alphabetSize ^ position.val % alphabetSize := rfl

end NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcSampler.Transcript
