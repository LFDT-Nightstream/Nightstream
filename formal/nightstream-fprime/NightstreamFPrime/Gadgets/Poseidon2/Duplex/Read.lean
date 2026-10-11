import NightstreamFPrime.Circuit.Quadratic
import NightstreamFPrime.Gadgets.Poseidon2.Permutation

/-!
Owns one quadratic-extension read for the Poseidon2 duplex gadget. It reads
the rate lanes `2 · pair` and `2 · pair + 1` of the current state and does not
permute. It allocates no recipe and defines no protocol schedule.
-/

namespace NightstreamFPrime.Gadgets.Poseidon2.Duplex.Read

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Gadgets.Poseidon2

abbrev EState := Layer.EState

/-- The six quadratic-extension values of one rate chunk. -/
abbrev Pair := Fin (Spec.Poseidon2.rate / 2)

def lowLane (pair : Pair) : Fin 16 :=
  ⟨2 * pair.val, by have bound : pair.val < 6 := pair.isLt; omega⟩

def highLane (pair : Pair) : Fin 16 :=
  ⟨2 * pair.val + 1, by have bound : pair.val < 6 := pair.isLt; omega⟩

def sample (state : EState) (pair : Pair) : KExpr :=
  ⟨state (lowLane pair), state (highLane pair)⟩

def referenceSample (state : Spec.Poseidon2.State) (pair : Pair) : K :=
  ⟨state.getD (2 * pair.val) 0, state.getD (2 * pair.val + 1) 0⟩

private theorem evalState_getD (env : Env) (state : EState) (lane : Fin 16) :
    (List.ofFn (Layer.evalState env state)).getD lane.val 0 = (state lane).eval env := by
  rw [List.getD_eq_getElem _ _ (by simp)]
  simp only [List.getElem_ofFn]
  rfl

theorem sample_eval (env : Env) (state : EState) (pair : Pair) :
    (sample state pair).eval env =
      referenceSample (List.ofFn (Layer.evalState env state)) pair := by
  unfold sample referenceSample KExpr.eval
  exact congrArg₂ K.mk (evalState_getD env state (lowLane pair)).symm
    (evalState_getD env state (highLane pair)).symm

theorem sample_below (state : EState) (pair : Pair) (bound : Nat)
    (stateBelow : ∀ lane, (state lane).VarsBelow bound) :
    (sample state pair).c0.VarsBelow bound ∧
      (sample state pair).c1.VarsBelow bound :=
  ⟨stateBelow (lowLane pair), stateBelow (highLane pair)⟩

end NightstreamFPrime.Gadgets.Poseidon2.Duplex.Read
