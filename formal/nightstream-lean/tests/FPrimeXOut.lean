import Nightstream.Protocol.FPrime.XOut

/-!
`FPR-HASH` witnesses. Direct preimage coordinates change the state-output
message; deliberately omitted coordinates are protected by `StatePinned`.
The constant-hash case demonstrates that equal outputs for distinct authority
views reach the theorem's explicit collision branch rather than becoming a
false equality claim.
-/

namespace NightstreamTests.FPrimeXOut

open Nightstream.HyperNova.Construction2
open Nightstream.Protocol.FPrime.XOut

def optionCode : Option Nat → Nat
  | none => 0
  | some value => value + 1

/-- Readable test encoding; production instantiates the same message type with Poseidon2. -/
def toyHash : Message Nat Nat Nat Nat Nat → Nat
  | .verifier preimage =>
      1000000 + preimage.params + 10 * preimage.structureDigest +
        100 * preimage.piCcsHeader + 1000 * optionCode preimage.publicInputLength +
        10000 * preimage.initialSemanticState
  | .initialBoundary preimage =>
      2000000 + preimage.structureDigest +
        1000 * optionCode preimage.publicInputLength
  | .publicTraceSeed preimage =>
      2500000 + preimage.structureDigest
  | .stateOutput preimage =>
      3000000 + preimage.vkFsDigest + 3 * preimage.piCcsHeader +
        5 * preimage.chunkCount + 7 * preimage.stepCount + 11 * preimage.pc +
        13 * preimage.currentBoundary + 17 * optionCode preimage.semanticState +
        19 * preimage.construction2Accumulator + 23 * optionCode preimage.nebula

def semantics : Semantics Nat Nat Nat Nat Nat Nat where
  hash := toyHash
  nebulaDigest := id

def context : Context Nat Nat Nat Nat where
  params := 2
  structureDigest := 3
  piCcsHeader := 5
  publicInputLength := some 7
  initialSemanticState := 11

abbrev TestState := State Nat Unit Unit Nat

def pinnedState : TestState where
  chunkCount := 2
  stepCount := 4
  z0 := initialBoundary semantics context
  zi := 31
  initialSemanticState := 11
  semanticState := 41
  pc := 1
  accumulatorDigest := 41
  publicTrace := 31
  proof := .active () [()]
  nebula := some 13

example : StatePinned semantics .stateless context pinnedState := by decide
example : StatePinned semantics .stateful context pinnedState := by decide

-- Every coordinate directly owned by the compact Rust preimage is observable.
example : preimage semantics .stateless context pinnedState ≠
    preimage semantics .stateless context { pinnedState with chunkCount := 3 } := by decide

example : preimage semantics .stateless context pinnedState ≠
    preimage semantics .stateless context { pinnedState with stepCount := 5 } := by decide

example : preimage semantics .stateless context pinnedState ≠
    preimage semantics .stateless context { pinnedState with pc := 2 } := by decide

example : preimage semantics .stateless context pinnedState ≠
    preimage semantics .stateless context { pinnedState with zi := 32, publicTrace := 32 } := by
  decide

example : preimage semantics .stateless context pinnedState ≠
    preimage semantics .stateless context
      { pinnedState with accumulatorDigest := 42, semanticState := 42 } := by decide

example : preimage semantics .stateful context pinnedState ≠
    preimage semantics .stateful context { pinnedState with semanticState := 42 } := by decide

example : preimage semantics .stateless context pinnedState ≠
    preimage semantics .stateless context { pinnedState with nebula := some 14 } := by decide

-- Stateless mode deliberately omits a duplicate semantic lane; pinning protects it.
example : preimage semantics .stateless context pinnedState =
    preimage semantics .stateless context { pinnedState with semanticState := 42 } := by decide

example : ¬ StatePinned semantics .stateless context
    { pinnedState with semanticState := 42 } := by decide

-- The three other omitted fields are verifier-derived or equality-pinned.
example : ¬ StatePinned semantics .stateless context
    { pinnedState with z0 := pinnedState.z0 + 1 } := by decide

example : ¬ StatePinned semantics .stateless context
    { pinnedState with initialSemanticState := 12 } := by decide

example : ¬ StatePinned semantics .stateless context
    { pinnedState with publicTrace := 30 } := by decide

def constantSemantics : Semantics Nat Nat Nat Nat Nat Nat where
  hash := fun _ => 0
  nebulaDigest := id

def constantPinned : TestState :=
  { pinnedState with z0 := initialBoundary constantSemantics context }

/-- Distinct authority under equal outputs is exposed as a named compression failure. -/
example : BindingFailure constantSemantics := by
  let changed := { constantPinned with chunkCount := 3 }
  have leftPinned : StatePinned constantSemantics .stateless context constantPinned := by
    decide
  have rightPinned : StatePinned constantSemantics .stateless context changed := by
    decide
  rcases xOut_binding_or_collision constantSemantics .stateless .stateless
      context context constantPinned changed leftPinned rightPinned rfl with
    sameAuthority | collision
  · have differentAuthority :
        authorityView constantSemantics .stateless context constantPinned ≠
          authorityView constantSemantics .stateless context changed := by decide
    exact False.elim (differentAuthority sameAuthority)
  · exact collision

-- The optional lane's inner compression is an independently named boundary.
example : NebulaDigestCollision (fun _ : Nat => 0) := by
  exact ⟨13, 14, by decide, rfl⟩

end NightstreamTests.FPrimeXOut
