import Nightstream.Implementation.Rust.Terminal

/-! Executable positive and one-coordinate-negative witnesses for TERM-CE/RUST-REFINE. -/

namespace NightstreamTests.TerminalCE

open Nightstream.Protocol
open Nightstream.Implementation.Rust

abbrev Assignment := List Nat
abbrev TestClaim := TerminalCE.Claim (List Nat) Nat Nat Nat Nat Bool
abbrev TestInstance := TerminalCE.Instance Nat Assignment (List Nat) Nat Nat Nat Nat Bool

def semantics : TerminalCE.Semantics Nat Assignment (List Nat) Nat Nat Nat Nat Bool where
  commit := List.sum
  projectPublicInput := fun width witness =>
    if witness.length = width then some (witness.take width) else none
  normBounded := fun bound witness => witness.all (fun value => value < bound)
  evaluationPointValid := fun relation point => relation == point
  evaluations := fun relation witness point =>
    if relation = point then some (witness.map (fun value => value + point)) else none
  constantTerm := id
  sidecarValid := fun _ _ sidecar => sidecar

def context : TerminalCE.Context Nat where
  relation := 3
  normBound := 10
  expectedPublicWidth := some 2

def claimFor (witness : Assignment) : TestClaim where
  commitment := witness.sum
  publicWidth := witness.length
  publicInput := witness
  point := 3
  evaluations := witness.map (fun value => value + 3)
  constantTerms := witness.map (fun value => value + 3)
  sidecar := true

def honestWitness : Assignment := [2, 3]
def honestClaim : TestClaim := claimFor honestWitness

def honest : TestInstance where
  context := context
  verifierChildren := [honestClaim]
  recordedClaims := [honestClaim]
  witnesses := [honestWitness]

def resultCode : Except Terminal.Error Unit → Nat
  | .ok _ => 0
  | .error .childAuthority => 1
  | .error .witnessCount => 2
  | .error .publicWidth => 3
  | .error .commitment => 4
  | .error .publicProjection => 5
  | .error .norm => 6
  | .error .evaluationPoint => 7
  | .error .evaluations => 8
  | .error .constantTerms => 9
  | .error .sidecar => 10

example : TerminalCE.check semantics honest = true := by native_decide

example : TerminalCE.Holds semantics honest :=
  TerminalCE.terminalCE_sound semantics honest (by native_decide)

example : resultCode (Terminal.verify semantics honest) = 0 := by native_decide

example : resultCode (Terminal.verify semantics { honest with recordedClaims := [] }) = 1 := by
  native_decide

example : resultCode (Terminal.verify semantics { honest with witnesses := [] }) = 2 := by
  native_decide

example : resultCode (Terminal.verify semantics {
      honest with
      verifierChildren := [{ honestClaim with publicWidth := 1 }]
      recordedClaims := [{ honestClaim with publicWidth := 1 }]
    }) = 3 := by native_decide

example : resultCode (Terminal.verify semantics {
      honest with
      verifierChildren := [{ honestClaim with commitment := 6 }]
      recordedClaims := [{ honestClaim with commitment := 6 }]
    }) = 4 := by native_decide

example : resultCode (Terminal.verify semantics {
      honest with
      verifierChildren := [{ honestClaim with publicInput := [3, 2] }]
      recordedClaims := [{ honestClaim with publicInput := [3, 2] }]
    }) = 5 := by native_decide

def highWitness : Assignment := [12, 3]
def highClaim : TestClaim := claimFor highWitness

example : resultCode (Terminal.verify semantics {
      honest with
      verifierChildren := [highClaim]
      recordedClaims := [highClaim]
      witnesses := [highWitness]
    }) = 6 := by native_decide

example : resultCode (Terminal.verify semantics {
      honest with
      verifierChildren := [{ honestClaim with point := 4 }]
      recordedClaims := [{ honestClaim with point := 4 }]
    }) = 7 := by native_decide

example : resultCode (Terminal.verify semantics {
      honest with
      verifierChildren := [{ honestClaim with evaluations := [5, 7] }]
      recordedClaims := [{ honestClaim with evaluations := [5, 7] }]
    }) = 8 := by native_decide

example : resultCode (Terminal.verify semantics {
      honest with
      verifierChildren := [{ honestClaim with constantTerms := [5, 7] }]
      recordedClaims := [{ honestClaim with constantTerms := [5, 7] }]
    }) = 9 := by native_decide

example : resultCode (Terminal.verify semantics {
      honest with
      verifierChildren := [{ honestClaim with sidecar := false }]
      recordedClaims := [{ honestClaim with sidecar := false }]
    }) = 10 := by native_decide

example : TerminalCE.check semantics {
      honest with
      verifierChildren := [{ honestClaim with commitment := 6 }]
      recordedClaims := [{ honestClaim with commitment := 6 }]
    } = false := by native_decide

end NightstreamTests.TerminalCE
