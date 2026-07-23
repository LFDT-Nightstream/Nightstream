import Nightstream.Implementation.Rust.CanonicalConformance.OneSlot

namespace Nightstream.Tests.FPrimeCanonicalConformanceOneSlot

open Nightstream.Implementation.Rust.CanonicalConformance.OneSlot

def key : Key := ⟨1⟩
def z0 : State := ⟨10⟩
def zBaseNext : State := ⟨11⟩
def zRecursive : State := ⟨12⟩
def zRecursiveNext : State := ⟨13⟩
def wrongState : State := ⟨14⟩
def defaultRunning : Running := ⟨20⟩
def running : Running := ⟨21⟩
def folded : Running := ⟨22⟩
def wrongRunning : Running := ⟨23⟩
def fresh : Fresh := ⟨30⟩
def witness : Witness := ⟨40⟩
def priorDigest : Digest := ⟨51⟩
def nextDigest : Digest := ⟨52⟩
def wrongDigest : Digest := ⟨53⟩
def priorEncoded : Encoded := ⟨61⟩
def nifsProof : NifsProof := ⟨70⟩
def runningWitness : RunningWitness := ⟨80⟩
def freshWitness : FreshWitness := ⟨81⟩

def baseNextHash : HashReceipt where
  input := {
    verifierKey := key
    iteration := 1
    z0 := z0
    current := zBaseNext
    running := defaultRunning
    pc := 1
  }
  output := nextDigest

def baseStep : StepCase where
  verifierKey := key
  defaultRunning := defaultRunning
  iteration := 0
  z0 := z0
  zi := z0
  running := running
  fresh := fresh
  priorPc := 1
  witness := witness
  nifsProof := nifsProof
  stepReceipt := { state := z0, witness := witness, output := zBaseNext }
  trace := .base baseNextHash
  claim := {
    zNext := zBaseNext
    runningNext := defaultRunning
    pcNext := 0
    x := nextDigest
  }
  rustAccepted := true

def baseClaimMismatch : StepCase :=
  { baseStep with
    claim := {
      zNext := wrongState
      runningNext := defaultRunning
      pcNext := 0
      x := nextDigest
    }
    rustAccepted := false }

def recursivePriorHash : HashReceipt where
  input := {
    verifierKey := key
    iteration := 1
    z0 := z0
    current := zRecursive
    running := running
    pc := 1
  }
  output := priorDigest

def recursiveNextHash : HashReceipt where
  input := {
    verifierKey := key
    iteration := 2
    z0 := z0
    current := zRecursiveNext
    running := folded
    pc := 1
  }
  output := nextDigest

def recursiveTrace : StepTrace := .recursive
  recursivePriorHash
  { input := fresh, output := priorEncoded }
  { input := priorDigest, output := priorEncoded }
  {
    key := key
    running := running
    fresh := fresh
    proof := nifsProof
    output := folded
  }
  recursiveNextHash

def recursiveStep : StepCase where
  verifierKey := key
  defaultRunning := defaultRunning
  iteration := 1
  z0 := z0
  zi := zRecursive
  running := running
  fresh := fresh
  priorPc := 1
  witness := witness
  nifsProof := nifsProof
  stepReceipt := {
    state := zRecursive
    witness := witness
    output := zRecursiveNext
  }
  trace := recursiveTrace
  claim := {
    zNext := zRecursiveNext
    runningNext := folded
    pcNext := 0
    x := nextDigest
  }
  rustAccepted := true

def recursiveStateClaimMismatch : StepCase :=
  { recursiveStep with
    claim := { recursiveStep.claim with zNext := wrongState }
    rustAccepted := false }

def recursiveRunningClaimMismatch : StepCase :=
  { recursiveStep with
    claim := { recursiveStep.claim with runningNext := wrongRunning }
    rustAccepted := false }

def recursivePcClaimMismatch : StepCase :=
  { recursiveStep with
    claim := { recursiveStep.claim with pcNext := 1 }
    rustAccepted := false }

def recursiveDigestClaimMismatch : StepCase :=
  { recursiveStep with
    claim := { recursiveStep.claim with x := wrongDigest }
    rustAccepted := false }

/-- Exactly seven proof-free cases.  Every stored receipt belongs to an actual
call in its branch; mutations affect only the compared public claim. -/
def stepCases : List StepCase := [
  baseStep,
  baseClaimMismatch,
  recursiveStep,
  recursiveStateClaimMismatch,
  recursiveRunningClaimMismatch,
  recursivePcClaimMismatch,
  recursiveDigestClaimMismatch
]

def terminalPriorHash : HashReceipt where
  input := {
    verifierKey := key
    iteration := 1
    z0 := z0
    current := zRecursive
    running := running
    pc := 1
  }
  output := priorDigest

def terminalTrace (runningAccepted freshAccepted : Bool) : TerminalTrace :=
  .recursive
    terminalPriorHash
    { input := fresh, output := priorEncoded }
    { input := priorDigest, output := priorEncoded }
    {
      key := key
      value := running
      witness := runningWitness
      accepted := runningAccepted
    }
    {
      key := key
      value := fresh
      witness := freshWitness
      accepted := freshAccepted
    }

def recursiveTerminalWith
    (runningAccepted freshAccepted rustAccepted : Bool) : TerminalCase where
  verifierKey := key
  defaultRunning := defaultRunning
  iteration := 1
  z0 := z0
  zi := zRecursive
  running := running
  runningWitness := runningWitness
  fresh := fresh
  freshWitness := freshWitness
  pc := 1
  trace := terminalTrace runningAccepted freshAccepted
  rustAccepted := rustAccepted

def baseTerminal : TerminalCase where
  verifierKey := key
  defaultRunning := defaultRunning
  iteration := 0
  z0 := z0
  zi := z0
  running := running
  runningWitness := runningWitness
  fresh := fresh
  freshWitness := freshWitness
  pc := 1
  trace := .base
  rustAccepted := true

def baseTerminalEndpointMismatch : TerminalCase :=
  { baseTerminal with zi := zRecursive, rustAccepted := false }

/-- A branch-tag mismatch has no orphan receipts: the base trace carries none
and the schema rejects it before canonical evaluation. -/
def terminalBranchTagMismatch : TerminalCase :=
  { baseTerminal with iteration := 1, rustAccepted := false }

def recursiveTerminal : TerminalCase :=
  recursiveTerminalWith true true true

def recursiveTerminalRunningRejected : TerminalCase :=
  recursiveTerminalWith false true false

def recursiveTerminalFreshRejected : TerminalCase :=
  recursiveTerminalWith true false false

def recursiveTerminalBothRejected : TerminalCase :=
  recursiveTerminalWith false false false

/-- Exactly seven proof-free cases with a branch tag or one receipt per
canonical terminal call position. -/
def terminalCases : List TerminalCase := [
  baseTerminal,
  baseTerminalEndpointMismatch,
  terminalBranchTagMismatch,
  recursiveTerminal,
  recursiveTerminalRunningRejected,
  recursiveTerminalFreshRejected,
  recursiveTerminalBothRejected
]

example : stepCases.length = 7 := by decide
example : terminalCases.length = 7 := by decide
example : stepCases.length + terminalCases.length = 14 := by decide

example : stepCases.all stepAgrees = true := by decide
example : terminalCases.all terminalAgrees = true := by decide

#check stepAgrees_eq_true_iff
#check terminalAgrees_eq_true_iff

end Nightstream.Tests.FPrimeCanonicalConformanceOneSlot
