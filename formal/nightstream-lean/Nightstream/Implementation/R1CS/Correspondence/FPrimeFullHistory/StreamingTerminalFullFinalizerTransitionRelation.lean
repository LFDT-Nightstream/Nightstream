import Nightstream.SuperNeo.Concrete.Algebra

/-!
Contract: typed transition relation for the exact 50-field terminal Nebula
lane used by Rust.

The relation owns the field order, delayed-step agreement, three
domain-separated chain updates, one-step advance, complete terminal product
and root checks, and the canonical closed output. Hash evaluation is an
explicit semantic input until exact Poseidon2 rows prove compatibility.

It does not own row generation, delayed-input decoding, commitment openings,
hash collision resistance, or HyperNova recursive-size closure.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerTransitionRelation

open Nightstream.SuperNeo.Concrete

abbrev Digest := Fin 4 → F

inductive LinkRole where
  | operations
  | memory
deriving DecidableEq, Repr

structure HashSemantics where
  chainLink : LinkRole → Digest → Digest → Digest
  operationsHeader : Digest
  memoryHeader : Digest

def HashSemantics.header
    (hash : HashSemantics) (lane : Fin 3) : Digest :=
  if lane.val = 0 then hash.operationsHeader else hash.memoryHeader

def chainRole (lane : Fin 3) : LinkRole :=
  if lane.val = 0 then .operations else .memory

/-- Exact Rust `NebulaLaneWires` value shape. -/
structure Lane where
  programBindingDigest : Digest
  isOpen : Bool
  segmentIndex : Nat
  stepIndex : Nat
  timestamp : Nat
  gamma : Fin 2 → K
  products : Fin 4 → K
  stackPointers : Fin 2 → Nat
  dPre : Fin 3 → Digest
  dSeen : Fin 3 → Digest
  dMem : Digest

@[ext]
theorem Lane.ext {left right : Lane}
    (programBindingDigest :
      left.programBindingDigest = right.programBindingDigest)
    (isOpen : left.isOpen = right.isOpen)
    (segmentIndex : left.segmentIndex = right.segmentIndex)
    (stepIndex : left.stepIndex = right.stepIndex)
    (timestamp : left.timestamp = right.timestamp)
    (gamma : left.gamma = right.gamma)
    (products : left.products = right.products)
    (stackPointers : left.stackPointers = right.stackPointers)
    (dPre : left.dPre = right.dPre)
    (dSeen : left.dSeen = right.dSeen)
    (dMem : left.dMem = right.dMem) :
    left = right := by
  cases left
  cases right
  simp_all

/-- Exact Rust `NebulaStepXWires` value shape after Boolean decoding. -/
structure StepInput where
  segmentIndex : Nat
  stepIndex : Nat
  timestampIn : Nat
  timestampOut : Nat
  gamma : Fin 2 → K
  productsIn : Fin 4 → K
  productsOut : Fin 4 → K
  stackPointersIn : Fin 2 → Nat
  stackPointersOut : Fin 2 → Nat

abbrev LeafDigests := Fin 3 → Digest

def boolValue : Bool → Nat
  | false => 0
  | true => 1

def digestValues (digest : Digest) : List Nat :=
  List.ofFn fun lane => (digest lane).val

def kValues (value : K) : List Nat :=
  [value.c0.val, value.c1.val]

/-- Canonical Rust field order:
program digest, phase and counters, gamma, products, stack pointers,
precommit roots, seen roots, and memory root. -/
def Lane.fields (lane : Lane) : List Nat :=
  digestValues lane.programBindingDigest ++
    [boolValue lane.isOpen, lane.segmentIndex, lane.stepIndex, lane.timestamp] ++
    (List.ofFn lane.gamma).flatMap kValues ++
    (List.ofFn lane.products).flatMap kValues ++
    List.ofFn lane.stackPointers ++
    (List.ofFn lane.dPre).flatMap digestValues ++
    (List.ofFn lane.dSeen).flatMap digestValues ++
    digestValues lane.dMem

theorem Lane.fields_length (lane : Lane) : lane.fields.length = 50 := by
  simp [Lane.fields, digestValues, kValues]

def Lane.Canonical (lane : Lane) : Prop :=
  ∀ value ∈ lane.fields, value < goldilocksModulus

/-- The delayed claim must be the claim for the current open lane. -/
structure StepMatches (before : Lane) (step : StepInput) : Prop where
  laneOpen : before.isOpen = true
  segmentIndex : step.segmentIndex = before.segmentIndex
  stepIndex : step.stepIndex = before.stepIndex
  timestampIn : step.timestampIn = before.timestamp
  gamma : step.gamma = before.gamma
  productsIn : step.productsIn = before.products
  stackPointersIn : step.stackPointersIn = before.stackPointers

/-- Deterministic open-lane advance before the terminal close. -/
def advanceLane
    (hash : HashSemantics) (before : Lane) (step : StepInput)
    (leaves : LeafDigests) : Lane where
  programBindingDigest := before.programBindingDigest
  isOpen := before.isOpen
  segmentIndex := before.segmentIndex
  stepIndex := before.stepIndex + 1
  timestamp := step.timestampOut
  gamma := step.gamma
  products := step.productsOut
  stackPointers := step.stackPointersOut
  dPre := before.dPre
  dSeen := fun lane =>
    hash.chainLink (chainRole lane) (before.dSeen lane) (leaves lane)
  dMem := before.dMem

/-- Complete close-only obligations after the one accepted terminal step. -/
structure CloseChecks (advanced : Lane) : Prop where
  closeIndex : advanced.stepIndex = 1
  stackPointersZero : ∀ index, advanced.stackPointers index = 0
  seenEqualsPrecommit : advanced.dSeen = advanced.dPre
  productsBalanced :
    K.mul (advanced.products ⟨2, by decide⟩)
        (advanced.products ⟨1, by decide⟩) =
      K.mul (advanced.products ⟨0, by decide⟩)
        (advanced.products ⟨3, by decide⟩)
  initialMemoryExact : advanced.dSeen ⟨1, by decide⟩ = advanced.dMem

/-- Canonical reset state returned by the forced close branch. -/
def closeLane
    (hash : HashSemantics) (advanced : Lane) : Lane where
  programBindingDigest := advanced.programBindingDigest
  isOpen := false
  segmentIndex := advanced.segmentIndex + 1
  stepIndex := 0
  timestamp := advanced.timestamp
  gamma := fun _ => K.one
  products := fun _ => K.one
  stackPointers := fun _ => 0
  dPre := hash.header
  dSeen := hash.header
  dMem := advanced.dSeen ⟨2, by decide⟩

/-- Exact canonical post-close predicate enforced by the separate terminal
closed-lane family. Program binding, segment index, timestamp, and memory root
are carried values and are not pinned here. -/
def Closed (hash : HashSemantics) (lane : Lane) : Prop :=
  lane.isOpen = false ∧
    lane.stepIndex = 0 ∧
    lane.gamma = (fun _ => K.one) ∧
    lane.products = (fun _ => K.one) ∧
    lane.stackPointers = (fun _ => 0) ∧
    lane.dPre = hash.header ∧
    lane.dSeen = hash.header

theorem close_closed (hash : HashSemantics) (advanced : Lane) :
    Closed hash (closeLane hash advanced) := by
  exact ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩

/-- Smallest complete typed target for the fixed one-step terminal profile. -/
structure TerminalTransition
    (hash : HashSemantics) (before : Lane) (step : StepInput)
    (leaves : LeafDigests) (after : Lane) : Prop where
  stepMatches : StepMatches before step
  closeChecks : CloseChecks (advanceLane hash before step leaves)
  outputExact : after = closeLane hash (advanceLane hash before step leaves)

theorem TerminalTransition.after_closed
    {hash : HashSemantics} {before : Lane} {step : StepInput}
    {leaves : LeafDigests} {after : Lane}
    (transition : TerminalTransition hash before step leaves after) :
    Closed hash after := by
  rw [transition.outputExact]
  exact close_closed hash (advanceLane hash before step leaves)

theorem TerminalTransition.memory_handoff
    {hash : HashSemantics} {before : Lane} {step : StepInput}
    {leaves : LeafDigests} {after : Lane}
    (transition : TerminalTransition hash before step leaves after) :
    after.dMem =
      (advanceLane hash before step leaves).dSeen ⟨2, by decide⟩ := by
  rw [transition.outputExact]
  rfl

end Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerTransitionRelation
