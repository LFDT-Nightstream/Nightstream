import Nightstream.Implementation.R1CS.Correspondence.FPrime.FPrimeTerminalLinkBatch

/-!
Contract: typed source schedule interpreted by the production terminal-link
R1CS emitter for every fresh claim.

Owns:
- exact claim/producer offsets and range lengths;
- a definitional scalar-row cost;
- fail-closed expansion into the selected batch row owners;
- a fail-closed compiler into the receipt-owned arbitrary-batch rows.

Does not own: compiled Rust semantics, host shape checks, producer-column
placement, generated program data, or the surrounding decider.

Emits constraints: no.
-/

namespace Nightstream.Implementation.Rust.CanonicalConformance.TerminalLink.Program

open Nightstream.Implementation.R1CS.FPrimeTerminalLinkBatch
open Nightstream.Implementation.R1CS

set_option maxRecDepth 32768

inductive Instruction where
  | affineOne (claimOffset : Nat)
  | bodyRange (claimOffset producerOffset length : Nat)
  | paddingZeroRange (claimOffset length : Nat)
deriving Repr, DecidableEq

def Instruction.cost : Instruction → Nat
  | .affineOne _ => 1
  | .bodyRange _ _ length => length
  | .paddingZeroRange _ length => length

/-- Expansion rejects any offset or width not belonging to the selected plain
terminal-link encoding. -/
def Instruction.expand? : Instruction → Option (List LocalOwner)
  | .affineOne claimOffset =>
      if claimOffset = 0 then
        some [LocalOwner.affineOne]
      else
        none
  | .bodyRange claimOffset producerOffset length =>
      if claimOffset = 1 ∧ producerOffset = 0 ∧ length = 256 then
        some (List.ofFn fun bit : Fin 256 => LocalOwner.linked bit)
      else
        none
  | .paddingZeroRange claimOffset length =>
      if claimOffset = 257 ∧ length = 13 then
        some (List.ofFn fun padding : Fin 13 =>
          LocalOwner.paddingZero padding)
      else
        none

abbrev Program := List Instruction

def cost (program : Program) : Nat :=
  (program.map Instruction.cost).sum

def expand? : Program → Option (List LocalOwner)
  | [] => some []
  | instruction :: rest => do
      let head ← instruction.expand?
      let tail ← expand? rest
      pure (head ++ tail)

def plain : Program :=
  [ .affineOne 0
  , .bodyRange 1 0 256
  , .paddingZeroRange 257 13
  ]

/-- Receipt-owner order selected independently by the batch row compiler. -/
def selectedOwnerOrder : List LocalOwner :=
  List.ofFn ownerAt

theorem plain_cost :
    cost plain = 270 := by
  decide

/-- The three source phases expand to every selected scalar receipt exactly in
physical local-row order. -/
theorem plain_expansion :
    expand? plain = some selectedOwnerOrder := by
  decide

/-- Compile only a complete program whose expanded owner sequence is exactly
the selected physical owner order. Incomplete, duplicated, reordered, or
otherwise malformed schedules emit no rows. The successful branch delegates
to the receipt-owned arbitrary-batch compiler. -/
def compile (program : Program) (batchSize : Nat) : Option (List Row) := do
  let owners ← expand? program
  if owners = selectedOwnerOrder then
    some (rows batchSize)
  else
    none

/-- The selected source schedule compiles to the exact receipt-owned rows for
every batch size. -/
theorem compile_plain (batchSize : Nat) :
    compile plain batchSize = some (rows batchSize) := by
  simp [compile, plain_expansion]

/-- Semantic acceptance of a source schedule is satisfaction of the rows
returned by its checked compiler. A rejected program has no accepted
assignment. -/
def Accepts
    (program : Program)
    (batchSize : Nat)
    (assignment : Nat -> Nat) : Prop :=
  exists emitted,
    compile program batchSize = some emitted /\
      Satisfies emitted assignment

/-- The selected program accepts exactly the selected receipt-owned physical
relation. -/
theorem accepts_plain_iff
    (batchSize : Nat)
    (assignment : Nat -> Nat) :
    Accepts plain batchSize assignment <->
      Satisfies (rows batchSize) assignment := by
  simp [Accepts, compile_plain]

end Nightstream.Implementation.Rust.CanonicalConformance.TerminalLink.Program
