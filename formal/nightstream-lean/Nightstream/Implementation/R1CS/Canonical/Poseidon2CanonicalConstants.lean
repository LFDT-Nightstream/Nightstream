import Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23Ownership

/-!
Contract: the protocol's own width-eight Goldilocks Poseidon2 round constants.

Owns: the selected constant table, as Lean data, and its shape and canonicality.
This file is **authority**. The eighty-six values below are the protocol's
choice; any implementation that uses a different table implements a different
protocol and is wrong.

Does not own: derivation of the table from the ChaCha8 seed. Lean fixes *which*
table F′ uses; it does not yet prove that table is the seed's image. That gap is
named `POSEIDON2-CONSTANT-DERIVATION` and is deliberately not closed here — a
seed derivation would replace one selection with another, and the selection is
what the encoding needs pinned.

## Why the values are inlined rather than imported

They were first observed as the output of the Rust generator. Reading them from
a generated file made Rust the authority: had the generator changed, every
canonical theorem would have silently followed it. Writing them here inverts
that. `Poseidon2RustConformance` now checks Rust against this file, so drift
fails as *Rust* being wrong rather than as Lean quietly re-deriving.

This is a change of direction, not of values, and it is the whole content of the
change. Nothing here claims the values are canonical in any deeper sense than
"the protocol selects them".
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.Poseidon2CanonicalConstants

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Core
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Schedule

/-! ## The selected table

Thirty-two initial full-round constants, twenty-two partial-round constants, and
thirty-two terminal full-round constants: eighty-six in total, one per S-box. -/

/-- Round constants for the four initial full rounds, eight lanes each. -/
def initial : List (List Nat) :=
  [[15504881536434223753, 2212164856944708396, 1885257220781225929, 17531637481572944510, 16769640728293682348, 445908668462176974, 1308472042479836079, 17465001500823438575],
   [1922033642430128704, 2657514617275794404, 17238706657248448792, 7348277157222259646, 10777112892842897939, 1771261721914735482, 9409693344407549465, 16619731096074499912],
   [1922036059108268922, 2681686362645798986, 12432722052283819565, 2826979200512189741, 5080805286413226676, 16827966425431695029, 9196241087337510154, 2350771591198563053],
   [2989012136977041732, 4359939046747977080, 16089932437481530267, 6601984573273403484, 13005272261058756234, 17128237926164276121, 8240789415616872849, 8676316357341090631]]

/-- Round constants for the twenty-two partial rounds: lane zero only. -/
def internal : List Nat :=
  [7482194551502142718, 3471957803411196592, 8846669050136897522, 4431017908497072775, 14382646627736292998, 15636596632746594248, 14521990061611210983, 4351091752509404379, 14119848206371842921, 528205008764728916, 15379406877060454284, 13572057177474709483, 780214424511389757, 10591233664360718633, 1849508423779478786, 7345390174439848870, 14580881241235634775, 8777273265976228774, 1758781345554053863, 9701442189086298420, 15685565327448534444, 5672331717709479627]

/-- Round constants for the four terminal full rounds, eight lanes each. -/
def terminal : List (List Nat) :=
  [[16452552554259143025, 17874550554210084887, 3031715677034868367, 18215520516675091549, 18186005068527139405, 11138995707668647102, 15098195648006184282, 2025927025270509469],
   [9957669227203243937, 11554336633716867616, 9729067570563846225, 4239770196713589268, 4390607796152185292, 17647511975646925721, 7671337049037340193, 4209452938403606590],
   [6593973666654839090, 8390781086037206386, 7324343054784993307, 17780748563735894140, 15974082699116886783, 13213371256836887512, 7312926934405385057, 10393853239698468203],
   [2710107888698774842, 2801523468128575786, 15894340394120906162, 13510783799941644149, 7917164295139071913, 13839801071899888959, 6672989303670154677, 4519956214037211385]]

def rowValue (rows : List (List Nat)) (round : Nat) (lane : Fin width) : Nat :=
  (rows.getD round []).getD lane.val 0

/-- **The protocol's constant schedule.** -/
def selected : Constants where
  initial := rowValue initial
  internal := fun round => internal.getD round 0
  terminal := rowValue terminal

/-! ## Shape

The tables have exactly one constant per S-box, so `getD`'s defaults are never
reached at any index the schedule visits. Without this the table could be short
and `selected` would silently read zeros. -/

theorem initial_shape :
    initial.length = halfFullRounds ∧ ∀ row ∈ initial, row.length = width := by
  decide

theorem internal_shape : internal.length = partialRounds := by
  decide

theorem terminal_shape :
    terminal.length = halfFullRounds ∧ ∀ row ∈ terminal, row.length = width := by
  decide

/-! ## Canonicality -/

theorem selected_canonical :
    (∀ round : Fin halfFullRounds, ∀ lane : Fin width,
      selected.initial round.val lane < goldilocksP) ∧
    (∀ round : Fin partialRounds,
      selected.internal round.val < goldilocksP) ∧
    (∀ round : Fin halfFullRounds, ∀ lane : Fin width,
      selected.terminal round.val lane < goldilocksP) := by
  decide

end Nightstream.Implementation.R1CS.Canonical.Poseidon2CanonicalConstants
