import Nightstream.Assurance.Nebula.IdealTranscript

set_option autoImplicit false

namespace Nightstream.Tests.NebulaIdealTranscript

open Nightstream.Assurance.Nebula.IdealTranscript
open Nightstream.Protocol.Nebula.Transcript

example : Function.Injective
    (fun position : Fin 2 × Fin 2 =>
      coordinateIndex position.1 position.2) :=
  coordinateIndex_injective

example {ChallengeField : Type} (table : ChallengeTable ChallengeField) :
    tableEquiv ChallengeField table =
      ( (fun coordinate => table (coordinateIndex 0 coordinate.rev))
      , (fun coordinate => table (coordinateIndex 1 coordinate.rev)) ) :=
  tableEquiv_apply table

example {Digest ChallengeField : Type}
    (table : ChallengeTable ChallengeField)
    (frame : Frame Digest) :
    Nightstream.Assurance.Nebula.FingerprintSecurity.repeatedPoint
        (derive (tableOracle table) frame) =
      tableEquiv ChallengeField table :=
  derive_repeatedPoint_eq_tableEquiv table frame

example {Digest ChallengeField : Type}
    (oracle : Oracle Digest ChallengeField)
    (frame : Frame Digest) :
    Nightstream.Assurance.Nebula.FingerprintSecurity.repeatedPoint
        (derive oracle frame) =
      tableEquiv ChallengeField (tableAt oracle frame) :=
  derive_repeatedPoint_eq_tableEquiv_at oracle frame

namespace ReusedChallengePair

/-- An invalid sampler that reuses coordinates zero and one for both
repetitions. It ignores two oracle coordinates and is not injective. -/
def reuseFirstPair (table : ChallengeTable Nat) : ChallengeTable Nat :=
  fun coordinate =>
    if coordinate.val % 2 = 0 then table 0 else table 1

def leftTable : ChallengeTable Nat := fun coordinate => coordinate.val

def rightTable : ChallengeTable Nat := fun coordinate =>
  if coordinate.val = 2 then 99 else coordinate.val

theorem source_tables_differ : leftTable ≠ rightTable := by
  intro equal
  have atTwo := congrFun equal (2 : Fin 4)
  change 2 = 99 at atTwo
  omega

theorem reused_pair_loses_two_coordinates :
    reuseFirstPair leftTable = reuseFirstPair rightTable := by
  funext coordinate
  fin_cases coordinate <;> rfl

theorem reused_pair_sampler_is_not_injective :
    ¬ Function.Injective reuseFirstPair := by
  intro injective
  exact source_tables_differ
    (injective reused_pair_loses_two_coordinates)

end ReusedChallengePair

end Nightstream.Tests.NebulaIdealTranscript
