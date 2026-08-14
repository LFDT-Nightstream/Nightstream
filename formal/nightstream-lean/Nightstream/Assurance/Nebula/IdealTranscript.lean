import Nightstream.Assurance.Nebula.FingerprintSecurity
import Nightstream.Protocol.Nebula.Transcript

/-!
Contract: fixed-frame ideal sampling bridge for the four V2 fingerprint
coordinates.

Owns the exact bijection between a four-coordinate oracle table and the two
independent polynomial points, plus its link to `Transcript.derive`.

Does not own adaptive Fiat--Shamir security, Poseidon2, frame commitment
timing, oracle programming, collision resistance, or a query-loss theorem.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Assurance.Nebula.IdealTranscript

open Nightstream.Implementation.Nebula.ConcreteField
open Nightstream.Assurance.Nebula.FingerprintSecurity
open Nightstream.Protocol.Nebula
open Nightstream.Protocol.Nebula.IdealFingerprint
open Nightstream.Protocol.Nebula.Transcript

abbrev ChallengeTable (ChallengeField : Type) := Fin 4 → ChallengeField

/-- Convert polynomial-coordinate order `(gamma2, gamma1)` into transcript
coordinate order `(gamma1, gamma2)` before selecting one of four squeezes. -/
def sampleCoordinateEquiv : Fin 2 × Fin 2 ≃ Fin 4 :=
  ((Equiv.refl (Fin 2)).prodCongr (@Fin.revPerm 2)).trans coordinateEquiv

/-- Reindex four oracle coordinates as two independent two-coordinate
polynomial points. -/
def tableEquiv (ChallengeField : Type) :
    ChallengeTable ChallengeField ≃
      ((Fin 2 → ChallengeField) × (Fin 2 → ChallengeField)) :=
  (Equiv.arrowCongr sampleCoordinateEquiv.symm (Equiv.refl ChallengeField)).trans
    ((Equiv.curry (Fin 2) (Fin 2) ChallengeField).trans
      (finTwoArrowEquiv (Fin 2 → ChallengeField)))

theorem tableEquiv_apply
    {ChallengeField : Type} (table : ChallengeTable ChallengeField) :
    tableEquiv ChallengeField table =
      ( (fun coordinate => table (coordinateIndex 0 coordinate.rev))
      , (fun coordinate => table (coordinateIndex 1 coordinate.rev)) ) := by
  rfl

/-- Ideal oracle for one fixed, already-committed frame. Other frame inputs
are intentionally ignored; the adaptive multi-frame ROM theorem is outside
this fixed-frame lemma. -/
def tableOracle
    {Digest ChallengeField : Type}
    (table : ChallengeTable ChallengeField) :
    Oracle Digest ChallengeField :=
  fun _frame coordinate => table coordinate

/-- The four answers used by one concrete, fixed frame. -/
def tableAt
    {Digest ChallengeField : Type}
    (oracle : Oracle Digest ChallengeField)
    (frame : Frame Digest) : ChallengeTable ChallengeField :=
  fun coordinate => oracle (encode frame) coordinate

/-- The exact transcript implementation consumes the ideal table through the
same bijection used by the public-coin probability theorem. -/
theorem derive_repeatedPoint_eq_tableEquiv
    {Digest ChallengeField : Type}
    (table : ChallengeTable ChallengeField)
    (frame : Frame Digest) :
    repeatedPoint (derive (tableOracle table) frame) =
      tableEquiv ChallengeField table := by
  rw [tableEquiv_apply]
  apply Prod.ext
  · funext coordinate
    fin_cases coordinate
    · rfl
    · change table (coordinateIndex 0 0) = table (coordinateIndex 0 0)
      rfl
  · funext coordinate
    fin_cases coordinate
    · rfl
    · change table (coordinateIndex 1 0) = table (coordinateIndex 1 0)
      rfl

/-- For any oracle implementation, one fixed frame depends only on its four
answers at that frame. This is an exact data-flow theorem, not a randomness
claim. -/
theorem derive_repeatedPoint_eq_tableEquiv_at
    {Digest ChallengeField : Type}
    (oracle : Oracle Digest ChallengeField)
    (frame : Frame Digest) :
    repeatedPoint (derive oracle frame) =
      tableEquiv ChallengeField (tableAt oracle frame) := by
  exact derive_repeatedPoint_eq_tableEquiv (tableAt oracle frame) frame

theorem challengeTable_cardinality :
    Fintype.card (ChallengeTable ChallengeField) =
      Fintype.card ChallengeField ^ 4 := by
  simp

/-- A full table is equivalent to the exact repeated challenge sample space;
there is no challenge-pair reuse or missing sample at the fixed-frame ideal
boundary. -/
theorem table_to_repeated_points_bijective :
    Function.Bijective (tableEquiv ChallengeField) :=
  (tableEquiv ChallengeField).bijective

end Nightstream.Assurance.Nebula.IdealTranscript
