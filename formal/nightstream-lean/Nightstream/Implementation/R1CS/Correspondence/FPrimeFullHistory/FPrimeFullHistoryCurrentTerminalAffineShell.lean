import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryCurrentTerminalAffineShell
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.FPrimeFullHistoryCurrentTerminalLinkPlacement

/-!
Contract: coefficient-exact semantics of the bounded current fixed-one
terminal affine shell.

Assurance tier: artifact-checked bounded current placement.

Owns:
- the exact partition of rows `[9657286, 9673659)` into four running-digest
  equalities, 16,099 parent-authority equalities, and the 270-row
  prior/latest public-input link;
- soundness and completeness of those exact generated rows with respect to
  their coefficient-derived equality/constant relation;
- exact identification of the final 270 rows with the independently captured
  current plain-carrier owner.

Does not own: current terminal NIFS semantics, output-accumulator semantics,
direct terminal-CE semantics, either frozen unary running/fresh relation,
`SourceAuthority`, or a whole-current artifact.

Emits constraints: no; Rust emitted the imported bounded rows.
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryCurrentTerminalAffineShellSound

open Nightstream.Implementation.R1CS

set_option maxRecDepth 131072

namespace Captured

abbrev pins : List AffinePins.Pin :=
  FPrimeFullHistoryCurrentTerminalAffineShell.pins

abbrev rows : List Row :=
  FPrimeFullHistoryCurrentTerminalAffineShell.rows

def runningDigestPins : List AffinePins.Pin :=
  pins.take 4

def parentAuthorityPins : List AffinePins.Pin :=
  (pins.drop 4).take 16099

def priorLatestPins : List AffinePins.Pin :=
  pins.drop (4 + 16099)

/-- The relation actually expressed by the selected coefficients. No protocol
validity proposition is stored in this record. -/
structure Holds (assignment : Nat → Nat) : Prop where
  runningDigest :
    ∀ pin ∈ runningDigestPins, pin.Holds assignment
  parentAuthority :
    ∀ pin ∈ parentAuthorityPins, pin.Holds assignment
  priorLatest :
    ∀ pin ∈ priorLatestPins, pin.Holds assignment

theorem pins_eq_partition :
    pins =
      runningDigestPins ++ parentAuthorityPins ++ priorLatestPins := by
  rw [runningDigestPins, parentAuthorityPins, priorLatestPins]
  calc
    pins =
        pins.take 4 ++ pins.drop 4 :=
      (List.take_append_drop 4 pins).symm
    _ =
        pins.take 4 ++
          ((pins.drop 4).take 16099 ++
            (pins.drop 4).drop 16099) := by
      rw [List.take_append_drop 16099 (pins.drop 4)]
    _ =
        (pins.take 4 ++ (pins.drop 4).take 16099) ++
          pins.drop (4 + 16099) := by
      rw [← List.append_assoc]
      simp only [List.drop_drop]

/-- Kernel computation over the compact run schedule; this proof deliberately
does not consume the generated `native_decide` certificate. -/
theorem pinsCanonical :
    AffinePins.PinsCanonical pins := by
  decide

theorem rows_iff_holds
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1) :
    Satisfies rows assignment ↔ Holds assignment := by
  constructor
  · intro satisfies
    have equations :=
      AffinePins.rows_sound pinsCanonical canonical one satisfies
    refine ⟨?_, ?_, ?_⟩
    · intro pin member
      exact equations pin (by
        rw [pins_eq_partition]
        simp only [List.mem_append]
        exact Or.inl (Or.inl member))
    · intro pin member
      exact equations pin (by
        rw [pins_eq_partition]
        simp only [List.mem_append]
        exact Or.inl (Or.inr member))
    · intro pin member
      exact equations pin (by
        rw [pins_eq_partition]
        simp only [List.mem_append]
        exact Or.inr member)
  · intro holds
    apply AffinePins.rows_complete pinsCanonical canonical one
    intro pin member
    rw [pins_eq_partition] at member
    simp only [List.mem_append] at member
    rcases member with (member | member) | member
    · exact holds.runningDigest pin member
    · exact holds.parentAuthority pin member
    · exact holds.priorLatest pin member

theorem priorLatestPins_eq_currentPlacement :
    priorLatestPins =
      FPrimeFullHistoryCurrentTerminalLinkPlacement.pins := by
  decide

theorem priorLatestRows_eq_currentPlacement :
    AffinePins.rows priorLatestPins =
      FPrimeFullHistoryCurrentTerminalLinkPlacement.rows := by
  rw [priorLatestPins_eq_currentPlacement]
  rfl

theorem priorLatestPinsCanonical :
    AffinePins.PinsCanonical priorLatestPins := by
  intro pin member
  exact pinsCanonical pin (by
    rw [pins_eq_partition]
    simp only [List.mem_append]
    exact Or.inr member)

/-- The final captured block is exactly the current prior-public link. This
theorem does not identify that link with the frozen terminal `freshCheck`. -/
theorem priorLatest_iff_currentPlacementRows
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1) :
    (∀ pin ∈ priorLatestPins, pin.Holds assignment) ↔
      Satisfies
        FPrimeFullHistoryCurrentTerminalLinkPlacement.rows assignment := by
  rw [← priorLatestRows_eq_currentPlacement]
  constructor
  · exact AffinePins.rows_complete priorLatestPinsCanonical canonical one
  · intro satisfies
    exact AffinePins.rows_sound priorLatestPinsCanonical canonical one satisfies

end Captured

end Nightstream.Implementation.R1CS.FPrimeFullHistoryCurrentTerminalAffineShellSound
