import NightstreamFPrime.Spec.Checkpoint.Structure

/-! The checkpoint structure of a suffix of the stages of a protocol (Akita, ePrint 2026/1983,
revision of 2026-10-15, Definition 10.16, as the proof of Proposition 10.44 uses it: the stages of
the opening suffix are the stages of its levels in their order).

- `Structure.LawfulFrom protocol start` has the conditions of Definition 10.16 for the stages from
  `start`, for the stage inputs of `protocol`. `LawfulFrom.lawful` states that the conditions from
  the stage 0, with item (viii), are `Structure.Lawful`.
- `Structure.chargeFrom` is the sum of the stage errors of the stages from one stage.
- `finish` is the structure after the last stage: a certificate is an accepting transcript. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Checkpoint

open scoped BigOperators
open NightstreamFPrime.Spec.RandomOracle (mass)

variable {Point Coord Slot Proof Cert Break Cand Key Witness : Type}

namespace Structure

variable (structure_ : Structure Point Coord Slot Proof Cert Break Cand Key Witness)

/-- The conditions (i) to (vii) of Akita Definition 10.16 for the stages from `start`. The stage
inputs are those of `protocol`. -/
structure LawfulFrom [Fintype Coord] [DecidableEq Coord] [Fintype Slot]
    (protocol : Protocol Point (Coord → Slot) Proof) (start : ℕ) : Prop where
  segment : ∀ level span, start ≤ level → structure_.kind level = .checkpoint span →
    level + span < protocol.stages ∧
      ∀ later, level < later → later ≤ level + span → structure_.kind later = .plain
  key_some : ∀ level coordinates coordinate point deeper, start ≤ level →
    structure_.kind level = .indexed coordinates → coordinate ∈ coordinates →
    protocol.accept point → protocol.cut (level + 1) point = some deeper →
    (structure_.key level coordinate (protocol.last deeper coordinate)).isSome
  key_mass : ∀ level coordinates coordinate value, start ≤ level →
    structure_.kind level = .indexed coordinates → coordinate ∈ coordinates →
    mass {slot | structure_.key level coordinate slot = some value} ≤
      structure_.weight level coordinate
  weight_nonnegative : ∀ level coordinate, 0 ≤ structure_.weight level coordinate
  leaf_point : ∀ point, structure_.point (structure_.leaf point) = point
  leaf_holds : ∀ point, protocol.accept point →
    structure_.holds protocol.stages (structure_.leaf point)
  plain_holds : ∀ level certificate, start ≤ level → structure_.kind level = .plain →
    structure_.holds (level + 1) certificate → structure_.holds level certificate
  combine_holds : ∀ level coordinates reference family anchor deeper, start ≤ level →
    structure_.kind level = .indexed coordinates → structure_.holds (level + 1) reference →
    protocol.cut level (structure_.point reference) = some anchor →
    protocol.cut (level + 1) (structure_.point reference) = some deeper →
    (∀ coordinate ∈ coordinates, structure_.holds (level + 1) (family coordinate) ∧
      protocol.cut level (structure_.point (family coordinate)) = some anchor ∧
      ∃ other slot,
        protocol.cut (level + 1) (structure_.point (family coordinate)) = some other ∧
        protocol.last other = Function.update (protocol.last deeper) coordinate slot ∧
        structure_.key level coordinate slot ≠
          structure_.key level coordinate (protocol.last deeper coordinate)) →
    Sum.elim
      (fun certificate => structure_.holds level certificate ∧ ∀ earlier ≤ level,
        protocol.cut earlier (structure_.point certificate) =
          protocol.cut earlier (structure_.point reference))
      structure_.breaks (structure_.combine level reference family)
  good_holds : ∀ level span certificate, start ≤ level →
    structure_.kind level = .checkpoint span →
    structure_.holds (level + 1) certificate → structure_.good level certificate →
    structure_.holds level certificate
  collide_breaks : ∀ level span certificate other anchor, start ≤ level →
    structure_.kind level = .checkpoint span →
    structure_.holds (level + 1) certificate → structure_.holds (level + 1) other →
    protocol.cut level (structure_.point certificate) = some anchor →
    protocol.cut level (structure_.point other) = some anchor →
    structure_.candidate level certificate ≠ structure_.candidate level other →
    structure_.breaks (structure_.collide level certificate other)
  refuted_start : ∀ level span certificate anchor, start ≤ level →
    structure_.kind level = .checkpoint span →
    structure_.holds (level + 1) certificate → ¬ structure_.good level certificate →
    protocol.cut level (structure_.point certificate) = some anchor →
    structure_.refuted level anchor (structure_.candidate level certificate)
  refuted_step : ∀ level span input candidate reply proof, start ≤ level →
    structure_.kind level = .checkpoint span →
    level ≤ protocol.stage input → protocol.stage input ≤ level + span →
    structure_.refuted level input candidate → reply ∉ structure_.bad level input candidate →
    structure_.refuted level (protocol.step proof input reply) candidate
  refuted_end : ∀ level span certificate input, start ≤ level →
    structure_.kind level = .checkpoint span →
    structure_.holds (level + 1) certificate →
    protocol.cut (level + span + 1) (structure_.point certificate) = some input →
    ¬ structure_.refuted level input (structure_.candidate level certificate)
  bad_mass : ∀ level span input candidate, start ≤ level →
    structure_.kind level = .checkpoint span →
    level ≤ protocol.stage input → protocol.stage input ≤ level + span →
    mass (structure_.bad level input candidate) ≤ structure_.error (protocol.stage input)
  error_nonnegative : ∀ level, 0 ≤ structure_.error level

variable {structure_}

/-- The conditions from the stage 0 with item (viii) are the conditions of Definition 10.16. -/
theorem LawfulFrom.lawful [Fintype Coord] [DecidableEq Coord] [Fintype Slot]
    (from_ : structure_.LawfulFrom structure_.protocol 0)
    (witness_holds : ∀ certificate statement, structure_.holds 0 certificate →
      structure_.protocol.cut 0 (structure_.point certificate) = some statement →
      structure_.relation statement (structure_.witness certificate)) :
    structure_.Lawful where
  segment level span := from_.segment level span (Nat.zero_le _)
  key_some level coordinates coordinate point deeper :=
    from_.key_some level coordinates coordinate point deeper (Nat.zero_le _)
  key_mass level coordinates coordinate value :=
    from_.key_mass level coordinates coordinate value (Nat.zero_le _)
  weight_nonnegative := from_.weight_nonnegative
  leaf_point := from_.leaf_point
  leaf_holds := from_.leaf_holds
  plain_holds level certificate := from_.plain_holds level certificate (Nat.zero_le _)
  combine_holds level coordinates reference family anchor deeper :=
    from_.combine_holds level coordinates reference family anchor deeper (Nat.zero_le _)
  good_holds level span certificate := from_.good_holds level span certificate (Nat.zero_le _)
  collide_breaks level span certificate other anchor :=
    from_.collide_breaks level span certificate other anchor (Nat.zero_le _)
  refuted_start level span certificate anchor :=
    from_.refuted_start level span certificate anchor (Nat.zero_le _)
  refuted_step level span input candidate reply proof :=
    from_.refuted_step level span input candidate reply proof (Nat.zero_le _)
  refuted_end level span certificate input :=
    from_.refuted_end level span certificate input (Nat.zero_le _)
  bad_mass level span input candidate :=
    from_.bad_mass level span input candidate (Nat.zero_le _)
  error_nonnegative := from_.error_nonnegative
  witness_holds := witness_holds

variable (structure_)

/-- The sum of the stage errors `ε_r` of the stages from `level`. -/
noncomputable def chargeFrom (level : ℕ) : ℝ :=
  ∑ later ∈ Finset.Ico level structure_.protocol.stages, structure_.charge later

theorem checkError_eq_chargeFrom : structure_.checkError = structure_.chargeFrom 0 := by
  unfold checkError chargeFrom
  rw [Nat.Ico_zero_eq_range]

end Structure

/-- The structure after the last stage of a protocol: each stage is plain, and a certificate is
an accepting transcript (Akita Definition 10.16 (ii)). It has no witness of `R_bind`. -/
def finish (protocol : Protocol Point (Coord → Slot) Proof) :
    Structure Point Coord Slot Proof Point Unit Unit Unit Unit where
  protocol := protocol
  kind _ := .plain
  key _ _ _ := none
  weight _ _ := 0
  error _ := 0
  leaf point := point
  point certificate := certificate
  holds _ certificate := protocol.accept certificate
  breaks _ := False
  combine _ reference _ := .inl reference
  candidate _ _ := ()
  good _ _ := True
  collide _ _ _ := ()
  refuted _ _ _ := False
  bad _ _ _ := ∅
  witness _ := ()
  relation _ _ := True

theorem finish_lawfulFrom [Fintype Coord] [DecidableEq Coord] [Fintype Slot]
    (protocol : Protocol Point (Coord → Slot) Proof) (start : ℕ) :
    (finish protocol).LawfulFrom protocol start where
  segment _ _ _ isCheckpoint := absurd isCheckpoint (by simp [finish])
  key_some _ _ _ _ _ _ isIndexed := absurd isIndexed (by simp [finish])
  key_mass _ _ _ _ _ isIndexed := absurd isIndexed (by simp [finish])
  weight_nonnegative _ _ := le_rfl
  leaf_point _ := rfl
  leaf_holds _ accepted := accepted
  plain_holds _ _ _ _ held := held
  combine_holds _ _ _ _ _ _ _ isIndexed := absurd isIndexed (by simp [finish])
  good_holds _ _ _ _ isCheckpoint := absurd isCheckpoint (by simp [finish])
  collide_breaks _ _ _ _ _ _ isCheckpoint := absurd isCheckpoint (by simp [finish])
  refuted_start _ _ _ _ _ isCheckpoint := absurd isCheckpoint (by simp [finish])
  refuted_step _ _ _ _ _ _ _ isCheckpoint := absurd isCheckpoint (by simp [finish])
  refuted_end _ _ _ _ _ isCheckpoint := absurd isCheckpoint (by simp [finish])
  bad_mass _ _ _ _ _ isCheckpoint := absurd isCheckpoint (by simp [finish])
  error_nonnegative _ := le_rfl

end NightstreamFPrime.Spec.Checkpoint
