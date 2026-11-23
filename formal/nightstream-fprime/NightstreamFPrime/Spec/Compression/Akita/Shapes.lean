import NightstreamFPrime.Spec.Compression.Akita.Commit

/-! The commitment shape of the compression key in the schedule files `key20.aks` and `key30.aks`
of `crates/ns-compress/artifacts/` (Akita, ePrint 2026/1983, revision of 2026-11-23, (28)).

- `Shape.key20` and `Shape.key30` have the numbers of the first group of the root level, which is
  the group of the key polynomial. The two rows of `key30.aks` have one key group.
  `tests/AkitaSchedules.lean` compares the numbers with the files.
- A file has no field for the certified base `b⋆` of the commitment contract (Akita Section 4.6).
  It has the bound 7 of the Module-SIS instance of `B`, which is `b⋆ − 1` (Akita (49)), so `b⋆`
  is 8, the opening base `b_1`.
- A file has `payload_mode = Compressed` and no shape of `F_1` and `F_2`. Their ring degrees 64
  and 32 and their rank 1 are fixed in the Akita code (revision `d50f700f`,
  `crates/akita-params/src/compression.rs`), and each width has one position for each of the 64
  binary digits of each coefficient of the image.
- `Shape.key20_honest` and `Shape.key30_honest` give `Shape.Honest`.
- `Descriptor.key20` and `Descriptor.key30` have the numbers of the opening descriptor of the key
  group. A fold challenge has 31 coefficients `±1` and no other nonzero coefficient, so `κ_1` is
  31. A file has one base for the opening, and it is `b_z`. `Descriptor.key20_innerBound` and
  `Descriptor.key30_innerBound` give `η_A` of Akita (50), which is the bound of the file for the
  Module-SIS instance of `A`. `Descriptor.key20_covers` and `Descriptor.key30_covers` give that
  the bound `β̄_∞ + κ̄_1·b/2` of the reduction `collision` is at most `η_A`. -/

namespace NightstreamFPrime.Spec.Compression.Akita

theorem binaryDigits_le : binaryDigits ≤ 64 := Nat.clog_le_of_le_pow (by decide)

namespace Shape

/-- The key group of `key20.aks`: a table of `2^20` cells in 32 blocks of 32 elements of
`R_{q,1024}`. -/
def key20 : Shape where
  innerDegree := 1024
  outerDegree := 128
  divides := ⟨8, rfl⟩
  blocks := 32
  blockLength := 32
  sourceBase := 2 ^ 10
  sourceDigits := 7
  innerRank := 1
  openingBase := 8
  openingDigits := 22
  certifiedBase := 8
  outerRank := 1
  slices := 4
  chain := some {
    firstDegree := 64, firstRank := 1, firstWidth := 512
    secondDegree := 32, secondRank := 1, secondWidth := 128 }

/-- The key group of each row of `key30.aks`: a table of `2^30` cells in 512 blocks of 1024
elements of `R_{q,2048}`. -/
def key30 : Shape where
  innerDegree := 2048
  outerDegree := 256
  divides := ⟨8, rfl⟩
  blocks := 512
  blockLength := 1024
  sourceBase := 2 ^ 16
  sourceDigits := 4
  innerRank := 1
  openingBase := 8
  openingDigits := 22
  certifiedBase := 8
  outerRank := 1
  slices := 4
  chain := some {
    firstDegree := 64, firstRank := 1, firstWidth := 1024
    secondDegree := 32, secondRank := 1, secondWidth := 128 }

theorem key20_honest : key20.Honest where
  source := ⟨by decide, by decide, by decide, by decide⟩
  opening := ⟨by decide, by decide, by decide, by decide⟩
  certified := le_rfl
  block := by decide
  path := by
    have digits := binaryDigits_le
    exact show 4 * 1 * 128 * binaryDigits ≤ 512 * 64 ∧ 1 * 64 * binaryDigits ≤ 128 * 32 from
      ⟨by omega, by omega⟩

theorem key30_honest : key30.Honest where
  source := ⟨by decide, by decide, by decide, by decide⟩
  opening := ⟨by decide, by decide, by decide, by decide⟩
  certified := le_rfl
  block := by decide
  path := by
    have digits := binaryDigits_le
    exact show 4 * 1 * 256 * binaryDigits ≤ 1024 * 64 ∧ 1 * 64 * binaryDigits ≤ 128 * 32 from
      ⟨by omega, by omega⟩

end Shape

namespace Descriptor

/-- The opening descriptor of the key group of the root of `key20.aks`. -/
def key20 : Descriptor where
  challengeMass := 31
  responseBase := 8
  responseDigits := 6

/-- The opening descriptor of the key group of the root of each row of `key30.aks`. -/
def key30 : Descriptor where
  challengeMass := 31
  responseBase := 8
  responseDigits := 9

theorem key20_innerBound : key20.innerBound Shape.key20 = 32505732 := by decide

theorem key30_innerBound : key30.innerBound Shape.key30 = 16642998148 := by decide

theorem key20_covers :
    key20.sourceBound Shape.key20 + key20.factorBound * (Shape.key20.sourceBase / 2) ≤
      key20.innerBound Shape.key20 :=
  key20.mixed_le Shape.key20 (by decide) (by decide)

theorem key30_covers :
    key30.sourceBound Shape.key30 + key30.factorBound * (Shape.key30.sourceBase / 2) ≤
      key30.innerBound Shape.key30 :=
  key30.mixed_le Shape.key30 (by decide) (by decide)

end Descriptor

end NightstreamFPrime.Spec.Compression.Akita
