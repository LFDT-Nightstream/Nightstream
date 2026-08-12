import Nightstream.Implementation.R1CS.Artifacts.CanonicalU64
import Mathlib.Tactic.IntervalCases

/-!
Contract: artifact-level soundness of the canonical-u64 decomposition gadget.

Claim: any assignment satisfying the exact exported rows of
`decompose_var_to_u64_bits` (see `CanonicalU64Artifact.lean`) recomposes, over
the integers, to the canonical value of the decomposed field element — the
non-canonical `x + p` bit pattern is impossible. Property `CIR-U64CANON`.

Assumes: canonical-residue assignments, constant-one wire at column 0, and the
Euclid divisor property of the Goldilocks modulus as a typed hypothesis.

Non-goals: compiler completeness (owned by `CanonicalU64Complete.lean`),
commitment binding, and any claim about the surrounding F' circuit rows.
-/

namespace Nightstream.Implementation.R1CS

open CanonicalU64

/-- Every indexed bit wire of a satisfying canonical-u64 block is Boolean.
This lets a larger compiler construct its typed word from row-derived columns
instead of accepting that word as parser authority. -/
theorem canonicalU64_bit_lt_two (hq : EuclidPrime goldilocksP)
    {z : Nat → Nat}
    (hcanon : ∀ i, z i < goldilocksP)
    (hone : z 0 = 1)
    (hsat : Satisfies rows z)
    (index : Nat) (bounded : index < 64) :
    z (bitCol index) < 2 := by
  interval_cases index <;>
    exact Nat.lt_succ_iff.mpr
      (bitRow_le_one hq (hcanon _) hone (hsat _ (by decide)))

/-- Little-endian integer value of the 64 exported bit columns. -/
def bitsValue (z : Nat → Nat) : Nat :=
  (List.range 64).foldl (fun acc i => acc + 2 ^ i * z (bitCol i)) 0

/--
Soundness of the exported canonical-u64 decomposition artifact: the bit
columns recompose **as integers** to the decomposed field element, and the
recomposed value stays below the modulus. Together these rule out the
`x + p` second representation the gadget's canonicity gate exists to reject.
-/
theorem canonicalU64_sound (hq : EuclidPrime goldilocksP)
    {z : Nat → Nat}
    (hcanon : ∀ i, z i < goldilocksP)
    (hone : z 0 = 1)
    (hsat : Satisfies rows z) :
    z varCol = bitsValue z ∧ bitsValue z < goldilocksP := by
  -- Literal-modulus view of the Euclid hypothesis, so `omega` can consume
  -- the disjuncts it produces.
  have hq' : ∀ a b : Nat,
      a * b % 18446744069414584321 = 0 →
      a % 18446744069414584321 = 0 ∨ b % 18446744069414584321 = 0 := hq
  -- Every bit column is boolean (rows 0–63 and 65 are `enforce_bit` rows).
  have hb2 : z 2 ≤ 1 := bitRow_le_one hq (hcanon 2) hone (hsat _ (by decide))
  have hb3 : z 3 ≤ 1 := bitRow_le_one hq (hcanon 3) hone (hsat _ (by decide))
  have hb4 : z 4 ≤ 1 := bitRow_le_one hq (hcanon 4) hone (hsat _ (by decide))
  have hb5 : z 5 ≤ 1 := bitRow_le_one hq (hcanon 5) hone (hsat _ (by decide))
  have hb6 : z 6 ≤ 1 := bitRow_le_one hq (hcanon 6) hone (hsat _ (by decide))
  have hb7 : z 7 ≤ 1 := bitRow_le_one hq (hcanon 7) hone (hsat _ (by decide))
  have hb8 : z 8 ≤ 1 := bitRow_le_one hq (hcanon 8) hone (hsat _ (by decide))
  have hb9 : z 9 ≤ 1 := bitRow_le_one hq (hcanon 9) hone (hsat _ (by decide))
  have hb10 : z 10 ≤ 1 := bitRow_le_one hq (hcanon 10) hone (hsat _ (by decide))
  have hb11 : z 11 ≤ 1 := bitRow_le_one hq (hcanon 11) hone (hsat _ (by decide))
  have hb12 : z 12 ≤ 1 := bitRow_le_one hq (hcanon 12) hone (hsat _ (by decide))
  have hb13 : z 13 ≤ 1 := bitRow_le_one hq (hcanon 13) hone (hsat _ (by decide))
  have hb14 : z 14 ≤ 1 := bitRow_le_one hq (hcanon 14) hone (hsat _ (by decide))
  have hb15 : z 15 ≤ 1 := bitRow_le_one hq (hcanon 15) hone (hsat _ (by decide))
  have hb16 : z 16 ≤ 1 := bitRow_le_one hq (hcanon 16) hone (hsat _ (by decide))
  have hb17 : z 17 ≤ 1 := bitRow_le_one hq (hcanon 17) hone (hsat _ (by decide))
  have hb18 : z 18 ≤ 1 := bitRow_le_one hq (hcanon 18) hone (hsat _ (by decide))
  have hb19 : z 19 ≤ 1 := bitRow_le_one hq (hcanon 19) hone (hsat _ (by decide))
  have hb20 : z 20 ≤ 1 := bitRow_le_one hq (hcanon 20) hone (hsat _ (by decide))
  have hb21 : z 21 ≤ 1 := bitRow_le_one hq (hcanon 21) hone (hsat _ (by decide))
  have hb22 : z 22 ≤ 1 := bitRow_le_one hq (hcanon 22) hone (hsat _ (by decide))
  have hb23 : z 23 ≤ 1 := bitRow_le_one hq (hcanon 23) hone (hsat _ (by decide))
  have hb24 : z 24 ≤ 1 := bitRow_le_one hq (hcanon 24) hone (hsat _ (by decide))
  have hb25 : z 25 ≤ 1 := bitRow_le_one hq (hcanon 25) hone (hsat _ (by decide))
  have hb26 : z 26 ≤ 1 := bitRow_le_one hq (hcanon 26) hone (hsat _ (by decide))
  have hb27 : z 27 ≤ 1 := bitRow_le_one hq (hcanon 27) hone (hsat _ (by decide))
  have hb28 : z 28 ≤ 1 := bitRow_le_one hq (hcanon 28) hone (hsat _ (by decide))
  have hb29 : z 29 ≤ 1 := bitRow_le_one hq (hcanon 29) hone (hsat _ (by decide))
  have hb30 : z 30 ≤ 1 := bitRow_le_one hq (hcanon 30) hone (hsat _ (by decide))
  have hb31 : z 31 ≤ 1 := bitRow_le_one hq (hcanon 31) hone (hsat _ (by decide))
  have hb32 : z 32 ≤ 1 := bitRow_le_one hq (hcanon 32) hone (hsat _ (by decide))
  have hb33 : z 33 ≤ 1 := bitRow_le_one hq (hcanon 33) hone (hsat _ (by decide))
  have hb34 : z 34 ≤ 1 := bitRow_le_one hq (hcanon 34) hone (hsat _ (by decide))
  have hb35 : z 35 ≤ 1 := bitRow_le_one hq (hcanon 35) hone (hsat _ (by decide))
  have hb36 : z 36 ≤ 1 := bitRow_le_one hq (hcanon 36) hone (hsat _ (by decide))
  have hb37 : z 37 ≤ 1 := bitRow_le_one hq (hcanon 37) hone (hsat _ (by decide))
  have hb38 : z 38 ≤ 1 := bitRow_le_one hq (hcanon 38) hone (hsat _ (by decide))
  have hb39 : z 39 ≤ 1 := bitRow_le_one hq (hcanon 39) hone (hsat _ (by decide))
  have hb40 : z 40 ≤ 1 := bitRow_le_one hq (hcanon 40) hone (hsat _ (by decide))
  have hb41 : z 41 ≤ 1 := bitRow_le_one hq (hcanon 41) hone (hsat _ (by decide))
  have hb42 : z 42 ≤ 1 := bitRow_le_one hq (hcanon 42) hone (hsat _ (by decide))
  have hb43 : z 43 ≤ 1 := bitRow_le_one hq (hcanon 43) hone (hsat _ (by decide))
  have hb44 : z 44 ≤ 1 := bitRow_le_one hq (hcanon 44) hone (hsat _ (by decide))
  have hb45 : z 45 ≤ 1 := bitRow_le_one hq (hcanon 45) hone (hsat _ (by decide))
  have hb46 : z 46 ≤ 1 := bitRow_le_one hq (hcanon 46) hone (hsat _ (by decide))
  have hb47 : z 47 ≤ 1 := bitRow_le_one hq (hcanon 47) hone (hsat _ (by decide))
  have hb48 : z 48 ≤ 1 := bitRow_le_one hq (hcanon 48) hone (hsat _ (by decide))
  have hb49 : z 49 ≤ 1 := bitRow_le_one hq (hcanon 49) hone (hsat _ (by decide))
  have hb50 : z 50 ≤ 1 := bitRow_le_one hq (hcanon 50) hone (hsat _ (by decide))
  have hb51 : z 51 ≤ 1 := bitRow_le_one hq (hcanon 51) hone (hsat _ (by decide))
  have hb52 : z 52 ≤ 1 := bitRow_le_one hq (hcanon 52) hone (hsat _ (by decide))
  have hb53 : z 53 ≤ 1 := bitRow_le_one hq (hcanon 53) hone (hsat _ (by decide))
  have hb54 : z 54 ≤ 1 := bitRow_le_one hq (hcanon 54) hone (hsat _ (by decide))
  have hb55 : z 55 ≤ 1 := bitRow_le_one hq (hcanon 55) hone (hsat _ (by decide))
  have hb56 : z 56 ≤ 1 := bitRow_le_one hq (hcanon 56) hone (hsat _ (by decide))
  have hb57 : z 57 ≤ 1 := bitRow_le_one hq (hcanon 57) hone (hsat _ (by decide))
  have hb58 : z 58 ≤ 1 := bitRow_le_one hq (hcanon 58) hone (hsat _ (by decide))
  have hb59 : z 59 ≤ 1 := bitRow_le_one hq (hcanon 59) hone (hsat _ (by decide))
  have hb60 : z 60 ≤ 1 := bitRow_le_one hq (hcanon 60) hone (hsat _ (by decide))
  have hb61 : z 61 ≤ 1 := bitRow_le_one hq (hcanon 61) hone (hsat _ (by decide))
  have hb62 : z 62 ≤ 1 := bitRow_le_one hq (hcanon 62) hone (hsat _ (by decide))
  have hb63 : z 63 ≤ 1 := bitRow_le_one hq (hcanon 63) hone (hsat _ (by decide))
  have hb64 : z 64 ≤ 1 := bitRow_le_one hq (hcanon 64) hone (hsat _ (by decide))
  have hb65 : z 65 ≤ 1 := bitRow_le_one hq (hcanon 65) hone (hsat _ (by decide))
  have hc1 : z 1 < 18446744069414584321 := hcanon 1
  have hc66 : z 66 < 18446744069414584321 := hcanon 66
  -- Row 64: `var = Σ 2^i · bit_i` in the field.
  have hrec : (z 1 + 18446744069414584320 * z 2 + 18446744069414584319 * z 3 + 18446744069414584317 * z 4 + 18446744069414584313 * z 5 + 18446744069414584305 * z 6 + 18446744069414584289 * z 7 + 18446744069414584257 * z 8 + 18446744069414584193 * z 9 + 18446744069414584065 * z 10 + 18446744069414583809 * z 11 + 18446744069414583297 * z 12 + 18446744069414582273 * z 13 + 18446744069414580225 * z 14 + 18446744069414576129 * z 15 + 18446744069414567937 * z 16 + 18446744069414551553 * z 17 + 18446744069414518785 * z 18 + 18446744069414453249 * z 19 + 18446744069414322177 * z 20 + 18446744069414060033 * z 21 + 18446744069413535745 * z 22 + 18446744069412487169 * z 23 + 18446744069410390017 * z 24 + 18446744069406195713 * z 25 + 18446744069397807105 * z 26 + 18446744069381029889 * z 27 + 18446744069347475457 * z 28 + 18446744069280366593 * z 29 + 18446744069146148865 * z 30 + 18446744068877713409 * z 31 + 18446744068340842497 * z 32 + 18446744067267100673 * z 33 + 18446744065119617025 * z 34 + 18446744060824649729 * z 35 + 18446744052234715137 * z 36 + 18446744035054845953 * z 37 + 18446744000695107585 * z 38 + 18446743931975630849 * z 39 + 18446743794536677377 * z 40 + 18446743519658770433 * z 41 + 18446742969902956545 * z 42 + 18446741870391328769 * z 43 + 18446739671368073217 * z 44 + 18446735273321562113 * z 45 + 18446726477228539905 * z 46 + 18446708885042495489 * z 47 + 18446673700670406657 * z 48 + 18446603331926228993 * z 49 + 18446462594437873665 * z 50 + 18446181119461163009 * z 51 + 18445618169507741697 * z 52 + 18444492269600899073 * z 53 + 18442240469787213825 * z 54 + 18437736870159843329 * z 55 + 18428729670905102337 * z 56 + 18410715272395620353 * z 57 + 18374686475376656385 * z 58 + 18302628881338728449 * z 59 + 18158513693262872577 * z 60 + 17870283317111160833 * z 61 + 17293822564807737345 * z 62 + 16140901060200890369 * z 63 + 13835058050987196417 * z 64 + 9223372032559808513 * z 65) % 18446744069414584321 = 0 := by
    simpa [RowHolds, lcEval, goldilocksP, hone] using
      hsat ⟨[(1, 1), (2, 18446744069414584320), (3, 18446744069414584319), (4, 18446744069414584317), (5, 18446744069414584313), (6, 18446744069414584305), (7, 18446744069414584289), (8, 18446744069414584257), (9, 18446744069414584193), (10, 18446744069414584065), (11, 18446744069414583809), (12, 18446744069414583297), (13, 18446744069414582273), (14, 18446744069414580225), (15, 18446744069414576129), (16, 18446744069414567937), (17, 18446744069414551553), (18, 18446744069414518785), (19, 18446744069414453249), (20, 18446744069414322177), (21, 18446744069414060033), (22, 18446744069413535745), (23, 18446744069412487169), (24, 18446744069410390017), (25, 18446744069406195713), (26, 18446744069397807105), (27, 18446744069381029889), (28, 18446744069347475457), (29, 18446744069280366593), (30, 18446744069146148865), (31, 18446744068877713409), (32, 18446744068340842497), (33, 18446744067267100673), (34, 18446744065119617025), (35, 18446744060824649729), (36, 18446744052234715137), (37, 18446744035054845953), (38, 18446744000695107585), (39, 18446743931975630849), (40, 18446743794536677377), (41, 18446743519658770433), (42, 18446742969902956545), (43, 18446741870391328769), (44, 18446739671368073217), (45, 18446735273321562113), (46, 18446726477228539905), (47, 18446708885042495489), (48, 18446673700670406657), (49, 18446603331926228993), (50, 18446462594437873665), (51, 18446181119461163009), (52, 18445618169507741697), (53, 18444492269600899073), (54, 18442240469787213825), (55, 18437736870159843329), (56, 18428729670905102337), (57, 18410715272395620353), (58, 18374686475376656385), (59, 18302628881338728449), (60, 18158513693262872577), (61, 17870283317111160833), (62, 17293822564807737345), (63, 16140901060200890369), (64, 13835058050987196417), (65, 9223372032559808513)], [(0, 1)], []⟩ (by decide)
  -- Row 68 (canonicity gate): `hi_is_max · lo = 0`.
  have hC : z 66 * (z 2 + 2 * z 3 + 4 * z 4 + 8 * z 5 + 16 * z 6 + 32 * z 7 + 64 * z 8 + 128 * z 9 + 256 * z 10 + 512 * z 11 + 1024 * z 12 + 2048 * z 13 + 4096 * z 14 + 8192 * z 15 + 16384 * z 16 + 32768 * z 17 + 65536 * z 18 + 131072 * z 19 + 262144 * z 20 + 524288 * z 21 + 1048576 * z 22 + 2097152 * z 23 + 4194304 * z 24 + 8388608 * z 25 + 16777216 * z 26 + 33554432 * z 27 + 67108864 * z 28 + 134217728 * z 29 + 268435456 * z 30 + 536870912 * z 31 + 1073741824 * z 32 + 2147483648 * z 33) % 18446744069414584321 = 0 := by
    simpa [RowHolds, lcEval, goldilocksP, hone] using
      hsat ⟨[(66, 1)], [(2, 1), (3, 2), (4, 4), (5, 8), (6, 16), (7, 32), (8, 64), (9, 128), (10, 256), (11, 512), (12, 1024), (13, 2048), (14, 4096), (15, 8192), (16, 16384), (17, 32768), (18, 65536), (19, 131072), (20, 262144), (21, 524288), (22, 1048576), (23, 2097152), (24, 4194304), (25, 8388608), (26, 16777216), (27, 33554432), (28, 67108864), (29, 134217728), (30, 268435456), (31, 536870912), (32, 1073741824), (33, 2147483648)], []⟩ (by decide)
  -- Row 67 (inverse row): `(hi − 0xFFFFFFFF) · inv = 1 − hi_is_max`.
  have hB : (z 34 + 2 * z 35 + 4 * z 36 + 8 * z 37 + 16 * z 38 + 32 * z 39 + 64 * z 40 + 128 * z 41 + 256 * z 42 + 512 * z 43 + 1024 * z 44 + 2048 * z 45 + 4096 * z 46 + 8192 * z 47 + 16384 * z 48 + 32768 * z 49 + 65536 * z 50 + 131072 * z 51 + 262144 * z 52 + 524288 * z 53 + 1048576 * z 54 + 2097152 * z 55 + 4194304 * z 56 + 8388608 * z 57 + 16777216 * z 58 + 33554432 * z 59 + 67108864 * z 60 + 134217728 * z 61 + 268435456 * z 62 + 536870912 * z 63 + 1073741824 * z 64 + 2147483648 * z 65 + 18446744065119617026) * z 67 % 18446744069414584321 = (18446744069414584320 * z 66 + 1) % 18446744069414584321 := by
    simpa [RowHolds, lcEval, goldilocksP, hone] using
      hsat ⟨[(34, 1), (35, 2), (36, 4), (37, 8), (38, 16), (39, 32), (40, 64), (41, 128), (42, 256), (43, 512), (44, 1024), (45, 2048), (46, 4096), (47, 8192), (48, 16384), (49, 32768), (50, 65536), (51, 131072), (52, 262144), (53, 524288), (54, 1048576), (55, 2097152), (56, 4194304), (57, 8388608), (58, 16777216), (59, 33554432), (60, 67108864), (61, 134217728), (62, 268435456), (63, 536870912), (64, 1073741824), (65, 2147483648), (0, 18446744065119617026)], [(67, 1)], [(66, 18446744069414584320), (0, 1)]⟩ (by decide)
  -- The recomposed integer stays below the modulus.
  have hSlt : z 2 + 2 * z 3 + 4 * z 4 + 8 * z 5 + 16 * z 6 + 32 * z 7 + 64 * z 8 + 128 * z 9 + 256 * z 10 + 512 * z 11 + 1024 * z 12 + 2048 * z 13 + 4096 * z 14 + 8192 * z 15 + 16384 * z 16 + 32768 * z 17 + 65536 * z 18 + 131072 * z 19 + 262144 * z 20 + 524288 * z 21 + 1048576 * z 22 + 2097152 * z 23 + 4194304 * z 24 + 8388608 * z 25 + 16777216 * z 26 + 33554432 * z 27 + 67108864 * z 28 + 134217728 * z 29 + 268435456 * z 30 + 536870912 * z 31 + 1073741824 * z 32 + 2147483648 * z 33 + 4294967296 * z 34 + 8589934592 * z 35 + 17179869184 * z 36 + 34359738368 * z 37 + 68719476736 * z 38 + 137438953472 * z 39 + 274877906944 * z 40 + 549755813888 * z 41 + 1099511627776 * z 42 + 2199023255552 * z 43 + 4398046511104 * z 44 + 8796093022208 * z 45 + 17592186044416 * z 46 + 35184372088832 * z 47 + 70368744177664 * z 48 + 140737488355328 * z 49 + 281474976710656 * z 50 + 562949953421312 * z 51 + 1125899906842624 * z 52 + 2251799813685248 * z 53 + 4503599627370496 * z 54 + 9007199254740992 * z 55 + 18014398509481984 * z 56 + 36028797018963968 * z 57 + 72057594037927936 * z 58 + 144115188075855872 * z 59 + 288230376151711744 * z 60 + 576460752303423488 * z 61 + 1152921504606846976 * z 62 + 2305843009213693952 * z 63 + 4611686018427387904 * z 64 + 9223372036854775808 * z 65 < 18446744069414584321 := by
    rcases hq' _ _ hC with h66 | hlo
    · -- `hi_is_max = 0`: the inverse row forces `hi ≠ 0xFFFFFFFF`.
      by_cases hM : z 34 + 2 * z 35 + 4 * z 36 + 8 * z 37 + 16 * z 38 + 32 * z 39 + 64 * z 40 + 128 * z 41 + 256 * z 42 + 512 * z 43 + 1024 * z 44 + 2048 * z 45 + 4096 * z 46 + 8192 * z 47 + 16384 * z 48 + 32768 * z 49 + 65536 * z 50 + 131072 * z 51 + 262144 * z 52 + 524288 * z 53 + 1048576 * z 54 + 2097152 * z 55 + 4194304 * z 56 + 8388608 * z 57 + 16777216 * z 58 + 33554432 * z 59 + 67108864 * z 60 + 134217728 * z 61 + 268435456 * z 62 + 536870912 * z 63 + 1073741824 * z 64 + 2147483648 * z 65 = 4294967295
      · exfalso
        rw [hM] at hB
        omega
      · omega
    · -- `lo = 0`: the value is a multiple of 2^32, hence below the modulus.
      omega
  have hbv : bitsValue z = z 2 + 2 * z 3 + 4 * z 4 + 8 * z 5 + 16 * z 6 + 32 * z 7 + 64 * z 8 + 128 * z 9 + 256 * z 10 + 512 * z 11 + 1024 * z 12 + 2048 * z 13 + 4096 * z 14 + 8192 * z 15 + 16384 * z 16 + 32768 * z 17 + 65536 * z 18 + 131072 * z 19 + 262144 * z 20 + 524288 * z 21 + 1048576 * z 22 + 2097152 * z 23 + 4194304 * z 24 + 8388608 * z 25 + 16777216 * z 26 + 33554432 * z 27 + 67108864 * z 28 + 134217728 * z 29 + 268435456 * z 30 + 536870912 * z 31 + 1073741824 * z 32 + 2147483648 * z 33 + 4294967296 * z 34 + 8589934592 * z 35 + 17179869184 * z 36 + 34359738368 * z 37 + 68719476736 * z 38 + 137438953472 * z 39 + 274877906944 * z 40 + 549755813888 * z 41 + 1099511627776 * z 42 + 2199023255552 * z 43 + 4398046511104 * z 44 + 8796093022208 * z 45 + 17592186044416 * z 46 + 35184372088832 * z 47 + 70368744177664 * z 48 + 140737488355328 * z 49 + 281474976710656 * z 50 + 562949953421312 * z 51 + 1125899906842624 * z 52 + 2251799813685248 * z 53 + 4503599627370496 * z 54 + 9007199254740992 * z 55 + 18014398509481984 * z 56 + 36028797018963968 * z 57 + 72057594037927936 * z 58 + 144115188075855872 * z 59 + 288230376151711744 * z 60 + 576460752303423488 * z 61 + 1152921504606846976 * z 62 + 2305843009213693952 * z 63 + 4611686018427387904 * z 64 + 9223372036854775808 * z 65 := by
    have hr : List.range 64 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] := by decide
    simp [bitsValue, bitCol, hr]
  rw [hbv]
  simp only [varCol]
  exact ⟨by omega, hSlt⟩

end Nightstream.Implementation.R1CS
