import Nightstream.SuperNeo.Relations
import Nightstream.SuperNeo.Concrete.Algebra

/-!
Verifier-owned production parameter instantiation for the frozen Nightstream
Goldilocks profile.

The values mirror `neo_params::nightstream_goldilocks_k16` and
`paper::params::Params::production`. The Definition-14 inequality is carried
as proof data and inherited by every permitted smaller batch arity.
-/

namespace Nightstream.SuperNeo

/-- Every arity below the verifier-owned cap inherits Definition 14. -/
theorem GlobalParams.rlc_bound_for (p : GlobalParams) {fresh : Nat}
    (hFresh : fresh ≤ p.maxFresh) :
    (fresh + p.k) * p.expansionT * (p.b - 1) < p.bigB := by
  have hsum : fresh + p.k ≤ p.maxFresh + p.k :=
    Nat.add_le_add_right hFresh p.k
  have hscaled : (fresh + p.k) * p.expansionT ≤
      (p.maxFresh + p.k) * p.expansionT :=
    Nat.mul_le_mul_right p.expansionT hsum
  have hfull : (fresh + p.k) * p.expansionT * (p.b - 1) ≤
      (p.maxFresh + p.k) * p.expansionT * (p.b - 1) :=
    Nat.mul_le_mul_right (p.b - 1) hscaled
  exact Nat.lt_of_le_of_lt hfull p.rlc_bound

namespace Concrete

def productionGlobalParams : GlobalParams where
  q := goldilocksModulus
  b := 2
  k := 16
  maxFresh := 287
  expansionT := 216
  rlc_bound := by decide

structure ProductionProfile where
  global : GlobalParams
  eta : Nat
  ringDegree : Nat
  commitmentWidth : Nat
  extensionDegree : Nat
  /-- Floor of `log2(5^54)`. This is challenge-set size, not end-to-end or
  statistical security. -/
  challengeSetBitsFloor : Nat

def productionProfile : ProductionProfile where
  global := productionGlobalParams
  eta := 81
  ringDegree := Concrete.ringDegree
  commitmentWidth := 18
  extensionDegree := 2
  challengeSetBitsFloor := 125

theorem production_parameter_values :
    productionProfile.global.q = 18446744069414584321 ∧
    productionProfile.global.b = 2 ∧
    productionProfile.global.k = 16 ∧
    productionProfile.global.maxFresh = 287 ∧
    productionProfile.global.expansionT = 216 ∧
    productionProfile.global.bigB = 65536 ∧
    productionProfile.eta = 81 ∧
    productionProfile.ringDegree = 54 ∧
    productionProfile.commitmentWidth = 18 ∧
    productionProfile.extensionDegree = 2 ∧
    productionProfile.challengeSetBitsFloor = 125 := by
  decide

theorem production_norm_stages :
    NormStage.bound productionGlobalParams .fresh = 2 ∧
    NormStage.bound productionGlobalParams .combined = 65536 ∧
    NormStage.bound productionGlobalParams .ambient = 9223372034707292161 := by
  decide

theorem production_msis_norm_bound :
    productionGlobalParams.msisNormBound = 113246208 := by
  decide

theorem production_allows_every_advertised_batch {fresh : Nat}
    (hFresh : fresh ≤ 287) :
    (fresh + 16) * 216 < 65536 := by
  simpa [productionGlobalParams, GlobalParams.bigB] using
    productionGlobalParams.rlc_bound_for hFresh

/-! ## Unsupported radix-four reference

This namespace remains only for isolated proof-model comparisons. It is not
an allowed Nightstream profile and must not be used by generated artifacts,
the production relation, or lifecycle verification.
-/

namespace Radix4Candidate

def globalParams : GlobalParams where
  q := goldilocksModulus
  b := 4
  k := 7
  maxFresh := 18
  expansionT := 216
  rlc_bound := by decide

theorem parameter_values :
    globalParams.q = goldilocksModulus ∧
      globalParams.b = 4 ∧
      globalParams.k = 7 ∧
      globalParams.maxFresh = 18 ∧
      globalParams.expansionT = 216 ∧
      globalParams.bigB = 16384 := by
  decide

theorem maxFresh_exact :
    (18 + 7) * 216 * (4 - 1) < 16384 ∧
      ¬((19 + 7) * 216 * (4 - 1) < 16384) := by
  decide

theorem oneFresh_rlc_bound :
    (1 + globalParams.k) * globalParams.expansionT *
        (globalParams.b - 1) < globalParams.bigB := by
  decide

theorem msisNormBound_lt_production :
    globalParams.msisNormBound < productionGlobalParams.msisNormBound := by
  decide

theorem degreeEight_verifierDegree_eq_production :
    max (8 + 1) (2 * globalParams.b) =
      max (8 + 1) (2 * productionGlobalParams.b) := by
  decide

theorem runningSourceCount_lt_production :
    2 * globalParams.k < productionGlobalParams.k := by
  decide

end Radix4Candidate

end Concrete

end Nightstream.SuperNeo
