import Nightstream.SuperNeo.Relations
import Nightstream.SuperNeo.Concrete.Algebra

/-!
Verifier-owned production parameter instantiation for SuperNeo Appendix B.2.

The values mirror `neo_params::goldilocks_paper_b2` and
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
  k := 14
  maxFresh := 61
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
    productionProfile.global.k = 14 ∧
    productionProfile.global.maxFresh = 61 ∧
    productionProfile.global.expansionT = 216 ∧
    productionProfile.global.bigB = 16384 ∧
    productionProfile.eta = 81 ∧
    productionProfile.ringDegree = 54 ∧
    productionProfile.commitmentWidth = 18 ∧
    productionProfile.extensionDegree = 2 ∧
    productionProfile.challengeSetBitsFloor = 125 := by
  decide

theorem production_norm_stages :
    NormStage.bound productionGlobalParams .fresh = 2 ∧
    NormStage.bound productionGlobalParams .combined = 16384 ∧
    NormStage.bound productionGlobalParams .ambient = 9223372034707292161 := by
  decide

theorem production_msis_norm_bound :
    productionGlobalParams.msisNormBound = 28311552 := by
  decide

theorem production_allows_every_advertised_batch {fresh : Nat}
    (hFresh : fresh ≤ 61) :
    (fresh + 14) * 216 < 16384 := by
  simpa [productionGlobalParams, GlobalParams.bigB] using
    productionGlobalParams.rlc_bound_for hFresh

/-! ## Radix-four width candidate

This candidate changes only the decomposition radix and source count. It is
not a production profile. In particular, these arithmetic facts do not supply
the canonical radix-four PiDEC implementation, its circuit refinement, or a
concrete Module-SIS estimator receipt.
-/

namespace Radix4Candidate

/-- Width-reduction candidate with the same combined norm bound as the active
radix-two profile. -/
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

/-- Eighteen is the largest fresh arity permitted by Definition 14 for this
candidate. Nebula needs only one fresh source. -/
theorem maxFresh_exact :
    (18 + 7) * 216 * (4 - 1) < 16384 ∧
      ¬((19 + 7) * 216 * (4 - 1) < 16384) := by
  decide

theorem oneFresh_rlc_bound :
    (1 + globalParams.k) * globalParams.expansionT *
        (globalParams.b - 1) < globalParams.bigB := by
  decide

/-- The relaxed-binding Module-SIS norm parameter is unchanged because it
depends on `T * B`, not on the selected radix separately. -/
theorem msisNormBound_eq_production :
    globalParams.msisNormBound = productionGlobalParams.msisNormBound := by
  decide

/-- For a degree-eight CCS relation, radix four does not increase the joint
PiCCS verifier degree: both profiles use degree nine. -/
theorem degreeEight_verifierDegree_eq_production :
    max (8 + 1) (2 * globalParams.b) =
      max (8 + 1) (2 * productionGlobalParams.b) := by
  decide

theorem runningSourceCount_halved :
    2 * globalParams.k = productionGlobalParams.k := by
  decide

end Radix4Candidate

end Concrete

end Nightstream.SuperNeo
