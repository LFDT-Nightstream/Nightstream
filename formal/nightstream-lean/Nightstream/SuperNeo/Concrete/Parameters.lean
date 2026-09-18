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
  securityBits : Nat

def productionProfile : ProductionProfile where
  global := productionGlobalParams
  eta := 81
  ringDegree := Concrete.ringDegree
  commitmentWidth := 18
  extensionDegree := 2
  securityBits := 125

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
    productionProfile.securityBits = 125 := by
  decide

theorem production_norm_stages :
    NormStage.bound productionGlobalParams .fresh = 2 ∧
    NormStage.bound productionGlobalParams .combined = 16384 ∧
    NormStage.bound productionGlobalParams .ambient = 9223372034707292160 := by
  decide

theorem production_msis_norm_bound :
    productionGlobalParams.msisNormBound = 28311552 := by
  decide

theorem production_allows_every_advertised_batch {fresh : Nat}
    (hFresh : fresh ≤ 61) :
    (fresh + 14) * 216 < 16384 := by
  simpa [productionGlobalParams, GlobalParams.bigB] using
    productionGlobalParams.rlc_bound_for hFresh

end Concrete

end Nightstream.SuperNeo
