import NightstreamFPrime.Spec.Algebra
import NightstreamFPrime.Spec.Relation

/-!
Owns the one Nightstream Goldilocks production profile: `b = 2`,
`k_rho = 16`, `B = 2^16`, `η = 81`, `d = 54`, `κ = 18`, quadratic
extension. Every other module reads the profile from here; no module restates
these values.

Provenance: production part of
`formal/nightstream-lean/Nightstream/SuperNeo/Concrete/Parameters.lean` at
commit `fb7a8a99aefbb8ebb5474681ecf80f1b95a1b7a2`; only the production namespace is copied.
-/

namespace NightstreamFPrime.Spec

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
  /-- Floor of `log2(5^54)`: challenge-set size, not end-to-end security. -/
  challengeSetBitsFloor : Nat
  /-- One fresh source per step. -/
  freshSources : Nat
  /-- Running sources, equal to `k_rho`. -/
  runningSources : Nat
  /-- CCS matrices in the F′ structure. -/
  ccsMatrices : Nat

def productionProfile : ProductionProfile where
  global := productionGlobalParams
  eta := 81
  ringDegree := ringDegree
  commitmentWidth := 18
  extensionDegree := 2
  challengeSetBitsFloor := 125
  freshSources := 1
  runningSources := 16
  ccsMatrices := 14

/-- PiRLC inputs in exact `K + k` order. -/
def ProductionProfile.piRlcInputs (p : ProductionProfile) : Nat :=
  p.freshSources + p.runningSources

/-- PiDEC children, equal to `k_rho`. -/
def ProductionProfile.piDecChildren (p : ProductionProfile) : Nat :=
  p.global.k

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
    productionProfile.challengeSetBitsFloor = 125 ∧
    productionProfile.piRlcInputs = 17 ∧
    productionProfile.piDecChildren = 16 ∧
    productionProfile.ccsMatrices = 14 := by
  decide

theorem production_norm_stages :
    NormStage.bound productionGlobalParams .fresh = 2 ∧
    NormStage.bound productionGlobalParams .combined = 65536 ∧
    NormStage.bound productionGlobalParams .ambient = 9223372034707292161 := by
  decide

theorem production_msis_norm_bound :
    productionGlobalParams.msisNormBound = 113246208 := by
  decide

/-- Definition 14 at the production arity `K + k = 17`. -/
theorem production_rlc_bound_one_fresh :
    (1 + 16) * 216 * (2 - 1) < 65536 := by decide

end NightstreamFPrime.Spec
