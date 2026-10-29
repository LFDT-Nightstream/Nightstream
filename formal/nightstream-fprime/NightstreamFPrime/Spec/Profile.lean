import NightstreamFPrime.Spec.Algebra
import NightstreamFPrime.Spec.Relation

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
  challengeSetBitsFloor : Nat
  freshSources : Nat
  runningSources : Nat
  ccsMatrices : Nat

def productionProfile : ProductionProfile where
  global := productionGlobalParams
  eta := 81
  ringDegree := ringDegree
  commitmentWidth := 22
  extensionDegree := 2
  challengeSetBitsFloor := 125
  freshSources := 1
  runningSources := 16
  ccsMatrices := 4

def productionChallengeSetCardinality : Nat := 5 ^ productionProfile.ringDegree

def ProductionProfile.piRlcInputs (p : ProductionProfile) : Nat :=
  p.freshSources + p.runningSources

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
    productionProfile.commitmentWidth = 22 ∧
    productionProfile.extensionDegree = 2 ∧
    productionProfile.challengeSetBitsFloor = 125 ∧
    productionProfile.piRlcInputs = 17 ∧
    productionProfile.piDecChildren = 16 ∧
    productionProfile.ccsMatrices = 4 := by
  decide

theorem production_norm_stages :
    NormStage.bound productionGlobalParams .fresh = 2 ∧
    NormStage.bound productionGlobalParams .combined = 65536 ∧
    NormStage.bound productionGlobalParams .ambient = 9223372034707292161 := by
  decide

theorem production_bigB_below_half_modulus :
    2 * productionGlobalParams.bigB < productionGlobalParams.q := by
  decide

theorem production_msis_norm_bound :
    productionGlobalParams.msisNormBound = 113246208 := by
  decide

/-- SuperNeo v1.2 Definition 22. -/
theorem production_rlc_bound_one_fresh :
    (1 + 16) * 216 * (2 - 1) < 65536 := by decide

end NightstreamFPrime.Spec
