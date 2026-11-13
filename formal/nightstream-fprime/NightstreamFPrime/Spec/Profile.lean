import NightstreamFPrime.Spec.Algebra
import NightstreamFPrime.Spec.Relation

namespace NightstreamFPrime.Spec

def productionGlobalParams : GlobalParams where
  q := fieldModulus
  b := 2
  k := 16
  maxFresh := 495
  expansionT := 128
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
  eta := 128
  ringDegree := ringDegree
  commitmentWidth := 18
  extensionDegree := 2
  challengeSetBitsFloor := 148
  freshSources := 1
  runningSources := 16
  ccsMatrices := 4

def productionChallengeSetCardinality : Nat := 5 ^ productionProfile.ringDegree

def ProductionProfile.piRlcInputs (p : ProductionProfile) : Nat :=
  p.freshSources + p.runningSources

def ProductionProfile.piDecChildren (p : ProductionProfile) : Nat :=
  p.global.k

theorem production_parameter_values :
    productionProfile.global.q = 18446744073709551557 ∧
    productionProfile.global.b = 2 ∧
    productionProfile.global.k = 16 ∧
    productionProfile.global.maxFresh = 495 ∧
    productionProfile.global.expansionT = 128 ∧
    productionProfile.global.bigB = 65536 ∧
    productionProfile.eta = 128 ∧
    productionProfile.ringDegree = 64 ∧
    productionProfile.commitmentWidth = 18 ∧
    productionProfile.extensionDegree = 2 ∧
    productionProfile.challengeSetBitsFloor = 148 ∧
    productionProfile.piRlcInputs = 17 ∧
    productionProfile.piDecChildren = 16 ∧
    productionProfile.ccsMatrices = 4 := by
  decide

theorem production_norm_stages :
    NormStage.bound productionGlobalParams .fresh = 2 ∧
    NormStage.bound productionGlobalParams .combined = 65536 ∧
    NormStage.bound productionGlobalParams .ambient = 9223372036854775779 := by
  decide

theorem production_bigB_below_half_modulus :
    2 * productionGlobalParams.bigB < productionGlobalParams.q := by
  decide

theorem production_msis_norm_bound :
    productionGlobalParams.msisNormBound = 67108864 := by
  decide

/-- SuperNeo v1.2 Definition 22. -/
theorem production_rlc_bound_one_fresh :
    (1 + 16) * 128 * (2 - 1) < 65536 := by decide

end NightstreamFPrime.Spec
