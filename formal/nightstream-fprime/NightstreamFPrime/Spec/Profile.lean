import NightstreamFPrime.Spec.Algebra
import NightstreamFPrime.Spec.Relation

namespace NightstreamFPrime.Spec

def productionGlobalParams : GlobalParams where
  q := fieldModulus
  b := 2
  k := 14
  maxFresh := 113
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
  runningSources := 14
  ccsMatrices := 4

def productionChallengeSetCardinality : Nat := 5 ^ productionProfile.ringDegree

def ProductionProfile.piRlcInputs (p : ProductionProfile) : Nat :=
  p.freshSources + p.runningSources

def ProductionProfile.piDecChildren (p : ProductionProfile) : Nat :=
  p.global.k

theorem production_parameter_values :
    productionProfile.global.q = 18446744073709551557 ∧
    productionProfile.global.b = 2 ∧
    productionProfile.global.k = 14 ∧
    productionProfile.global.maxFresh = 113 ∧
    productionProfile.global.expansionT = 128 ∧
    productionProfile.global.bigB = 16384 ∧
    productionProfile.eta = 128 ∧
    productionProfile.ringDegree = 64 ∧
    productionProfile.commitmentWidth = 18 ∧
    productionProfile.extensionDegree = 2 ∧
    productionProfile.challengeSetBitsFloor = 148 ∧
    productionProfile.piRlcInputs = 15 ∧
    productionProfile.piDecChildren = 14 ∧
    productionProfile.ccsMatrices = 4 := by
  decide

theorem production_norm_stages :
    NormStage.bound productionGlobalParams .fresh = 2 ∧
    NormStage.bound productionGlobalParams .combined = 16384 ∧
    NormStage.bound productionGlobalParams .ambient = 9223372036854775779 := by
  decide

theorem production_bigB_below_half_modulus :
    2 * productionGlobalParams.bigB < productionGlobalParams.q := by
  decide

theorem production_msis_norm_bound :
    productionGlobalParams.msisNormBound = 16777216 := by
  decide

/-- SuperNeo v1.2 Definition 22. -/
theorem production_rlc_bound_one_fresh :
    (1 + 14) * 128 * (2 - 1) < 16384 := by decide

end NightstreamFPrime.Spec
