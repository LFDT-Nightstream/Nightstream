import SuperNeo.Primitives.Dimensions
import SuperNeo.FPrimeRecursiveVerifier.PiRlcChallenge.Parameters

/-!
Owns: fixed dimensions for the Pi_RLC algebra verifier.

Does not own: algebra equations, transcript state, or Rust configuration.

Emits constraints: no.

Authority boundary: these constants define the theorem-level fixed shape; a
concrete bridge must prove the runtime circuit uses the same dimensions.

| Name | Value | Owned shape fact |
|---|---:|---|
| `inputCount` | 15 | Number of Pi_CCS output claims and rho values |
| `commitmentLanes` | 18 | Ajtai commitment ring lanes |
| `activeXColumns` | 5 | Ring columns populated by fixed public X |
| `yRingRows` | 3 | Active `y_ring` rows |
| `extensionLimbs` | 2 | Base-field limbs in one K value |
| `paddedDegree` | 64 | Split-NC vector length; active ring degree is 54 |

These are the fixed F-prime relation parameters, not configurable protocol
defaults.  Changing one requires a new measured Rust circuit and conformance
evidence.
-/

namespace SuperNeo.FPrimeRecursiveVerifier.PiRlcAlgebra

def inputCount : Nat := PiRlcChallenge.rhoCount
def commitmentLanes : Nat := 18
def activeXColumns : Nat := 5
def yRingRows : Nat := 3
def extensionLimbs : Nat := 2
def paddedDegree : Nat := 64

theorem inputCount_eq : inputCount = 15 := rfl
theorem commitmentLanes_eq : commitmentLanes = 18 := rfl
theorem activeXColumns_eq : activeXColumns = 5 := rfl
theorem yRingRows_eq : yRingRows = 3 := rfl
theorem extensionLimbs_eq : extensionLimbs = 2 := rfl
theorem paddedDegree_eq : paddedDegree = 64 := rfl
theorem ringDegree_lt_paddedDegree : SuperNeo.d < paddedDegree := by decide

end SuperNeo.FPrimeRecursiveVerifier.PiRlcAlgebra
