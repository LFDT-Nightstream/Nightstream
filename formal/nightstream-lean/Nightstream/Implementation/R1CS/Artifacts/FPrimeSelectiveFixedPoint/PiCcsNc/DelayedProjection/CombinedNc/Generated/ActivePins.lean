/-
Generated file: production combined-NC artifact; do not hand-edit.

Owns: the active constant/selector pins and their exact selector-domain and one-hot row owners.

Does not own: decoding, row satisfaction, transcript authority, commitment
binding, semantic acceptance, costs, or permission to remove rows.

Emits constraints: no.

| Stable stage path | Obligation | Authority class |
|---|---|---|
| `f_prime.pi_ccs_nc.delayed.combined.generated` | The generated payload named by `Owns` above | computed artifact |
-/

import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Schema
import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.ActivePins.PackedCoordinates

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.ActivePins

def raw : RawActivePins :=
  { schemaVersion := 1
    sourceRows := 11308137
    sourceColumns := 10997363
    finalRows := 14944219
    finalColumns := 11437038
    constantOneColumn := 0
    constantOneValue := 1
    selectorColumns := [270, 271, 272]
    recursiveSelectorValues := [0, 0, 1]
    packedLaneCount := 54
    packedBlockCount := 5
    publicCoordinateCount := 270
    selectorDomainRows := [
  { schemaVersion := 1
    rows := 14944219
    columns := 11437038
    emittedRow := 0
    runIndex := 0
    family := .selectorDomain
    arm := none
    ports := [
        { explicit := [{ column := 270, coefficient := 1 }], geometric := [] }
      , { explicit := [{ column := 0, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      ] }
,
  { schemaVersion := 1
    rows := 14944219
    columns := 11437038
    emittedRow := 1
    runIndex := 0
    family := .selectorDomain
    arm := none
    ports := [
        { explicit := [{ column := 271, coefficient := 1 }], geometric := [] }
      , { explicit := [{ column := 0, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      ] }
,
  { schemaVersion := 1
    rows := 14944219
    columns := 11437038
    emittedRow := 2
    runIndex := 0
    family := .selectorDomain
    arm := none
    ports := [
        { explicit := [{ column := 272, coefficient := 1 }], geometric := [] }
      , { explicit := [{ column := 0, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      ] }
    ]
    oneHotRow :=
  { schemaVersion := 1
    rows := 14944219
    columns := 11437038
    emittedRow := 4729579
    runIndex := 5
    family := .oneHot
    arm := none
    ports := [
        { explicit := [], geometric := [] }
      , { explicit := [{ column := 0, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [{ column := 0, coefficient := 18446744069414584320 }, { column := 270, coefficient := 1 }, { column := 271, coefficient := 1 }, { column := 272, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      ] }
  }

def packedCoordinates : List RawPackedPublicCoordinate := PackedCoordinates.values

end Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.ActivePins
