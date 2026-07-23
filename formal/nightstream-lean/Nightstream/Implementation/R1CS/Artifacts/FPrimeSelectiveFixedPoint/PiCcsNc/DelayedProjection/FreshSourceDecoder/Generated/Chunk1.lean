import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.FreshSourceDecoder.Schema

/-!
Generated file: exact fresh public-X source decoder chunk; do not hand-edit.

The records describe only `prior_link.fresh_public_inputs[0]`, coordinates
0 through 269. This is the public-X source prefix consumed by the recursive
step, not the full private witness `Z` and not commitment authority.

The current Rust wire surface does not identify the exact binding row owned by
each coordinate. Consequently this artifact records normalized column and
selective-decoder provenance only; the row-level prior-link bridge remains
open.

Owns: one exact 14-record proof-free decoder shard.

Does not own: source values, full-witness coordinates, per-coordinate binding
rows, commitment binding, or permission to remove constraints.

Emits constraints: none; generated certificate data only.

| Stage path | Mathematical obligation | Authority class | Artifact owner |
|---|---|---|---|
| `pi_ccs.nc.fresh_x.generated.chunk1` | exact ordered source column and fail-closed selective disposition | generated/checked | `fresh_source.rs` |
-/

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.FreshSourceDecoder.Generated.Chunk1

open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.FreshSourceDecoder

def schemaVersion : Nat := 1
def sourceArm : Nat := 2
def sourceCount : Nat := 1
def logicalColumnCount : Nat := 270
def finalColumnCount : Nat := 11437038
def records : List SourceColumnRecord := [
  { logicalColumn := 256, sourceArmColumn := 21803, resolution := .equalityAlias 8058 46786 1 false }
, { logicalColumn := 257, sourceArmColumn := 21804, resolution := .direct 133464 41 false }
, { logicalColumn := 258, sourceArmColumn := 21805, resolution := .direct 133505 41 false }
, { logicalColumn := 259, sourceArmColumn := 21806, resolution := .direct 133546 41 false }
, { logicalColumn := 260, sourceArmColumn := 21807, resolution := .direct 133587 41 false }
, { logicalColumn := 261, sourceArmColumn := 21808, resolution := .direct 133628 41 false }
, { logicalColumn := 262, sourceArmColumn := 21809, resolution := .direct 133669 41 false }
, { logicalColumn := 263, sourceArmColumn := 21810, resolution := .direct 133710 41 false }
, { logicalColumn := 264, sourceArmColumn := 21811, resolution := .direct 133751 41 false }
, { logicalColumn := 265, sourceArmColumn := 21812, resolution := .direct 133792 41 false }
, { logicalColumn := 266, sourceArmColumn := 21813, resolution := .direct 133833 41 false }
, { logicalColumn := 267, sourceArmColumn := 21814, resolution := .direct 133874 41 false }
, { logicalColumn := 268, sourceArmColumn := 21815, resolution := .direct 133915 41 false }
, { logicalColumn := 269, sourceArmColumn := 21816, resolution := .direct 133956 41 false }
]

end Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.FreshSourceDecoder.Generated.Chunk1
