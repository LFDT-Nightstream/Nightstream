import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.Schema

/-!
Generated file: authoritative raw-running assignment decoder chunk; do not
hand-edit.

Each provenance record carries both the normalized source-arm column and its
complete final selective-assignment scalar encoding. The generator fails
closed unless the final interval and encoding kind come from the exact direct
slot for the record's actual
`running[child].x[(logicalColumn % 54) * x_cols + logicalColumn / 54]` wire.

`balancedTernary` means the field value is reconstructed as
`sum(digit[i] * 3^i)` from exactly 41 signed-unit digits. It is not a binary
encoding and the first digit is not the scalar value.

This data does not establish delayed-projection acceptance, raw-child semantic
authority, commitment binding, or row-removal permission.

Owns: one exact 252-record raw-running physical-column provenance shard.

Does not own: assignment values, combined-NC acceptance, transcript scheduling,
commitment binding, or permission to remove rows.

Emits constraints: none; generated data only.

| Stable stage path | Obligation | Authority |
|---|---|---|
| `pi_ccs_nc.delayed_projection.raw_running_decoder.generated.chunk` | Exact generated coordinate-to-column records | computed artifact |
-/

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.Generated.Chunk9

open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder

def schemaVersion : Nat := 2
def sourceArm : Nat := 2
def childCount : Nat := 14
def logicalColumnCount : Nat := 270
def finalColumnCount : Nat := 11437038
def allocationRecords : List AllocationRecord := [
  { child := 8, logicalColumn := 108, sourceArmColumn := 39997, finalStart := 757689, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 109, sourceArmColumn := 40002, finalStart := 757894, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 110, sourceArmColumn := 40007, finalStart := 758099, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 111, sourceArmColumn := 40012, finalStart := 758304, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 112, sourceArmColumn := 40017, finalStart := 758509, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 113, sourceArmColumn := 40022, finalStart := 758714, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 114, sourceArmColumn := 40027, finalStart := 758919, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 115, sourceArmColumn := 40032, finalStart := 759124, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 116, sourceArmColumn := 40037, finalStart := 759329, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 117, sourceArmColumn := 40042, finalStart := 759534, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 118, sourceArmColumn := 40047, finalStart := 759739, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 119, sourceArmColumn := 40052, finalStart := 759944, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 120, sourceArmColumn := 40057, finalStart := 760149, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 121, sourceArmColumn := 40062, finalStart := 760354, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 122, sourceArmColumn := 40067, finalStart := 760559, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 123, sourceArmColumn := 40072, finalStart := 760764, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 124, sourceArmColumn := 40077, finalStart := 760969, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 125, sourceArmColumn := 40082, finalStart := 761174, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 126, sourceArmColumn := 40087, finalStart := 761379, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 127, sourceArmColumn := 40092, finalStart := 761584, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 128, sourceArmColumn := 40097, finalStart := 761789, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 129, sourceArmColumn := 40102, finalStart := 761994, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 130, sourceArmColumn := 40107, finalStart := 762199, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 131, sourceArmColumn := 40112, finalStart := 762404, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 132, sourceArmColumn := 40117, finalStart := 762609, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 133, sourceArmColumn := 40122, finalStart := 762814, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 134, sourceArmColumn := 40127, finalStart := 763019, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 135, sourceArmColumn := 40132, finalStart := 763224, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 136, sourceArmColumn := 40137, finalStart := 763429, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 137, sourceArmColumn := 40142, finalStart := 763634, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 138, sourceArmColumn := 40147, finalStart := 763839, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 139, sourceArmColumn := 40152, finalStart := 764044, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 140, sourceArmColumn := 40157, finalStart := 764249, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 141, sourceArmColumn := 40162, finalStart := 764454, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 142, sourceArmColumn := 40167, finalStart := 764659, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 143, sourceArmColumn := 40172, finalStart := 764864, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 144, sourceArmColumn := 40177, finalStart := 765069, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 145, sourceArmColumn := 40182, finalStart := 765274, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 146, sourceArmColumn := 40187, finalStart := 765479, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 147, sourceArmColumn := 40192, finalStart := 765684, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 148, sourceArmColumn := 40197, finalStart := 765889, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 149, sourceArmColumn := 40202, finalStart := 766094, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 150, sourceArmColumn := 40207, finalStart := 766299, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 151, sourceArmColumn := 40212, finalStart := 766504, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 152, sourceArmColumn := 40217, finalStart := 766709, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 153, sourceArmColumn := 40222, finalStart := 766914, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 154, sourceArmColumn := 40227, finalStart := 767119, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 155, sourceArmColumn := 40232, finalStart := 767324, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 156, sourceArmColumn := 40237, finalStart := 767529, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 157, sourceArmColumn := 40242, finalStart := 767734, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 158, sourceArmColumn := 40247, finalStart := 767939, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 159, sourceArmColumn := 40252, finalStart := 768144, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 160, sourceArmColumn := 40257, finalStart := 768349, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 161, sourceArmColumn := 40262, finalStart := 768554, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 162, sourceArmColumn := 39998, finalStart := 757730, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 163, sourceArmColumn := 40003, finalStart := 757935, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 164, sourceArmColumn := 40008, finalStart := 758140, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 165, sourceArmColumn := 40013, finalStart := 758345, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 166, sourceArmColumn := 40018, finalStart := 758550, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 167, sourceArmColumn := 40023, finalStart := 758755, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 168, sourceArmColumn := 40028, finalStart := 758960, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 169, sourceArmColumn := 40033, finalStart := 759165, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 170, sourceArmColumn := 40038, finalStart := 759370, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 171, sourceArmColumn := 40043, finalStart := 759575, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 172, sourceArmColumn := 40048, finalStart := 759780, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 173, sourceArmColumn := 40053, finalStart := 759985, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 174, sourceArmColumn := 40058, finalStart := 760190, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 175, sourceArmColumn := 40063, finalStart := 760395, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 176, sourceArmColumn := 40068, finalStart := 760600, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 177, sourceArmColumn := 40073, finalStart := 760805, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 178, sourceArmColumn := 40078, finalStart := 761010, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 179, sourceArmColumn := 40083, finalStart := 761215, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 180, sourceArmColumn := 40088, finalStart := 761420, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 181, sourceArmColumn := 40093, finalStart := 761625, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 182, sourceArmColumn := 40098, finalStart := 761830, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 183, sourceArmColumn := 40103, finalStart := 762035, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 184, sourceArmColumn := 40108, finalStart := 762240, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 185, sourceArmColumn := 40113, finalStart := 762445, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 186, sourceArmColumn := 40118, finalStart := 762650, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 187, sourceArmColumn := 40123, finalStart := 762855, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 188, sourceArmColumn := 40128, finalStart := 763060, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 189, sourceArmColumn := 40133, finalStart := 763265, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 190, sourceArmColumn := 40138, finalStart := 763470, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 191, sourceArmColumn := 40143, finalStart := 763675, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 192, sourceArmColumn := 40148, finalStart := 763880, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 193, sourceArmColumn := 40153, finalStart := 764085, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 194, sourceArmColumn := 40158, finalStart := 764290, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 195, sourceArmColumn := 40163, finalStart := 764495, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 196, sourceArmColumn := 40168, finalStart := 764700, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 197, sourceArmColumn := 40173, finalStart := 764905, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 198, sourceArmColumn := 40178, finalStart := 765110, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 199, sourceArmColumn := 40183, finalStart := 765315, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 200, sourceArmColumn := 40188, finalStart := 765520, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 201, sourceArmColumn := 40193, finalStart := 765725, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 202, sourceArmColumn := 40198, finalStart := 765930, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 203, sourceArmColumn := 40203, finalStart := 766135, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 204, sourceArmColumn := 40208, finalStart := 766340, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 205, sourceArmColumn := 40213, finalStart := 766545, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 206, sourceArmColumn := 40218, finalStart := 766750, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 207, sourceArmColumn := 40223, finalStart := 766955, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 208, sourceArmColumn := 40228, finalStart := 767160, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 209, sourceArmColumn := 40233, finalStart := 767365, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 210, sourceArmColumn := 40238, finalStart := 767570, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 211, sourceArmColumn := 40243, finalStart := 767775, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 212, sourceArmColumn := 40248, finalStart := 767980, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 213, sourceArmColumn := 40253, finalStart := 768185, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 214, sourceArmColumn := 40258, finalStart := 768390, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 215, sourceArmColumn := 40263, finalStart := 768595, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 216, sourceArmColumn := 39999, finalStart := 757771, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 217, sourceArmColumn := 40004, finalStart := 757976, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 218, sourceArmColumn := 40009, finalStart := 758181, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 219, sourceArmColumn := 40014, finalStart := 758386, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 220, sourceArmColumn := 40019, finalStart := 758591, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 221, sourceArmColumn := 40024, finalStart := 758796, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 222, sourceArmColumn := 40029, finalStart := 759001, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 223, sourceArmColumn := 40034, finalStart := 759206, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 224, sourceArmColumn := 40039, finalStart := 759411, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 225, sourceArmColumn := 40044, finalStart := 759616, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 226, sourceArmColumn := 40049, finalStart := 759821, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 227, sourceArmColumn := 40054, finalStart := 760026, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 228, sourceArmColumn := 40059, finalStart := 760231, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 229, sourceArmColumn := 40064, finalStart := 760436, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 230, sourceArmColumn := 40069, finalStart := 760641, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 231, sourceArmColumn := 40074, finalStart := 760846, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 232, sourceArmColumn := 40079, finalStart := 761051, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 233, sourceArmColumn := 40084, finalStart := 761256, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 234, sourceArmColumn := 40089, finalStart := 761461, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 235, sourceArmColumn := 40094, finalStart := 761666, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 236, sourceArmColumn := 40099, finalStart := 761871, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 237, sourceArmColumn := 40104, finalStart := 762076, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 238, sourceArmColumn := 40109, finalStart := 762281, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 239, sourceArmColumn := 40114, finalStart := 762486, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 240, sourceArmColumn := 40119, finalStart := 762691, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 241, sourceArmColumn := 40124, finalStart := 762896, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 242, sourceArmColumn := 40129, finalStart := 763101, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 243, sourceArmColumn := 40134, finalStart := 763306, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 244, sourceArmColumn := 40139, finalStart := 763511, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 245, sourceArmColumn := 40144, finalStart := 763716, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 246, sourceArmColumn := 40149, finalStart := 763921, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 247, sourceArmColumn := 40154, finalStart := 764126, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 248, sourceArmColumn := 40159, finalStart := 764331, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 249, sourceArmColumn := 40164, finalStart := 764536, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 250, sourceArmColumn := 40169, finalStart := 764741, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 251, sourceArmColumn := 40174, finalStart := 764946, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 252, sourceArmColumn := 40179, finalStart := 765151, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 253, sourceArmColumn := 40184, finalStart := 765356, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 254, sourceArmColumn := 40189, finalStart := 765561, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 255, sourceArmColumn := 40194, finalStart := 765766, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 256, sourceArmColumn := 40199, finalStart := 765971, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 257, sourceArmColumn := 40204, finalStart := 766176, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 258, sourceArmColumn := 40209, finalStart := 766381, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 259, sourceArmColumn := 40214, finalStart := 766586, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 260, sourceArmColumn := 40219, finalStart := 766791, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 261, sourceArmColumn := 40224, finalStart := 766996, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 262, sourceArmColumn := 40229, finalStart := 767201, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 263, sourceArmColumn := 40234, finalStart := 767406, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 264, sourceArmColumn := 40239, finalStart := 767611, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 265, sourceArmColumn := 40244, finalStart := 767816, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 266, sourceArmColumn := 40249, finalStart := 768021, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 267, sourceArmColumn := 40254, finalStart := 768226, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 268, sourceArmColumn := 40259, finalStart := 768431, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 269, sourceArmColumn := 40264, finalStart := 768636, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 0, sourceArmColumn := 42267, finalStart := 835097, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 1, sourceArmColumn := 42272, finalStart := 835302, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 2, sourceArmColumn := 42277, finalStart := 835507, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 3, sourceArmColumn := 42282, finalStart := 835712, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 4, sourceArmColumn := 42287, finalStart := 835917, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 5, sourceArmColumn := 42292, finalStart := 836122, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 6, sourceArmColumn := 42297, finalStart := 836327, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 7, sourceArmColumn := 42302, finalStart := 836532, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 8, sourceArmColumn := 42307, finalStart := 836737, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 9, sourceArmColumn := 42312, finalStart := 836942, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 10, sourceArmColumn := 42317, finalStart := 837147, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 11, sourceArmColumn := 42322, finalStart := 837352, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 12, sourceArmColumn := 42327, finalStart := 837557, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 13, sourceArmColumn := 42332, finalStart := 837762, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 14, sourceArmColumn := 42337, finalStart := 837967, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 15, sourceArmColumn := 42342, finalStart := 838172, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 16, sourceArmColumn := 42347, finalStart := 838377, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 17, sourceArmColumn := 42352, finalStart := 838582, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 18, sourceArmColumn := 42357, finalStart := 838787, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 19, sourceArmColumn := 42362, finalStart := 838992, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 20, sourceArmColumn := 42367, finalStart := 839197, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 21, sourceArmColumn := 42372, finalStart := 839402, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 22, sourceArmColumn := 42377, finalStart := 839607, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 23, sourceArmColumn := 42382, finalStart := 839812, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 24, sourceArmColumn := 42387, finalStart := 840017, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 25, sourceArmColumn := 42392, finalStart := 840222, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 26, sourceArmColumn := 42397, finalStart := 840427, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 27, sourceArmColumn := 42402, finalStart := 840632, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 28, sourceArmColumn := 42407, finalStart := 840837, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 29, sourceArmColumn := 42412, finalStart := 841042, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 30, sourceArmColumn := 42417, finalStart := 841247, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 31, sourceArmColumn := 42422, finalStart := 841452, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 32, sourceArmColumn := 42427, finalStart := 841657, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 33, sourceArmColumn := 42432, finalStart := 841862, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 34, sourceArmColumn := 42437, finalStart := 842067, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 35, sourceArmColumn := 42442, finalStart := 842272, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 36, sourceArmColumn := 42447, finalStart := 842477, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 37, sourceArmColumn := 42452, finalStart := 842682, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 38, sourceArmColumn := 42457, finalStart := 842887, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 39, sourceArmColumn := 42462, finalStart := 843092, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 40, sourceArmColumn := 42467, finalStart := 843297, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 41, sourceArmColumn := 42472, finalStart := 843502, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 42, sourceArmColumn := 42477, finalStart := 843707, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 43, sourceArmColumn := 42482, finalStart := 843912, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 44, sourceArmColumn := 42487, finalStart := 844117, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 45, sourceArmColumn := 42492, finalStart := 844322, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 46, sourceArmColumn := 42497, finalStart := 844527, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 47, sourceArmColumn := 42502, finalStart := 844732, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 48, sourceArmColumn := 42507, finalStart := 844937, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 49, sourceArmColumn := 42512, finalStart := 845142, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 50, sourceArmColumn := 42517, finalStart := 845347, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 51, sourceArmColumn := 42522, finalStart := 845552, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 52, sourceArmColumn := 42527, finalStart := 845757, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 53, sourceArmColumn := 42532, finalStart := 845962, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 54, sourceArmColumn := 42268, finalStart := 835138, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 55, sourceArmColumn := 42273, finalStart := 835343, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 56, sourceArmColumn := 42278, finalStart := 835548, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 57, sourceArmColumn := 42283, finalStart := 835753, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 58, sourceArmColumn := 42288, finalStart := 835958, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 59, sourceArmColumn := 42293, finalStart := 836163, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 60, sourceArmColumn := 42298, finalStart := 836368, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 61, sourceArmColumn := 42303, finalStart := 836573, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 62, sourceArmColumn := 42308, finalStart := 836778, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 63, sourceArmColumn := 42313, finalStart := 836983, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 64, sourceArmColumn := 42318, finalStart := 837188, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 65, sourceArmColumn := 42323, finalStart := 837393, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 66, sourceArmColumn := 42328, finalStart := 837598, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 67, sourceArmColumn := 42333, finalStart := 837803, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 68, sourceArmColumn := 42338, finalStart := 838008, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 69, sourceArmColumn := 42343, finalStart := 838213, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 70, sourceArmColumn := 42348, finalStart := 838418, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 71, sourceArmColumn := 42353, finalStart := 838623, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 72, sourceArmColumn := 42358, finalStart := 838828, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 73, sourceArmColumn := 42363, finalStart := 839033, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 74, sourceArmColumn := 42368, finalStart := 839238, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 75, sourceArmColumn := 42373, finalStart := 839443, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 76, sourceArmColumn := 42378, finalStart := 839648, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 77, sourceArmColumn := 42383, finalStart := 839853, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 78, sourceArmColumn := 42388, finalStart := 840058, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 79, sourceArmColumn := 42393, finalStart := 840263, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 80, sourceArmColumn := 42398, finalStart := 840468, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 81, sourceArmColumn := 42403, finalStart := 840673, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 82, sourceArmColumn := 42408, finalStart := 840878, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 83, sourceArmColumn := 42413, finalStart := 841083, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 84, sourceArmColumn := 42418, finalStart := 841288, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 85, sourceArmColumn := 42423, finalStart := 841493, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 86, sourceArmColumn := 42428, finalStart := 841698, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 87, sourceArmColumn := 42433, finalStart := 841903, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 88, sourceArmColumn := 42438, finalStart := 842108, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 89, sourceArmColumn := 42443, finalStart := 842313, width := 41, encoding := .balancedTernary }
]

def records : List SourceColumnRecord :=
  allocationRecords.map AllocationRecord.sourceRecord

end Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.Generated.Chunk9
