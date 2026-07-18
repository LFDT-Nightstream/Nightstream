import Nightstream.Implementation.R1CS.Correspondence.PiCcsOutputDigest.ActiveSourceLayout.Counts

/-!
Focused regressions for the typed active `Pi_CCS` output role tree.

| Stage path | Property under test | Failure caught |
|---|---|---|
| `nifs.pi_ccs.output_message_hashes.digest.preimage.outer_header` | exact eight-field outer header | domain/count drift |
| `nifs.pi_ccs.output_message_hashes.digest.preimage.source_headers` | exact nine-field source header | tag/matrix-count drift |
| `nifs.pi_ccs.output_message_hashes.digest.preimage.y_ring` | active matrix/lane/limb family is complete | truncated or reordered `y_ring` |
| `nifs.pi_ccs.output_message_hashes.digest.preimage.y_zcol` | active lane/limb family is complete | omitted or reordered `y_zcol` |
| `nifs.pi_ccs.output_message_hashes.digest.preimage` | role values equal the independent serializer | role/serializer schedule drift |
-/

open Nightstream.Implementation.R1CS.PiCcsOutputDigest
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc

#check ActiveSourceLayout.SourceRole
#check ActiveSourceLayout.InputOwner
#check ActiveSourceLayout.sourceRoles_length
#check ActiveSourceLayout.ownerFieldCounts_reconcile
#check ActiveSourceLayout.sourceRoleValues_eq_serialize
#check ActiveSourceLayout.decodedFields_eq_serialize

def activeSourceLayoutShape : SemanticShape where
  rowVariables := 0
  logicalWidth := 0
  freshCount := 1
  runningCount := 14
  matrixCount := 13

example :
    (ActiveSourceLayout.sourceRoles activeSourceLayoutShape).length = 23033 := by
  rw [ActiveSourceLayout.sourceRoles_length]
  decide

example :
    ActiveSourceLayout.ownerFieldCount activeSourceLayoutShape
        .verifierShape = 353 := by
  rw [ActiveSourceLayout.ownerFieldCount_verifierShape]
  decide

example :
    ActiveSourceLayout.ownerFieldCount activeSourceLayoutShape
        .yRingOutput = 21060 := by
  rw [ActiveSourceLayout.ownerFieldCount_yRingOutput]
  decide

example :
    ActiveSourceLayout.ownerFieldCount activeSourceLayoutShape
        .yZcolOutput = 1620 := by
  rw [ActiveSourceLayout.ownerFieldCount_yZcolOutput]
  decide
