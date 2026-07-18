import Nightstream.Implementation.R1CS.Correspondence.PiCcsOutputDigest.ActiveProfile

/-!
Focused regressions for the complete shape-indexed `Pi_CCS` output encoding.

| Stage path | Property under test | Failure caught |
|---|---|---|
| `nifs.pi_ccs.output_digest.domain` | shared tags have exact packed lengths | tag drift or inconsistent packing |
| `nifs.pi_ccs.output_digest.source.y_ring` | all matrix/lane/limb coordinates are serialized | silent three-row truncation |
| `nifs.pi_ccs.output_digest.source.y_zcol` | the complete active sidecar is serialized | omitted output authority surface |
| `nifs.pi_ccs.output_digest.injective` | equal field messages imply equal typed outputs | representation alias before compression |
| `nifs.pi_ccs.output_digest.profile.15x13` | active profile field count is explicit and typed | diagnostic 3-matrix count reused as active |
-/

open Nightstream.Implementation.R1CS.PiCcsOutputDigest
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier

#check Encoding.outputsDomainFields_length
#check Encoding.outputMessageDomainFields_length
#check Encoding.encodeKVector_injective
#check Encoding.encodeKVectorFamily_injective
#check ActiveSemantics.SourcePayload
#check ActiveSemantics.sourcePayloads_injective
#check ActiveSemantics.encodeSourcePayload_injective
#check ActiveSemantics.encodeSource_injective
#check ActiveSemantics.serialize
#check ActiveSemantics.serialize_injective
#check ActiveSemantics.serialize_length_15_sources_13_matrices
#check ActiveProfile.context_sourceCount_eq_15
#check ActiveProfile.serialize_length
#check ActiveProfile.selectiveShape
#check ActiveProfile.relationShape_eq
#check ActiveProfile.selectiveShape_not_legacyProfile
#check ActiveProfile.selective_serialize_length

def activeProfileShape : SemanticShape where
  rowVariables := 0
  logicalWidth := 0
  freshCount := 1
  runningCount := 14
  matrixCount := 13

example (message : OutputMessage activeProfileShape) :
    (ActiveSemantics.serialize message).length = 23033 := by
  exact ActiveSemantics.serialize_length_15_sources_13_matrices
    (by rfl) (by rfl) message
