import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactWire
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf0
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf1
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf2
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf3
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf4
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf5
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf6
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf7
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf8
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf9
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf10
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf11

/-!
GENERATED FILE - do not edit by hand.

Assembly of the chunk-aligned compact source artifact. All heavy
facts live in the bounded leaf modules; this module only
dispatches them and applies the structural composition theorems.
Exact validation is discharged by proof, never by evaluation.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifact

open Nightstream.Assurance.CompactSourceArtifact
open Nightstream.Assurance.ConstraintMinimization
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactWire

set_option maxHeartbeats 2000000
set_option maxRecDepth 65536

export Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactWire (families wire sourceArtifact reviewedPlan reviewedPlan_subset chunkRows_eq totalRows_eq chunkCount_eq)

theorem censuses :
    ∀ k, k < wire.chunkCount →
      (rowsChunk wire k).map (fun row => row.sourceIndex) =
        List.range' (wire.chunkStart k) (wire.chunkLength k) := by
  intro k bound
  rw [chunkCount_eq] at bound
  match k, bound with
  | 0, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf0.chunkLeaf0).1
  | 1, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf0.chunkLeaf1).1
  | 2, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf0.chunkLeaf2).1
  | 3, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf0.chunkLeaf3).1
  | 4, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf0.chunkLeaf4).1
  | 5, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf0.chunkLeaf5).1
  | 6, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf0.chunkLeaf6).1
  | 7, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf0.chunkLeaf7).1
  | 8, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf0.chunkLeaf8).1
  | 9, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf0.chunkLeaf9).1
  | 10, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf0.chunkLeaf10).1
  | 11, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf0.chunkLeaf11).1
  | 12, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf0.chunkLeaf12).1
  | 13, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf0.chunkLeaf13).1
  | 14, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf1.chunkLeaf14).1
  | 15, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf1.chunkLeaf15).1
  | 16, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf1.chunkLeaf16).1
  | 17, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf1.chunkLeaf17).1
  | 18, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf1.chunkLeaf18).1
  | 19, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf1.chunkLeaf19).1
  | 20, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf1.chunkLeaf20).1
  | 21, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf1.chunkLeaf21).1
  | 22, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf1.chunkLeaf22).1
  | 23, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf1.chunkLeaf23).1
  | 24, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf1.chunkLeaf24).1
  | 25, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf1.chunkLeaf25).1
  | 26, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf1.chunkLeaf26).1
  | 27, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf1.chunkLeaf27).1
  | 28, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf2.chunkLeaf28).1
  | 29, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf2.chunkLeaf29).1
  | 30, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf2.chunkLeaf30).1
  | 31, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf2.chunkLeaf31).1
  | 32, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf2.chunkLeaf32).1
  | 33, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf2.chunkLeaf33).1
  | 34, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf2.chunkLeaf34).1
  | 35, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf2.chunkLeaf35).1
  | 36, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf2.chunkLeaf36).1
  | 37, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf2.chunkLeaf37).1
  | 38, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf2.chunkLeaf38).1
  | 39, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf2.chunkLeaf39).1
  | 40, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf2.chunkLeaf40).1
  | 41, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf2.chunkLeaf41).1
  | 42, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf3.chunkLeaf42).1
  | 43, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf3.chunkLeaf43).1
  | 44, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf3.chunkLeaf44).1
  | 45, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf3.chunkLeaf45).1
  | 46, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf3.chunkLeaf46).1
  | 47, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf3.chunkLeaf47).1
  | 48, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf3.chunkLeaf48).1
  | 49, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf3.chunkLeaf49).1
  | 50, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf3.chunkLeaf50).1
  | 51, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf3.chunkLeaf51).1
  | 52, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf3.chunkLeaf52).1
  | 53, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf3.chunkLeaf53).1
  | 54, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf3.chunkLeaf54).1
  | 55, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf3.chunkLeaf55).1
  | 56, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf4.chunkLeaf56).1
  | 57, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf4.chunkLeaf57).1
  | 58, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf4.chunkLeaf58).1
  | 59, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf4.chunkLeaf59).1
  | 60, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf4.chunkLeaf60).1
  | 61, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf4.chunkLeaf61).1
  | 62, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf4.chunkLeaf62).1
  | 63, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf4.chunkLeaf63).1
  | 64, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf4.chunkLeaf64).1
  | 65, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf4.chunkLeaf65).1
  | 66, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf4.chunkLeaf66).1
  | 67, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf4.chunkLeaf67).1
  | 68, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf4.chunkLeaf68).1
  | 69, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf4.chunkLeaf69).1
  | 70, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf5.chunkLeaf70).1
  | 71, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf5.chunkLeaf71).1
  | 72, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf5.chunkLeaf72).1
  | 73, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf5.chunkLeaf73).1
  | 74, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf5.chunkLeaf74).1
  | 75, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf5.chunkLeaf75).1
  | 76, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf5.chunkLeaf76).1
  | 77, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf5.chunkLeaf77).1
  | 78, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf5.chunkLeaf78).1
  | 79, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf5.chunkLeaf79).1
  | 80, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf5.chunkLeaf80).1
  | 81, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf5.chunkLeaf81).1
  | 82, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf5.chunkLeaf82).1
  | 83, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf5.chunkLeaf83).1
  | 84, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf6.chunkLeaf84).1
  | 85, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf6.chunkLeaf85).1
  | 86, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf6.chunkLeaf86).1
  | 87, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf6.chunkLeaf87).1
  | 88, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf6.chunkLeaf88).1
  | 89, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf6.chunkLeaf89).1
  | 90, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf6.chunkLeaf90).1
  | 91, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf6.chunkLeaf91).1
  | 92, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf6.chunkLeaf92).1
  | 93, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf6.chunkLeaf93).1
  | 94, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf6.chunkLeaf94).1
  | 95, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf6.chunkLeaf95).1
  | 96, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf6.chunkLeaf96).1
  | 97, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf6.chunkLeaf97).1
  | 98, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf7.chunkLeaf98).1
  | 99, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf7.chunkLeaf99).1
  | 100, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf7.chunkLeaf100).1
  | 101, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf7.chunkLeaf101).1
  | 102, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf7.chunkLeaf102).1
  | 103, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf7.chunkLeaf103).1
  | 104, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf7.chunkLeaf104).1
  | 105, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf7.chunkLeaf105).1
  | 106, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf7.chunkLeaf106).1
  | 107, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf7.chunkLeaf107).1
  | 108, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf7.chunkLeaf108).1
  | 109, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf7.chunkLeaf109).1
  | 110, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf7.chunkLeaf110).1
  | 111, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf7.chunkLeaf111).1
  | 112, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf8.chunkLeaf112).1
  | 113, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf8.chunkLeaf113).1
  | 114, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf8.chunkLeaf114).1
  | 115, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf8.chunkLeaf115).1
  | 116, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf8.chunkLeaf116).1
  | 117, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf8.chunkLeaf117).1
  | 118, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf8.chunkLeaf118).1
  | 119, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf8.chunkLeaf119).1
  | 120, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf8.chunkLeaf120).1
  | 121, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf8.chunkLeaf121).1
  | 122, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf8.chunkLeaf122).1
  | 123, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf8.chunkLeaf123).1
  | 124, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf8.chunkLeaf124).1
  | 125, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf8.chunkLeaf125).1
  | 126, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf9.chunkLeaf126).1
  | 127, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf9.chunkLeaf127).1
  | 128, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf9.chunkLeaf128).1
  | 129, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf9.chunkLeaf129).1
  | 130, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf9.chunkLeaf130).1
  | 131, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf9.chunkLeaf131).1
  | 132, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf9.chunkLeaf132).1
  | 133, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf9.chunkLeaf133).1
  | 134, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf9.chunkLeaf134).1
  | 135, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf9.chunkLeaf135).1
  | 136, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf9.chunkLeaf136).1
  | 137, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf9.chunkLeaf137).1
  | 138, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf9.chunkLeaf138).1
  | 139, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf9.chunkLeaf139).1
  | 140, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf10.chunkLeaf140).1
  | 141, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf10.chunkLeaf141).1
  | 142, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf10.chunkLeaf142).1
  | 143, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf10.chunkLeaf143).1
  | 144, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf10.chunkLeaf144).1
  | 145, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf10.chunkLeaf145).1
  | 146, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf10.chunkLeaf146).1
  | 147, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf10.chunkLeaf147).1
  | 148, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf10.chunkLeaf148).1
  | 149, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf10.chunkLeaf149).1
  | 150, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf10.chunkLeaf150).1
  | 151, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf10.chunkLeaf151).1
  | 152, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf10.chunkLeaf152).1
  | 153, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf10.chunkLeaf153).1
  | 154, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf11.chunkLeaf154).1
  | 155, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf11.chunkLeaf155).1
  | 156, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf11.chunkLeaf156).1
  | n + 157, bound => exact absurd bound (by omega)

theorem rowsWf :
    ∀ k, k < wire.chunkCount →
      (rowsChunk wire k).all (rowWellFormedAt 39949 38626) = true := by
  intro k bound
  rw [chunkCount_eq] at bound
  match k, bound with
  | 0, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf0.chunkLeaf0).2.1
  | 1, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf0.chunkLeaf1).2.1
  | 2, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf0.chunkLeaf2).2.1
  | 3, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf0.chunkLeaf3).2.1
  | 4, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf0.chunkLeaf4).2.1
  | 5, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf0.chunkLeaf5).2.1
  | 6, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf0.chunkLeaf6).2.1
  | 7, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf0.chunkLeaf7).2.1
  | 8, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf0.chunkLeaf8).2.1
  | 9, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf0.chunkLeaf9).2.1
  | 10, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf0.chunkLeaf10).2.1
  | 11, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf0.chunkLeaf11).2.1
  | 12, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf0.chunkLeaf12).2.1
  | 13, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf0.chunkLeaf13).2.1
  | 14, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf1.chunkLeaf14).2.1
  | 15, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf1.chunkLeaf15).2.1
  | 16, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf1.chunkLeaf16).2.1
  | 17, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf1.chunkLeaf17).2.1
  | 18, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf1.chunkLeaf18).2.1
  | 19, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf1.chunkLeaf19).2.1
  | 20, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf1.chunkLeaf20).2.1
  | 21, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf1.chunkLeaf21).2.1
  | 22, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf1.chunkLeaf22).2.1
  | 23, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf1.chunkLeaf23).2.1
  | 24, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf1.chunkLeaf24).2.1
  | 25, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf1.chunkLeaf25).2.1
  | 26, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf1.chunkLeaf26).2.1
  | 27, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf1.chunkLeaf27).2.1
  | 28, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf2.chunkLeaf28).2.1
  | 29, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf2.chunkLeaf29).2.1
  | 30, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf2.chunkLeaf30).2.1
  | 31, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf2.chunkLeaf31).2.1
  | 32, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf2.chunkLeaf32).2.1
  | 33, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf2.chunkLeaf33).2.1
  | 34, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf2.chunkLeaf34).2.1
  | 35, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf2.chunkLeaf35).2.1
  | 36, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf2.chunkLeaf36).2.1
  | 37, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf2.chunkLeaf37).2.1
  | 38, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf2.chunkLeaf38).2.1
  | 39, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf2.chunkLeaf39).2.1
  | 40, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf2.chunkLeaf40).2.1
  | 41, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf2.chunkLeaf41).2.1
  | 42, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf3.chunkLeaf42).2.1
  | 43, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf3.chunkLeaf43).2.1
  | 44, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf3.chunkLeaf44).2.1
  | 45, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf3.chunkLeaf45).2.1
  | 46, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf3.chunkLeaf46).2.1
  | 47, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf3.chunkLeaf47).2.1
  | 48, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf3.chunkLeaf48).2.1
  | 49, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf3.chunkLeaf49).2.1
  | 50, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf3.chunkLeaf50).2.1
  | 51, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf3.chunkLeaf51).2.1
  | 52, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf3.chunkLeaf52).2.1
  | 53, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf3.chunkLeaf53).2.1
  | 54, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf3.chunkLeaf54).2.1
  | 55, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf3.chunkLeaf55).2.1
  | 56, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf4.chunkLeaf56).2.1
  | 57, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf4.chunkLeaf57).2.1
  | 58, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf4.chunkLeaf58).2.1
  | 59, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf4.chunkLeaf59).2.1
  | 60, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf4.chunkLeaf60).2.1
  | 61, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf4.chunkLeaf61).2.1
  | 62, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf4.chunkLeaf62).2.1
  | 63, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf4.chunkLeaf63).2.1
  | 64, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf4.chunkLeaf64).2.1
  | 65, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf4.chunkLeaf65).2.1
  | 66, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf4.chunkLeaf66).2.1
  | 67, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf4.chunkLeaf67).2.1
  | 68, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf4.chunkLeaf68).2.1
  | 69, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf4.chunkLeaf69).2.1
  | 70, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf5.chunkLeaf70).2.1
  | 71, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf5.chunkLeaf71).2.1
  | 72, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf5.chunkLeaf72).2.1
  | 73, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf5.chunkLeaf73).2.1
  | 74, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf5.chunkLeaf74).2.1
  | 75, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf5.chunkLeaf75).2.1
  | 76, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf5.chunkLeaf76).2.1
  | 77, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf5.chunkLeaf77).2.1
  | 78, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf5.chunkLeaf78).2.1
  | 79, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf5.chunkLeaf79).2.1
  | 80, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf5.chunkLeaf80).2.1
  | 81, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf5.chunkLeaf81).2.1
  | 82, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf5.chunkLeaf82).2.1
  | 83, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf5.chunkLeaf83).2.1
  | 84, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf6.chunkLeaf84).2.1
  | 85, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf6.chunkLeaf85).2.1
  | 86, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf6.chunkLeaf86).2.1
  | 87, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf6.chunkLeaf87).2.1
  | 88, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf6.chunkLeaf88).2.1
  | 89, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf6.chunkLeaf89).2.1
  | 90, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf6.chunkLeaf90).2.1
  | 91, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf6.chunkLeaf91).2.1
  | 92, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf6.chunkLeaf92).2.1
  | 93, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf6.chunkLeaf93).2.1
  | 94, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf6.chunkLeaf94).2.1
  | 95, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf6.chunkLeaf95).2.1
  | 96, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf6.chunkLeaf96).2.1
  | 97, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf6.chunkLeaf97).2.1
  | 98, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf7.chunkLeaf98).2.1
  | 99, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf7.chunkLeaf99).2.1
  | 100, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf7.chunkLeaf100).2.1
  | 101, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf7.chunkLeaf101).2.1
  | 102, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf7.chunkLeaf102).2.1
  | 103, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf7.chunkLeaf103).2.1
  | 104, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf7.chunkLeaf104).2.1
  | 105, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf7.chunkLeaf105).2.1
  | 106, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf7.chunkLeaf106).2.1
  | 107, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf7.chunkLeaf107).2.1
  | 108, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf7.chunkLeaf108).2.1
  | 109, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf7.chunkLeaf109).2.1
  | 110, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf7.chunkLeaf110).2.1
  | 111, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf7.chunkLeaf111).2.1
  | 112, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf8.chunkLeaf112).2.1
  | 113, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf8.chunkLeaf113).2.1
  | 114, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf8.chunkLeaf114).2.1
  | 115, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf8.chunkLeaf115).2.1
  | 116, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf8.chunkLeaf116).2.1
  | 117, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf8.chunkLeaf117).2.1
  | 118, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf8.chunkLeaf118).2.1
  | 119, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf8.chunkLeaf119).2.1
  | 120, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf8.chunkLeaf120).2.1
  | 121, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf8.chunkLeaf121).2.1
  | 122, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf8.chunkLeaf122).2.1
  | 123, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf8.chunkLeaf123).2.1
  | 124, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf8.chunkLeaf124).2.1
  | 125, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf8.chunkLeaf125).2.1
  | 126, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf9.chunkLeaf126).2.1
  | 127, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf9.chunkLeaf127).2.1
  | 128, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf9.chunkLeaf128).2.1
  | 129, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf9.chunkLeaf129).2.1
  | 130, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf9.chunkLeaf130).2.1
  | 131, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf9.chunkLeaf131).2.1
  | 132, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf9.chunkLeaf132).2.1
  | 133, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf9.chunkLeaf133).2.1
  | 134, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf9.chunkLeaf134).2.1
  | 135, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf9.chunkLeaf135).2.1
  | 136, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf9.chunkLeaf136).2.1
  | 137, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf9.chunkLeaf137).2.1
  | 138, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf9.chunkLeaf138).2.1
  | 139, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf9.chunkLeaf139).2.1
  | 140, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf10.chunkLeaf140).2.1
  | 141, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf10.chunkLeaf141).2.1
  | 142, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf10.chunkLeaf142).2.1
  | 143, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf10.chunkLeaf143).2.1
  | 144, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf10.chunkLeaf144).2.1
  | 145, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf10.chunkLeaf145).2.1
  | 146, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf10.chunkLeaf146).2.1
  | 147, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf10.chunkLeaf147).2.1
  | 148, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf10.chunkLeaf148).2.1
  | 149, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf10.chunkLeaf149).2.1
  | 150, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf10.chunkLeaf150).2.1
  | 151, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf10.chunkLeaf151).2.1
  | 152, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf10.chunkLeaf152).2.1
  | 153, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf10.chunkLeaf153).2.1
  | 154, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf11.chunkLeaf154).2.1
  | 155, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf11.chunkLeaf155).2.1
  | 156, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf11.chunkLeaf156).2.1
  | n + 157, bound => exact absurd bound (by omega)

theorem familiesCovered :
    ∀ k, k < wire.chunkCount →
      (rowsChunk wire k).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true := by
  intro k bound
  rw [chunkCount_eq] at bound
  match k, bound with
  | 0, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf0.chunkLeaf0).2.2.1
  | 1, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf0.chunkLeaf1).2.2.1
  | 2, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf0.chunkLeaf2).2.2.1
  | 3, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf0.chunkLeaf3).2.2.1
  | 4, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf0.chunkLeaf4).2.2.1
  | 5, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf0.chunkLeaf5).2.2.1
  | 6, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf0.chunkLeaf6).2.2.1
  | 7, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf0.chunkLeaf7).2.2.1
  | 8, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf0.chunkLeaf8).2.2.1
  | 9, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf0.chunkLeaf9).2.2.1
  | 10, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf0.chunkLeaf10).2.2.1
  | 11, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf0.chunkLeaf11).2.2.1
  | 12, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf0.chunkLeaf12).2.2.1
  | 13, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf0.chunkLeaf13).2.2.1
  | 14, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf1.chunkLeaf14).2.2.1
  | 15, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf1.chunkLeaf15).2.2.1
  | 16, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf1.chunkLeaf16).2.2.1
  | 17, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf1.chunkLeaf17).2.2.1
  | 18, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf1.chunkLeaf18).2.2.1
  | 19, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf1.chunkLeaf19).2.2.1
  | 20, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf1.chunkLeaf20).2.2.1
  | 21, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf1.chunkLeaf21).2.2.1
  | 22, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf1.chunkLeaf22).2.2.1
  | 23, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf1.chunkLeaf23).2.2.1
  | 24, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf1.chunkLeaf24).2.2.1
  | 25, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf1.chunkLeaf25).2.2.1
  | 26, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf1.chunkLeaf26).2.2.1
  | 27, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf1.chunkLeaf27).2.2.1
  | 28, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf2.chunkLeaf28).2.2.1
  | 29, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf2.chunkLeaf29).2.2.1
  | 30, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf2.chunkLeaf30).2.2.1
  | 31, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf2.chunkLeaf31).2.2.1
  | 32, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf2.chunkLeaf32).2.2.1
  | 33, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf2.chunkLeaf33).2.2.1
  | 34, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf2.chunkLeaf34).2.2.1
  | 35, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf2.chunkLeaf35).2.2.1
  | 36, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf2.chunkLeaf36).2.2.1
  | 37, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf2.chunkLeaf37).2.2.1
  | 38, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf2.chunkLeaf38).2.2.1
  | 39, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf2.chunkLeaf39).2.2.1
  | 40, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf2.chunkLeaf40).2.2.1
  | 41, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf2.chunkLeaf41).2.2.1
  | 42, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf3.chunkLeaf42).2.2.1
  | 43, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf3.chunkLeaf43).2.2.1
  | 44, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf3.chunkLeaf44).2.2.1
  | 45, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf3.chunkLeaf45).2.2.1
  | 46, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf3.chunkLeaf46).2.2.1
  | 47, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf3.chunkLeaf47).2.2.1
  | 48, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf3.chunkLeaf48).2.2.1
  | 49, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf3.chunkLeaf49).2.2.1
  | 50, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf3.chunkLeaf50).2.2.1
  | 51, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf3.chunkLeaf51).2.2.1
  | 52, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf3.chunkLeaf52).2.2.1
  | 53, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf3.chunkLeaf53).2.2.1
  | 54, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf3.chunkLeaf54).2.2.1
  | 55, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf3.chunkLeaf55).2.2.1
  | 56, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf4.chunkLeaf56).2.2.1
  | 57, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf4.chunkLeaf57).2.2.1
  | 58, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf4.chunkLeaf58).2.2.1
  | 59, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf4.chunkLeaf59).2.2.1
  | 60, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf4.chunkLeaf60).2.2.1
  | 61, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf4.chunkLeaf61).2.2.1
  | 62, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf4.chunkLeaf62).2.2.1
  | 63, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf4.chunkLeaf63).2.2.1
  | 64, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf4.chunkLeaf64).2.2.1
  | 65, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf4.chunkLeaf65).2.2.1
  | 66, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf4.chunkLeaf66).2.2.1
  | 67, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf4.chunkLeaf67).2.2.1
  | 68, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf4.chunkLeaf68).2.2.1
  | 69, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf4.chunkLeaf69).2.2.1
  | 70, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf5.chunkLeaf70).2.2.1
  | 71, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf5.chunkLeaf71).2.2.1
  | 72, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf5.chunkLeaf72).2.2.1
  | 73, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf5.chunkLeaf73).2.2.1
  | 74, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf5.chunkLeaf74).2.2.1
  | 75, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf5.chunkLeaf75).2.2.1
  | 76, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf5.chunkLeaf76).2.2.1
  | 77, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf5.chunkLeaf77).2.2.1
  | 78, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf5.chunkLeaf78).2.2.1
  | 79, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf5.chunkLeaf79).2.2.1
  | 80, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf5.chunkLeaf80).2.2.1
  | 81, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf5.chunkLeaf81).2.2.1
  | 82, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf5.chunkLeaf82).2.2.1
  | 83, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf5.chunkLeaf83).2.2.1
  | 84, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf6.chunkLeaf84).2.2.1
  | 85, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf6.chunkLeaf85).2.2.1
  | 86, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf6.chunkLeaf86).2.2.1
  | 87, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf6.chunkLeaf87).2.2.1
  | 88, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf6.chunkLeaf88).2.2.1
  | 89, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf6.chunkLeaf89).2.2.1
  | 90, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf6.chunkLeaf90).2.2.1
  | 91, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf6.chunkLeaf91).2.2.1
  | 92, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf6.chunkLeaf92).2.2.1
  | 93, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf6.chunkLeaf93).2.2.1
  | 94, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf6.chunkLeaf94).2.2.1
  | 95, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf6.chunkLeaf95).2.2.1
  | 96, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf6.chunkLeaf96).2.2.1
  | 97, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf6.chunkLeaf97).2.2.1
  | 98, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf7.chunkLeaf98).2.2.1
  | 99, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf7.chunkLeaf99).2.2.1
  | 100, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf7.chunkLeaf100).2.2.1
  | 101, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf7.chunkLeaf101).2.2.1
  | 102, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf7.chunkLeaf102).2.2.1
  | 103, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf7.chunkLeaf103).2.2.1
  | 104, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf7.chunkLeaf104).2.2.1
  | 105, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf7.chunkLeaf105).2.2.1
  | 106, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf7.chunkLeaf106).2.2.1
  | 107, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf7.chunkLeaf107).2.2.1
  | 108, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf7.chunkLeaf108).2.2.1
  | 109, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf7.chunkLeaf109).2.2.1
  | 110, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf7.chunkLeaf110).2.2.1
  | 111, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf7.chunkLeaf111).2.2.1
  | 112, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf8.chunkLeaf112).2.2.1
  | 113, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf8.chunkLeaf113).2.2.1
  | 114, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf8.chunkLeaf114).2.2.1
  | 115, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf8.chunkLeaf115).2.2.1
  | 116, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf8.chunkLeaf116).2.2.1
  | 117, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf8.chunkLeaf117).2.2.1
  | 118, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf8.chunkLeaf118).2.2.1
  | 119, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf8.chunkLeaf119).2.2.1
  | 120, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf8.chunkLeaf120).2.2.1
  | 121, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf8.chunkLeaf121).2.2.1
  | 122, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf8.chunkLeaf122).2.2.1
  | 123, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf8.chunkLeaf123).2.2.1
  | 124, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf8.chunkLeaf124).2.2.1
  | 125, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf8.chunkLeaf125).2.2.1
  | 126, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf9.chunkLeaf126).2.2.1
  | 127, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf9.chunkLeaf127).2.2.1
  | 128, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf9.chunkLeaf128).2.2.1
  | 129, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf9.chunkLeaf129).2.2.1
  | 130, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf9.chunkLeaf130).2.2.1
  | 131, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf9.chunkLeaf131).2.2.1
  | 132, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf9.chunkLeaf132).2.2.1
  | 133, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf9.chunkLeaf133).2.2.1
  | 134, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf9.chunkLeaf134).2.2.1
  | 135, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf9.chunkLeaf135).2.2.1
  | 136, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf9.chunkLeaf136).2.2.1
  | 137, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf9.chunkLeaf137).2.2.1
  | 138, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf9.chunkLeaf138).2.2.1
  | 139, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf9.chunkLeaf139).2.2.1
  | 140, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf10.chunkLeaf140).2.2.1
  | 141, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf10.chunkLeaf141).2.2.1
  | 142, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf10.chunkLeaf142).2.2.1
  | 143, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf10.chunkLeaf143).2.2.1
  | 144, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf10.chunkLeaf144).2.2.1
  | 145, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf10.chunkLeaf145).2.2.1
  | 146, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf10.chunkLeaf146).2.2.1
  | 147, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf10.chunkLeaf147).2.2.1
  | 148, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf10.chunkLeaf148).2.2.1
  | 149, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf10.chunkLeaf149).2.2.1
  | 150, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf10.chunkLeaf150).2.2.1
  | 151, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf10.chunkLeaf151).2.2.1
  | 152, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf10.chunkLeaf152).2.2.1
  | 153, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf10.chunkLeaf153).2.2.1
  | 154, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf11.chunkLeaf154).2.2.1
  | 155, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf11.chunkLeaf155).2.2.1
  | 156, _ => exact (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf11.chunkLeaf156).2.2.1
  | n + 157, bound => exact absurd bound (by omega)

theorem chunkArithmeticFull :
    ∀ k, k + 1 < wire.chunkCount → wire.chunkLength k = wire.chunkRows := by
  intro k bound
  rw [chunkCount_eq] at bound
  simp only [Wire.chunkLength, Wire.chunkStart, chunkRows_eq, totalRows_eq]
  omega

theorem chunkArithmeticLast :
    wire.chunkCount ≠ 0 →
      (wire.chunkCount - 1) * wire.chunkRows +
        wire.chunkLength (wire.chunkCount - 1) = wire.totalRows := by
  intro _
  simp only [Wire.chunkLength, Wire.chunkStart, chunkCount_eq, chunkRows_eq, totalRows_eq]
  omega

theorem chunkArithmeticLead :
    wire.chunkCount ≠ 0 →
      (wire.chunkCount - 1) * wire.chunkRows ≤ wire.totalRows := by
  intro _
  simp only [chunkCount_eq, chunkRows_eq, totalRows_eq]
  omega

theorem chunkArithmeticEmpty :
    wire.chunkCount = 0 → wire.totalRows = 0 := by
  intro h
  rw [chunkCount_eq] at h
  exact absurd h (by decide)

theorem familyPresence :
    sourceArtifact.completeFamilies.all
      (fun family =>
        sourceArtifact.rows.any
          (fun row => decide (row.family = family))) = true := by
  rw [List.all_eq_true]
  intro family membership
  have present : ∃ chunk, chunk < wire.chunkCount ∧
      (rowsChunk wire chunk).any
        (fun row => decide (row.family = family)) = true := by
    fin_cases membership
    · exact ⟨0, by rw [chunkCount_eq]; decide, Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf0.presence0⟩
    · exact ⟨57, by rw [chunkCount_eq]; decide, Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf4.presence1⟩
    · exact ⟨57, by rw [chunkCount_eq]; decide, Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf4.presence2⟩
    · exact ⟨59, by rw [chunkCount_eq]; decide, Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf4.presence3⟩
    · exact ⟨17, by rw [chunkCount_eq]; decide, Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf1.presence4⟩
    · exact ⟨55, by rw [chunkCount_eq]; decide, Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf3.presence5⟩
  rcases present with ⟨chunk, chunkBound, chunkAny⟩
  rw [List.any_eq_true] at chunkAny ⊢
  rcases chunkAny with ⟨row, rowMember, rowFamily⟩
  refine ⟨row, ?_, rowFamily⟩
  show row ∈ artifactRows wire
  unfold artifactRows
  exact List.mem_flatMap.mpr ⟨chunk, List.mem_range.mpr chunkBound, rowMember⟩

theorem scalarFacts :
    sourceArtifact.schema = Artifact.supportedSchema ∧
      sourceArtifact.profile ≠ "" ∧
      sourceArtifact.scope ∈ Artifact.scopes ∧
      sourceArtifact.diagnosticDigest ≠ "" ∧
      sourceArtifact.fieldModulus = Artifact.goldilocksModulusDecimal ∧
      0 < sourceArtifact.totalRows ∧
      0 < sourceArtifact.columnCount ∧
      0 < sourceArtifact.publicInputCount ∧
      sourceArtifact.publicInputCount ≤ sourceArtifact.columnCount ∧
      sourceArtifact.constantOneColumn < sourceArtifact.publicInputCount ∧
      sourceArtifact.completeFamilies.Nodup ∧
      sourceArtifact.completeFamilies.all
        (fun family => decide (family ≠ "")) = true := by
  native_decide

theorem sourceArtifact_indexCensus :
    (artifactRows wire).map (fun row => row.sourceIndex) =
      List.range wire.totalRows :=
  covers_indexes_of_chunks wire censuses chunkArithmeticFull
    chunkArithmeticLast chunkArithmeticLead chunkArithmeticEmpty

theorem sourceArtifact_coversFullRelation :
    sourceArtifact.CoversFullRelation :=
  coversFullRelation_of_chunks wire censuses chunkArithmeticFull
    chunkArithmeticLast chunkArithmeticLead chunkArithmeticEmpty familiesCovered

theorem sourceArtifact_wellFormed : sourceArtifact.WellFormed :=
  wellFormed_of_chunks wire scalarFacts sourceArtifact_indexCensus
    rowsWf familyPresence

theorem sourceArtifact_exactValidation :
    Artifact.ExactValidation sourceArtifact sourceArtifact = true :=
  exactValidation_self sourceArtifact_wellFormed

end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifact
