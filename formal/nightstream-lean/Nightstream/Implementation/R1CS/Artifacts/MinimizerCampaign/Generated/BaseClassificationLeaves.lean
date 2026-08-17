import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactWire
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment
import Nightstream.Assurance.ChunkLeaves
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf0
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf1
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf2
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf3
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf4
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf5
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf6
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf7
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf8
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf9
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf10
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf11

/-!
GENERATED FILE - do not edit by hand.

Dispatchers over the shared classification leaves.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeaves

open Nightstream.Assurance.CompactSourceArtifact
open Nightstream.Assurance.ConstraintMinimization
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactWire

def background : Nat → Field := backgroundFn Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values

def overridePairs : List (Nat × String) :=
  [(3811, "fprime.base.step.initial")]

theorem holdsAll :
    ∀ k, k < wire.chunkCount →
      (rowsChunk wire k).all
        (fun row => decide (Algebraic.Holds background row.row)) = true := by
  intro k bound
  rw [chunkCount_eq] at bound
  match k, bound with
  | 0, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf0.classLeaf0).1
  | 1, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf0.classLeaf1).1
  | 2, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf0.classLeaf2).1
  | 3, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf0.classLeaf3).1
  | 4, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf0.classLeaf4).1
  | 5, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf0.classLeaf5).1
  | 6, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf0.classLeaf6).1
  | 7, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf0.classLeaf7).1
  | 8, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf0.classLeaf8).1
  | 9, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf0.classLeaf9).1
  | 10, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf0.classLeaf10).1
  | 11, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf0.classLeaf11).1
  | 12, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf0.classLeaf12).1
  | 13, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf0.classLeaf13).1
  | 14, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf1.classLeaf14).1
  | 15, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf1.classLeaf15).1
  | 16, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf1.classLeaf16).1
  | 17, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf1.classLeaf17).1
  | 18, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf1.classLeaf18).1
  | 19, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf1.classLeaf19).1
  | 20, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf1.classLeaf20).1
  | 21, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf1.classLeaf21).1
  | 22, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf1.classLeaf22).1
  | 23, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf1.classLeaf23).1
  | 24, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf1.classLeaf24).1
  | 25, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf1.classLeaf25).1
  | 26, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf1.classLeaf26).1
  | 27, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf1.classLeaf27).1
  | 28, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf2.classLeaf28).1
  | 29, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf2.classLeaf29).1
  | 30, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf2.classLeaf30).1
  | 31, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf2.classLeaf31).1
  | 32, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf2.classLeaf32).1
  | 33, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf2.classLeaf33).1
  | 34, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf2.classLeaf34).1
  | 35, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf2.classLeaf35).1
  | 36, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf2.classLeaf36).1
  | 37, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf2.classLeaf37).1
  | 38, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf2.classLeaf38).1
  | 39, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf2.classLeaf39).1
  | 40, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf2.classLeaf40).1
  | 41, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf2.classLeaf41).1
  | 42, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf3.classLeaf42).1
  | 43, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf3.classLeaf43).1
  | 44, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf3.classLeaf44).1
  | 45, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf3.classLeaf45).1
  | 46, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf3.classLeaf46).1
  | 47, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf3.classLeaf47).1
  | 48, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf3.classLeaf48).1
  | 49, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf3.classLeaf49).1
  | 50, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf3.classLeaf50).1
  | 51, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf3.classLeaf51).1
  | 52, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf3.classLeaf52).1
  | 53, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf3.classLeaf53).1
  | 54, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf3.classLeaf54).1
  | 55, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf3.classLeaf55).1
  | 56, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf4.classLeaf56).1
  | 57, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf4.classLeaf57).1
  | 58, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf4.classLeaf58).1
  | 59, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf4.classLeaf59).1
  | 60, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf4.classLeaf60).1
  | 61, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf4.classLeaf61).1
  | 62, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf4.classLeaf62).1
  | 63, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf4.classLeaf63).1
  | 64, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf4.classLeaf64).1
  | 65, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf4.classLeaf65).1
  | 66, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf4.classLeaf66).1
  | 67, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf4.classLeaf67).1
  | 68, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf4.classLeaf68).1
  | 69, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf4.classLeaf69).1
  | 70, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf5.classLeaf70).1
  | 71, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf5.classLeaf71).1
  | 72, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf5.classLeaf72).1
  | 73, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf5.classLeaf73).1
  | 74, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf5.classLeaf74).1
  | 75, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf5.classLeaf75).1
  | 76, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf5.classLeaf76).1
  | 77, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf5.classLeaf77).1
  | 78, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf5.classLeaf78).1
  | 79, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf5.classLeaf79).1
  | 80, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf5.classLeaf80).1
  | 81, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf5.classLeaf81).1
  | 82, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf5.classLeaf82).1
  | 83, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf5.classLeaf83).1
  | 84, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf6.classLeaf84).1
  | 85, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf6.classLeaf85).1
  | 86, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf6.classLeaf86).1
  | 87, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf6.classLeaf87).1
  | 88, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf6.classLeaf88).1
  | 89, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf6.classLeaf89).1
  | 90, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf6.classLeaf90).1
  | 91, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf6.classLeaf91).1
  | 92, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf6.classLeaf92).1
  | 93, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf6.classLeaf93).1
  | 94, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf6.classLeaf94).1
  | 95, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf6.classLeaf95).1
  | 96, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf6.classLeaf96).1
  | 97, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf6.classLeaf97).1
  | 98, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf7.classLeaf98).1
  | 99, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf7.classLeaf99).1
  | 100, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf7.classLeaf100).1
  | 101, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf7.classLeaf101).1
  | 102, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf7.classLeaf102).1
  | 103, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf7.classLeaf103).1
  | 104, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf7.classLeaf104).1
  | 105, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf7.classLeaf105).1
  | 106, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf7.classLeaf106).1
  | 107, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf7.classLeaf107).1
  | 108, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf7.classLeaf108).1
  | 109, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf7.classLeaf109).1
  | 110, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf7.classLeaf110).1
  | 111, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf7.classLeaf111).1
  | 112, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf8.classLeaf112).1
  | 113, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf8.classLeaf113).1
  | 114, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf8.classLeaf114).1
  | 115, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf8.classLeaf115).1
  | 116, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf8.classLeaf116).1
  | 117, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf8.classLeaf117).1
  | 118, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf8.classLeaf118).1
  | 119, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf8.classLeaf119).1
  | 120, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf8.classLeaf120).1
  | 121, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf8.classLeaf121).1
  | 122, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf8.classLeaf122).1
  | 123, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf8.classLeaf123).1
  | 124, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf8.classLeaf124).1
  | 125, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf8.classLeaf125).1
  | 126, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf9.classLeaf126).1
  | 127, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf9.classLeaf127).1
  | 128, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf9.classLeaf128).1
  | 129, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf9.classLeaf129).1
  | 130, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf9.classLeaf130).1
  | 131, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf9.classLeaf131).1
  | 132, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf9.classLeaf132).1
  | 133, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf9.classLeaf133).1
  | 134, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf9.classLeaf134).1
  | 135, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf9.classLeaf135).1
  | 136, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf9.classLeaf136).1
  | 137, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf9.classLeaf137).1
  | 138, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf9.classLeaf138).1
  | 139, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf9.classLeaf139).1
  | 140, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf10.classLeaf140).1
  | 141, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf10.classLeaf141).1
  | 142, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf10.classLeaf142).1
  | 143, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf10.classLeaf143).1
  | 144, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf10.classLeaf144).1
  | 145, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf10.classLeaf145).1
  | 146, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf10.classLeaf146).1
  | 147, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf10.classLeaf147).1
  | 148, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf10.classLeaf148).1
  | 149, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf10.classLeaf149).1
  | 150, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf10.classLeaf150).1
  | 151, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf10.classLeaf151).1
  | 152, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf10.classLeaf152).1
  | 153, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf10.classLeaf153).1
  | 154, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf11.classLeaf154).1
  | 155, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf11.classLeaf155).1
  | 156, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf11.classLeaf156).1
  | n + 157, bound => exact absurd bound (by omega)

theorem guardsAll :
    ∀ k, k < wire.chunkCount →
      chunkGuardsOverrides overridePairs (rowsChunk wire k) = true := by
  intro k bound
  rw [chunkCount_eq] at bound
  match k, bound with
  | 0, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf0.classLeaf0).2
  | 1, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf0.classLeaf1).2
  | 2, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf0.classLeaf2).2
  | 3, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf0.classLeaf3).2
  | 4, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf0.classLeaf4).2
  | 5, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf0.classLeaf5).2
  | 6, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf0.classLeaf6).2
  | 7, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf0.classLeaf7).2
  | 8, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf0.classLeaf8).2
  | 9, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf0.classLeaf9).2
  | 10, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf0.classLeaf10).2
  | 11, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf0.classLeaf11).2
  | 12, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf0.classLeaf12).2
  | 13, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf0.classLeaf13).2
  | 14, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf1.classLeaf14).2
  | 15, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf1.classLeaf15).2
  | 16, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf1.classLeaf16).2
  | 17, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf1.classLeaf17).2
  | 18, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf1.classLeaf18).2
  | 19, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf1.classLeaf19).2
  | 20, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf1.classLeaf20).2
  | 21, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf1.classLeaf21).2
  | 22, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf1.classLeaf22).2
  | 23, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf1.classLeaf23).2
  | 24, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf1.classLeaf24).2
  | 25, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf1.classLeaf25).2
  | 26, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf1.classLeaf26).2
  | 27, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf1.classLeaf27).2
  | 28, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf2.classLeaf28).2
  | 29, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf2.classLeaf29).2
  | 30, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf2.classLeaf30).2
  | 31, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf2.classLeaf31).2
  | 32, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf2.classLeaf32).2
  | 33, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf2.classLeaf33).2
  | 34, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf2.classLeaf34).2
  | 35, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf2.classLeaf35).2
  | 36, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf2.classLeaf36).2
  | 37, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf2.classLeaf37).2
  | 38, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf2.classLeaf38).2
  | 39, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf2.classLeaf39).2
  | 40, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf2.classLeaf40).2
  | 41, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf2.classLeaf41).2
  | 42, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf3.classLeaf42).2
  | 43, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf3.classLeaf43).2
  | 44, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf3.classLeaf44).2
  | 45, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf3.classLeaf45).2
  | 46, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf3.classLeaf46).2
  | 47, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf3.classLeaf47).2
  | 48, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf3.classLeaf48).2
  | 49, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf3.classLeaf49).2
  | 50, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf3.classLeaf50).2
  | 51, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf3.classLeaf51).2
  | 52, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf3.classLeaf52).2
  | 53, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf3.classLeaf53).2
  | 54, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf3.classLeaf54).2
  | 55, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf3.classLeaf55).2
  | 56, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf4.classLeaf56).2
  | 57, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf4.classLeaf57).2
  | 58, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf4.classLeaf58).2
  | 59, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf4.classLeaf59).2
  | 60, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf4.classLeaf60).2
  | 61, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf4.classLeaf61).2
  | 62, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf4.classLeaf62).2
  | 63, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf4.classLeaf63).2
  | 64, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf4.classLeaf64).2
  | 65, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf4.classLeaf65).2
  | 66, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf4.classLeaf66).2
  | 67, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf4.classLeaf67).2
  | 68, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf4.classLeaf68).2
  | 69, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf4.classLeaf69).2
  | 70, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf5.classLeaf70).2
  | 71, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf5.classLeaf71).2
  | 72, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf5.classLeaf72).2
  | 73, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf5.classLeaf73).2
  | 74, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf5.classLeaf74).2
  | 75, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf5.classLeaf75).2
  | 76, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf5.classLeaf76).2
  | 77, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf5.classLeaf77).2
  | 78, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf5.classLeaf78).2
  | 79, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf5.classLeaf79).2
  | 80, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf5.classLeaf80).2
  | 81, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf5.classLeaf81).2
  | 82, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf5.classLeaf82).2
  | 83, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf5.classLeaf83).2
  | 84, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf6.classLeaf84).2
  | 85, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf6.classLeaf85).2
  | 86, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf6.classLeaf86).2
  | 87, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf6.classLeaf87).2
  | 88, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf6.classLeaf88).2
  | 89, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf6.classLeaf89).2
  | 90, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf6.classLeaf90).2
  | 91, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf6.classLeaf91).2
  | 92, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf6.classLeaf92).2
  | 93, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf6.classLeaf93).2
  | 94, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf6.classLeaf94).2
  | 95, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf6.classLeaf95).2
  | 96, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf6.classLeaf96).2
  | 97, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf6.classLeaf97).2
  | 98, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf7.classLeaf98).2
  | 99, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf7.classLeaf99).2
  | 100, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf7.classLeaf100).2
  | 101, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf7.classLeaf101).2
  | 102, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf7.classLeaf102).2
  | 103, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf7.classLeaf103).2
  | 104, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf7.classLeaf104).2
  | 105, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf7.classLeaf105).2
  | 106, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf7.classLeaf106).2
  | 107, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf7.classLeaf107).2
  | 108, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf7.classLeaf108).2
  | 109, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf7.classLeaf109).2
  | 110, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf7.classLeaf110).2
  | 111, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf7.classLeaf111).2
  | 112, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf8.classLeaf112).2
  | 113, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf8.classLeaf113).2
  | 114, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf8.classLeaf114).2
  | 115, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf8.classLeaf115).2
  | 116, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf8.classLeaf116).2
  | 117, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf8.classLeaf117).2
  | 118, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf8.classLeaf118).2
  | 119, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf8.classLeaf119).2
  | 120, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf8.classLeaf120).2
  | 121, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf8.classLeaf121).2
  | 122, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf8.classLeaf122).2
  | 123, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf8.classLeaf123).2
  | 124, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf8.classLeaf124).2
  | 125, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf8.classLeaf125).2
  | 126, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf9.classLeaf126).2
  | 127, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf9.classLeaf127).2
  | 128, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf9.classLeaf128).2
  | 129, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf9.classLeaf129).2
  | 130, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf9.classLeaf130).2
  | 131, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf9.classLeaf131).2
  | 132, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf9.classLeaf132).2
  | 133, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf9.classLeaf133).2
  | 134, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf9.classLeaf134).2
  | 135, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf9.classLeaf135).2
  | 136, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf9.classLeaf136).2
  | 137, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf9.classLeaf137).2
  | 138, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf9.classLeaf138).2
  | 139, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf9.classLeaf139).2
  | 140, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf10.classLeaf140).2
  | 141, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf10.classLeaf141).2
  | 142, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf10.classLeaf142).2
  | 143, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf10.classLeaf143).2
  | 144, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf10.classLeaf144).2
  | 145, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf10.classLeaf145).2
  | 146, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf10.classLeaf146).2
  | 147, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf10.classLeaf147).2
  | 148, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf10.classLeaf148).2
  | 149, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf10.classLeaf149).2
  | 150, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf10.classLeaf150).2
  | 151, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf10.classLeaf151).2
  | 152, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf10.classLeaf152).2
  | 153, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf10.classLeaf153).2
  | 154, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf11.classLeaf154).2
  | 155, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf11.classLeaf155).2
  | 156, _ => exact (classFacts_split Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf11.classLeaf156).2
  | n + 157, bound => exact absurd bound (by omega)

end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeaves
