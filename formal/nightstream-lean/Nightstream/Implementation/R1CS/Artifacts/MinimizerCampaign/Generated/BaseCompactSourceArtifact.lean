import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactWire
import Nightstream.Assurance.ChunkLeaves
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
  | 0, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf0.chunkLeaf0.1)).1
  | 1, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf0.chunkLeaf1.1)).1
  | 2, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf0.chunkLeaf2.1)).1
  | 3, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf0.chunkLeaf3.1)).1
  | 4, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf0.chunkLeaf4.1)).1
  | 5, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf0.chunkLeaf5.1)).1
  | 6, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf0.chunkLeaf6.1)).1
  | 7, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf0.chunkLeaf7.1)).1
  | 8, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf0.chunkLeaf8.1)).1
  | 9, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf0.chunkLeaf9.1)).1
  | 10, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf0.chunkLeaf10.1)).1
  | 11, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf0.chunkLeaf11.1)).1
  | 12, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf0.chunkLeaf12.1)).1
  | 13, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf0.chunkLeaf13.1)).1
  | 14, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf1.chunkLeaf14.1)).1
  | 15, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf1.chunkLeaf15.1)).1
  | 16, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf1.chunkLeaf16.1)).1
  | 17, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf1.chunkLeaf17.1)).1
  | 18, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf1.chunkLeaf18.1)).1
  | 19, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf1.chunkLeaf19.1)).1
  | 20, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf1.chunkLeaf20.1)).1
  | 21, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf1.chunkLeaf21.1)).1
  | 22, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf1.chunkLeaf22.1)).1
  | 23, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf1.chunkLeaf23.1)).1
  | 24, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf1.chunkLeaf24.1)).1
  | 25, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf1.chunkLeaf25.1)).1
  | 26, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf1.chunkLeaf26.1)).1
  | 27, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf1.chunkLeaf27.1)).1
  | 28, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf2.chunkLeaf28.1)).1
  | 29, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf2.chunkLeaf29.1)).1
  | 30, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf2.chunkLeaf30.1)).1
  | 31, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf2.chunkLeaf31.1)).1
  | 32, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf2.chunkLeaf32.1)).1
  | 33, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf2.chunkLeaf33.1)).1
  | 34, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf2.chunkLeaf34.1)).1
  | 35, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf2.chunkLeaf35.1)).1
  | 36, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf2.chunkLeaf36.1)).1
  | 37, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf2.chunkLeaf37.1)).1
  | 38, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf2.chunkLeaf38.1)).1
  | 39, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf2.chunkLeaf39.1)).1
  | 40, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf2.chunkLeaf40.1)).1
  | 41, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf2.chunkLeaf41.1)).1
  | 42, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf3.chunkLeaf42.1)).1
  | 43, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf3.chunkLeaf43.1)).1
  | 44, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf3.chunkLeaf44.1)).1
  | 45, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf3.chunkLeaf45.1)).1
  | 46, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf3.chunkLeaf46.1)).1
  | 47, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf3.chunkLeaf47.1)).1
  | 48, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf3.chunkLeaf48.1)).1
  | 49, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf3.chunkLeaf49.1)).1
  | 50, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf3.chunkLeaf50.1)).1
  | 51, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf3.chunkLeaf51.1)).1
  | 52, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf3.chunkLeaf52.1)).1
  | 53, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf3.chunkLeaf53.1)).1
  | 54, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf3.chunkLeaf54.1)).1
  | 55, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf3.chunkLeaf55.1)).1
  | 56, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf4.chunkLeaf56.1)).1
  | 57, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf4.chunkLeaf57.1)).1
  | 58, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf4.chunkLeaf58.1)).1
  | 59, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf4.chunkLeaf59.1)).1
  | 60, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf4.chunkLeaf60.1)).1
  | 61, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf4.chunkLeaf61.1)).1
  | 62, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf4.chunkLeaf62.1)).1
  | 63, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf4.chunkLeaf63.1)).1
  | 64, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf4.chunkLeaf64.1)).1
  | 65, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf4.chunkLeaf65.1)).1
  | 66, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf4.chunkLeaf66.1)).1
  | 67, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf4.chunkLeaf67.1)).1
  | 68, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf4.chunkLeaf68.1)).1
  | 69, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf4.chunkLeaf69.1)).1
  | 70, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf5.chunkLeaf70.1)).1
  | 71, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf5.chunkLeaf71.1)).1
  | 72, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf5.chunkLeaf72.1)).1
  | 73, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf5.chunkLeaf73.1)).1
  | 74, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf5.chunkLeaf74.1)).1
  | 75, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf5.chunkLeaf75.1)).1
  | 76, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf5.chunkLeaf76.1)).1
  | 77, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf5.chunkLeaf77.1)).1
  | 78, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf5.chunkLeaf78.1)).1
  | 79, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf5.chunkLeaf79.1)).1
  | 80, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf5.chunkLeaf80.1)).1
  | 81, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf5.chunkLeaf81.1)).1
  | 82, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf5.chunkLeaf82.1)).1
  | 83, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf5.chunkLeaf83.1)).1
  | 84, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf6.chunkLeaf84.1)).1
  | 85, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf6.chunkLeaf85.1)).1
  | 86, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf6.chunkLeaf86.1)).1
  | 87, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf6.chunkLeaf87.1)).1
  | 88, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf6.chunkLeaf88.1)).1
  | 89, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf6.chunkLeaf89.1)).1
  | 90, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf6.chunkLeaf90.1)).1
  | 91, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf6.chunkLeaf91.1)).1
  | 92, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf6.chunkLeaf92.1)).1
  | 93, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf6.chunkLeaf93.1)).1
  | 94, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf6.chunkLeaf94.1)).1
  | 95, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf6.chunkLeaf95.1)).1
  | 96, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf6.chunkLeaf96.1)).1
  | 97, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf6.chunkLeaf97.1)).1
  | 98, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf7.chunkLeaf98.1)).1
  | 99, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf7.chunkLeaf99.1)).1
  | 100, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf7.chunkLeaf100.1)).1
  | 101, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf7.chunkLeaf101.1)).1
  | 102, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf7.chunkLeaf102.1)).1
  | 103, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf7.chunkLeaf103.1)).1
  | 104, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf7.chunkLeaf104.1)).1
  | 105, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf7.chunkLeaf105.1)).1
  | 106, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf7.chunkLeaf106.1)).1
  | 107, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf7.chunkLeaf107.1)).1
  | 108, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf7.chunkLeaf108.1)).1
  | 109, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf7.chunkLeaf109.1)).1
  | 110, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf7.chunkLeaf110.1)).1
  | 111, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf7.chunkLeaf111.1)).1
  | 112, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf8.chunkLeaf112.1)).1
  | 113, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf8.chunkLeaf113.1)).1
  | 114, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf8.chunkLeaf114.1)).1
  | 115, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf8.chunkLeaf115.1)).1
  | 116, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf8.chunkLeaf116.1)).1
  | 117, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf8.chunkLeaf117.1)).1
  | 118, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf8.chunkLeaf118.1)).1
  | 119, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf8.chunkLeaf119.1)).1
  | 120, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf8.chunkLeaf120.1)).1
  | 121, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf8.chunkLeaf121.1)).1
  | 122, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf8.chunkLeaf122.1)).1
  | 123, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf8.chunkLeaf123.1)).1
  | 124, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf8.chunkLeaf124.1)).1
  | 125, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf8.chunkLeaf125.1)).1
  | 126, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf9.chunkLeaf126.1)).1
  | 127, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf9.chunkLeaf127.1)).1
  | 128, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf9.chunkLeaf128.1)).1
  | 129, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf9.chunkLeaf129.1)).1
  | 130, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf9.chunkLeaf130.1)).1
  | 131, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf9.chunkLeaf131.1)).1
  | 132, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf9.chunkLeaf132.1)).1
  | 133, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf9.chunkLeaf133.1)).1
  | 134, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf9.chunkLeaf134.1)).1
  | 135, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf9.chunkLeaf135.1)).1
  | 136, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf9.chunkLeaf136.1)).1
  | 137, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf9.chunkLeaf137.1)).1
  | 138, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf9.chunkLeaf138.1)).1
  | 139, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf9.chunkLeaf139.1)).1
  | 140, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf10.chunkLeaf140.1)).1
  | 141, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf10.chunkLeaf141.1)).1
  | 142, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf10.chunkLeaf142.1)).1
  | 143, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf10.chunkLeaf143.1)).1
  | 144, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf10.chunkLeaf144.1)).1
  | 145, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf10.chunkLeaf145.1)).1
  | 146, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf10.chunkLeaf146.1)).1
  | 147, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf10.chunkLeaf147.1)).1
  | 148, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf10.chunkLeaf148.1)).1
  | 149, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf10.chunkLeaf149.1)).1
  | 150, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf10.chunkLeaf150.1)).1
  | 151, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf10.chunkLeaf151.1)).1
  | 152, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf10.chunkLeaf152.1)).1
  | 153, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf10.chunkLeaf153.1)).1
  | 154, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf11.chunkLeaf154.1)).1
  | 155, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf11.chunkLeaf155.1)).1
  | 156, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf11.chunkLeaf156.1)).1
  | n + 157, bound => exact absurd bound (by omega)

theorem rowsWf :
    ∀ k, k < wire.chunkCount →
      (rowsChunk wire k).all (rowWellFormedAt 39949 38626) = true := by
  intro k bound
  rw [chunkCount_eq] at bound
  match k, bound with
  | 0, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf0.chunkLeaf0.1)).2.1
  | 1, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf0.chunkLeaf1.1)).2.1
  | 2, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf0.chunkLeaf2.1)).2.1
  | 3, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf0.chunkLeaf3.1)).2.1
  | 4, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf0.chunkLeaf4.1)).2.1
  | 5, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf0.chunkLeaf5.1)).2.1
  | 6, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf0.chunkLeaf6.1)).2.1
  | 7, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf0.chunkLeaf7.1)).2.1
  | 8, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf0.chunkLeaf8.1)).2.1
  | 9, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf0.chunkLeaf9.1)).2.1
  | 10, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf0.chunkLeaf10.1)).2.1
  | 11, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf0.chunkLeaf11.1)).2.1
  | 12, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf0.chunkLeaf12.1)).2.1
  | 13, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf0.chunkLeaf13.1)).2.1
  | 14, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf1.chunkLeaf14.1)).2.1
  | 15, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf1.chunkLeaf15.1)).2.1
  | 16, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf1.chunkLeaf16.1)).2.1
  | 17, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf1.chunkLeaf17.1)).2.1
  | 18, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf1.chunkLeaf18.1)).2.1
  | 19, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf1.chunkLeaf19.1)).2.1
  | 20, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf1.chunkLeaf20.1)).2.1
  | 21, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf1.chunkLeaf21.1)).2.1
  | 22, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf1.chunkLeaf22.1)).2.1
  | 23, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf1.chunkLeaf23.1)).2.1
  | 24, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf1.chunkLeaf24.1)).2.1
  | 25, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf1.chunkLeaf25.1)).2.1
  | 26, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf1.chunkLeaf26.1)).2.1
  | 27, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf1.chunkLeaf27.1)).2.1
  | 28, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf2.chunkLeaf28.1)).2.1
  | 29, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf2.chunkLeaf29.1)).2.1
  | 30, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf2.chunkLeaf30.1)).2.1
  | 31, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf2.chunkLeaf31.1)).2.1
  | 32, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf2.chunkLeaf32.1)).2.1
  | 33, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf2.chunkLeaf33.1)).2.1
  | 34, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf2.chunkLeaf34.1)).2.1
  | 35, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf2.chunkLeaf35.1)).2.1
  | 36, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf2.chunkLeaf36.1)).2.1
  | 37, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf2.chunkLeaf37.1)).2.1
  | 38, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf2.chunkLeaf38.1)).2.1
  | 39, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf2.chunkLeaf39.1)).2.1
  | 40, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf2.chunkLeaf40.1)).2.1
  | 41, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf2.chunkLeaf41.1)).2.1
  | 42, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf3.chunkLeaf42.1)).2.1
  | 43, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf3.chunkLeaf43.1)).2.1
  | 44, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf3.chunkLeaf44.1)).2.1
  | 45, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf3.chunkLeaf45.1)).2.1
  | 46, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf3.chunkLeaf46.1)).2.1
  | 47, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf3.chunkLeaf47.1)).2.1
  | 48, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf3.chunkLeaf48.1)).2.1
  | 49, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf3.chunkLeaf49.1)).2.1
  | 50, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf3.chunkLeaf50.1)).2.1
  | 51, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf3.chunkLeaf51.1)).2.1
  | 52, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf3.chunkLeaf52.1)).2.1
  | 53, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf3.chunkLeaf53.1)).2.1
  | 54, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf3.chunkLeaf54.1)).2.1
  | 55, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf3.chunkLeaf55.1)).2.1
  | 56, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf4.chunkLeaf56.1)).2.1
  | 57, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf4.chunkLeaf57.1)).2.1
  | 58, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf4.chunkLeaf58.1)).2.1
  | 59, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf4.chunkLeaf59.1)).2.1
  | 60, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf4.chunkLeaf60.1)).2.1
  | 61, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf4.chunkLeaf61.1)).2.1
  | 62, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf4.chunkLeaf62.1)).2.1
  | 63, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf4.chunkLeaf63.1)).2.1
  | 64, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf4.chunkLeaf64.1)).2.1
  | 65, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf4.chunkLeaf65.1)).2.1
  | 66, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf4.chunkLeaf66.1)).2.1
  | 67, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf4.chunkLeaf67.1)).2.1
  | 68, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf4.chunkLeaf68.1)).2.1
  | 69, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf4.chunkLeaf69.1)).2.1
  | 70, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf5.chunkLeaf70.1)).2.1
  | 71, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf5.chunkLeaf71.1)).2.1
  | 72, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf5.chunkLeaf72.1)).2.1
  | 73, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf5.chunkLeaf73.1)).2.1
  | 74, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf5.chunkLeaf74.1)).2.1
  | 75, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf5.chunkLeaf75.1)).2.1
  | 76, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf5.chunkLeaf76.1)).2.1
  | 77, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf5.chunkLeaf77.1)).2.1
  | 78, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf5.chunkLeaf78.1)).2.1
  | 79, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf5.chunkLeaf79.1)).2.1
  | 80, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf5.chunkLeaf80.1)).2.1
  | 81, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf5.chunkLeaf81.1)).2.1
  | 82, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf5.chunkLeaf82.1)).2.1
  | 83, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf5.chunkLeaf83.1)).2.1
  | 84, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf6.chunkLeaf84.1)).2.1
  | 85, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf6.chunkLeaf85.1)).2.1
  | 86, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf6.chunkLeaf86.1)).2.1
  | 87, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf6.chunkLeaf87.1)).2.1
  | 88, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf6.chunkLeaf88.1)).2.1
  | 89, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf6.chunkLeaf89.1)).2.1
  | 90, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf6.chunkLeaf90.1)).2.1
  | 91, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf6.chunkLeaf91.1)).2.1
  | 92, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf6.chunkLeaf92.1)).2.1
  | 93, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf6.chunkLeaf93.1)).2.1
  | 94, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf6.chunkLeaf94.1)).2.1
  | 95, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf6.chunkLeaf95.1)).2.1
  | 96, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf6.chunkLeaf96.1)).2.1
  | 97, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf6.chunkLeaf97.1)).2.1
  | 98, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf7.chunkLeaf98.1)).2.1
  | 99, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf7.chunkLeaf99.1)).2.1
  | 100, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf7.chunkLeaf100.1)).2.1
  | 101, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf7.chunkLeaf101.1)).2.1
  | 102, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf7.chunkLeaf102.1)).2.1
  | 103, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf7.chunkLeaf103.1)).2.1
  | 104, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf7.chunkLeaf104.1)).2.1
  | 105, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf7.chunkLeaf105.1)).2.1
  | 106, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf7.chunkLeaf106.1)).2.1
  | 107, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf7.chunkLeaf107.1)).2.1
  | 108, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf7.chunkLeaf108.1)).2.1
  | 109, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf7.chunkLeaf109.1)).2.1
  | 110, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf7.chunkLeaf110.1)).2.1
  | 111, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf7.chunkLeaf111.1)).2.1
  | 112, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf8.chunkLeaf112.1)).2.1
  | 113, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf8.chunkLeaf113.1)).2.1
  | 114, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf8.chunkLeaf114.1)).2.1
  | 115, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf8.chunkLeaf115.1)).2.1
  | 116, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf8.chunkLeaf116.1)).2.1
  | 117, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf8.chunkLeaf117.1)).2.1
  | 118, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf8.chunkLeaf118.1)).2.1
  | 119, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf8.chunkLeaf119.1)).2.1
  | 120, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf8.chunkLeaf120.1)).2.1
  | 121, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf8.chunkLeaf121.1)).2.1
  | 122, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf8.chunkLeaf122.1)).2.1
  | 123, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf8.chunkLeaf123.1)).2.1
  | 124, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf8.chunkLeaf124.1)).2.1
  | 125, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf8.chunkLeaf125.1)).2.1
  | 126, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf9.chunkLeaf126.1)).2.1
  | 127, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf9.chunkLeaf127.1)).2.1
  | 128, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf9.chunkLeaf128.1)).2.1
  | 129, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf9.chunkLeaf129.1)).2.1
  | 130, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf9.chunkLeaf130.1)).2.1
  | 131, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf9.chunkLeaf131.1)).2.1
  | 132, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf9.chunkLeaf132.1)).2.1
  | 133, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf9.chunkLeaf133.1)).2.1
  | 134, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf9.chunkLeaf134.1)).2.1
  | 135, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf9.chunkLeaf135.1)).2.1
  | 136, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf9.chunkLeaf136.1)).2.1
  | 137, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf9.chunkLeaf137.1)).2.1
  | 138, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf9.chunkLeaf138.1)).2.1
  | 139, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf9.chunkLeaf139.1)).2.1
  | 140, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf10.chunkLeaf140.1)).2.1
  | 141, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf10.chunkLeaf141.1)).2.1
  | 142, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf10.chunkLeaf142.1)).2.1
  | 143, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf10.chunkLeaf143.1)).2.1
  | 144, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf10.chunkLeaf144.1)).2.1
  | 145, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf10.chunkLeaf145.1)).2.1
  | 146, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf10.chunkLeaf146.1)).2.1
  | 147, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf10.chunkLeaf147.1)).2.1
  | 148, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf10.chunkLeaf148.1)).2.1
  | 149, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf10.chunkLeaf149.1)).2.1
  | 150, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf10.chunkLeaf150.1)).2.1
  | 151, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf10.chunkLeaf151.1)).2.1
  | 152, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf10.chunkLeaf152.1)).2.1
  | 153, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf10.chunkLeaf153.1)).2.1
  | 154, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf11.chunkLeaf154.1)).2.1
  | 155, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf11.chunkLeaf155.1)).2.1
  | 156, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf11.chunkLeaf156.1)).2.1
  | n + 157, bound => exact absurd bound (by omega)

theorem familiesCovered :
    ∀ k, k < wire.chunkCount →
      (rowsChunk wire k).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true := by
  intro k bound
  rw [chunkCount_eq] at bound
  match k, bound with
  | 0, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf0.chunkLeaf0.1)).2.2.1
  | 1, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf0.chunkLeaf1.1)).2.2.1
  | 2, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf0.chunkLeaf2.1)).2.2.1
  | 3, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf0.chunkLeaf3.1)).2.2.1
  | 4, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf0.chunkLeaf4.1)).2.2.1
  | 5, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf0.chunkLeaf5.1)).2.2.1
  | 6, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf0.chunkLeaf6.1)).2.2.1
  | 7, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf0.chunkLeaf7.1)).2.2.1
  | 8, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf0.chunkLeaf8.1)).2.2.1
  | 9, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf0.chunkLeaf9.1)).2.2.1
  | 10, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf0.chunkLeaf10.1)).2.2.1
  | 11, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf0.chunkLeaf11.1)).2.2.1
  | 12, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf0.chunkLeaf12.1)).2.2.1
  | 13, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf0.chunkLeaf13.1)).2.2.1
  | 14, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf1.chunkLeaf14.1)).2.2.1
  | 15, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf1.chunkLeaf15.1)).2.2.1
  | 16, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf1.chunkLeaf16.1)).2.2.1
  | 17, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf1.chunkLeaf17.1)).2.2.1
  | 18, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf1.chunkLeaf18.1)).2.2.1
  | 19, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf1.chunkLeaf19.1)).2.2.1
  | 20, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf1.chunkLeaf20.1)).2.2.1
  | 21, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf1.chunkLeaf21.1)).2.2.1
  | 22, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf1.chunkLeaf22.1)).2.2.1
  | 23, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf1.chunkLeaf23.1)).2.2.1
  | 24, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf1.chunkLeaf24.1)).2.2.1
  | 25, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf1.chunkLeaf25.1)).2.2.1
  | 26, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf1.chunkLeaf26.1)).2.2.1
  | 27, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf1.chunkLeaf27.1)).2.2.1
  | 28, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf2.chunkLeaf28.1)).2.2.1
  | 29, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf2.chunkLeaf29.1)).2.2.1
  | 30, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf2.chunkLeaf30.1)).2.2.1
  | 31, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf2.chunkLeaf31.1)).2.2.1
  | 32, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf2.chunkLeaf32.1)).2.2.1
  | 33, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf2.chunkLeaf33.1)).2.2.1
  | 34, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf2.chunkLeaf34.1)).2.2.1
  | 35, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf2.chunkLeaf35.1)).2.2.1
  | 36, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf2.chunkLeaf36.1)).2.2.1
  | 37, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf2.chunkLeaf37.1)).2.2.1
  | 38, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf2.chunkLeaf38.1)).2.2.1
  | 39, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf2.chunkLeaf39.1)).2.2.1
  | 40, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf2.chunkLeaf40.1)).2.2.1
  | 41, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf2.chunkLeaf41.1)).2.2.1
  | 42, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf3.chunkLeaf42.1)).2.2.1
  | 43, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf3.chunkLeaf43.1)).2.2.1
  | 44, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf3.chunkLeaf44.1)).2.2.1
  | 45, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf3.chunkLeaf45.1)).2.2.1
  | 46, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf3.chunkLeaf46.1)).2.2.1
  | 47, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf3.chunkLeaf47.1)).2.2.1
  | 48, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf3.chunkLeaf48.1)).2.2.1
  | 49, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf3.chunkLeaf49.1)).2.2.1
  | 50, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf3.chunkLeaf50.1)).2.2.1
  | 51, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf3.chunkLeaf51.1)).2.2.1
  | 52, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf3.chunkLeaf52.1)).2.2.1
  | 53, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf3.chunkLeaf53.1)).2.2.1
  | 54, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf3.chunkLeaf54.1)).2.2.1
  | 55, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf3.chunkLeaf55.1)).2.2.1
  | 56, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf4.chunkLeaf56.1)).2.2.1
  | 57, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf4.chunkLeaf57.1)).2.2.1
  | 58, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf4.chunkLeaf58.1)).2.2.1
  | 59, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf4.chunkLeaf59.1)).2.2.1
  | 60, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf4.chunkLeaf60.1)).2.2.1
  | 61, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf4.chunkLeaf61.1)).2.2.1
  | 62, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf4.chunkLeaf62.1)).2.2.1
  | 63, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf4.chunkLeaf63.1)).2.2.1
  | 64, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf4.chunkLeaf64.1)).2.2.1
  | 65, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf4.chunkLeaf65.1)).2.2.1
  | 66, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf4.chunkLeaf66.1)).2.2.1
  | 67, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf4.chunkLeaf67.1)).2.2.1
  | 68, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf4.chunkLeaf68.1)).2.2.1
  | 69, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf4.chunkLeaf69.1)).2.2.1
  | 70, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf5.chunkLeaf70.1)).2.2.1
  | 71, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf5.chunkLeaf71.1)).2.2.1
  | 72, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf5.chunkLeaf72.1)).2.2.1
  | 73, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf5.chunkLeaf73.1)).2.2.1
  | 74, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf5.chunkLeaf74.1)).2.2.1
  | 75, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf5.chunkLeaf75.1)).2.2.1
  | 76, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf5.chunkLeaf76.1)).2.2.1
  | 77, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf5.chunkLeaf77.1)).2.2.1
  | 78, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf5.chunkLeaf78.1)).2.2.1
  | 79, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf5.chunkLeaf79.1)).2.2.1
  | 80, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf5.chunkLeaf80.1)).2.2.1
  | 81, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf5.chunkLeaf81.1)).2.2.1
  | 82, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf5.chunkLeaf82.1)).2.2.1
  | 83, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf5.chunkLeaf83.1)).2.2.1
  | 84, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf6.chunkLeaf84.1)).2.2.1
  | 85, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf6.chunkLeaf85.1)).2.2.1
  | 86, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf6.chunkLeaf86.1)).2.2.1
  | 87, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf6.chunkLeaf87.1)).2.2.1
  | 88, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf6.chunkLeaf88.1)).2.2.1
  | 89, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf6.chunkLeaf89.1)).2.2.1
  | 90, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf6.chunkLeaf90.1)).2.2.1
  | 91, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf6.chunkLeaf91.1)).2.2.1
  | 92, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf6.chunkLeaf92.1)).2.2.1
  | 93, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf6.chunkLeaf93.1)).2.2.1
  | 94, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf6.chunkLeaf94.1)).2.2.1
  | 95, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf6.chunkLeaf95.1)).2.2.1
  | 96, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf6.chunkLeaf96.1)).2.2.1
  | 97, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf6.chunkLeaf97.1)).2.2.1
  | 98, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf7.chunkLeaf98.1)).2.2.1
  | 99, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf7.chunkLeaf99.1)).2.2.1
  | 100, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf7.chunkLeaf100.1)).2.2.1
  | 101, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf7.chunkLeaf101.1)).2.2.1
  | 102, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf7.chunkLeaf102.1)).2.2.1
  | 103, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf7.chunkLeaf103.1)).2.2.1
  | 104, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf7.chunkLeaf104.1)).2.2.1
  | 105, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf7.chunkLeaf105.1)).2.2.1
  | 106, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf7.chunkLeaf106.1)).2.2.1
  | 107, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf7.chunkLeaf107.1)).2.2.1
  | 108, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf7.chunkLeaf108.1)).2.2.1
  | 109, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf7.chunkLeaf109.1)).2.2.1
  | 110, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf7.chunkLeaf110.1)).2.2.1
  | 111, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf7.chunkLeaf111.1)).2.2.1
  | 112, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf8.chunkLeaf112.1)).2.2.1
  | 113, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf8.chunkLeaf113.1)).2.2.1
  | 114, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf8.chunkLeaf114.1)).2.2.1
  | 115, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf8.chunkLeaf115.1)).2.2.1
  | 116, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf8.chunkLeaf116.1)).2.2.1
  | 117, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf8.chunkLeaf117.1)).2.2.1
  | 118, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf8.chunkLeaf118.1)).2.2.1
  | 119, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf8.chunkLeaf119.1)).2.2.1
  | 120, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf8.chunkLeaf120.1)).2.2.1
  | 121, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf8.chunkLeaf121.1)).2.2.1
  | 122, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf8.chunkLeaf122.1)).2.2.1
  | 123, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf8.chunkLeaf123.1)).2.2.1
  | 124, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf8.chunkLeaf124.1)).2.2.1
  | 125, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf8.chunkLeaf125.1)).2.2.1
  | 126, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf9.chunkLeaf126.1)).2.2.1
  | 127, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf9.chunkLeaf127.1)).2.2.1
  | 128, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf9.chunkLeaf128.1)).2.2.1
  | 129, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf9.chunkLeaf129.1)).2.2.1
  | 130, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf9.chunkLeaf130.1)).2.2.1
  | 131, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf9.chunkLeaf131.1)).2.2.1
  | 132, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf9.chunkLeaf132.1)).2.2.1
  | 133, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf9.chunkLeaf133.1)).2.2.1
  | 134, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf9.chunkLeaf134.1)).2.2.1
  | 135, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf9.chunkLeaf135.1)).2.2.1
  | 136, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf9.chunkLeaf136.1)).2.2.1
  | 137, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf9.chunkLeaf137.1)).2.2.1
  | 138, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf9.chunkLeaf138.1)).2.2.1
  | 139, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf9.chunkLeaf139.1)).2.2.1
  | 140, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf10.chunkLeaf140.1)).2.2.1
  | 141, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf10.chunkLeaf141.1)).2.2.1
  | 142, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf10.chunkLeaf142.1)).2.2.1
  | 143, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf10.chunkLeaf143.1)).2.2.1
  | 144, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf10.chunkLeaf144.1)).2.2.1
  | 145, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf10.chunkLeaf145.1)).2.2.1
  | 146, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf10.chunkLeaf146.1)).2.2.1
  | 147, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf10.chunkLeaf147.1)).2.2.1
  | 148, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf10.chunkLeaf148.1)).2.2.1
  | 149, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf10.chunkLeaf149.1)).2.2.1
  | 150, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf10.chunkLeaf150.1)).2.2.1
  | 151, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf10.chunkLeaf151.1)).2.2.1
  | 152, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf10.chunkLeaf152.1)).2.2.1
  | 153, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf10.chunkLeaf153.1)).2.2.1
  | 154, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf11.chunkLeaf154.1)).2.2.1
  | 155, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf11.chunkLeaf155.1)).2.2.1
  | 156, _ => exact (chunkFacts_split (Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf11.chunkLeaf156.1)).2.2.1
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
