import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifact
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment

/-!
GENERATED FILE - do not edit by hand.

Shared classification leaves: every chunk proves once that the
background satisfies its rows and that each row either belongs to
an override's family or avoids the override's column. All family
necessity modules of the batch reuse these leaves.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeaves

open Nightstream.Assurance.CompactSourceArtifact
open Nightstream.Assurance.ConstraintMinimization
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifact

set_option maxHeartbeats 2000000
set_option maxRecDepth 65536

def background : Nat → Field := backgroundFn Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values

def overridePairs : List (Nat × String) :=
  [(3811, "fprime.base.step.initial")]

theorem holdsLeaf0 :
    (rowsChunk wire 0).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf0 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 0) = true := by
  native_decide

theorem holdsLeaf1 :
    (rowsChunk wire 1).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf1 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 1) = true := by
  native_decide

theorem holdsLeaf2 :
    (rowsChunk wire 2).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf2 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 2) = true := by
  native_decide

theorem holdsLeaf3 :
    (rowsChunk wire 3).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf3 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 3) = true := by
  native_decide

theorem holdsLeaf4 :
    (rowsChunk wire 4).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf4 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 4) = true := by
  native_decide

theorem holdsLeaf5 :
    (rowsChunk wire 5).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf5 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 5) = true := by
  native_decide

theorem holdsLeaf6 :
    (rowsChunk wire 6).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf6 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 6) = true := by
  native_decide

theorem holdsLeaf7 :
    (rowsChunk wire 7).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf7 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 7) = true := by
  native_decide

theorem holdsLeaf8 :
    (rowsChunk wire 8).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf8 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 8) = true := by
  native_decide

theorem holdsLeaf9 :
    (rowsChunk wire 9).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf9 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 9) = true := by
  native_decide

theorem holdsLeaf10 :
    (rowsChunk wire 10).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf10 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 10) = true := by
  native_decide

theorem holdsLeaf11 :
    (rowsChunk wire 11).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf11 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 11) = true := by
  native_decide

theorem holdsLeaf12 :
    (rowsChunk wire 12).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf12 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 12) = true := by
  native_decide

theorem holdsLeaf13 :
    (rowsChunk wire 13).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf13 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 13) = true := by
  native_decide

theorem holdsLeaf14 :
    (rowsChunk wire 14).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf14 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 14) = true := by
  native_decide

theorem holdsLeaf15 :
    (rowsChunk wire 15).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf15 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 15) = true := by
  native_decide

theorem holdsLeaf16 :
    (rowsChunk wire 16).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf16 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 16) = true := by
  native_decide

theorem holdsLeaf17 :
    (rowsChunk wire 17).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf17 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 17) = true := by
  native_decide

theorem holdsLeaf18 :
    (rowsChunk wire 18).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf18 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 18) = true := by
  native_decide

theorem holdsLeaf19 :
    (rowsChunk wire 19).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf19 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 19) = true := by
  native_decide

theorem holdsLeaf20 :
    (rowsChunk wire 20).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf20 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 20) = true := by
  native_decide

theorem holdsLeaf21 :
    (rowsChunk wire 21).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf21 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 21) = true := by
  native_decide

theorem holdsLeaf22 :
    (rowsChunk wire 22).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf22 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 22) = true := by
  native_decide

theorem holdsLeaf23 :
    (rowsChunk wire 23).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf23 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 23) = true := by
  native_decide

theorem holdsLeaf24 :
    (rowsChunk wire 24).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf24 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 24) = true := by
  native_decide

theorem holdsLeaf25 :
    (rowsChunk wire 25).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf25 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 25) = true := by
  native_decide

theorem holdsLeaf26 :
    (rowsChunk wire 26).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf26 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 26) = true := by
  native_decide

theorem holdsLeaf27 :
    (rowsChunk wire 27).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf27 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 27) = true := by
  native_decide

theorem holdsLeaf28 :
    (rowsChunk wire 28).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf28 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 28) = true := by
  native_decide

theorem holdsLeaf29 :
    (rowsChunk wire 29).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf29 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 29) = true := by
  native_decide

theorem holdsLeaf30 :
    (rowsChunk wire 30).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf30 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 30) = true := by
  native_decide

theorem holdsLeaf31 :
    (rowsChunk wire 31).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf31 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 31) = true := by
  native_decide

theorem holdsLeaf32 :
    (rowsChunk wire 32).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf32 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 32) = true := by
  native_decide

theorem holdsLeaf33 :
    (rowsChunk wire 33).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf33 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 33) = true := by
  native_decide

theorem holdsLeaf34 :
    (rowsChunk wire 34).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf34 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 34) = true := by
  native_decide

theorem holdsLeaf35 :
    (rowsChunk wire 35).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf35 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 35) = true := by
  native_decide

theorem holdsLeaf36 :
    (rowsChunk wire 36).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf36 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 36) = true := by
  native_decide

theorem holdsLeaf37 :
    (rowsChunk wire 37).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf37 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 37) = true := by
  native_decide

theorem holdsLeaf38 :
    (rowsChunk wire 38).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf38 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 38) = true := by
  native_decide

theorem holdsLeaf39 :
    (rowsChunk wire 39).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf39 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 39) = true := by
  native_decide

theorem holdsLeaf40 :
    (rowsChunk wire 40).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf40 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 40) = true := by
  native_decide

theorem holdsLeaf41 :
    (rowsChunk wire 41).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf41 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 41) = true := by
  native_decide

theorem holdsLeaf42 :
    (rowsChunk wire 42).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf42 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 42) = true := by
  native_decide

theorem holdsLeaf43 :
    (rowsChunk wire 43).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf43 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 43) = true := by
  native_decide

theorem holdsLeaf44 :
    (rowsChunk wire 44).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf44 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 44) = true := by
  native_decide

theorem holdsLeaf45 :
    (rowsChunk wire 45).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf45 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 45) = true := by
  native_decide

theorem holdsLeaf46 :
    (rowsChunk wire 46).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf46 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 46) = true := by
  native_decide

theorem holdsLeaf47 :
    (rowsChunk wire 47).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf47 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 47) = true := by
  native_decide

theorem holdsLeaf48 :
    (rowsChunk wire 48).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf48 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 48) = true := by
  native_decide

theorem holdsLeaf49 :
    (rowsChunk wire 49).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf49 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 49) = true := by
  native_decide

theorem holdsLeaf50 :
    (rowsChunk wire 50).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf50 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 50) = true := by
  native_decide

theorem holdsLeaf51 :
    (rowsChunk wire 51).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf51 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 51) = true := by
  native_decide

theorem holdsLeaf52 :
    (rowsChunk wire 52).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf52 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 52) = true := by
  native_decide

theorem holdsLeaf53 :
    (rowsChunk wire 53).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf53 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 53) = true := by
  native_decide

theorem holdsLeaf54 :
    (rowsChunk wire 54).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf54 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 54) = true := by
  native_decide

theorem holdsLeaf55 :
    (rowsChunk wire 55).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf55 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 55) = true := by
  native_decide

theorem holdsLeaf56 :
    (rowsChunk wire 56).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf56 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 56) = true := by
  native_decide

theorem holdsLeaf57 :
    (rowsChunk wire 57).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf57 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 57) = true := by
  native_decide

theorem holdsLeaf58 :
    (rowsChunk wire 58).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf58 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 58) = true := by
  native_decide

theorem holdsLeaf59 :
    (rowsChunk wire 59).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf59 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 59) = true := by
  native_decide

theorem holdsLeaf60 :
    (rowsChunk wire 60).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf60 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 60) = true := by
  native_decide

theorem holdsLeaf61 :
    (rowsChunk wire 61).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf61 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 61) = true := by
  native_decide

theorem holdsLeaf62 :
    (rowsChunk wire 62).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf62 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 62) = true := by
  native_decide

theorem holdsLeaf63 :
    (rowsChunk wire 63).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf63 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 63) = true := by
  native_decide

theorem holdsLeaf64 :
    (rowsChunk wire 64).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf64 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 64) = true := by
  native_decide

theorem holdsLeaf65 :
    (rowsChunk wire 65).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf65 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 65) = true := by
  native_decide

theorem holdsLeaf66 :
    (rowsChunk wire 66).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf66 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 66) = true := by
  native_decide

theorem holdsLeaf67 :
    (rowsChunk wire 67).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf67 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 67) = true := by
  native_decide

theorem holdsLeaf68 :
    (rowsChunk wire 68).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf68 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 68) = true := by
  native_decide

theorem holdsLeaf69 :
    (rowsChunk wire 69).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf69 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 69) = true := by
  native_decide

theorem holdsLeaf70 :
    (rowsChunk wire 70).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf70 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 70) = true := by
  native_decide

theorem holdsLeaf71 :
    (rowsChunk wire 71).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf71 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 71) = true := by
  native_decide

theorem holdsLeaf72 :
    (rowsChunk wire 72).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf72 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 72) = true := by
  native_decide

theorem holdsLeaf73 :
    (rowsChunk wire 73).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf73 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 73) = true := by
  native_decide

theorem holdsLeaf74 :
    (rowsChunk wire 74).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf74 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 74) = true := by
  native_decide

theorem holdsLeaf75 :
    (rowsChunk wire 75).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf75 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 75) = true := by
  native_decide

theorem holdsLeaf76 :
    (rowsChunk wire 76).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf76 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 76) = true := by
  native_decide

theorem holdsLeaf77 :
    (rowsChunk wire 77).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf77 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 77) = true := by
  native_decide

theorem holdsLeaf78 :
    (rowsChunk wire 78).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf78 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 78) = true := by
  native_decide

theorem holdsLeaf79 :
    (rowsChunk wire 79).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf79 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 79) = true := by
  native_decide

theorem holdsLeaf80 :
    (rowsChunk wire 80).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf80 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 80) = true := by
  native_decide

theorem holdsLeaf81 :
    (rowsChunk wire 81).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf81 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 81) = true := by
  native_decide

theorem holdsLeaf82 :
    (rowsChunk wire 82).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf82 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 82) = true := by
  native_decide

theorem holdsLeaf83 :
    (rowsChunk wire 83).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf83 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 83) = true := by
  native_decide

theorem holdsLeaf84 :
    (rowsChunk wire 84).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf84 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 84) = true := by
  native_decide

theorem holdsLeaf85 :
    (rowsChunk wire 85).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf85 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 85) = true := by
  native_decide

theorem holdsLeaf86 :
    (rowsChunk wire 86).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf86 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 86) = true := by
  native_decide

theorem holdsLeaf87 :
    (rowsChunk wire 87).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf87 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 87) = true := by
  native_decide

theorem holdsLeaf88 :
    (rowsChunk wire 88).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf88 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 88) = true := by
  native_decide

theorem holdsLeaf89 :
    (rowsChunk wire 89).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf89 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 89) = true := by
  native_decide

theorem holdsLeaf90 :
    (rowsChunk wire 90).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf90 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 90) = true := by
  native_decide

theorem holdsLeaf91 :
    (rowsChunk wire 91).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf91 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 91) = true := by
  native_decide

theorem holdsLeaf92 :
    (rowsChunk wire 92).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf92 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 92) = true := by
  native_decide

theorem holdsLeaf93 :
    (rowsChunk wire 93).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf93 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 93) = true := by
  native_decide

theorem holdsLeaf94 :
    (rowsChunk wire 94).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf94 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 94) = true := by
  native_decide

theorem holdsLeaf95 :
    (rowsChunk wire 95).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf95 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 95) = true := by
  native_decide

theorem holdsLeaf96 :
    (rowsChunk wire 96).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf96 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 96) = true := by
  native_decide

theorem holdsLeaf97 :
    (rowsChunk wire 97).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf97 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 97) = true := by
  native_decide

theorem holdsLeaf98 :
    (rowsChunk wire 98).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf98 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 98) = true := by
  native_decide

theorem holdsLeaf99 :
    (rowsChunk wire 99).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf99 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 99) = true := by
  native_decide

theorem holdsLeaf100 :
    (rowsChunk wire 100).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf100 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 100) = true := by
  native_decide

theorem holdsLeaf101 :
    (rowsChunk wire 101).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf101 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 101) = true := by
  native_decide

theorem holdsLeaf102 :
    (rowsChunk wire 102).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf102 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 102) = true := by
  native_decide

theorem holdsLeaf103 :
    (rowsChunk wire 103).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf103 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 103) = true := by
  native_decide

theorem holdsLeaf104 :
    (rowsChunk wire 104).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf104 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 104) = true := by
  native_decide

theorem holdsLeaf105 :
    (rowsChunk wire 105).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf105 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 105) = true := by
  native_decide

theorem holdsLeaf106 :
    (rowsChunk wire 106).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf106 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 106) = true := by
  native_decide

theorem holdsLeaf107 :
    (rowsChunk wire 107).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf107 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 107) = true := by
  native_decide

theorem holdsLeaf108 :
    (rowsChunk wire 108).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf108 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 108) = true := by
  native_decide

theorem holdsLeaf109 :
    (rowsChunk wire 109).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf109 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 109) = true := by
  native_decide

theorem holdsLeaf110 :
    (rowsChunk wire 110).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf110 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 110) = true := by
  native_decide

theorem holdsLeaf111 :
    (rowsChunk wire 111).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf111 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 111) = true := by
  native_decide

theorem holdsLeaf112 :
    (rowsChunk wire 112).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf112 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 112) = true := by
  native_decide

theorem holdsLeaf113 :
    (rowsChunk wire 113).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf113 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 113) = true := by
  native_decide

theorem holdsLeaf114 :
    (rowsChunk wire 114).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf114 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 114) = true := by
  native_decide

theorem holdsLeaf115 :
    (rowsChunk wire 115).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf115 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 115) = true := by
  native_decide

theorem holdsLeaf116 :
    (rowsChunk wire 116).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf116 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 116) = true := by
  native_decide

theorem holdsLeaf117 :
    (rowsChunk wire 117).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf117 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 117) = true := by
  native_decide

theorem holdsLeaf118 :
    (rowsChunk wire 118).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf118 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 118) = true := by
  native_decide

theorem holdsLeaf119 :
    (rowsChunk wire 119).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf119 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 119) = true := by
  native_decide

theorem holdsLeaf120 :
    (rowsChunk wire 120).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf120 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 120) = true := by
  native_decide

theorem holdsLeaf121 :
    (rowsChunk wire 121).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf121 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 121) = true := by
  native_decide

theorem holdsLeaf122 :
    (rowsChunk wire 122).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf122 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 122) = true := by
  native_decide

theorem holdsLeaf123 :
    (rowsChunk wire 123).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf123 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 123) = true := by
  native_decide

theorem holdsLeaf124 :
    (rowsChunk wire 124).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf124 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 124) = true := by
  native_decide

theorem holdsLeaf125 :
    (rowsChunk wire 125).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf125 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 125) = true := by
  native_decide

theorem holdsLeaf126 :
    (rowsChunk wire 126).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf126 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 126) = true := by
  native_decide

theorem holdsLeaf127 :
    (rowsChunk wire 127).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf127 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 127) = true := by
  native_decide

theorem holdsLeaf128 :
    (rowsChunk wire 128).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf128 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 128) = true := by
  native_decide

theorem holdsLeaf129 :
    (rowsChunk wire 129).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf129 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 129) = true := by
  native_decide

theorem holdsLeaf130 :
    (rowsChunk wire 130).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf130 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 130) = true := by
  native_decide

theorem holdsLeaf131 :
    (rowsChunk wire 131).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf131 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 131) = true := by
  native_decide

theorem holdsLeaf132 :
    (rowsChunk wire 132).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf132 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 132) = true := by
  native_decide

theorem holdsLeaf133 :
    (rowsChunk wire 133).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf133 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 133) = true := by
  native_decide

theorem holdsLeaf134 :
    (rowsChunk wire 134).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf134 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 134) = true := by
  native_decide

theorem holdsLeaf135 :
    (rowsChunk wire 135).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf135 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 135) = true := by
  native_decide

theorem holdsLeaf136 :
    (rowsChunk wire 136).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf136 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 136) = true := by
  native_decide

theorem holdsLeaf137 :
    (rowsChunk wire 137).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf137 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 137) = true := by
  native_decide

theorem holdsLeaf138 :
    (rowsChunk wire 138).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf138 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 138) = true := by
  native_decide

theorem holdsLeaf139 :
    (rowsChunk wire 139).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf139 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 139) = true := by
  native_decide

theorem holdsLeaf140 :
    (rowsChunk wire 140).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf140 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 140) = true := by
  native_decide

theorem holdsLeaf141 :
    (rowsChunk wire 141).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf141 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 141) = true := by
  native_decide

theorem holdsLeaf142 :
    (rowsChunk wire 142).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf142 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 142) = true := by
  native_decide

theorem holdsLeaf143 :
    (rowsChunk wire 143).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf143 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 143) = true := by
  native_decide

theorem holdsLeaf144 :
    (rowsChunk wire 144).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf144 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 144) = true := by
  native_decide

theorem holdsLeaf145 :
    (rowsChunk wire 145).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf145 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 145) = true := by
  native_decide

theorem holdsLeaf146 :
    (rowsChunk wire 146).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf146 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 146) = true := by
  native_decide

theorem holdsLeaf147 :
    (rowsChunk wire 147).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf147 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 147) = true := by
  native_decide

theorem holdsLeaf148 :
    (rowsChunk wire 148).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf148 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 148) = true := by
  native_decide

theorem holdsLeaf149 :
    (rowsChunk wire 149).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf149 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 149) = true := by
  native_decide

theorem holdsLeaf150 :
    (rowsChunk wire 150).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf150 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 150) = true := by
  native_decide

theorem holdsLeaf151 :
    (rowsChunk wire 151).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf151 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 151) = true := by
  native_decide

theorem holdsLeaf152 :
    (rowsChunk wire 152).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf152 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 152) = true := by
  native_decide

theorem holdsLeaf153 :
    (rowsChunk wire 153).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf153 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 153) = true := by
  native_decide

theorem holdsLeaf154 :
    (rowsChunk wire 154).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf154 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 154) = true := by
  native_decide

theorem holdsLeaf155 :
    (rowsChunk wire 155).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf155 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 155) = true := by
  native_decide

theorem holdsLeaf156 :
    (rowsChunk wire 156).all
      (fun row => decide (Algebraic.Holds background row.row)) = true := by
  native_decide

theorem guardsLeaf156 :
    chunkGuardsOverrides overridePairs (rowsChunk wire 156) = true := by
  native_decide

theorem holdsAll :
    ∀ k, k < wire.chunkCount →
      (rowsChunk wire k).all
        (fun row => decide (Algebraic.Holds background row.row)) = true := by
  intro k bound
  match k with
  | 0 => exact holdsLeaf0
  | 1 => exact holdsLeaf1
  | 2 => exact holdsLeaf2
  | 3 => exact holdsLeaf3
  | 4 => exact holdsLeaf4
  | 5 => exact holdsLeaf5
  | 6 => exact holdsLeaf6
  | 7 => exact holdsLeaf7
  | 8 => exact holdsLeaf8
  | 9 => exact holdsLeaf9
  | 10 => exact holdsLeaf10
  | 11 => exact holdsLeaf11
  | 12 => exact holdsLeaf12
  | 13 => exact holdsLeaf13
  | 14 => exact holdsLeaf14
  | 15 => exact holdsLeaf15
  | 16 => exact holdsLeaf16
  | 17 => exact holdsLeaf17
  | 18 => exact holdsLeaf18
  | 19 => exact holdsLeaf19
  | 20 => exact holdsLeaf20
  | 21 => exact holdsLeaf21
  | 22 => exact holdsLeaf22
  | 23 => exact holdsLeaf23
  | 24 => exact holdsLeaf24
  | 25 => exact holdsLeaf25
  | 26 => exact holdsLeaf26
  | 27 => exact holdsLeaf27
  | 28 => exact holdsLeaf28
  | 29 => exact holdsLeaf29
  | 30 => exact holdsLeaf30
  | 31 => exact holdsLeaf31
  | 32 => exact holdsLeaf32
  | 33 => exact holdsLeaf33
  | 34 => exact holdsLeaf34
  | 35 => exact holdsLeaf35
  | 36 => exact holdsLeaf36
  | 37 => exact holdsLeaf37
  | 38 => exact holdsLeaf38
  | 39 => exact holdsLeaf39
  | 40 => exact holdsLeaf40
  | 41 => exact holdsLeaf41
  | 42 => exact holdsLeaf42
  | 43 => exact holdsLeaf43
  | 44 => exact holdsLeaf44
  | 45 => exact holdsLeaf45
  | 46 => exact holdsLeaf46
  | 47 => exact holdsLeaf47
  | 48 => exact holdsLeaf48
  | 49 => exact holdsLeaf49
  | 50 => exact holdsLeaf50
  | 51 => exact holdsLeaf51
  | 52 => exact holdsLeaf52
  | 53 => exact holdsLeaf53
  | 54 => exact holdsLeaf54
  | 55 => exact holdsLeaf55
  | 56 => exact holdsLeaf56
  | 57 => exact holdsLeaf57
  | 58 => exact holdsLeaf58
  | 59 => exact holdsLeaf59
  | 60 => exact holdsLeaf60
  | 61 => exact holdsLeaf61
  | 62 => exact holdsLeaf62
  | 63 => exact holdsLeaf63
  | 64 => exact holdsLeaf64
  | 65 => exact holdsLeaf65
  | 66 => exact holdsLeaf66
  | 67 => exact holdsLeaf67
  | 68 => exact holdsLeaf68
  | 69 => exact holdsLeaf69
  | 70 => exact holdsLeaf70
  | 71 => exact holdsLeaf71
  | 72 => exact holdsLeaf72
  | 73 => exact holdsLeaf73
  | 74 => exact holdsLeaf74
  | 75 => exact holdsLeaf75
  | 76 => exact holdsLeaf76
  | 77 => exact holdsLeaf77
  | 78 => exact holdsLeaf78
  | 79 => exact holdsLeaf79
  | 80 => exact holdsLeaf80
  | 81 => exact holdsLeaf81
  | 82 => exact holdsLeaf82
  | 83 => exact holdsLeaf83
  | 84 => exact holdsLeaf84
  | 85 => exact holdsLeaf85
  | 86 => exact holdsLeaf86
  | 87 => exact holdsLeaf87
  | 88 => exact holdsLeaf88
  | 89 => exact holdsLeaf89
  | 90 => exact holdsLeaf90
  | 91 => exact holdsLeaf91
  | 92 => exact holdsLeaf92
  | 93 => exact holdsLeaf93
  | 94 => exact holdsLeaf94
  | 95 => exact holdsLeaf95
  | 96 => exact holdsLeaf96
  | 97 => exact holdsLeaf97
  | 98 => exact holdsLeaf98
  | 99 => exact holdsLeaf99
  | 100 => exact holdsLeaf100
  | 101 => exact holdsLeaf101
  | 102 => exact holdsLeaf102
  | 103 => exact holdsLeaf103
  | 104 => exact holdsLeaf104
  | 105 => exact holdsLeaf105
  | 106 => exact holdsLeaf106
  | 107 => exact holdsLeaf107
  | 108 => exact holdsLeaf108
  | 109 => exact holdsLeaf109
  | 110 => exact holdsLeaf110
  | 111 => exact holdsLeaf111
  | 112 => exact holdsLeaf112
  | 113 => exact holdsLeaf113
  | 114 => exact holdsLeaf114
  | 115 => exact holdsLeaf115
  | 116 => exact holdsLeaf116
  | 117 => exact holdsLeaf117
  | 118 => exact holdsLeaf118
  | 119 => exact holdsLeaf119
  | 120 => exact holdsLeaf120
  | 121 => exact holdsLeaf121
  | 122 => exact holdsLeaf122
  | 123 => exact holdsLeaf123
  | 124 => exact holdsLeaf124
  | 125 => exact holdsLeaf125
  | 126 => exact holdsLeaf126
  | 127 => exact holdsLeaf127
  | 128 => exact holdsLeaf128
  | 129 => exact holdsLeaf129
  | 130 => exact holdsLeaf130
  | 131 => exact holdsLeaf131
  | 132 => exact holdsLeaf132
  | 133 => exact holdsLeaf133
  | 134 => exact holdsLeaf134
  | 135 => exact holdsLeaf135
  | 136 => exact holdsLeaf136
  | 137 => exact holdsLeaf137
  | 138 => exact holdsLeaf138
  | 139 => exact holdsLeaf139
  | 140 => exact holdsLeaf140
  | 141 => exact holdsLeaf141
  | 142 => exact holdsLeaf142
  | 143 => exact holdsLeaf143
  | 144 => exact holdsLeaf144
  | 145 => exact holdsLeaf145
  | 146 => exact holdsLeaf146
  | 147 => exact holdsLeaf147
  | 148 => exact holdsLeaf148
  | 149 => exact holdsLeaf149
  | 150 => exact holdsLeaf150
  | 151 => exact holdsLeaf151
  | 152 => exact holdsLeaf152
  | 153 => exact holdsLeaf153
  | 154 => exact holdsLeaf154
  | 155 => exact holdsLeaf155
  | 156 => exact holdsLeaf156
  | n + 157 => exact absurd bound (by omega)

theorem guardsAll :
    ∀ k, k < wire.chunkCount →
      chunkGuardsOverrides overridePairs (rowsChunk wire k) = true := by
  intro k bound
  match k with
  | 0 => exact guardsLeaf0
  | 1 => exact guardsLeaf1
  | 2 => exact guardsLeaf2
  | 3 => exact guardsLeaf3
  | 4 => exact guardsLeaf4
  | 5 => exact guardsLeaf5
  | 6 => exact guardsLeaf6
  | 7 => exact guardsLeaf7
  | 8 => exact guardsLeaf8
  | 9 => exact guardsLeaf9
  | 10 => exact guardsLeaf10
  | 11 => exact guardsLeaf11
  | 12 => exact guardsLeaf12
  | 13 => exact guardsLeaf13
  | 14 => exact guardsLeaf14
  | 15 => exact guardsLeaf15
  | 16 => exact guardsLeaf16
  | 17 => exact guardsLeaf17
  | 18 => exact guardsLeaf18
  | 19 => exact guardsLeaf19
  | 20 => exact guardsLeaf20
  | 21 => exact guardsLeaf21
  | 22 => exact guardsLeaf22
  | 23 => exact guardsLeaf23
  | 24 => exact guardsLeaf24
  | 25 => exact guardsLeaf25
  | 26 => exact guardsLeaf26
  | 27 => exact guardsLeaf27
  | 28 => exact guardsLeaf28
  | 29 => exact guardsLeaf29
  | 30 => exact guardsLeaf30
  | 31 => exact guardsLeaf31
  | 32 => exact guardsLeaf32
  | 33 => exact guardsLeaf33
  | 34 => exact guardsLeaf34
  | 35 => exact guardsLeaf35
  | 36 => exact guardsLeaf36
  | 37 => exact guardsLeaf37
  | 38 => exact guardsLeaf38
  | 39 => exact guardsLeaf39
  | 40 => exact guardsLeaf40
  | 41 => exact guardsLeaf41
  | 42 => exact guardsLeaf42
  | 43 => exact guardsLeaf43
  | 44 => exact guardsLeaf44
  | 45 => exact guardsLeaf45
  | 46 => exact guardsLeaf46
  | 47 => exact guardsLeaf47
  | 48 => exact guardsLeaf48
  | 49 => exact guardsLeaf49
  | 50 => exact guardsLeaf50
  | 51 => exact guardsLeaf51
  | 52 => exact guardsLeaf52
  | 53 => exact guardsLeaf53
  | 54 => exact guardsLeaf54
  | 55 => exact guardsLeaf55
  | 56 => exact guardsLeaf56
  | 57 => exact guardsLeaf57
  | 58 => exact guardsLeaf58
  | 59 => exact guardsLeaf59
  | 60 => exact guardsLeaf60
  | 61 => exact guardsLeaf61
  | 62 => exact guardsLeaf62
  | 63 => exact guardsLeaf63
  | 64 => exact guardsLeaf64
  | 65 => exact guardsLeaf65
  | 66 => exact guardsLeaf66
  | 67 => exact guardsLeaf67
  | 68 => exact guardsLeaf68
  | 69 => exact guardsLeaf69
  | 70 => exact guardsLeaf70
  | 71 => exact guardsLeaf71
  | 72 => exact guardsLeaf72
  | 73 => exact guardsLeaf73
  | 74 => exact guardsLeaf74
  | 75 => exact guardsLeaf75
  | 76 => exact guardsLeaf76
  | 77 => exact guardsLeaf77
  | 78 => exact guardsLeaf78
  | 79 => exact guardsLeaf79
  | 80 => exact guardsLeaf80
  | 81 => exact guardsLeaf81
  | 82 => exact guardsLeaf82
  | 83 => exact guardsLeaf83
  | 84 => exact guardsLeaf84
  | 85 => exact guardsLeaf85
  | 86 => exact guardsLeaf86
  | 87 => exact guardsLeaf87
  | 88 => exact guardsLeaf88
  | 89 => exact guardsLeaf89
  | 90 => exact guardsLeaf90
  | 91 => exact guardsLeaf91
  | 92 => exact guardsLeaf92
  | 93 => exact guardsLeaf93
  | 94 => exact guardsLeaf94
  | 95 => exact guardsLeaf95
  | 96 => exact guardsLeaf96
  | 97 => exact guardsLeaf97
  | 98 => exact guardsLeaf98
  | 99 => exact guardsLeaf99
  | 100 => exact guardsLeaf100
  | 101 => exact guardsLeaf101
  | 102 => exact guardsLeaf102
  | 103 => exact guardsLeaf103
  | 104 => exact guardsLeaf104
  | 105 => exact guardsLeaf105
  | 106 => exact guardsLeaf106
  | 107 => exact guardsLeaf107
  | 108 => exact guardsLeaf108
  | 109 => exact guardsLeaf109
  | 110 => exact guardsLeaf110
  | 111 => exact guardsLeaf111
  | 112 => exact guardsLeaf112
  | 113 => exact guardsLeaf113
  | 114 => exact guardsLeaf114
  | 115 => exact guardsLeaf115
  | 116 => exact guardsLeaf116
  | 117 => exact guardsLeaf117
  | 118 => exact guardsLeaf118
  | 119 => exact guardsLeaf119
  | 120 => exact guardsLeaf120
  | 121 => exact guardsLeaf121
  | 122 => exact guardsLeaf122
  | 123 => exact guardsLeaf123
  | 124 => exact guardsLeaf124
  | 125 => exact guardsLeaf125
  | 126 => exact guardsLeaf126
  | 127 => exact guardsLeaf127
  | 128 => exact guardsLeaf128
  | 129 => exact guardsLeaf129
  | 130 => exact guardsLeaf130
  | 131 => exact guardsLeaf131
  | 132 => exact guardsLeaf132
  | 133 => exact guardsLeaf133
  | 134 => exact guardsLeaf134
  | 135 => exact guardsLeaf135
  | 136 => exact guardsLeaf136
  | 137 => exact guardsLeaf137
  | 138 => exact guardsLeaf138
  | 139 => exact guardsLeaf139
  | 140 => exact guardsLeaf140
  | 141 => exact guardsLeaf141
  | 142 => exact guardsLeaf142
  | 143 => exact guardsLeaf143
  | 144 => exact guardsLeaf144
  | 145 => exact guardsLeaf145
  | 146 => exact guardsLeaf146
  | 147 => exact guardsLeaf147
  | 148 => exact guardsLeaf148
  | 149 => exact guardsLeaf149
  | 150 => exact guardsLeaf150
  | 151 => exact guardsLeaf151
  | 152 => exact guardsLeaf152
  | 153 => exact guardsLeaf153
  | 154 => exact guardsLeaf154
  | 155 => exact guardsLeaf155
  | 156 => exact guardsLeaf156
  | n + 157 => exact absurd bound (by omega)

end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeaves
