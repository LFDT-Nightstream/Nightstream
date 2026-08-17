import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPiRLCFamilyBodyDecoder

/-! Focused checks for the compact production PiRLC family-body decoder. -/

namespace tests.FPrimeFullHistoryStreamingPiRLCFamilyBodyDecoder

open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyBodyDecoder

example : EvenValid := even_valid

example : OddValid := odd_valid

example :
    templateColumnCount
          Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyDecoder.evenArm +
        residualColumnCount
          Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyDecoder.evenArm =
      1301125 :=
  even_column_census_exact.2.2

example :
    templateColumnCount
          Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyDecoder.oddArm +
        residualColumnCount
          Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyDecoder.oddArm =
      1302325 :=
  odd_column_census_exact.2.2

end tests.FPrimeFullHistoryStreamingPiRLCFamilyBodyDecoder
