import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Transcript.SliceComposition

namespace Nightstream.Tests.Poseidon2TranscriptSliceComposition

open Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.ColumnReplay
open Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.SliceComposition

example (assignment : Nat → Nat) (canonical : CanonicalAssignment assignment)
    (run : SemanticRun) (left right : List Operation) :
    semanticExecuteSlice assignment canonical run (left ++ right) =
      semanticExecuteSlice assignment canonical
        (semanticExecuteSlice assignment canonical run left) right :=
  semanticExecuteSlice_append assignment canonical run left right

example (assignment : Nat → Nat) (canonical : CanonicalAssignment assignment)
    (run : SemanticRun) (slices : List (List Operation)) :
    semanticExecuteSlices assignment canonical run slices =
      semanticExecuteSlice assignment canonical run slices.flatten :=
  semanticExecuteSlices_eq_flatten assignment canonical run slices

end Nightstream.Tests.Poseidon2TranscriptSliceComposition
