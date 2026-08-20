import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalSourceBindingSchema

/-! Generated compact certificate for the exact Rust terminal source-binding rows.

Emits constraints: no. Rust emits the rows reconstructed by these decoder groups.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalSourceBinding

open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalSourceBinding.Artifact

def decoderGroups : List DecoderGroup := [
    DecoderGroup.block { owner := "x_out", sourceFields := { start := 30382557, stop := 30382558 }, decodedColumns := { start := 28041899, stop := 28041900 }, finalColumns := { start := 1, stop := 2 }, width := 1, radix := 2, scale := 1313210370 },
    DecoderGroup.block { owner := "x_out", sourceFields := { start := 642, stop := 650 }, decodedColumns := { start := 28041900, stop := 28041908 }, finalColumns := { start := 767, stop := 1095 }, width := 41, radix := 3, scale := 1 },
    DecoderGroup.block { owner := "x_out", sourceFields := { start := 30382558, stop := 30382560 }, decodedColumns := { start := 28041908, stop := 28041910 }, finalColumns := { start := 21909951, stop := 21910015 }, width := 32, radix := 2, scale := 1 },
    DecoderGroup.block { owner := "x_out", sourceFields := { start := 30382560, stop := 30382562 }, decodedColumns := { start := 28041910, stop := 28041912 }, finalColumns := { start := 578, stop := 642 }, width := 32, radix := 2, scale := 1 },
    DecoderGroup.block { owner := "x_out", sourceFields := { start := 30382562, stop := 30382564 }, decodedColumns := { start := 28041912, stop := 28041914 }, finalColumns := { start := 703, stop := 767 }, width := 32, radix := 2, scale := 1 },
    DecoderGroup.block { owner := "x_out", sourceFields := { start := 722, stop := 726 }, decodedColumns := { start := 28041914, stop := 28041918 }, finalColumns := { start := 3784, stop := 3948 }, width := 41, radix := 3, scale := 1 },
    DecoderGroup.block { owner := "x_out", sourceFields := { start := 30362945, stop := 30362949 }, decodedColumns := { start := 28041918, stop := 28041922 }, finalColumns := { start := 21909787, stop := 21909951 }, width := 41, radix := 3, scale := 1 },
    DecoderGroup.composite { owner := "x_out", sourceField := 30362937, decodedColumn := 28041922, segments := [{ finalColumns := { start := 21909459, stop := 21909500 }, radix := 3, scale := 4 }, { finalColumns := { start := 21909500, stop := 21909541 }, radix := 3, scale := 6 }, { finalColumns := { start := 21909541, stop := 21909582 }, radix := 3, scale := 2 }, { finalColumns := { start := 21909582, stop := 21909623 }, radix := 3, scale := 2 }, { finalColumns := { start := 21909623, stop := 21909664 }, radix := 3, scale := 2 }, { finalColumns := { start := 21909664, stop := 21909705 }, radix := 3, scale := 3 }, { finalColumns := { start := 21909705, stop := 21909746 }, radix := 3, scale := 1 }, { finalColumns := { start := 21909746, stop := 21909787 }, radix := 3, scale := 1 }] },
    DecoderGroup.composite { owner := "x_out", sourceField := 30362938, decodedColumn := 28041923, segments := [{ finalColumns := { start := 21909459, stop := 21909500 }, radix := 3, scale := 2 }, { finalColumns := { start := 21909500, stop := 21909541 }, radix := 3, scale := 4 }, { finalColumns := { start := 21909541, stop := 21909582 }, radix := 3, scale := 6 }, { finalColumns := { start := 21909582, stop := 21909623 }, radix := 3, scale := 2 }, { finalColumns := { start := 21909623, stop := 21909664 }, radix := 3, scale := 1 }, { finalColumns := { start := 21909664, stop := 21909705 }, radix := 3, scale := 2 }, { finalColumns := { start := 21909705, stop := 21909746 }, radix := 3, scale := 3 }, { finalColumns := { start := 21909746, stop := 21909787 }, radix := 3, scale := 1 }] },
    DecoderGroup.composite { owner := "x_out", sourceField := 30362939, decodedColumn := 28041924, segments := [{ finalColumns := { start := 21909459, stop := 21909500 }, radix := 3, scale := 2 }, { finalColumns := { start := 21909500, stop := 21909541 }, radix := 3, scale := 2 }, { finalColumns := { start := 21909541, stop := 21909582 }, radix := 3, scale := 4 }, { finalColumns := { start := 21909582, stop := 21909623 }, radix := 3, scale := 6 }, { finalColumns := { start := 21909623, stop := 21909664 }, radix := 3, scale := 1 }, { finalColumns := { start := 21909664, stop := 21909705 }, radix := 3, scale := 1 }, { finalColumns := { start := 21909705, stop := 21909746 }, radix := 3, scale := 2 }, { finalColumns := { start := 21909746, stop := 21909787 }, radix := 3, scale := 3 }] },
    DecoderGroup.composite { owner := "x_out", sourceField := 30362940, decodedColumn := 28041925, segments := [{ finalColumns := { start := 21909459, stop := 21909500 }, radix := 3, scale := 6 }, { finalColumns := { start := 21909500, stop := 21909541 }, radix := 3, scale := 2 }, { finalColumns := { start := 21909541, stop := 21909582 }, radix := 3, scale := 2 }, { finalColumns := { start := 21909582, stop := 21909623 }, radix := 3, scale := 4 }, { finalColumns := { start := 21909623, stop := 21909664 }, radix := 3, scale := 3 }, { finalColumns := { start := 21909664, stop := 21909705 }, radix := 3, scale := 1 }, { finalColumns := { start := 21909705, stop := 21909746 }, radix := 3, scale := 1 }, { finalColumns := { start := 21909746, stop := 21909787 }, radix := 3, scale := 2 }] },
    DecoderGroup.block { owner := "x_out", sourceFields := { start := 30382564, stop := 30382565 }, decodedColumns := { start := 28041926, stop := 28041927 }, finalColumns := { start := 1, stop := 2 }, width := 1, radix := 2, scale := 1312967745 },
    DecoderGroup.block { owner := "x_out", sourceFields := { start := 30382553, stop := 30382557 }, decodedColumns := { start := 28041927, stop := 28041931 }, finalColumns := { start := 22022995, stop := 22023159 }, width := 41, radix := 3, scale := 1 },
    DecoderGroup.block { owner := "nebula_lane", sourceFields := { start := 672, stop := 676 }, decodedColumns := { start := 28041931, stop := 28041935 }, finalColumns := { start := 1815, stop := 1979 }, width := 41, radix := 3, scale := 1 },
    DecoderGroup.block { owner := "nebula_lane", sourceFields := { start := 19341627, stop := 19341628 }, decodedColumns := { start := 28041935, stop := 28041936 }, finalColumns := { start := 19949237, stop := 19949238 }, width := 1, radix := 2, scale := 1 },
    DecoderGroup.block { owner := "nebula_lane", sourceFields := { start := 19341628, stop := 19341630 }, decodedColumns := { start := 28041936, stop := 28041938 }, finalColumns := { start := 19949238, stop := 19949320 }, width := 41, radix := 3, scale := 1 },
    DecoderGroup.block { owner := "nebula_lane", sourceFields := { start := 19231123, stop := 19231124 }, decodedColumns := { start := 28041938, stop := 28041939 }, finalColumns := { start := 19526378, stop := 19526422 }, width := 44, radix := 2, scale := 1 },
    DecoderGroup.block { owner := "nebula_lane", sourceFields := { start := 19341630, stop := 19341672 }, decodedColumns := { start := 28041939, stop := 28041981 }, finalColumns := { start := 19949320, stop := 19951042 }, width := 41, radix := 3, scale := 1 },
    DecoderGroup.block { owner := "local_state", sourceFields := { start := 30400385, stop := 30400389 }, decodedColumns := { start := 28041981, stop := 28041985 }, finalColumns := { start := 22126493, stop := 22126657 }, width := 41, radix := 3, scale := 1 },
    DecoderGroup.block { owner := "delayed_payload", sourceFields := { start := 30400389, stop := 30402558 }, decodedColumns := { start := 28041985, stop := 28044154 }, finalColumns := { start := 22126657, stop := 22128826 }, width := 1, radix := 2, scale := 1 },
  ]

def rawArtifact : RawArtifact :=
  { schemaVersion := 1,
    profileId := "nightstream/goldilocks/streaming-terminal-slice/v1",
    sourceArtifactIdentity := "rust:nightstream/streaming-terminal-lifecycle/source-rows/v1",
    finalArtifactIdentity := "rust:nightstream/streaming-selective-ccs/final-rows/v1", lifecycleScope := "recursive-terminal-arm-435",
    rowFamily := "terminal.streaming.source_binding", rowStart := 6, rowStop := 2261,
    columnCount := 28863843, finalAssignmentColumns := { start := 1, stop := 28038961 }, decodedColumns := { start := 28041899, stop := 28044154 },
    decoderGroups := decoderGroups }

end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalSourceBinding
