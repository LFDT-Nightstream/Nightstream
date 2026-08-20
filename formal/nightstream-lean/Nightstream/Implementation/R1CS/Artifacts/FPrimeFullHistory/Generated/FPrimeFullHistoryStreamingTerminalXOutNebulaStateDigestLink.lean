import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalXOutNebulaStateDigestLinkSchema

/-! Generated compact recipe for the exact Rust terminal Nebula-state-digest family.

Rust checks both Poseidon2 branches, the Boolean selector, four mux rows,
and four final links against all 19,353 source rows.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalXOutNebulaStateDigestLink

open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalXOutNebulaStateDigestLink.Artifact

def artifactSha256 : String := "f025862ff7936c683ae16bd422518d3ade69d3d69fea8af512ce0794226cab39"

def absentConstantValues : List Nat := [36, 30521782141150574, 31069335676202596, 27422324158721583, 28252386919279663, 33266224450594665, 52, 0, 0, 0, 0, 0, 4]

def presentConstantValues : List Nat := [36, 30521782141150574, 31069335676202596, 27422324158721583, 28252386919279663, 33266224450594665, 52, 1, 2, 4]

def rawArtifact : RawArtifact :=
  { schemaVersion := 2, profileId := "nightstream/goldilocks/streaming-terminal-nebula-state-digest/v2",
    sourceIdentity := "rust:streaming-terminal-nebula-state-digest/v2",
    sourceRowsSha256 := "89aae9a5eb9aa1f455cb97d60b648c7fdd03d729935d6d6cc87fe5419773173d", rowCount := 19353, columnCount := 352017,
    sourceRowStart := 330425, finalRowStart := 330425,
    openColumn := 2210,
    absentConstantValues := absentConstantValues, absentConstantStartColumn := 332669,
    absentInputColumns := [332669, 332670, 332671, 332672, 332673, 332674, 332675, 2206, 2207, 2208, 2209, 2211, 2212, 2213, 2226, 2227, 332676, 332677, 332678, 332679, 332680, 332681, 2218, 2219, 2220, 2221, 2222, 2223, 2224, 2225, 2228, 2229, 2230, 2231, 2232, 2233, 2234, 2235, 2236, 2237, 2238, 2239, 2240, 2241, 2242, 2243, 2244, 2245, 2246, 2247, 2248, 2249, 2250, 2251, 2252, 2253, 2254, 2255], absentOutputColumns := [342334, 342335, 342336, 342337],
    presentConstantValues := presentConstantValues, presentConstantStartColumn := 342342,
    presentInputColumns := [342342, 342343, 342344, 342345, 342346, 342347, 342348, 2206, 2207, 2208, 2209, 2211, 2212, 2213, 2226, 2227, 342349, 342350, 2214, 2215, 2216, 2217, 342351, 2218, 2219, 2220, 2221, 2222, 2223, 2224, 2225, 2228, 2229, 2230, 2231, 2232, 2233, 2234, 2235, 2236, 2237, 2238, 2239, 2240, 2241, 2242, 2243, 2244, 2245, 2246, 2247, 2248, 2249, 2250, 2251, 2252, 2253, 2254, 2255], presentOutputColumns := [352005, 352006, 352007, 352008],
    hashOutputColumns := [352013, 352014, 352015, 352016], xOutStateColumns := [29, 30, 31, 32],
    baselineDigestValue := 6284679863123074783,
    absentRowStart := 1, presentRowStart := 9674, muxRowStart := 19345,
    equalityRowStart := 19349, selectedSourceRow := 349774 }

end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalXOutNebulaStateDigestLink
