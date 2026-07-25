import Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.CanonicalPlainCarrierLink
import Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.FixedOneLoweringAdapter

/-!
Contract: shared production fixed-one lowering context for independently
selected call recipes.

Owns only the concrete digest/fresh carrier choice and the exact
`FixedOneLoweringAdapter.parameters` instantiation. Call codecs, footprints,
field laws, physical recipes, and receipts remain owned by each call module.

Emits constraints: no.
-/

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionCallContext

open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
open Nightstream.Implementation.Rust.CanonicalConformance.NativeStep

abbrev ProductionDigest :=
  Nightstream.Implementation.Encoding.FPrime.Digest

abbrev AdapterParameters
    (Params StructureDigest Header Running NifsProof Nebula NebulaDigest
      NebulaOpen : Type) :=
  FixedOneCanonicalAdapter.Parameters
    Params StructureDigest Header ProductionDigest Running
      CanonicalPlainCarrierLink.RawClaim NifsProof Nebula NebulaDigest
      NebulaOpen

abbrev Configuration
    {Params StructureDigest Header Running NifsProof Nebula NebulaDigest
      NebulaOpen : Type}
    (adapter :
      AdapterParameters Params StructureDigest Header Running NifsProof
        Nebula NebulaDigest NebulaOpen) :=
  FixedOneLoweringAdapter.Configuration adapter

abbrev parameters
    {Params StructureDigest Header Running NifsProof Nebula NebulaDigest
      NebulaOpen : Type}
    [DecidableEq ProductionDigest]
    [DecidableEq Running]
    [DecidableEq CanonicalPlainCarrierLink.RawClaim]
    [DecidableEq NifsProof]
    [DecidableEq Nebula]
    [DecidableEq NebulaOpen]
    (adapter :
      AdapterParameters Params StructureDigest Header Running NifsProof
        Nebula NebulaDigest NebulaOpen)
    (configuration : Configuration adapter) :
    Parameters :=
  FixedOneLoweringAdapter.parameters
    (Params := Params)
    (StructureDigest := StructureDigest)
    (Header := Header)
    (Digest := ProductionDigest)
    (Running := Running)
    (Fresh := CanonicalPlainCarrierLink.RawClaim)
    (NifsProof := NifsProof)
    (Nebula := Nebula)
    (NebulaDigest := NebulaDigest)
    (NebulaOpen := NebulaOpen)
    adapter configuration

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionCallContext
