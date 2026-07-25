import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls.EncodeInstance
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionCallContext
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionDigestCodecs

/-!
Contract: package the audited production `encodeInstance` codec/map as one
complete typed lowering call recipe.

Assurance tier: artifact-independent encoding refinement.

Owns:
- exact alignment of a supplied full codec profile at the optional-digest and
  compact-encoded ports;
- the minimal `EncodeInstanceProfile`, independent of the nonlinear
  `freshPublic` operation;
- a complete activation-aware `CallRecipe` with a nonoptional receipt;
- the exact six-row/no-temporary footprint derived from that recipe.

Does not own: the other data codecs or production call implementations, a
complete canonical Step/Terminal recipe family, Rust emission, generated
rows, or compiled-Rust semantics.

Emits constraints: exactly six gated affine rows per call occurrence.
-/

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionEncodeInstanceCallRecipe

open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Rust.CanonicalConformance.NativeStep
open DirectCalls
open ProductionDigestCodecs

abbrev ProductionDigest :=
  ProductionCallContext.ProductionDigest

variable
  {Params : Type}
  {StructureDigest : Type}
  {Header : Type}
  {Running : Type}
  {NifsProof : Type}
  {Nebula : Type}
  {NebulaDigest : Type}
  {NebulaOpen : Type}

local notation "AdapterParameters" =>
  ProductionCallContext.AdapterParameters
    Params StructureDigest Header Running NifsProof Nebula NebulaDigest
      NebulaOpen

abbrev AdapterConfiguration (adapter : AdapterParameters) :=
  ProductionCallContext.Configuration adapter

abbrev loweringParameters
    [DecidableEq ProductionDigest]
    [DecidableEq Running]
    [DecidableEq CanonicalPlainCarrierLink.RawClaim]
    [DecidableEq NifsProof]
    [DecidableEq Nebula]
    [DecidableEq NebulaOpen]
    (adapter : AdapterParameters)
    (configuration : AdapterConfiguration adapter) :
    Parameters :=
  ProductionCallContext.parameters adapter configuration

/-- Exact two-codec and footprint boundary needed to select the production
affine call without making claims about any unrelated codec. -/
structure Alignment
    [DecidableEq ProductionDigest]
    [DecidableEq Running]
    [DecidableEq CanonicalPlainCarrierLink.RawClaim]
    [DecidableEq NifsProof]
    [DecidableEq Nebula]
    [DecidableEq NebulaOpen]
    (adapter : AdapterParameters)
    (configuration : AdapterConfiguration adapter)
    (profile :
      Profile (loweringParameters adapter configuration)) : Prop where
  digestCodecExact :
    profile.codecs.digest = optionalDigestCodec
  encodedCodecExact :
    profile.codecs.encoded = adapterEncodedCodec
  footprintExact :
    configuration.footprints.encodeInstance =
      affineFootprint adapterEncodedCodec.width

/-- Minimal exact profile accepted by the generic affine call compiler. -/
def profile
    [DecidableEq ProductionDigest]
    [DecidableEq Running]
    [DecidableEq CanonicalPlainCarrierLink.RawClaim]
    [DecidableEq NifsProof]
    [DecidableEq Nebula]
    [DecidableEq NebulaOpen]
    (adapter : AdapterParameters)
    (configuration : AdapterConfiguration adapter)
    (fullProfile :
      Profile (loweringParameters adapter configuration))
    (alignment : Alignment adapter configuration fullProfile) :
    EncodeInstanceProfile (loweringParameters adapter configuration) := by
  refine {
    toProfile := fullProfile
    encodeInstanceMap := ?_
    encodeInstanceFootprint := ?_
  }
  · rw [alignment.digestCodecExact, alignment.encodedCodecExact]
    exact encodeInstanceAffineMap
  · rw [alignment.encodedCodecExact]
    exact alignment.footprintExact

/-- Complete recipe: soundness, honest completeness, inactive
satisfiability, support, row ownership, and the emission receipt are fields of
this value, not later assumptions. -/
def recipe
    [DecidableEq ProductionDigest]
    [DecidableEq Running]
    [DecidableEq CanonicalPlainCarrierLink.RawClaim]
    [DecidableEq NifsProof]
    [DecidableEq Nebula]
    [DecidableEq NebulaOpen]
    (adapter : AdapterParameters)
    (configuration : AdapterConfiguration adapter)
    (fullProfile :
      Profile (loweringParameters adapter configuration))
    (alignment : Alignment adapter configuration fullProfile) :
    CallRecipe
      (signature (loweringParameters adapter configuration))
      ((profile adapter configuration fullProfile alignment).family
        (loweringParameters adapter configuration))
      Call.encodeInstance :=
  encodeInstanceRecipeForProfile
    (loweringParameters adapter configuration)
    (profile adapter configuration fullProfile alignment)

theorem selected_footprint_exact
    [DecidableEq ProductionDigest]
    [DecidableEq Running]
    [DecidableEq CanonicalPlainCarrierLink.RawClaim]
    [DecidableEq NifsProof]
    [DecidableEq Nebula]
    [DecidableEq NebulaOpen]
    (adapter : AdapterParameters)
    (configuration : AdapterConfiguration adapter)
    (fullProfile :
      Profile (loweringParameters adapter configuration))
    (alignment : Alignment adapter configuration fullProfile) :
    (signature (loweringParameters adapter configuration)).callFootprint
        Call.encodeInstance =
      { recurringRows := 6, temporaries := [] } := by
  change configuration.footprints.encodeInstance =
    { recurringRows := 6, temporaries := [] }
  rw [alignment.footprintExact]
  rfl

/-- Every occurrence emits exactly six rows, as computed from the selected
program rather than measured from an artifact. -/
theorem recipe_row_count
    [DecidableEq ProductionDigest]
    [DecidableEq Running]
    [DecidableEq CanonicalPlainCarrierLink.RawClaim]
    [DecidableEq NifsProof]
    [DecidableEq Nebula]
    [DecidableEq NebulaOpen]
    (adapter : AdapterParameters)
    (configuration : AdapterConfiguration adapter)
    (fullProfile :
      Profile (loweringParameters adapter configuration))
    (alignment : Alignment adapter configuration fullProfile)
    {context :
      Nightstream.Implementation.Lowering.Typed.Schema
        (typeSystem (loweringParameters adapter configuration))}
    {references :
      Nightstream.Implementation.Lowering.Typed.Refs
        (typeSystem (loweringParameters adapter configuration))
        context
        ((signature (loweringParameters adapter configuration)).callInputs
            Call.encodeInstance)}
    (frame :
      CallFrame
        (signature := signature
          (loweringParameters adapter configuration))
        ((profile adapter configuration fullProfile alignment).family
          (loweringParameters adapter configuration))
        Call.encodeInstance references) :
    ((recipe adapter configuration fullProfile alignment).rows frame).length =
      6 := by
  rw [(recipe adapter configuration fullProfile alignment).rowCount frame]
  rw [selected_footprint_exact adapter configuration fullProfile alignment]

/-- The emitted receipt is nonoptional and contains exactly the call's output
allocation, declared temporary allocation, and six selected rows. -/
theorem receipt_exact
    [DecidableEq ProductionDigest]
    [DecidableEq Running]
    [DecidableEq CanonicalPlainCarrierLink.RawClaim]
    [DecidableEq NifsProof]
    [DecidableEq Nebula]
    [DecidableEq NebulaOpen]
    (adapter : AdapterParameters)
    (configuration : AdapterConfiguration adapter)
    (fullProfile :
      Profile (loweringParameters adapter configuration))
    (alignment : Alignment adapter configuration fullProfile)
    {context :
      Nightstream.Implementation.Lowering.Typed.Schema
        (typeSystem (loweringParameters adapter configuration))}
    {references :
      Nightstream.Implementation.Lowering.Typed.Refs
        (typeSystem (loweringParameters adapter configuration))
        context
        ((signature (loweringParameters adapter configuration)).callInputs
            Call.encodeInstance)}
    (frame :
      CallFrame
        (signature := signature
          (loweringParameters adapter configuration))
        ((profile adapter configuration fullProfile alignment).family
          (loweringParameters adapter configuration))
        Call.encodeInstance references) :
    (recipe adapter configuration fullProfile alignment).receipt frame =
      { outputBundles := frame.outputs.portColumns
        temporaryBundles := frame.temporaries.bundleColumns
        rows :=
          (recipe adapter configuration fullProfile alignment).rows frame } :=
  CallRecipe.receipt_exact
    (recipe adapter configuration fullProfile alignment) frame

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionEncodeInstanceCallRecipe
