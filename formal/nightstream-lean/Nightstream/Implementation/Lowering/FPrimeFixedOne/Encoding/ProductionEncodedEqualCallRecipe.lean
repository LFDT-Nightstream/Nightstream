import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls.EncodedEqual
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionCallContext
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionDigestCodecs

/-!
Contract: package equality of the production six-coordinate fixed-one
encoding as one complete typed lowering call recipe.

Assurance tier: artifact-independent encoding refinement.

Owns:
- exact alignment of a supplied full codec profile at the compact-encoded
  port;
- the permitted Goldilocks field/inversion contracts used by equality;
- a complete activation-aware `CallRecipe` and nonoptional receipt;
- the exact eighteen-row and three-temporary-bundle footprint.

Does not own: any other production call recipe, a complete Step/Terminal
recipe family, Rust emission, generated rows, or compiled-Rust semantics.

Emits constraints: exactly eighteen rows and auxiliary temporary bundles of
widths six, six, and five per call occurrence.
-/

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionEncodedEqualCallRecipe

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

/-- Exact codec, permitted algebraic contracts, and footprint needed by
compact encoded-value equality. No unrelated call semantics enter this
boundary. -/
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
      Profile (loweringParameters adapter configuration)) : Type where
  encodedCodecExact :
    profile.codecs.encoded = adapterEncodedCodec
  fieldLaws : FieldLaws
  inverseLaw : InverseLaw
  footprintExact :
    configuration.footprints.encodedEqual =
      equalityFootprint adapterEncodedCodec.width

/-- Minimal exact profile accepted by the encoded-equality call compiler. -/
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
    EncodedEqualProfile (loweringParameters adapter configuration) := by
  refine {
    toProfile := fullProfile
    fieldLaws := alignment.fieldLaws
    inverseLaw := alignment.inverseLaw
    encodedEqualFootprint := ?_
  }
  rw [alignment.encodedCodecExact]
  exact alignment.footprintExact

/-- Complete recipe with soundness, honest completeness, inactive
satisfiability, ownership, support, and receipt conservation. -/
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
      Call.encodedEqual :=
  encodedEqualRecipeForProfile
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
        Call.encodedEqual =
      { recurringRows := 18
        temporaries :=
          [auxiliaryLayout 6, auxiliaryLayout 6, auxiliaryLayout 5] } := by
  change configuration.footprints.encodedEqual =
    { recurringRows := 18
      temporaries :=
        [auxiliaryLayout 6, auxiliaryLayout 6, auxiliaryLayout 5] }
  rw [alignment.footprintExact]
  rfl

/-- Every occurrence emits exactly eighteen rows, computed from the selected
equality program. -/
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
          Call.encodedEqual)}
    (frame :
      CallFrame
        (signature := signature
          (loweringParameters adapter configuration))
        ((profile adapter configuration fullProfile alignment).family
          (loweringParameters adapter configuration))
        Call.encodedEqual references) :
    ((recipe adapter configuration fullProfile alignment).rows frame).length =
      18 := by
  rw [(recipe adapter configuration fullProfile alignment).rowCount frame]
  rw [selected_footprint_exact adapter configuration fullProfile alignment]

/-- The emitted receipt contains exactly the output allocation, all three
declared temporary bundles, and the eighteen selected rows. -/
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
          Call.encodedEqual)}
    (frame :
      CallFrame
        (signature := signature
          (loweringParameters adapter configuration))
        ((profile adapter configuration fullProfile alignment).family
          (loweringParameters adapter configuration))
        Call.encodedEqual references) :
    (recipe adapter configuration fullProfile alignment).receipt frame =
      { outputBundles := frame.outputs.portColumns
        temporaryBundles := frame.temporaries.bundleColumns
        rows :=
          (recipe adapter configuration fullProfile alignment).rows frame } :=
  CallRecipe.receipt_exact
    (recipe adapter configuration fullProfile alignment) frame

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionEncodedEqualCallRecipe
