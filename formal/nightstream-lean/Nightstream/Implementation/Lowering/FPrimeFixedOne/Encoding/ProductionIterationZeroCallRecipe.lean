import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls.IterationZero
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionCallContext

/-!
Contract: package the selected fixed-one `iterationZero` test as one complete
typed call recipe without depending on unrelated production call maps.

Assurance tier: artifact-independent encoding refinement.

Owns:
- the minimal zero-test profile over a supplied full codec profile;
- the permitted Goldilocks field/inversion contracts used by the selected
  witness construction;
- a complete activation-aware `CallRecipe` and nonoptional receipt;
- the exact three-row/two-one-coordinate-temporary footprint.

Does not own: any other production call recipe, a complete Step/Terminal
recipe family, Rust emission, generated rows, or compiled-Rust semantics.

Emits constraints: exactly three rows and two auxiliary one-coordinate
temporary bundles per call occurrence.
-/

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionIterationZeroCallRecipe

open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Rust.CanonicalConformance.NativeStep
open DirectCalls

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

/-- Exact permitted algebraic contracts and footprint needed by the
zero-test recipe. No unrelated call semantics enter this boundary. -/
structure Alignment
    [DecidableEq ProductionDigest]
    [DecidableEq Running]
    [DecidableEq CanonicalPlainCarrierLink.RawClaim]
    [DecidableEq NifsProof]
    [DecidableEq Nebula]
    [DecidableEq NebulaOpen]
    (adapter : AdapterParameters)
    (configuration : AdapterConfiguration adapter) : Type where
  fieldLaws : FieldLaws
  inverseLaw : InverseLaw
  footprintExact :
    configuration.footprints.iterationZero = zeroFootprint

/-- Minimal exact profile accepted by the zero-test call compiler. -/
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
    (alignment : Alignment adapter configuration) :
    IterationZeroProfile (loweringParameters adapter configuration) where
  toProfile := fullProfile
  fieldLaws := alignment.fieldLaws
  inverseLaw := alignment.inverseLaw
  iterationZeroFootprint := alignment.footprintExact

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
    (alignment : Alignment adapter configuration) :
    CallRecipe
      (signature (loweringParameters adapter configuration))
      ((profile adapter configuration fullProfile alignment).family
        (loweringParameters adapter configuration))
      Call.iterationZero :=
  iterationZeroRecipeForProfile
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
    (alignment : Alignment adapter configuration) :
    (signature (loweringParameters adapter configuration)).callFootprint
        Call.iterationZero =
      { recurringRows := 3
        temporaries := [auxiliaryLayout 1, auxiliaryLayout 1] } := by
  change configuration.footprints.iterationZero =
    { recurringRows := 3
      temporaries := [auxiliaryLayout 1, auxiliaryLayout 1] }
  rw [alignment.footprintExact]
  rfl

/-- Every occurrence emits exactly three rows, computed from the selected
zero-test program. -/
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
    (alignment : Alignment adapter configuration)
    {context :
      Nightstream.Implementation.Lowering.Typed.Schema
        (typeSystem (loweringParameters adapter configuration))}
    {references :
      Nightstream.Implementation.Lowering.Typed.Refs
        (typeSystem (loweringParameters adapter configuration))
        context
        ((signature (loweringParameters adapter configuration)).callInputs
          Call.iterationZero)}
    (frame :
      CallFrame
        (signature := signature
          (loweringParameters adapter configuration))
        ((profile adapter configuration fullProfile alignment).family
          (loweringParameters adapter configuration))
        Call.iterationZero references) :
    ((recipe adapter configuration fullProfile alignment).rows frame).length =
      3 := by
  rw [(recipe adapter configuration fullProfile alignment).rowCount frame]
  rw [selected_footprint_exact adapter configuration alignment]

/-- The emitted receipt contains exactly the output allocation, both declared
temporary bundles, and the three selected rows. -/
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
    (alignment : Alignment adapter configuration)
    {context :
      Nightstream.Implementation.Lowering.Typed.Schema
        (typeSystem (loweringParameters adapter configuration))}
    {references :
      Nightstream.Implementation.Lowering.Typed.Refs
        (typeSystem (loweringParameters adapter configuration))
        context
        ((signature (loweringParameters adapter configuration)).callInputs
          Call.iterationZero)}
    (frame :
      CallFrame
        (signature := signature
          (loweringParameters adapter configuration))
        ((profile adapter configuration fullProfile alignment).family
          (loweringParameters adapter configuration))
        Call.iterationZero references) :
    (recipe adapter configuration fullProfile alignment).receipt frame =
      { outputBundles := frame.outputs.portColumns
        temporaryBundles := frame.temporaries.bundleColumns
        rows :=
          (recipe adapter configuration fullProfile alignment).rows frame } :=
  CallRecipe.receipt_exact
    (recipe adapter configuration fullProfile alignment) frame

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionIterationZeroCallRecipe
