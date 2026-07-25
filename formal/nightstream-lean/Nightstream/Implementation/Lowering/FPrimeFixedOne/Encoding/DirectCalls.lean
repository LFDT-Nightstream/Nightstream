import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls.EncodedEqual
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls.EncodeInstance
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls.FreshPublic
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls.IterationZero
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls.StateEqual

/-!
Contract: the exact certified subset of direct fixed-one call recipes.

This module packages only calls whose selected physical row programs already
have artifact-independent soundness, honest completeness, exact ownership,
and exact footprint proofs.  It intentionally does not construct a
`CallRecipes`: application, the two hashes, NIFS verification, and the two
terminal checks remain separate refinement obligations.
-/

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls

open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary

/-- The five call recipes currently selected and certified by the direct
fixed-one encoding. -/
structure CertifiedSubset
    (parameters : Parameters)
    (profile : DirectProfile parameters) where
  iterationZero :
    CallRecipe (signature parameters) profile.family Call.iterationZero
  stateEqual :
    CallRecipe (signature parameters) profile.family Call.stateEqual
  freshPublic :
    CallRecipe (signature parameters) profile.family Call.freshPublic
  encodeInstance :
    CallRecipe (signature parameters) profile.family Call.encodeInstance
  encodedEqual :
    CallRecipe (signature parameters) profile.family Call.encodedEqual

/-- Exact package of the five kernel-checked recipes. -/
def certifiedSubset
    (parameters : Parameters)
    (profile : DirectProfile parameters) :
    CertifiedSubset parameters profile where
  iterationZero := iterationZeroRecipe parameters profile
  stateEqual := stateEqualRecipe parameters profile
  freshPublic := freshPublicRecipe parameters profile
  encodeInstance := encodeInstanceRecipe parameters profile
  encodedEqual := encodedEqualRecipe parameters profile

/-- Exactly the six recipes not yet constructed by `certifiedSubset`.

Keeping this boundary typed prevents a caller from replacing any missing
operation with a generic acceptance proposition. -/
structure RemainingRecipes
    (parameters : Parameters)
    (profile : DirectProfile parameters) where
  step :
    CallRecipe (signature parameters) profile.family Call.step
  hashPrior :
    CallRecipe (signature parameters) profile.family Call.hashPrior
  hashNext :
    CallRecipe (signature parameters) profile.family Call.hashNext
  nifsVerify :
    CallRecipe (signature parameters) profile.family Call.nifsVerify
  runningCheck :
    CallRecipe (signature parameters) profile.family Call.runningCheck
  freshCheck :
    CallRecipe (signature parameters) profile.family Call.freshCheck

/-- A complete recipe family assembled only after all six remaining typed
operations have their own certified physical recipes. -/
def allRecipes
    (parameters : Parameters)
    (profile : DirectProfile parameters)
    (remaining : RemainingRecipes parameters profile) :
    CallRecipes (signature parameters) profile.family where
  recipe
    | .iterationZero => iterationZeroRecipe parameters profile
    | .stateEqual => stateEqualRecipe parameters profile
    | .step => remaining.step
    | .hashPrior => remaining.hashPrior
    | .hashNext => remaining.hashNext
    | .freshPublic => freshPublicRecipe parameters profile
    | .encodeInstance => encodeInstanceRecipe parameters profile
    | .encodedEqual => encodedEqualRecipe parameters profile
    | .nifsVerify => remaining.nifsVerify
    | .runningCheck => remaining.runningCheck
    | .freshCheck => remaining.freshCheck

/-- Calls still requiring selected physical recipes before a complete
`CallRecipes` value can be constructed. -/
def remainingCalls : List Call :=
  [.step, .hashPrior, .hashNext, .nifsVerify, .runningCheck, .freshCheck]

theorem remainingCalls_exact :
    remainingCalls =
      [.step, .hashPrior, .hashNext, .nifsVerify,
        .runningCheck, .freshCheck] :=
  rfl

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls
