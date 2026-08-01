import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CompleteApplicationCertification
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPlain270Profile
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsVerifyCallRecipe

/-!
Contract: instantiate the existing complete fixed-one assembly with the
Lean-owned selected `nifsVerify` recipe.

The application-owned `step` remains an explicit proof-carrying
`CallRecipe`, as required by HyperNova setup.  The selected NIFS verifier,
hash calls, direct calls, and the two independent terminal checks are all
constructed internally.

Emits constraints: no new rows.  This module assembles the existing recipes.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCompleteApplication

open Nightstream.HyperNova.Construction2.Paper
open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsParameters
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc

private abbrev TranscriptState := Poseidon2Duplex.State

section SelectedFrame

variable {shape : SemanticShape}
variable {Digest AppState Witness Encoded RunningWitness FreshWitness : Type}
variable [DecidableEq AppState] [DecidableEq Encoded]
variable {publicRingColumns verifierRows : Nat}
variable {publicFits :
  ringDegree * publicRingColumns <= shape.carrierWidth}
variable {keys : Fin 1 →
  SelectedKey shape TranscriptState publicRingColumns publicFits verifierRows}
variable {defaultRunning :
  SelectedRunning shape publicRingColumns publicFits verifierRows}
variable {machine :
  Machine
    (SelectedKey shape TranscriptState publicRingColumns publicFits
      verifierRows)
    Digest AppState Witness
    (SelectedRunning shape publicRingColumns publicFits verifierRows)
    (SelectedFresh shape publicRingColumns publicFits verifierRows)
    Encoded 1}
variable {terminalRelations :
  TerminalRelations
    (SelectedKey shape TranscriptState publicRingColumns publicFits
      verifierRows)
    (SelectedRunning shape publicRingColumns publicFits verifierRows)
    RunningWitness
    (SelectedFresh shape publicRingColumns publicFits verifierRows)
    FreshWitness 1}
variable {terminalChecks :
  Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.RelationChecks
    terminalRelations}
variable {widths : Widths} {footprints : Footprints}

local notation "Selected" =>
  ConcreteNifsParameters.selected keys defaultRunning machine
    terminalRelations terminalChecks widths footprints

/-- Complete protocol-owned assembly for every proof-carrying application
`step`.  No application semantics or numeric step cost is fabricated here. -/
def complete
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs :
      ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step :
      CallRecipe (signature Selected)
        (application.profile.family Selected) Call.step)
    (defaultRunningAdmissible :
      ((application.profile.family Selected).codecFor
        (.data .running)).Admissible
        defaultRunning) :
    CompleteApplicationCertification Selected where
  profile := application.profile
  runningCheck := application.runningCheck
  freshCheck := application.freshCheck
  phase5 := {
    step := step
    nifsVerify :=
      ConcreteNifsVerifyCallRecipe.recipe application.profile nifs
  }
  defaultRunningAdmissible := defaultRunningAdmissible

@[simp] theorem complete_step
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs :
      ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step :
      CallRecipe (signature Selected)
        (application.profile.family Selected) Call.step)
    (defaultRunningAdmissible :
      ((application.profile.family Selected).codecFor
        (.data .running)).Admissible
        defaultRunning) :
    (complete application nifs step defaultRunningAdmissible).phase5.step =
      step :=
  rfl

@[simp] theorem complete_nifsVerify
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs :
      ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step :
      CallRecipe (signature Selected)
        (application.profile.family Selected) Call.step)
    (defaultRunningAdmissible :
      ((application.profile.family Selected).codecFor
        (.data .running)).Admissible
        defaultRunning) :
    (complete application nifs step
      defaultRunningAdmissible).phase5.nifsVerify =
        ConcreteNifsVerifyCallRecipe.recipe application.profile nifs :=
  rfl

/-- The complete eleven-recipe family contains the selected NIFS program
definitionally. -/
@[simp] theorem allRecipes_nifsVerify
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs :
      ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step :
      CallRecipe (signature Selected)
        (application.profile.family Selected) Call.step)
    (defaultRunningAdmissible :
      ((application.profile.family Selected).codecFor
        (.data .running)).Admissible
        defaultRunning) :
    (complete application nifs step defaultRunningAdmissible
      ).allRecipes.recipe Call.nifsVerify =
        ConcreteNifsVerifyCallRecipe.recipe application.profile nifs :=
  rfl

/-- The complete eleven-recipe family retains the caller's certified
application step exactly once and unchanged. -/
@[simp] theorem allRecipes_step
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs :
      ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step :
      CallRecipe (signature Selected)
        (application.profile.family Selected) Call.step)
    (defaultRunningAdmissible :
      ((application.profile.family Selected).codecFor
        (.data .running)).Admissible
        defaultRunning) :
    (complete application nifs step defaultRunningAdmissible
      ).allRecipes.recipe Call.step = step :=
  rfl

end SelectedFrame

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCompleteApplication
