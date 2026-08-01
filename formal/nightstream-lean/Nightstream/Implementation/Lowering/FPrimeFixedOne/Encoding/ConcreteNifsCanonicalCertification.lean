import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalOperationalProfile
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsRustManifest
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsStepPaperRefinement
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ApplicationCodecRecovery
import Nightstream.Implementation.R1CS.Canonical.GoldilocksField
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.CanonicalOpeningSplitNc.SelectedVerifierRefinement

/-!
Contract: close the selected fixed-one NIFS certification and complete
application manifest from Lean-owned verifier data.

The deployment supplies only its true HyperNova application boundary:

* the Phase-4 application profile and its exact canonical NIFS codecs;
* one proof-carrying application `step` recipe;
* admissibility of the setup-selected default running value; and
* equality between the setup footprint slot and the Lean-derived NIFS
  footprint.

This module constructs the operational NIFS profile, Goldilocks arithmetic
support, complete NIFS recipe certification, eleven-call application
certification, and proof-free manifest.  No Rust row, measured cost,
acceptance proposition, transcript challenge, or generated artifact is an
input.

The selected canonical-opening refinement is exported here as application
construction evidence. It does not claim that an arbitrary supplied `step`
recipe emits those rows; the application recipe remains proof-carrying.

Emits constraints: none.  It closes and serializes existing Lean programs.
-/

set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 2200000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalCertification

open Nightstream.HyperNova.Construction2.Paper
open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalKey
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalOperationalProfile
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsParameters
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPlain270Profile
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Protocol.FPrime.ConcretePhi81
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc

private abbrev TranscriptState := Poseidon2Duplex.State

/-- Selected physical canonical-opening rows are canonical on the Split-NC
sound branch. The other branch retains the exact verifier security event.

This is the theorem an application-step certification consumes when it uses
the fixed-one selective CCS opening layout. -/
noncomputable def selectedCanonicalOpening_refines_or_securityEvent :=
  Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.CanonicalOpeningSplitNc.SelectedVerifierRefinement.selectedVerifierAndPhysicalRows_encoded_lt_modulus_or_securityEvent

section SelectedApplication

variable {Digest AppState Witness Encoded RunningWitness FreshWitness : Type}
variable [DecidableEq AppState] [DecidableEq Encoded]
variable {dimensions : Dimensions}
variable {verifierRows : Nat}
variable (setup : RelationSetup dimensions verifierRows)
variable (defaultRunning : Running dimensions verifierRows)
variable
  (machine :
    Machine
      (Key dimensions TranscriptState verifierRows)
      Digest AppState Witness
      (Running dimensions verifierRows)
      (Fresh dimensions verifierRows)
      Encoded 1)
variable
  (terminalRelations :
    TerminalRelations
      (Key dimensions TranscriptState verifierRows)
      (Running dimensions verifierRows)
      RunningWitness
      (Fresh dimensions verifierRows)
      FreshWitness 1)
variable
  (terminalChecks :
    Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.RelationChecks
      terminalRelations)
variable (widths : Widths) (footprints : Footprints)

local notation "Selected" =>
  ConcreteNifsPlain270Profile.selected dimensions
    (selectedKeys setup) defaultRunning machine terminalRelations
      terminalChecks widths footprints

private abbrev CanonicalApplication :=
  ConcreteNifsCanonicalOperationalProfile.Application
    setup defaultRunning machine terminalRelations terminalChecks
      widths footprints

/-- Complete deployment boundary after protocol-owned NIFS choices.

The footprint field is a static equality because `Footprints` is part of
HyperNova application setup.  The equality quantifies over every physical
call frame, so it cannot price one occurrence and emit another. -/
structure Deployment where
  application : CanonicalApplication setup defaultRunning machine
    terminalRelations terminalChecks widths footprints
  applicationCodecRecovery :
    ApplicationCodecRecovery Selected
      application.phase4.profile.codecs
  footprintExact :
    ConcreteNifsActivatedProgram.FootprintAlignment
      application.phase4.profile
      (ConcreteNifsCanonicalOperationalProfile.operational
        setup defaultRunning machine terminalRelations terminalChecks
          widths footprints application)
  step :
    CallRecipe (signature Selected)
      (application.phase4.profile.family Selected) Call.step
  defaultRunningAdmissible :
    ((application.phase4.profile.family Selected).codecFor
      (.data .running)).Admissible defaultRunning

/-- Complete selected NIFS call certification.

Primality and inversion are constructed from the Lean-owned Goldilocks
certificate.  Only setup footprint equality remains in the deployment
boundary. -/
noncomputable def nifs
    (deployment :
      Deployment setup defaultRunning machine terminalRelations
        terminalChecks widths footprints) :
    ConcreteNifsVerifyCallRecipe.Certification
      deployment.application.phase4.profile where
  operational :=
    ConcreteNifsCanonicalOperationalProfile.operational
      setup defaultRunning machine terminalRelations terminalChecks
        widths footprints deployment.application
  prime := GoldilocksField.goldilocks_euclidPrime
  field := GoldilocksField.goldilocksFieldInverse
  footprint := deployment.footprintExact

/-- Complete eleven-call Step and Terminal certification. -/
noncomputable def complete
    (deployment :
      Deployment setup defaultRunning machine terminalRelations
        terminalChecks widths footprints) :
    CompleteApplicationCertification Selected :=
  ConcreteNifsCompleteApplication.complete
    deployment.application.phase4
    (nifs setup defaultRunning machine terminalRelations terminalChecks
      widths footprints deployment)
    deployment.step
    deployment.defaultRunningAdmissible

/-- The canonical deployment carries the occurrence-bound paper event through
the exact recursive `nifsVerify` receipt in the complete Step program. -/
noncomputable def recursiveNifs_refinesPaper_or_boundEvent
    (deployment :
      Deployment setup defaultRunning machine terminalRelations
        terminalChecks widths footprints) :=
  ConcreteNifsStepPaperRefinement.recursiveNifs_refinesPaper_or_boundEvent
    deployment.application.phase4
    (nifs setup defaultRunning machine terminalRelations terminalChecks
      widths footprints deployment)
    deployment.step
    deployment.defaultRunningAdmissible

/-- Proof-free Rust-ready image of the complete Lean-owned program.

The application step remains an exact proof-carrying recipe inside this
manifest. -/
noncomputable def manifest
    (deployment :
      Deployment setup defaultRunning machine terminalRelations
        terminalChecks widths footprints) :
    ConcreteNifsRustManifest.Manifest :=
  ConcreteNifsRustManifest.manifest
    deployment.application.phase4
    (nifs setup defaultRunning machine terminalRelations terminalChecks
      widths footprints deployment)
    deployment.step
    deployment.defaultRunningAdmissible

@[simp] theorem nifs_operational
    (deployment :
      Deployment setup defaultRunning machine terminalRelations
        terminalChecks widths footprints) :
    (nifs setup defaultRunning machine terminalRelations terminalChecks
      widths footprints deployment).operational =
      ConcreteNifsCanonicalOperationalProfile.operational
        setup defaultRunning machine terminalRelations terminalChecks
          widths footprints deployment.application :=
  rfl

@[simp] theorem nifs_prime
    (deployment :
      Deployment setup defaultRunning machine terminalRelations
        terminalChecks widths footprints) :
    (nifs setup defaultRunning machine terminalRelations terminalChecks
      widths footprints deployment).prime =
      GoldilocksField.goldilocks_euclidPrime :=
  rfl

@[simp] theorem complete_step
    (deployment :
      Deployment setup defaultRunning machine terminalRelations
        terminalChecks widths footprints) :
    (complete setup defaultRunning machine terminalRelations terminalChecks
      widths footprints deployment).phase5.step = deployment.step :=
  rfl

@[simp] theorem manifest_profile
    (deployment :
      Deployment setup defaultRunning machine terminalRelations
        terminalChecks widths footprints) :
    (manifest setup defaultRunning machine terminalRelations terminalChecks
      widths footprints deployment).profile =
      ConcreteNifsRustManifest.profileIdentifier dimensions :=
  rfl

/-- Exact application-parametric Step cost.  Both terms are receipt folds
from the constructed program; the application term is not a supplied
number. -/
theorem manifest_stepCost_split
    (deployment :
      Deployment setup defaultRunning machine terminalRelations
        terminalChecks widths footprints) :
    let value :=
      manifest setup defaultRunning machine terminalRelations terminalChecks
        widths footprints deployment
    value.stepCost =
      value.fixedProtocolCost + value.applicationStepCost := by
  exact
    ConcreteNifsRustManifest.stepCost_eq_fixedProtocol_add_application
      deployment.application.phase4
      (nifs setup defaultRunning machine terminalRelations terminalChecks
        widths footprints deployment)
      deployment.step deployment.defaultRunningAdmissible

/-- The fixed protocol cost is the manifest's receipt-derived value. -/
noncomputable def fixedProtocolCost
    (deployment :
      Deployment setup defaultRunning machine terminalRelations
        terminalChecks widths footprints) : Cost :=
  (manifest setup defaultRunning machine terminalRelations terminalChecks
    widths footprints deployment).fixedProtocolCost

/-- The application term is exactly the selected physical `step` call cost. -/
theorem applicationStepCost_exact
    (deployment :
      Deployment setup defaultRunning machine terminalRelations
        terminalChecks widths footprints) :
    (manifest setup defaultRunning machine terminalRelations terminalChecks
      widths footprints deployment).applicationStepCost =
      (signature Selected).callCost Call.step := by
  exact
    ApplicationStepCostSplit.CompleteApplicationCertification.applicationStepCost_eq_callCost
      (complete setup defaultRunning machine terminalRelations terminalChecks
        widths footprints deployment)

end SelectedApplication

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalCertification
