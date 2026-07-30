import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsParameters
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ApplicationCertification
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.Poseidon23SeparatorConformance
import Nightstream.Protocol.FPrime.ConcretePhi81.ProductionRelation
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.PiCcsDomains

/-!
Protocol-owned fixed-one/plain specialization of the concrete NIFS adapter.

Assurance tier: model-level.

Owns: selection of the one-fresh/fourteen-running production relation, the
five-ring 270-coordinate public carrier, and the verifier-owned fixed-point
FE/block-lane transcript domain.

Does not own: an application machine, terminal witnesses, an application
`step`, physical NIFS rows, Rust layouts, generated artifacts, or costs.

The 270-coordinate fact concerns the authoritative public carrier. It does not
replace the full relation witness domain with the smaller five-ring diagnostic
domain.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPlain270Profile

open Nightstream.HyperNova.Construction2.Paper
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsParameters
open Nightstream.Protocol.FPrime.ConcretePhi81
open Nightstream.Protocol.FPrime.ConcretePhi81.ProductionRelation
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270
open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.LogicalCarrier
open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.PiCcsSources
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier

/-- The exact production semantic shape: one fresh source, fourteen running
sources, and the full aligned relation witness. -/
abbrev Shape (dimensions : Dimensions) : SemanticShape :=
  ProductionRelation.Shape dimensions

/-- The five complete public Phi81 rings fit the selected full relation. -/
def publicFits (dimensions : Dimensions) :
    ringDegree * publicRingColumns <= (Shape dimensions).carrierWidth :=
  ProductionRelation.publicFits dimensions

/-- Static selected-key carrier at the exact production relation shape. -/
abbrev Key
    (dimensions : Dimensions)
    (TranscriptState : Type)
    (verifierRows : Nat) :=
  SelectedKey (Shape dimensions) TranscriptState publicRingColumns
    (publicFits dimensions) verifierRows

/-- Complete selected running carrier at the exact production relation shape. -/
abbrev Running
    (dimensions : Dimensions)
    (verifierRows : Nat) :=
  SelectedRunning (Shape dimensions) publicRingColumns
    (publicFits dimensions) verifierRows

/-- Selected fresh carrier at the exact production relation shape. -/
abbrev Fresh
    (dimensions : Dimensions)
    (verifierRows : Nat) :=
  SelectedFresh (Shape dimensions) publicRingColumns
    (publicFits dimensions) verifierRows

/-- Raw selected NIFS proof carrier at the exact production relation shape. -/
abbrev Proof
    (dimensions : Dimensions)
    (TranscriptState : Type)
    (verifierRows : Nat) :=
  SelectedProof (Shape dimensions) TranscriptState publicRingColumns
    (publicFits dimensions) verifierRows

/-- The fixed-one source arity is exactly one fresh plus fourteen running
sources. -/
theorem source_arity_exact (dimensions : Dimensions) :
    (Shape dimensions).freshCount = 1 /\
      (Shape dimensions).runningCount = 14 /\
      (Shape dimensions).sourceCount = 15 :=
  ProductionRelation.sourceArity_exact dimensions

/-- The authoritative public carrier is exactly five Phi81 rings, hence 270
field coordinates. -/
theorem public_carrier_exact (dimensions : Dimensions) :
    ringDegree * publicRingColumns = 270 /\
      (RelationShape (Shape dimensions) publicRingColumns
        (publicFits dimensions)).publicWidth = 270 := by
  constructor
  · decide
  · exact ProductionRelation.publicWidth_eq dimensions

/-- The fresh public source is exactly the legacy 257-coordinate input
followed by thirteen verifier-owned zero coordinates. -/
theorem fresh_public_input_exact
    (dimensions : Dimensions)
    (legacy : LegacyAssignment dimensions) :
    sourcePublicInput publicRingColumns (publicFits dimensions)
        (assignment dimensions legacy) =
      expectedPublicInput dimensions legacy :=
  ProductionRelation.freshPublicInput_exact dimensions legacy

/-- Running sources retain the complete relation assignment. In particular no
257-coordinate projection is used after folding. -/
theorem running_assignment_exact
    (dimensions : Dimensions)
    (inputs : Inputs dimensions 1 14)
    (source : Fin 14) :
    inputs.data.assignment (Data.runningIndex source) =
      inputs.runningAssignments source :=
  ProductionRelation.runningAssignment_exact dimensions inputs source

/-- The thirteen fresh completion coordinates, in their authoritative carrier
order. This list is derived from the logical 257-to-270 encoder. -/
def freshCompletionCoordinates
    (dimensions : Dimensions)
    (external : ExternalInput) : List F :=
  List.ofFn fun padding : Fin fixedPaddingWidth =>
    paddingValue dimensions (encodeFresh dimensions external) padding

@[simp] theorem freshCompletionCoordinates_length
    (dimensions : Dimensions)
    (external : ExternalInput) :
    (freshCompletionCoordinates dimensions external).length = 13 := by
  simp [freshCompletionCoordinates, fixedPaddingWidth]

/-- Fresh construction fixes exactly all thirteen completion coordinates to
zero. -/
theorem freshCompletionCoordinates_eq_zeros
    (dimensions : Dimensions)
    (external : ExternalInput) :
    freshCompletionCoordinates dimensions external =
      List.replicate 13 0 := by
  apply List.ext_get
  · simp [freshCompletionCoordinates, fixedPaddingWidth]
  · intro index left right
    have leftEq :
        (freshCompletionCoordinates dimensions external).get
            ⟨index, left⟩ = 0 := by
      simp [freshCompletionCoordinates, paddingValue]
    have indexLt : index < 13 := by simpa using right
    have rightEq :
        (List.replicate 13 (0 : F)).get ⟨index, right⟩ = 0 := by
      simpa only [List.get_eq_getElem] using
        (List.getElem_replicate
          (a := (0 : F)) (n := 13) (i := index) indexLt)
    exact leftEq.trans rightEq.symm

/-- The full running carrier admits a legitimate nonzero completion tail that
the 257-coordinate external view cannot observe. -/
theorem running_tail_nonzero_witness (dimensions : Dimensions) :
    exists input : LIn dimensions,
      projectExternal dimensions input =
          projectExternal dimensions (zeroLIn dimensions) /\
        paddingValue dimensions input firstPadding = 1 := by
  refine ⟨firstPaddingOne dimensions,
    projectExternal_firstPaddingOne_eq_zero dimensions, ?_⟩
  simp [paddingValue, firstPaddingOne]

/-- The selected sampler contains a valid shift that moves external
coordinate 256 into running coordinate 257. -/
theorem sampler_shift_256_to_257 (dimensions : Dimensions) :
    PiRLCAlgebra.Challenge.challengeValid shiftChallenge /\
      PiRLCAlgebra.PublicInput.publicAct shiftChallenge
          (encodeFresh dimensions finalExternalOne)
          (firstPaddingColumn dimensions) =
        1 :=
  ⟨shiftChallenge_valid, shift_enters_first_padding dimensions⟩

/-- Application-owned Phase-4 data at the exact protocol boundary.

The profile remains application-selected, as HyperNova requires. The wrapper
adds only the proof that its existing hash plan is a `SeparatingPlan`; it does
not strengthen `CoordinatePlan` or select an application.

This structure does not claim that arbitrary `Parameters` have the production
shape. The fixed-one/plain/270 use applies it to `selected ...`, which
constructs those parameters from the protocol-owned relation above. -/
structure Phase4Application (parameters : Parameters) where
  profile : Poseidon23ApplicationProfile parameters
  separating :
    Poseidon23SeparatorConformance.SeparatingPlan
      profile.codecs profile.alignmentWidth
  separatingPlan_eq_hashPlan :
    separating.plan = profile.hashPlan

namespace Phase4Application

/-- The complete four-call Phase-4 certification. Both terminal calls are
constructed from the same proof-carrying application profile but remain
independent calls. -/
def certification
    {parameters : Parameters}
    (phase4 : Phase4Application parameters) :
    ApplicationCertification parameters :=
  ApplicationCertification.poseidon23 parameters phase4.profile

@[simp] theorem certification_runningCheck
    {parameters : Parameters}
    (phase4 : Phase4Application parameters) :
    phase4.certification.runningCheck =
      RunningCheckRecipe.recipe parameters
        phase4.profile.toTerminalEqualityProfile :=
  rfl

@[simp] theorem certification_freshCheck
    {parameters : Parameters}
    (phase4 : Phase4Application parameters) :
    phase4.certification.freshCheck =
      FreshCheckRecipe.recipe parameters
        phase4.profile.toTerminalEqualityProfile :=
  rfl

/-- The selected binding preimage has exactly 23 coordinates. -/
theorem selected_preimage_length
    {parameters : Parameters}
    (phase4 : Phase4Application parameters)
    (source : List Field) :
    (Poseidon23Hash.select source
      phase4.profile.hashPlan.preimage).length = 23 := by
  exact Poseidon23Hash.select_length source
    phase4.profile.hashPlan.preimage

/-- For fixed payload operands, prior and next modes differ only at slots that
read the normalized iteration coordinate. This is a mode-separation theorem,
not a link between the real F-prime prior and next calls, whose current and
running operands can differ. -/
theorem same_payload_next_preimage_is_separated
    {parameters : Parameters}
    (phase4 : Phase4Application parameters)
    (iteration : Nat)
    (z0 current : parameters.State)
    (running : parameters.Running)
    (slot : Fin 23) :
    (Poseidon23Hash.select
        (Poseidon23Hash.sourceCoordinates phase4.profile.codecs true
          iteration z0 current running)
        phase4.profile.hashPlan.preimage).getD slot.val 0 =
      if (phase4.profile.hashPlan.preimage slot).val = 0
      then Poseidon23Hash.normalizedIteration true iteration
      else
        (Poseidon23Hash.select
          (Poseidon23Hash.sourceCoordinates phase4.profile.codecs false
            iteration z0 current running)
          phase4.profile.hashPlan.preimage).getD slot.val 0 := by
  simpa [phase4.separatingPlan_eq_hashPlan] using
    phase4.separating.next_is_separated iteration z0 current running slot

/-- Phase 4 keeps the two unary terminal checks as distinct physical calls. -/
theorem terminal_calls_independent
    {parameters : Parameters}
    (_phase4 : Phase4Application parameters) :
    Call.runningCheck ≠ Call.freshCheck :=
  ApplicationCertification.terminal_calls_distinct

/-- The Phase-3/4 four-call cost is the receipt-derived cost of the four
constructed recipes. No separate cross-call equal-tail program is valid or
included. -/
theorem cost_exact
    {parameters : Parameters}
    (phase4 : Phase4Application parameters) :
    ApplicationCertification.phase34Cost parameters phase4.certification =
      ⟨(2 * phase4.profile.alignmentWidth +
            phase4.profile.alignmentWidth.pred + 2503) +
          (2 * phase4.profile.alignmentWidth +
            phase4.profile.alignmentWidth.pred + 2503) +
          (2 * phase4.profile.codecs.running.width +
            phase4.profile.codecs.running.width.pred + 1) +
          (2 * phase4.profile.codecs.fresh.width +
            phase4.profile.codecs.fresh.width.pred + 1),
        0,
        5,
        (2 * phase4.profile.alignmentWidth +
            phase4.profile.alignmentWidth.pred + 2499) +
          (2 * phase4.profile.alignmentWidth +
            phase4.profile.alignmentWidth.pred + 2494) +
          (2 * phase4.profile.codecs.running.width +
            phase4.profile.codecs.running.width.pred + 1) +
          (2 * phase4.profile.codecs.fresh.width +
            phase4.profile.codecs.fresh.width.pred + 1)⟩ :=
  ApplicationCertification.phase34Cost_exact parameters phase4.profile

end Phase4Application

/-- The operational verifier uses the full fixed-point FE domain, not the
five-ring public-prefix diagnostic domain. -/
theorem operational_fe_domain_exact :
    PiCcsDomains.production.fe.columnVariables = 25 /\
      PiCcsDomains.production.fe.laneVariables = 6 := by
  exact ⟨rfl, rfl⟩

/-- The operational verifier uses the full fixed-point block/lane NC domain. -/
theorem operational_nc_domain_exact :
    PiCcsDomains.production.nc.blockVariables = 19 /\
      PiCcsDomains.production.nc.laneVariables = 6 := by
  exact ⟨rfl, rfl⟩

/-- Construct the fixed-one lowering vocabulary at the selected 270-public
production relation. Application semantics and terminal relations remain
proof-carrying parameters, as required by HyperNova setup. -/
def selected
    {TranscriptState Digest AppState Witness Encoded
      RunningWitness FreshWitness : Type}
    [DecidableEq AppState]
    [DecidableEq Encoded]
    {verifierRows : Nat}
    (dimensions : Dimensions)
    (keys : Fin 1 -> Key dimensions TranscriptState verifierRows)
    (defaultRunning : Running dimensions verifierRows)
    (machine :
      Machine
        (Key dimensions TranscriptState verifierRows)
        Digest AppState Witness
        (Running dimensions verifierRows)
        (Fresh dimensions verifierRows)
        Encoded 1)
    (terminalRelations :
      TerminalRelations
        (Key dimensions TranscriptState verifierRows)
        (Running dimensions verifierRows)
        RunningWitness
        (Fresh dimensions verifierRows)
        FreshWitness 1)
    (terminalChecks :
      Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.RelationChecks
        terminalRelations)
    (widths : Widths)
    (footprints : Footprints) :
    Parameters :=
  ConcreteNifsParameters.selected keys defaultRunning machine
    terminalRelations terminalChecks widths footprints

@[simp] theorem selected_setup_nifs
    {TranscriptState Digest AppState Witness Encoded
      RunningWitness FreshWitness : Type}
    [DecidableEq AppState]
    [DecidableEq Encoded]
    {verifierRows : Nat}
    (dimensions : Dimensions)
    (keys : Fin 1 -> Key dimensions TranscriptState verifierRows)
    (defaultRunning : Running dimensions verifierRows)
    (machine :
      Machine
        (Key dimensions TranscriptState verifierRows)
        Digest AppState Witness
        (Running dimensions verifierRows)
        (Fresh dimensions verifierRows)
        Encoded 1)
    (terminalRelations :
      TerminalRelations
        (Key dimensions TranscriptState verifierRows)
        (Running dimensions verifierRows)
        RunningWitness
        (Fresh dimensions verifierRows)
        FreshWitness 1)
    (terminalChecks :
      Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.RelationChecks
        terminalRelations)
    (widths : Widths)
    (footprints : Footprints) :
    (selected dimensions keys defaultRunning machine terminalRelations
      terminalChecks widths footprints).setup.nifs =
      ConcreteNifsParameters.nifsVerifier :=
  rfl

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPlain270Profile
