import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Fe
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Nc.BlockLane
import Nightstream.SuperNeo.SumCheck.FixedPhase.SemanticView

/-!
Assignment-indexed semantic views of the actual Split-NC FE and block/lane-NC
wire transcripts.

Assurance tier: model-level.

Owns: separate FE and NC wire projections, reconstruction of their generic
`SumCheck.Instance` views from the independent protocol polynomials,
acceptance transport under exact source-bound terminal identities, independent
truth paths, and the two semantic claim implications.

Does not own: construction of `Sources.Data` from extracted assignments,
input/output commitment authority, challenge generation, mixing-root
probabilities, Fiat--Shamir, Rust, R1CS, costs, or rows.

The production certificates remain message-only. `trueInitial`, terminal
semantics, and expected rounds are recomputed only in the proof-side views.

Emits constraints: no.

| Stage path | Owned equation | Authority |
|---|---|---|
| `piccs.split_nc.semantic_view.fe` | FE wire replay equals the assignment-indexed FE truth path | derived |
| `piccs.split_nc.semantic_view.nc` | NC wire replay equals the independent assignment-indexed NC truth path | derived |
-/

set_option autoImplicit false

namespace Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.SemanticAdapter

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.SumCheck.Finite

private abbrev ops := ConcreteCarrier.extensionOps

/-! ## FE -/

/-- Ghost-free verifier data for the physical mixed-width FE certificate. -/
def feWire
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {input : PublicInput shape}
    (profile : Polynomial.Fe.SupportedProfile shape domain)
    (coins : Polynomial.Fe.Coins shape domain)
    (point : Polynomial.Fe.Point shape domain)
    (message : OutputMessage shape)
    (certificate : Fe.Certificate input domain)
    (challengeSetSize : Nat) :
    Nightstream.SuperNeo.SumCheck.Finite.FixedPhase.SemanticView.Wire
      K (Fe.Drow input) where
  initial := Polynomial.Fe.initial profile input coins
  terminal :=
    Polynomial.Fe.terminalFromMessage profile input coins point message
  challenges := point.coordinates
  certificate := { rounds := certificate.uniformRounds }
  challengeSetSize := challengeSetSize

/-- Recompute the FE semantic ghosts from the assignment-indexed FE
polynomial. -/
def feInstance
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (profile : Polynomial.Fe.SupportedProfile shape domain)
    (data : Data shape)
    (coins : Polynomial.Fe.Coins shape domain)
    (point : Polynomial.Fe.Point shape domain)
    (message : OutputMessage shape)
    (certificate : Fe.Certificate (PublicInput.ofSources data) domain)
    (challengeSetSize : Nat) :
    Nightstream.SuperNeo.SumCheck.Instance K K :=
  Nightstream.SuperNeo.SumCheck.Finite.FixedPhase.SemanticView.semanticInstance
    ops.toOps
    (Polynomial.Fe.InitialSum.sumcheckPolynomial profile data coins)
    (feWire profile coins point message certificate challengeSetSize)

/-- The FE wire predicate is definitionally the actual claimed-chain
interface. -/
theorem feWireAccepted_iff
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {input : PublicInput shape}
    (profile : Polynomial.Fe.SupportedProfile shape domain)
    (coins : Polynomial.Fe.Coins shape domain)
    (point : Polynomial.Fe.Point shape domain)
    (message : OutputMessage shape)
    (certificate : Fe.Certificate input domain)
    (challengeSetSize : Nat) :
    Nightstream.SuperNeo.SumCheck.Finite.FixedPhase.SemanticView.Accepted
        ops.toOps
        (feWire profile coins point message certificate challengeSetSize) ↔
      Fe.Accepted
        (Polynomial.Fe.initial profile input coins)
        (Polynomial.Fe.terminalFromMessage profile input coins point message)
        point certificate := by
  rfl

/-- Full source binding identifies the verifier-computed FE terminal with the
assignment-indexed polynomial terminal. -/
theorem feTerminalBinding
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (profile : Polynomial.Fe.SupportedProfile shape domain)
    (data : Data shape)
    (coins : Polynomial.Fe.Coins shape domain)
    (point : Polynomial.Fe.Point shape domain)
    (message : OutputMessage shape)
    (bound : message.yRing = Polynomial.Fe.sourceYRingAt data point.row) :
    Polynomial.Fe.terminalFromMessage profile (PublicInput.ofSources data)
        coins point message =
      Polynomial.Fe.InitialSum.sumcheckPolynomial profile data coins
        point.coordinates := by
  calc
    Polynomial.Fe.terminalFromMessage profile (PublicInput.ofSources data)
        coins point message =
        Polynomial.Fe.qAtPoint profile data coins point :=
      Polynomial.Fe.terminalFromMessage_eq_qAtPoint_of_yRing_eq
        profile data coins point message bound
    _ = Polynomial.Fe.InitialSum.sumcheckPolynomial profile data coins
          point.coordinates :=
      (Polynomial.Fe.InitialSum.sumcheckPolynomial_coordinates_eq_qAtPoint
        profile data coins point).symm

/-- Actual FE acceptance transports to the generic symbolic instance without
a caller-supplied semantic ghost. -/
theorem feAccepted_implies_genericAccepted
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (profile : Polynomial.Fe.SupportedProfile shape domain)
    (data : Data shape)
    (coins : Polynomial.Fe.Coins shape domain)
    (point : Polynomial.Fe.Point shape domain)
    (message : OutputMessage shape)
    (certificate : Fe.Certificate (PublicInput.ofSources data) domain)
    (challengeSetSize : Nat)
    (accepted :
      Fe.Accepted
        (Polynomial.Fe.initial profile (PublicInput.ofSources data) coins)
        (Polynomial.Fe.terminalFromMessage profile
          (PublicInput.ofSources data) coins point message)
        point certificate)
    (bound : message.yRing = Polynomial.Fe.sourceYRingAt data point.row) :
    Nightstream.SuperNeo.SumCheck.Accepted ops.toOps.toSymbolic
      (feInstance profile data coins point message certificate
        challengeSetSize) := by
  apply
    Nightstream.SuperNeo.SumCheck.Finite.FixedPhase.SemanticView.accepted_implies_symbolicAccepted
  · exact (feWireAccepted_iff profile coins point message certificate
      challengeSetSize).2 accepted
  · exact feTerminalBinding profile data coins point message bound

/-- The FE expected rounds form an independent generic truth path. -/
theorem feAccepted_implies_truthPath
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (profile : Polynomial.Fe.SupportedProfile shape domain)
    (data : Data shape)
    (coins : Polynomial.Fe.Coins shape domain)
    (point : Polynomial.Fe.Point shape domain)
    (message : OutputMessage shape)
    (certificate : Fe.Certificate (PublicInput.ofSources data) domain)
    (challengeSetSize : Nat)
    (accepted :
      Fe.Accepted
        (Polynomial.Fe.initial profile (PublicInput.ofSources data) coins)
        (Polynomial.Fe.terminalFromMessage profile
          (PublicInput.ofSources data) coins point message)
        point certificate)
    (bound : message.yRing = Polynomial.Fe.sourceYRingAt data point.row) :
    Nightstream.SuperNeo.SumCheck.TruthPath ops.toOps.toSymbolic
      (feInstance profile data coins point message certificate
        challengeSetSize) := by
  apply
    Nightstream.SuperNeo.SumCheck.Finite.FixedPhase.SemanticView.accepted_implies_truthPath
  · exact (feWireAccepted_iff profile coins point message certificate
      challengeSetSize).2 accepted
  · exact feTerminalBinding profile data coins point message bound

/-- Independent FE truth proves the recomputed initial claim. -/
theorem feClaimTrue_of_truth
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (profile : Polynomial.Fe.SupportedProfile shape domain)
    (data : Data shape)
    (coins : Polynomial.Fe.Coins shape domain)
    (point : Polynomial.Fe.Point shape domain)
    (message : OutputMessage shape)
    (certificate : Fe.Certificate (PublicInput.ofSources data) domain)
    (challengeSetSize : Nat)
    (truth : Semantics.Fe.Truth data) :
    Nightstream.SuperNeo.SumCheck.Claim.True
      (feInstance profile data coins point message certificate
        challengeSetSize) := by
  apply
    (Nightstream.SuperNeo.SumCheck.Finite.FixedPhase.SemanticView.claimTrue_iff
      ops.toOps
      (Polynomial.Fe.InitialSum.sumcheckPolynomial profile data coins)
      (feWire profile coins point message certificate challengeSetSize)).2
  change
    Polynomial.Fe.initial profile (PublicInput.ofSources data) coins =
      FixedPhase.semanticInitial ops.toOps
        (Polynomial.Fe.InitialSum.sumcheckPolynomial profile data coins)
        point.coordinates.length
  rw [Polynomial.Fe.InitialSum.CarriedBridge.initial_eq_sumcheckHypercubeSum_of_truth
    profile data coins truth]
  unfold Polynomial.Fe.InitialSum.sumcheckHypercubeSum
    FixedPhase.semanticInitial
  rw [point.coordinates_length]

/-! ## canonical block/lane NC -/

/-- Ghost-free verifier data for the physical block/lane NC certificate. -/
def ncWire
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (coins : Polynomial.Nc.BlockLane.Mixing.Coins domain)
    (point : Polynomial.Nc.BlockLane.Point domain)
    (message : OutputMessage shape)
    (certificate : Nc.Certificate)
    (challengeSetSize : Nat) :
    Nightstream.SuperNeo.SumCheck.Finite.FixedPhase.SemanticView.Wire
      K Polynomial.Nc.Degree.ncSumcheckDegreeBound where
  initial := Polynomial.Nc.BlockLane.InitialSum.claimedInitial
  terminal :=
    Polynomial.Nc.BlockLane.Terminal.terminalFromMessage message coins point
  challenges := point.coordinates
  certificate := certificate
  challengeSetSize := challengeSetSize

/-- Recompute the NC semantic ghosts from the assignment-indexed block/lane
polynomial. -/
def ncInstance
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Polynomial.Nc.BlockLane.Mixing.Coins domain)
    (point : Polynomial.Nc.BlockLane.Point domain)
    (message : OutputMessage shape)
    (certificate : Nc.Certificate)
    (challengeSetSize : Nat) :
    Nightstream.SuperNeo.SumCheck.Instance K K :=
  Nightstream.SuperNeo.SumCheck.Finite.FixedPhase.SemanticView.semanticInstance
    ops.toOps
    (Polynomial.Nc.BlockLane.InitialSum.sumcheckPolynomial covers data coins)
    (ncWire coins point message certificate challengeSetSize)

/-- The NC wire predicate is definitionally the actual claimed-chain
interface. -/
theorem ncWireAccepted_iff
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (coins : Polynomial.Nc.BlockLane.Mixing.Coins domain)
    (point : Polynomial.Nc.BlockLane.Point domain)
    (message : OutputMessage shape)
    (certificate : Nc.Certificate)
    (challengeSetSize : Nat) :
    Nightstream.SuperNeo.SumCheck.Finite.FixedPhase.SemanticView.Accepted
        ops.toOps
        (ncWire coins point message certificate challengeSetSize) ↔
      Nc.Accepted Polynomial.Nc.BlockLane.InitialSum.claimedInitial
        point.coordinates
        (Polynomial.Nc.BlockLane.Terminal.terminalFromMessage
          message coins point)
        certificate := by
  rfl

/-- Full packed-output binding identifies the verifier-computed NC terminal
with the assignment-indexed polynomial terminal. -/
theorem ncTerminalBinding
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Polynomial.Nc.BlockLane.Mixing.Coins domain)
    (point : Polynomial.Nc.BlockLane.Point domain)
    (message : OutputMessage shape)
    (bound :
      Polynomial.Nc.BlockLane.Terminal.PackedYZcolBoundAtBlock
        covers data point.block message) :
    Polynomial.Nc.BlockLane.Terminal.terminalFromMessage message coins point =
      Polynomial.Nc.BlockLane.InitialSum.sumcheckPolynomial covers data coins
        point.coordinates := by
  calc
    Polynomial.Nc.BlockLane.Terminal.terminalFromMessage message coins point =
        Polynomial.Nc.BlockLane.Mixing.qAtPoint covers data coins point :=
      Polynomial.Nc.BlockLane.Terminal.terminal_eq_qAtPoint_of_bound
        covers data coins point message bound
    _ = Polynomial.Nc.BlockLane.InitialSum.sumcheckPolynomial
          covers data coins point.coordinates :=
      (Polynomial.Nc.BlockLane.InitialSum.sumcheckPolynomial_coordinates_eq_qAtPoint
        covers data coins point).symm

/-- Actual block/lane NC acceptance transports to the generic symbolic
instance without a caller-supplied semantic ghost. -/
theorem ncAccepted_implies_genericAccepted
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Polynomial.Nc.BlockLane.Mixing.Coins domain)
    (point : Polynomial.Nc.BlockLane.Point domain)
    (message : OutputMessage shape)
    (certificate : Nc.Certificate)
    (challengeSetSize : Nat)
    (accepted :
      Nc.Accepted Polynomial.Nc.BlockLane.InitialSum.claimedInitial
        point.coordinates
        (Polynomial.Nc.BlockLane.Terminal.terminalFromMessage
          message coins point)
        certificate)
    (bound :
      Polynomial.Nc.BlockLane.Terminal.PackedYZcolBoundAtBlock
        covers data point.block message) :
    Nightstream.SuperNeo.SumCheck.Accepted ops.toOps.toSymbolic
      (ncInstance covers data coins point message certificate
        challengeSetSize) := by
  apply
    Nightstream.SuperNeo.SumCheck.Finite.FixedPhase.SemanticView.accepted_implies_symbolicAccepted
  · exact (ncWireAccepted_iff coins point message certificate
      challengeSetSize).2 accepted
  · exact ncTerminalBinding covers data coins point message bound

/-- The NC expected rounds form an independent generic truth path. -/
theorem ncAccepted_implies_truthPath
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Polynomial.Nc.BlockLane.Mixing.Coins domain)
    (point : Polynomial.Nc.BlockLane.Point domain)
    (message : OutputMessage shape)
    (certificate : Nc.Certificate)
    (challengeSetSize : Nat)
    (accepted :
      Nc.Accepted Polynomial.Nc.BlockLane.InitialSum.claimedInitial
        point.coordinates
        (Polynomial.Nc.BlockLane.Terminal.terminalFromMessage
          message coins point)
        certificate)
    (bound :
      Polynomial.Nc.BlockLane.Terminal.PackedYZcolBoundAtBlock
        covers data point.block message) :
    Nightstream.SuperNeo.SumCheck.TruthPath ops.toOps.toSymbolic
      (ncInstance covers data coins point message certificate
        challengeSetSize) := by
  apply
    Nightstream.SuperNeo.SumCheck.Finite.FixedPhase.SemanticView.accepted_implies_truthPath
  · exact (ncWireAccepted_iff coins point message certificate
      challengeSetSize).2 accepted
  · exact ncTerminalBinding covers data coins point message bound

/-- Independent full-carrier NC truth proves the recomputed zero initial
claim. -/
theorem ncClaimTrue_of_truth
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Polynomial.Nc.BlockLane.Mixing.Coins domain)
    (point : Polynomial.Nc.BlockLane.Point domain)
    (message : OutputMessage shape)
    (certificate : Nc.Certificate)
    (challengeSetSize : Nat)
    (truth : Semantics.Nc.Truth data) :
    Nightstream.SuperNeo.SumCheck.Claim.True
      (ncInstance covers data coins point message certificate
        challengeSetSize) := by
  apply
    (Nightstream.SuperNeo.SumCheck.Finite.FixedPhase.SemanticView.claimTrue_iff
      ops.toOps
      (Polynomial.Nc.BlockLane.InitialSum.sumcheckPolynomial covers data coins)
      (ncWire coins point message certificate challengeSetSize)).2
  change
    Polynomial.Nc.BlockLane.InitialSum.claimedInitial =
      FixedPhase.semanticInitial ops.toOps
        (Polynomial.Nc.BlockLane.InitialSum.sumcheckPolynomial
          covers data coins)
        point.coordinates.length
  rw [Polynomial.Nc.BlockLane.InitialSum.claimedInitial_eq_sumcheckHypercubeSum_of_truth
    covers data coins truth]
  unfold Polynomial.Nc.BlockLane.InitialSum.sumcheckHypercubeSum
    FixedPhase.semanticInitial
  rw [point.coordinates_length]

end Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.SemanticAdapter
