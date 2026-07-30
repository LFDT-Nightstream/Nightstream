import Nightstream.Implementation.R1CS.Canonical.KSplitNcEndpointSupport

/-!
Contract: whole-program column support for the selected Split-NC endpoint
composition.

The three endpoint emitters use one contiguous allocation, but their semantic
inputs are shared transcript outputs and authoritative call-frame reads. This
module proves that if those inputs precede the endpoint base, then every row
of the composed endpoint program stays below the exact end of its allocation.
It emits no rows and does not infer source authority from column numbers.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace Nightstream.Implementation.R1CS.Canonical.KSplitNcEndpointsSupport

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.KCompositeRowSupport
open Nightstream.Implementation.R1CS.Canonical.KSplitNcEndpointSupport
open Nightstream.Implementation.R1CS.Canonical.KMul
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.TranscriptAuthority.BlockLane

/-- Exact external-source obligation of the endpoint composition.

Every field is a column-support fact, never a decoded equation or an
acceptance proposition. Duplicated transcript values remain explicit because
the three subprograms consume them through different typed inputs. -/
structure InputsBelow
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (input : KSplitNcEndpoints.Input polynomialInput domains) : Prop where
  feInitialGamma :
    CarriedBelow (KSplitNcEndpoints.feInitialInput input).gamma input.frameBase
  feInitialAlpha :
    ∀ coordinate,
      CarriedBelow
        ((KSplitNcEndpoints.feInitialInput input).alpha coordinate)
        input.frameBase
  feInitialClaims :
    ∀ running matrix lane,
      CarriedBelow
        ((KSplitNcEndpoints.feInitialInput input).claimedYRing
          running matrix lane)
        input.frameBase
  feInitialEndpoint :
    CarriedBelow
      (KSplitNcEndpoints.feInitialInput input).initial input.frameBase
  feTerminalGamma :
    CarriedBelow (KSplitNcEndpoints.feTerminalInput input).gamma input.frameBase
  feTerminalAlpha :
    ∀ coordinate,
      CarriedBelow
        ((KSplitNcEndpoints.feTerminalInput input).alpha coordinate)
        input.frameBase
  feTerminalBetaA :
    ∀ coordinate,
      CarriedBelow
        ((KSplitNcEndpoints.feTerminalInput input).betaA coordinate)
        input.frameBase
  feTerminalBetaR :
    ∀ coordinate,
      CarriedBelow
        ((KSplitNcEndpoints.feTerminalInput input).betaR coordinate)
        input.frameBase
  feTerminalPointLane :
    ∀ coordinate,
      CarriedBelow
        ((KSplitNcEndpoints.feTerminalInput input).pointLane coordinate)
        input.frameBase
  feTerminalPointRow :
    ∀ coordinate,
      CarriedBelow
        ((KSplitNcEndpoints.feTerminalInput input).pointRow coordinate)
        input.frameBase
  feTerminalPriorPoint :
    ∀ coordinate,
      CarriedBelow
        ((KSplitNcEndpoints.feTerminalInput input).priorPoint coordinate)
        input.frameBase
  feTerminalMessage :
    ∀ source matrix lane,
      CarriedBelow
        ((KSplitNcEndpoints.feTerminalInput input).messageYRing
          source matrix lane)
        input.frameBase
  feTerminalEndpoint :
    CarriedBelow
      (KSplitNcEndpoints.feTerminalInput input).terminal input.frameBase
  ncGamma :
    CarriedBelow (KSplitNcEndpoints.ncInput input).gamma input.frameBase
  ncBetaBlock :
    ∀ coordinate,
      CarriedBelow
        ((KSplitNcEndpoints.ncInput input).betaBlock coordinate)
        input.frameBase
  ncBetaA :
    ∀ coordinate,
      CarriedBelow
        ((KSplitNcEndpoints.ncInput input).betaA coordinate)
        input.frameBase
  ncPointBlock :
    ∀ coordinate,
      CarriedBelow
        ((KSplitNcEndpoints.ncInput input).pointBlock coordinate)
        input.frameBase
  ncPointLane :
    ∀ coordinate,
      CarriedBelow
        ((KSplitNcEndpoints.ncInput input).pointLane coordinate)
        input.frameBase
  ncMessage :
    ∀ source lane,
      CarriedBelow
        ((KSplitNcEndpoints.ncInput input).messageYZcol source lane)
        input.frameBase
  ncInitialEndpoint :
    CarriedBelow (KSplitNcEndpoints.ncInput input).initial input.frameBase
  ncTerminalEndpoint :
    CarriedBelow (KSplitNcEndpoints.ncInput input).terminal input.frameBase

private theorem outer_mono
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (input : KSplitNcEndpoints.Input polynomialInput domains)
    (boundary : Nat)
    (allocationEnd :
      input.frameBase + KSplitNcEndpoints.allocationWidth input ≤ boundary)
    {value : Carried}
    (below : CarriedBelow value input.frameBase) :
    CarriedBelow value boundary :=
  carried_mono below
    (Nat.le_trans
      (Nat.le_add_right input.frameBase
        (KSplitNcEndpoints.allocationWidth input))
      allocationEnd)

private theorem feInitial_end
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (input : KSplitNcEndpoints.Input polynomialInput domains)
    (boundary : Nat)
    (allocationEnd :
      input.frameBase + KSplitNcEndpoints.allocationWidth input ≤ boundary) :
    (KSplitNcEndpoints.feInitialInput input).frameBase +
        KSplitNcFeInitial.allocationWidth
          (KSplitNcEndpoints.feInitialInput input) ≤
      boundary := by
  change input.frameBase +
      KSplitNcFeInitial.allocationWidth
        (KSplitNcEndpoints.feInitialInput input) ≤ boundary
  unfold KSplitNcEndpoints.allocationWidth at allocationEnd
  omega

private theorem feTerminal_end
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (input : KSplitNcEndpoints.Input polynomialInput domains)
    (boundary : Nat)
    (allocationEnd :
      input.frameBase + KSplitNcEndpoints.allocationWidth input ≤ boundary) :
    (KSplitNcEndpoints.feTerminalInput input).frameBase +
        KSplitNcFeTerminal.allocationWidth
          (KSplitNcEndpoints.feTerminalInput input) ≤
      boundary := by
  change KSplitNcEndpoints.feTerminalBase input +
      KSplitNcFeTerminal.allocationWidth
        (KSplitNcEndpoints.feTerminalInput input) ≤ boundary
  unfold KSplitNcEndpoints.feTerminalBase
  unfold KSplitNcEndpoints.allocationWidth at allocationEnd
  omega

private theorem nc_end
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (input : KSplitNcEndpoints.Input polynomialInput domains)
    (boundary : Nat)
    (allocationEnd :
      input.frameBase + KSplitNcEndpoints.allocationWidth input ≤ boundary) :
    (KSplitNcEndpoints.ncInput input).frameBase +
        KSplitNcNcEndpoint.allocationWidth
          (KSplitNcEndpoints.ncInput input) ≤
      boundary := by
  change KSplitNcEndpoints.ncBase input +
      KSplitNcNcEndpoint.allocationWidth
        (KSplitNcEndpoints.ncInput input) ≤ boundary
  unfold KSplitNcEndpoints.ncBase KSplitNcEndpoints.feTerminalBase
  unfold KSplitNcEndpoints.allocationWidth at allocationEnd
  omega

private theorem base_le_feTerminal
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (input : KSplitNcEndpoints.Input polynomialInput domains) :
    input.frameBase ≤ (KSplitNcEndpoints.feTerminalInput input).frameBase := by
  change input.frameBase ≤ KSplitNcEndpoints.feTerminalBase input
  unfold KSplitNcEndpoints.feTerminalBase
  omega

private theorem base_le_nc
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (input : KSplitNcEndpoints.Input polynomialInput domains) :
    input.frameBase ≤ (KSplitNcEndpoints.ncInput input).frameBase := by
  change input.frameBase ≤ KSplitNcEndpoints.ncBase input
  unfold KSplitNcEndpoints.ncBase KSplitNcEndpoints.feTerminalBase
  omega

/-- Every row of the exact three-program endpoint composition stays within
its declared source prefix and compact auxiliary allocation. -/
theorem rows_below
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (input : KSplitNcEndpoints.Input polynomialInput domains)
    (boundary : Nat) (positive : 0 < boundary)
    (sources : InputsBelow input)
    (allocationEnd :
      input.frameBase + KSplitNcEndpoints.allocationWidth input ≤ boundary) :
    RowsBelow (KSplitNcEndpoints.rows input) boundary := by
  have initialRows :
      RowsBelow
        (KSplitNcFeInitial.rows (KSplitNcEndpoints.feInitialInput input))
        boundary :=
    feInitial_rows_below
      (KSplitNcEndpoints.feInitialInput input) boundary positive
      sources.feInitialGamma sources.feInitialAlpha sources.feInitialClaims
      (outer_mono input boundary allocationEnd sources.feInitialEndpoint)
      (feInitial_end input boundary allocationEnd)
  have terminalBaseOrdered := base_le_feTerminal input
  have terminalRows :
      RowsBelow
        (KSplitNcFeTerminal.rows (KSplitNcEndpoints.feTerminalInput input))
        boundary :=
    feTerminal_rows_below
      (KSplitNcEndpoints.feTerminalInput input) boundary positive
      (carried_mono sources.feTerminalGamma terminalBaseOrdered)
      (fun coordinate =>
        carried_mono (sources.feTerminalAlpha coordinate)
          terminalBaseOrdered)
      (fun coordinate =>
        carried_mono (sources.feTerminalBetaA coordinate)
          terminalBaseOrdered)
      (fun coordinate =>
        carried_mono (sources.feTerminalBetaR coordinate)
          terminalBaseOrdered)
      (fun coordinate =>
        carried_mono (sources.feTerminalPointLane coordinate)
          terminalBaseOrdered)
      (fun coordinate =>
        carried_mono (sources.feTerminalPointRow coordinate)
          terminalBaseOrdered)
      (fun coordinate =>
        carried_mono (sources.feTerminalPriorPoint coordinate)
          terminalBaseOrdered)
      (fun source matrix lane =>
        carried_mono (sources.feTerminalMessage source matrix lane)
          terminalBaseOrdered)
      (outer_mono input boundary allocationEnd sources.feTerminalEndpoint)
      (feTerminal_end input boundary allocationEnd)
  have ncBaseOrdered := base_le_nc input
  have ncRows :
      RowsBelow (KSplitNcNcEndpoint.rows (KSplitNcEndpoints.ncInput input))
        boundary :=
    nc_rows_below (KSplitNcEndpoints.ncInput input) boundary positive
      (carried_mono sources.ncGamma ncBaseOrdered)
      (fun coordinate =>
        carried_mono (sources.ncBetaBlock coordinate) ncBaseOrdered)
      (fun coordinate =>
        carried_mono (sources.ncBetaA coordinate) ncBaseOrdered)
      (fun coordinate =>
        carried_mono (sources.ncPointBlock coordinate) ncBaseOrdered)
      (fun coordinate =>
        carried_mono (sources.ncPointLane coordinate) ncBaseOrdered)
      (fun source lane =>
        carried_mono (sources.ncMessage source lane) ncBaseOrdered)
      (outer_mono input boundary allocationEnd sources.ncInitialEndpoint)
      (outer_mono input boundary allocationEnd sources.ncTerminalEndpoint)
      (nc_end input boundary allocationEnd)
  intro row member column mentioned
  rcases List.mem_flatten.mp member with
    ⟨group, groupMember, rowMember⟩
  simp only [KSplitNcEndpoints.rowGroups, List.mem_cons,
    List.not_mem_nil, or_false] at groupMember
  rcases groupMember with rfl | rfl | rfl
  · exact initialRows row rowMember column mentioned
  · exact terminalRows row rowMember column mentioned
  · exact ncRows row rowMember column mentioned

end Nightstream.Implementation.R1CS.Canonical.KSplitNcEndpointsSupport
