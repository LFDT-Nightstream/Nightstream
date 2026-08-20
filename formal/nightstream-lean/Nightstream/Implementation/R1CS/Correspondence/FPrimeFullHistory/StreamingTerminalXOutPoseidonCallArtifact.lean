import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalXOutPoseidonCallBridge

/-!
Contract: exact call-to-round ownership for the recursive-terminal XOut
public Poseidon2 hash.

Owns small reusable certificates for the nine named Rust call placements.
Each certificate is proved from one generated leaf. It does not evaluate the
complete artifact.

Does not own final-row satisfaction, call semantics, call chaining, public
word decoding, lifecycle composition, or collision resistance.

Assurance tier: artifact-checked for the Nightstream b2/k16 terminal profile.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalXOutPoseidonCallArtifact

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Core
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalXOutPublicHash.Artifact
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalXOutPublicHash
open Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalXOutPoseidonCallBridge

def zeroPort : AbsolutePort := { explicit := [], geometric := [] }

/-- The first selected call is the first trace round and reads the first four
XOut fields followed by four zero state lanes. -/
theorem callPlacement0_round_exact :
    callPlacement0.roundIndex = 0 ∧
      callPlacement0.inputSourceColumns =
        (rounds.getD 0 default).call.inputColumns := by
  exact ⟨rfl, rfl⟩

theorem round0_kind :
    (rounds.getD 0 default).kind =
      .absorb [30382557, 642, 643, 644] := by
  rfl

theorem callPlacement0_input_ports :
    callPlacement0.inputImages.map SourceImage.port =
      (xOutImages.take 4).map SourceImage.port ++
        List.replicate 4 zeroPort := by
  rfl

def xOutValue
    (assignment : AbsoluteAssignment callPlacement0) (index : Nat) : F :=
  absolutePortAction (placement := callPlacement0) assignment
    (xOutImages.getD index emptySourceImage).port

/-- The call-0 source images and the XOut images have the same value on the
same final assignment. The remaining four lanes are the initial zero state. -/
theorem callPlacement0_input_values
    (assignment : AbsoluteAssignment callPlacement0) (lane : Fin width) :
    inputValue callPlacement0 assignment lane =
      if lane.val < 4 then xOutValue assignment lane.val else 0 := by
  fin_cases lane <;> rfl

/-- The second selected call is round one. Its rate lanes add XOut fields
four through seven to the complete call-0 output; its capacity lanes carry
the complete call-0 output unchanged. -/
theorem callPlacement1_round_exact :
    callPlacement1.roundIndex = 1 ∧
      callPlacement1.inputSourceColumns =
        (rounds.getD 1 default).call.inputColumns := by
  exact ⟨rfl, rfl⟩

theorem round1_kind :
    (rounds.getD 1 default).kind = .absorb [645, 646, 647, 648] := by
  rfl

theorem callPlacement1_input_port_exact (lane : Fin width) :
    (inputImage callPlacement1 lane).port =
      if lane.val < 4 then
        appendAbsolutePort (callOutputPort callPlacement0 lane)
          (xOutImages.getD (4 + lane.val) emptySourceImage).port
      else
        callOutputPort callPlacement0 lane := by
  fin_cases lane <;> rfl

def absorbInputPort (previous : PoseidonCallPlacement)
    (xOutOffset : Nat) (lane : Fin width) : AbsolutePort :=
  if lane.val < 4 then
    appendAbsolutePort (callOutputPort previous lane)
      (xOutImages.getD (xOutOffset + lane.val) emptySourceImage).port
  else
    callOutputPort previous lane

def onePort : AbsolutePort where
  explicit := [{ column := 0, coefficient := 1 }]
  geometric := []

def padInputPort (previous : PoseidonCallPlacement)
    (lane : Fin width) : AbsolutePort :=
  if lane.val = 0 then
    appendAbsolutePort (callOutputPort previous lane) onePort
  else
    callOutputPort previous lane

theorem round2_kind :
    (rounds.getD 2 default).kind =
      .absorb [649, 30382558, 30382559, 30382560] := by
  rfl

theorem callPlacement2_input_port_exact (lane : Fin width) :
    (inputImage callPlacement2 lane).port =
      absorbInputPort callPlacement1 8 lane := by
  fin_cases lane <;> rfl

theorem round3_kind :
    (rounds.getD 3 default).kind =
      .absorb [30382561, 30382562, 30382563, 722] := by
  rfl

theorem callPlacement3_input_port_exact (lane : Fin width) :
    (inputImage callPlacement3 lane).port =
      absorbInputPort callPlacement2 12 lane := by
  fin_cases lane <;> rfl

theorem round4_kind :
    (rounds.getD 4 default).kind =
      .absorb [723, 724, 725, 30362945] := by
  rfl

theorem callPlacement4_input_port_exact (lane : Fin width) :
    (inputImage callPlacement4 lane).port =
      absorbInputPort callPlacement3 16 lane := by
  fin_cases lane <;> rfl

theorem round5_kind :
    (rounds.getD 5 default).kind =
      .absorb [30362946, 30362947, 30362948, 30362937] := by
  rfl

theorem callPlacement5_input_port_exact (lane : Fin width) :
    (inputImage callPlacement5 lane).port =
      absorbInputPort callPlacement4 20 lane := by
  fin_cases lane <;> rfl

theorem round6_kind :
    (rounds.getD 6 default).kind =
      .absorb [30362938, 30362939, 30362940, 30382564] := by
  rfl

theorem callPlacement6_input_port_exact (lane : Fin width) :
    (inputImage callPlacement6 lane).port =
      absorbInputPort callPlacement5 24 lane := by
  fin_cases lane <;> rfl

theorem round7_kind :
    (rounds.getD 7 default).kind =
      .absorb [30382553, 30382554, 30382555, 30382556] := by
  rfl

theorem callPlacement7_input_port_exact (lane : Fin width) :
    (inputImage callPlacement7 lane).port =
      absorbInputPort callPlacement6 28 lane := by
  fin_cases lane <;> rfl

theorem round8_kind :
    (rounds.getD 8 default).kind = .pad := by
  rfl

theorem callPlacement8_input_port_exact (lane : Fin width) :
    (inputImage callPlacement8 lane).port =
      padInputPort callPlacement7 lane := by
  fin_cases lane <;> rfl

end Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalXOutPoseidonCallArtifact
