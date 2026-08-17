import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactWire

/-!
GENERATED FILE - do not edit by hand.

Bounded per-chunk leaf certificates for one slice of the artifact.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf24

open Nightstream.Assurance.CompactSourceArtifact
open Nightstream.Assurance.ConstraintMinimization
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactWire

set_option maxHeartbeats 2000000
set_option maxRecDepth 65536

theorem chunkLeaf126 :
    ((rowsChunk wire 126).map (fun row => row.sourceIndex) =
        List.range' 8257536 65536) ∧
      ((rowsChunk wire 126).all (rowWellFormedAt 11187825 11078210) = true) ∧
      ((rowsChunk wire 126).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true) := by
  native_decide

theorem chunkLeaf127 :
    ((rowsChunk wire 127).map (fun row => row.sourceIndex) =
        List.range' 8323072 65536) ∧
      ((rowsChunk wire 127).all (rowWellFormedAt 11187825 11078210) = true) ∧
      ((rowsChunk wire 127).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true) := by
  native_decide

theorem chunkLeaf128 :
    ((rowsChunk wire 128).map (fun row => row.sourceIndex) =
        List.range' 8388608 65536) ∧
      ((rowsChunk wire 128).all (rowWellFormedAt 11187825 11078210) = true) ∧
      ((rowsChunk wire 128).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true) := by
  native_decide

theorem presence5 :
    (rowsChunk wire 128).any
      (fun row => decide (row.family = "fprime.recursive.step.nebula")) = true := by
  native_decide

theorem presence8 :
    (rowsChunk wire 128).any
      (fun row => decide (row.family = "fprime.recursive.step.prior_link.carrier_padding")) = true := by
  native_decide

theorem presence9 :
    (rowsChunk wire 128).any
      (fun row => decide (row.family = "fprime.recursive.step.prior_link.digest")) = true := by
  native_decide

theorem presence10 :
    (rowsChunk wire 128).any
      (fun row => decide (row.family = "fprime.recursive.step.prior_link.enc_inst")) = true := by
  native_decide

theorem presence25 :
    (rowsChunk wire 128).any
      (fun row => decide (row.family = "nifs.pi_dec.verify")) = true := by
  native_decide

theorem presence42 :
    (rowsChunk wire 126).any
      (fun row => decide (row.family = "nifs.pi_rlc.verify.identities.adv.evaluations.inputs")) = true := by
  native_decide

theorem presence43 :
    (rowsChunk wire 126).any
      (fun row => decide (row.family = "nifs.pi_rlc.verify.identities.adv.evaluations.output")) = true := by
  native_decide

theorem presence44 :
    (rowsChunk wire 126).any
      (fun row => decide (row.family = "nifs.pi_rlc.verify.identities.adv.evaluations.quotient")) = true := by
  native_decide

theorem presence45 :
    (rowsChunk wire 126).any
      (fun row => decide (row.family = "nifs.pi_rlc.verify.identities.adv.final_limb_checks")) = true := by
  native_decide

theorem presence46 :
    (rowsChunk wire 126).any
      (fun row => decide (row.family = "nifs.pi_rlc.verify.identities.adv.k_products.quotient_times_phi")) = true := by
  native_decide

theorem presence47 :
    (rowsChunk wire 126).any
      (fun row => decide (row.family = "nifs.pi_rlc.verify.identities.adv.k_products.rho_times_input")) = true := by
  native_decide

theorem presence49 :
    (rowsChunk wire 126).any
      (fun row => decide (row.family = "nifs.pi_rlc.verify.identities.commitment.evaluations.output")) = true := by
  native_decide

theorem presence50 :
    (rowsChunk wire 126).any
      (fun row => decide (row.family = "nifs.pi_rlc.verify.identities.commitment.evaluations.quotient")) = true := by
  native_decide

theorem presence51 :
    (rowsChunk wire 126).any
      (fun row => decide (row.family = "nifs.pi_rlc.verify.identities.commitment.final_limb_checks")) = true := by
  native_decide

theorem presence52 :
    (rowsChunk wire 126).any
      (fun row => decide (row.family = "nifs.pi_rlc.verify.identities.commitment.k_products.quotient_times_phi")) = true := by
  native_decide

theorem presence54 :
    (rowsChunk wire 126).any
      (fun row => decide (row.family = "nifs.pi_rlc.verify.identities.x.evaluations.inputs")) = true := by
  native_decide

theorem presence55 :
    (rowsChunk wire 126).any
      (fun row => decide (row.family = "nifs.pi_rlc.verify.identities.x.evaluations.output")) = true := by
  native_decide

theorem presence56 :
    (rowsChunk wire 126).any
      (fun row => decide (row.family = "nifs.pi_rlc.verify.identities.x.evaluations.quotient")) = true := by
  native_decide

theorem presence57 :
    (rowsChunk wire 126).any
      (fun row => decide (row.family = "nifs.pi_rlc.verify.identities.x.final_limb_checks")) = true := by
  native_decide

theorem presence58 :
    (rowsChunk wire 126).any
      (fun row => decide (row.family = "nifs.pi_rlc.verify.identities.x.k_products.quotient_times_phi")) = true := by
  native_decide

theorem presence59 :
    (rowsChunk wire 126).any
      (fun row => decide (row.family = "nifs.pi_rlc.verify.identities.x.k_products.rho_times_input")) = true := by
  native_decide

theorem presence60 :
    (rowsChunk wire 127).any
      (fun row => decide (row.family = "nifs.pi_rlc.verify.identities.y_ring.evaluations.inputs")) = true := by
  native_decide

theorem presence61 :
    (rowsChunk wire 127).any
      (fun row => decide (row.family = "nifs.pi_rlc.verify.identities.y_ring.evaluations.output")) = true := by
  native_decide

theorem presence62 :
    (rowsChunk wire 127).any
      (fun row => decide (row.family = "nifs.pi_rlc.verify.identities.y_ring.evaluations.quotient")) = true := by
  native_decide

theorem presence63 :
    (rowsChunk wire 127).any
      (fun row => decide (row.family = "nifs.pi_rlc.verify.identities.y_ring.final_limb_checks")) = true := by
  native_decide

theorem presence64 :
    (rowsChunk wire 127).any
      (fun row => decide (row.family = "nifs.pi_rlc.verify.identities.y_ring.k_products.quotient_times_phi")) = true := by
  native_decide

theorem presence65 :
    (rowsChunk wire 127).any
      (fun row => decide (row.family = "nifs.pi_rlc.verify.identities.y_ring.k_products.rho_times_input")) = true := by
  native_decide

theorem presence66 :
    (rowsChunk wire 127).any
      (fun row => decide (row.family = "nifs.pi_rlc.verify.padding.y_ring")) = true := by
  native_decide

theorem presence80 :
    (rowsChunk wire 128).any
      (fun row => decide (row.family = "nifs.point_binding")) = true := by
  native_decide

end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf24
