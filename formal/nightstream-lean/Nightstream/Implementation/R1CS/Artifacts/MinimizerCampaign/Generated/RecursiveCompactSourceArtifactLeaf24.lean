import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactWire
import Nightstream.Assurance.ChunkLeaves

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
    chunkFacts (rowsChunk wire 126) 8257536 65536 11187825 11078210
      wire.completeFamilies
      ["nifs.pi_rlc.verify.identities.adv.evaluations.inputs",
       "nifs.pi_rlc.verify.identities.adv.evaluations.output",
       "nifs.pi_rlc.verify.identities.adv.evaluations.quotient",
       "nifs.pi_rlc.verify.identities.adv.final_limb_checks",
       "nifs.pi_rlc.verify.identities.adv.k_products.quotient_times_phi",
       "nifs.pi_rlc.verify.identities.adv.k_products.rho_times_input",
       "nifs.pi_rlc.verify.identities.commitment.evaluations.output",
       "nifs.pi_rlc.verify.identities.commitment.evaluations.quotient",
       "nifs.pi_rlc.verify.identities.commitment.final_limb_checks",
       "nifs.pi_rlc.verify.identities.commitment.k_products.quotient_times_phi",
       "nifs.pi_rlc.verify.identities.x.evaluations.inputs",
       "nifs.pi_rlc.verify.identities.x.evaluations.output",
       "nifs.pi_rlc.verify.identities.x.evaluations.quotient",
       "nifs.pi_rlc.verify.identities.x.final_limb_checks",
       "nifs.pi_rlc.verify.identities.x.k_products.quotient_times_phi",
       "nifs.pi_rlc.verify.identities.x.k_products.rho_times_input"] = true := by
  native_decide

theorem chunkLeaf127 :
    chunkFacts (rowsChunk wire 127) 8323072 65536 11187825 11078210
      wire.completeFamilies
      ["nifs.pi_rlc.verify.identities.y_ring.evaluations.inputs",
       "nifs.pi_rlc.verify.identities.y_ring.evaluations.output",
       "nifs.pi_rlc.verify.identities.y_ring.evaluations.quotient",
       "nifs.pi_rlc.verify.identities.y_ring.final_limb_checks",
       "nifs.pi_rlc.verify.identities.y_ring.k_products.quotient_times_phi",
       "nifs.pi_rlc.verify.identities.y_ring.k_products.rho_times_input",
       "nifs.pi_rlc.verify.padding.y_ring"] = true := by
  native_decide

theorem chunkLeaf128 :
    chunkFacts (rowsChunk wire 128) 8388608 65536 11187825 11078210
      wire.completeFamilies
      ["fprime.recursive.step.nebula",
       "fprime.recursive.step.prior_link.carrier_padding",
       "fprime.recursive.step.prior_link.digest",
       "fprime.recursive.step.prior_link.enc_inst",
       "nifs.pi_dec.verify",
       "nifs.point_binding"] = true := by
  native_decide

theorem presence5 :
    (rowsChunk wire 128).any
      (fun row => decide (row.family = "fprime.recursive.step.nebula")) = true :=
  presence_of_chunkFacts chunkLeaf128 (by decide)

theorem presence8 :
    (rowsChunk wire 128).any
      (fun row => decide (row.family = "fprime.recursive.step.prior_link.carrier_padding")) = true :=
  presence_of_chunkFacts chunkLeaf128 (by decide)

theorem presence9 :
    (rowsChunk wire 128).any
      (fun row => decide (row.family = "fprime.recursive.step.prior_link.digest")) = true :=
  presence_of_chunkFacts chunkLeaf128 (by decide)

theorem presence10 :
    (rowsChunk wire 128).any
      (fun row => decide (row.family = "fprime.recursive.step.prior_link.enc_inst")) = true :=
  presence_of_chunkFacts chunkLeaf128 (by decide)

theorem presence25 :
    (rowsChunk wire 128).any
      (fun row => decide (row.family = "nifs.pi_dec.verify")) = true :=
  presence_of_chunkFacts chunkLeaf128 (by decide)

theorem presence42 :
    (rowsChunk wire 126).any
      (fun row => decide (row.family = "nifs.pi_rlc.verify.identities.adv.evaluations.inputs")) = true :=
  presence_of_chunkFacts chunkLeaf126 (by decide)

theorem presence43 :
    (rowsChunk wire 126).any
      (fun row => decide (row.family = "nifs.pi_rlc.verify.identities.adv.evaluations.output")) = true :=
  presence_of_chunkFacts chunkLeaf126 (by decide)

theorem presence44 :
    (rowsChunk wire 126).any
      (fun row => decide (row.family = "nifs.pi_rlc.verify.identities.adv.evaluations.quotient")) = true :=
  presence_of_chunkFacts chunkLeaf126 (by decide)

theorem presence45 :
    (rowsChunk wire 126).any
      (fun row => decide (row.family = "nifs.pi_rlc.verify.identities.adv.final_limb_checks")) = true :=
  presence_of_chunkFacts chunkLeaf126 (by decide)

theorem presence46 :
    (rowsChunk wire 126).any
      (fun row => decide (row.family = "nifs.pi_rlc.verify.identities.adv.k_products.quotient_times_phi")) = true :=
  presence_of_chunkFacts chunkLeaf126 (by decide)

theorem presence47 :
    (rowsChunk wire 126).any
      (fun row => decide (row.family = "nifs.pi_rlc.verify.identities.adv.k_products.rho_times_input")) = true :=
  presence_of_chunkFacts chunkLeaf126 (by decide)

theorem presence49 :
    (rowsChunk wire 126).any
      (fun row => decide (row.family = "nifs.pi_rlc.verify.identities.commitment.evaluations.output")) = true :=
  presence_of_chunkFacts chunkLeaf126 (by decide)

theorem presence50 :
    (rowsChunk wire 126).any
      (fun row => decide (row.family = "nifs.pi_rlc.verify.identities.commitment.evaluations.quotient")) = true :=
  presence_of_chunkFacts chunkLeaf126 (by decide)

theorem presence51 :
    (rowsChunk wire 126).any
      (fun row => decide (row.family = "nifs.pi_rlc.verify.identities.commitment.final_limb_checks")) = true :=
  presence_of_chunkFacts chunkLeaf126 (by decide)

theorem presence52 :
    (rowsChunk wire 126).any
      (fun row => decide (row.family = "nifs.pi_rlc.verify.identities.commitment.k_products.quotient_times_phi")) = true :=
  presence_of_chunkFacts chunkLeaf126 (by decide)

theorem presence54 :
    (rowsChunk wire 126).any
      (fun row => decide (row.family = "nifs.pi_rlc.verify.identities.x.evaluations.inputs")) = true :=
  presence_of_chunkFacts chunkLeaf126 (by decide)

theorem presence55 :
    (rowsChunk wire 126).any
      (fun row => decide (row.family = "nifs.pi_rlc.verify.identities.x.evaluations.output")) = true :=
  presence_of_chunkFacts chunkLeaf126 (by decide)

theorem presence56 :
    (rowsChunk wire 126).any
      (fun row => decide (row.family = "nifs.pi_rlc.verify.identities.x.evaluations.quotient")) = true :=
  presence_of_chunkFacts chunkLeaf126 (by decide)

theorem presence57 :
    (rowsChunk wire 126).any
      (fun row => decide (row.family = "nifs.pi_rlc.verify.identities.x.final_limb_checks")) = true :=
  presence_of_chunkFacts chunkLeaf126 (by decide)

theorem presence58 :
    (rowsChunk wire 126).any
      (fun row => decide (row.family = "nifs.pi_rlc.verify.identities.x.k_products.quotient_times_phi")) = true :=
  presence_of_chunkFacts chunkLeaf126 (by decide)

theorem presence59 :
    (rowsChunk wire 126).any
      (fun row => decide (row.family = "nifs.pi_rlc.verify.identities.x.k_products.rho_times_input")) = true :=
  presence_of_chunkFacts chunkLeaf126 (by decide)

theorem presence60 :
    (rowsChunk wire 127).any
      (fun row => decide (row.family = "nifs.pi_rlc.verify.identities.y_ring.evaluations.inputs")) = true :=
  presence_of_chunkFacts chunkLeaf127 (by decide)

theorem presence61 :
    (rowsChunk wire 127).any
      (fun row => decide (row.family = "nifs.pi_rlc.verify.identities.y_ring.evaluations.output")) = true :=
  presence_of_chunkFacts chunkLeaf127 (by decide)

theorem presence62 :
    (rowsChunk wire 127).any
      (fun row => decide (row.family = "nifs.pi_rlc.verify.identities.y_ring.evaluations.quotient")) = true :=
  presence_of_chunkFacts chunkLeaf127 (by decide)

theorem presence63 :
    (rowsChunk wire 127).any
      (fun row => decide (row.family = "nifs.pi_rlc.verify.identities.y_ring.final_limb_checks")) = true :=
  presence_of_chunkFacts chunkLeaf127 (by decide)

theorem presence64 :
    (rowsChunk wire 127).any
      (fun row => decide (row.family = "nifs.pi_rlc.verify.identities.y_ring.k_products.quotient_times_phi")) = true :=
  presence_of_chunkFacts chunkLeaf127 (by decide)

theorem presence65 :
    (rowsChunk wire 127).any
      (fun row => decide (row.family = "nifs.pi_rlc.verify.identities.y_ring.k_products.rho_times_input")) = true :=
  presence_of_chunkFacts chunkLeaf127 (by decide)

theorem presence66 :
    (rowsChunk wire 127).any
      (fun row => decide (row.family = "nifs.pi_rlc.verify.padding.y_ring")) = true :=
  presence_of_chunkFacts chunkLeaf127 (by decide)

theorem presence80 :
    (rowsChunk wire 128).any
      (fun row => decide (row.family = "nifs.point_binding")) = true :=
  presence_of_chunkFacts chunkLeaf128 (by decide)

theorem censusGroup :
    ∀ k, 126 ≤ k → k < 129 →
      (rowsChunk wire k).map (fun row => row.sourceIndex) =
        List.range' (wire.chunkStart k) (wire.chunkLength k) := by
  intro k lower upper
  by_cases is126 : k = 126
  · subst is126
    exact (chunkFacts_split chunkLeaf126).1
  by_cases is127 : k = 127
  · subst is127
    exact (chunkFacts_split chunkLeaf127).1
  by_cases is128 : k = 128
  · subst is128
    exact (chunkFacts_split chunkLeaf128).1
  exact absurd upper (by omega)

theorem wfGroup :
    ∀ k, 126 ≤ k → k < 129 →
      (rowsChunk wire k).all (rowWellFormedAt 11187825 11078210) = true := by
  intro k lower upper
  by_cases is126 : k = 126
  · subst is126
    exact (chunkFacts_split chunkLeaf126).2.1
  by_cases is127 : k = 127
  · subst is127
    exact (chunkFacts_split chunkLeaf127).2.1
  by_cases is128 : k = 128
  · subst is128
    exact (chunkFacts_split chunkLeaf128).2.1
  exact absurd upper (by omega)

theorem coverGroup :
    ∀ k, 126 ≤ k → k < 129 →
      (rowsChunk wire k).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true := by
  intro k lower upper
  by_cases is126 : k = 126
  · subst is126
    exact (chunkFacts_split chunkLeaf126).2.2.1
  by_cases is127 : k = 127
  · subst is127
    exact (chunkFacts_split chunkLeaf127).2.2.1
  by_cases is128 : k = 128
  · subst is128
    exact (chunkFacts_split chunkLeaf128).2.2.1
  exact absurd upper (by omega)

end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf24
