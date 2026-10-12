import tests.AxiomAudit
import CompressionLean

/-! Axiom gate for every exported terminal theorem. -/

#audit_axioms CompressionLean.Terminal.circuitPackage_implies_terminal
#audit_axioms CompressionLean.Terminal.priorSpec
#audit_axioms CompressionLean.Terminal.piCcsPhaseHolds
#audit_axioms CompressionLean.Terminal.piCcsSpecHolds
#audit_axioms CompressionLean.Terminal.selectedVerifierContext
#audit_axioms CompressionLean.Terminal.piRlcPhaseHolds
#audit_axioms CompressionLean.Terminal.piRlcSpecHolds
#audit_axioms CompressionLean.Terminal.piRlcCombinationRows
#audit_axioms CompressionLean.Terminal.piRlcSamplerChain
#audit_axioms CompressionLean.Terminal.priorExtraRows_hold
#audit_axioms CompressionLean.Terminal.arithmeticRows_hold
#audit_axioms CompressionLean.Terminal.samplerInitialState_eq
#audit_axioms CompressionLean.Terminal.commitment_eval
#audit_axioms CompressionLean.Terminal.publicInput_eval
#audit_axioms CompressionLean.Terminal.point_eval
#audit_axioms CompressionLean.Terminal.eval_K_eval
#audit_axioms CompressionLean.Terminal.eval_A_eval
#audit_axioms CompressionLean.Terminal.state_eval

/-! Axiom gate for the layer-1 soundness proof. -/

#audit_axioms CompressionLean.Layer1.Game.prob_not_le
#audit_axioms CompressionLean.Layer1.Game.avoids_bind
#audit_axioms CompressionLean.Layer1.Game.costLe_bind
#audit_axioms CompressionLean.Layer1.sumcheck_avoids
#audit_axioms CompressionLean.Layer1.sumcheck_cost
#audit_axioms CompressionLean.Layer1.combRound_truth
#audit_axioms CompressionLean.Layer1.fraction_sum_bad
#audit_axioms CompressionLean.Layer1.Level.root_fraction
#audit_axioms CompressionLean.Layer1.gkr_avoids
#audit_axioms CompressionLean.Layer1.gkr_cost
#audit_axioms CompressionLean.Layer1.mle_ne_zero_of_path'
#audit_axioms CompressionLean.Layer1.twoSided
#audit_axioms CompressionLean.Layer1.norm_sound
#audit_axioms CompressionLean.Layer1.Lookup.sound
#audit_axioms CompressionLean.Layer1.Lookup.bad_card
#audit_axioms CompressionLean.Layer1.Scatter.sound
#audit_axioms CompressionLean.Layer1.Scatter.bad_card
#audit_axioms CompressionLean.Layer1.RingRows.ring_sound
#audit_axioms CompressionLean.Layer1.RingRows.zetaBad_card
#audit_axioms CompressionLean.Layer1.univ_eval
#audit_axioms CompressionLean.Layer1.univ_eq_zero
#audit_axioms CompressionLean.Layer1.leaf_formula
#audit_axioms CompressionLean.Layer1.agreement_card
#audit_axioms CompressionLean.Layer1.queryBad_density
#audit_axioms CompressionLean.Layer1.early_world
#audit_axioms CompressionLean.Layer1.late_world
#audit_axioms CompressionLean.Layer1.slotWeights_eq
#audit_axioms CompressionLean.Layer1.scaled_eq
#audit_axioms CompressionLean.Layer1.omegaTrue_parts
#audit_axioms CompressionLean.Layer1.linear_world
#audit_axioms CompressionLean.Layer1.honest_world
#audit_axioms CompressionLean.Layer1.earlyGame_avoids
#audit_axioms CompressionLean.Layer1.linearGame_avoids
#audit_axioms CompressionLean.Layer1.honestGame_avoids
#audit_axioms CompressionLean.Layer1.lateGame_avoids
#audit_axioms CompressionLean.Layer1.layer1Game_avoids
#audit_axioms CompressionLean.Layer1.earlyGame_cost
#audit_axioms CompressionLean.Layer1.linearGame_cost
#audit_axioms CompressionLean.Layer1.honestGame_cost
#audit_axioms CompressionLean.Layer1.lateGame_cost
#audit_axioms CompressionLean.Layer1.layer1Game_cost
#audit_axioms CompressionLean.Layer1.layer1_sound
