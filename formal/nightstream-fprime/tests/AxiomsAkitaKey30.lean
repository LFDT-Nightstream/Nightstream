import tests.AxiomAudit
import NightstreamFPrime.Export.BoundedDepth.FinalKey30All

/-! Axiom audits for the Akita opening of the schedule `key30.aks`: the numbers of the two rows
(`Spec/Compression/Akita/Key30`), the level with the row `w_cells = 0` in `R_{q,1}`
(`Extract/LiftSlim`), the checks of an offloaded fold (`Extract/Offload`), the group of the setup
prefix and the offloaded step of the suffix (`Extract/Groups`), the suffix of the opening with an
offloaded root (`Extract/OffTower`, `Key30Tower`), the opening with the decoders `draw`
(`Key30Opening`), the instances of Module-SIS (`Key30Instances`), the final verifier over the
opening (`Export/BoundedDepth/FinalKey30`), and the final verifier for each claim shape and
against a false history (`Export/BoundedDepth/FinalShapes`, `FinalKey30All`). -/

#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Key30.root_sized
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Key30.root_degrees
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Key30.level₁_sized
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Key30.level₂_sized
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Key30.level₃_sized
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Key30.level₃_tight
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Key30.level₄_sized
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Key30.tails_sized
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Key30.norms_sized
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Key30.norms_response
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Key30.quarters
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Key30.root_points
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Key30.level₁_points
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Key30.points
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Key30.honest
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Key30.live_cells
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Key30.not_covers
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Key30.tower_counts
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Key30.tower_suffix
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Key30.towerError_le
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.slimRows_below
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.slimRows_degree_le
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.slimRows_degree_sum
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.slimRelation_locked
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.slimRelation_quotientEnd
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.slimFold_satisfies
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.LevelSized.slim
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.setupCheck_avoids
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.SetupScan.direct
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.SetupScan.direct_value
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.offloadCheck_avoids
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.setupCosts_sum
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.offloadCheck_bare
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.setupCheck_staged
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.offloadCheck_staged
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.offBatch_valid
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.offBatch_uniform
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.Method.pinned_sound
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.Tier.pinned_valid
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Batch.pin_fixed
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Batch.registered_commit
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.root_suffix_pinned
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.Tier.suffix_push_offloaded
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.offTower_suffix
#audit_axioms NightstreamFPrime.Export.BoundedDepth.FinalKey30.key30_points
#audit_axioms NightstreamFPrime.Export.BoundedDepth.FinalKey30.Use.points
#audit_axioms NightstreamFPrime.Export.BoundedDepth.FinalKey30.sound
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Key30.Tower.Drawn.fits
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Key30.Tower.suffix
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Key30.Tower.extract_sound
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Key30.Tower.extraction_le
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Key30.views_covered
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Key30.instances_exact
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Key30.reduced
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Key30.reduced_key
#audit_axioms NightstreamFPrime.Export.BoundedDepth.FinalShapes.knowledge_all
#audit_axioms NightstreamFPrime.Export.BoundedDepth.FinalShapes.sound_all
#audit_axioms NightstreamFPrime.Export.BoundedDepth.FinalShapes.rejected_le
#audit_axioms NightstreamFPrime.Export.BoundedDepth.FinalShapes.adversary_depth
#audit_axioms NightstreamFPrime.Export.BoundedDepth.FinalShapes.history_sound
#audit_axioms NightstreamFPrime.Export.BoundedDepth.FinalShapes.history_knowledge
#audit_axioms NightstreamFPrime.Export.BoundedDepth.FinalKey30.Opening.knowledge
#audit_axioms NightstreamFPrime.Export.BoundedDepth.FinalKey30.Opening.reduced
#audit_axioms NightstreamFPrime.Export.BoundedDepth.FinalKey30.verifier_sound_drawn
#audit_axioms NightstreamFPrime.Export.BoundedDepth.FinalKey30.history_sound_drawn
#audit_axioms NightstreamFPrime.Export.BoundedDepth.FinalKey30.history_knowledge_drawn
