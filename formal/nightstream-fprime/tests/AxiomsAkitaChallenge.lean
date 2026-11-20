import tests.AxiomAudit
import NightstreamFPrime.Spec.Compression
import NightstreamFPrime.Export.BoundedDepth.FinalKey20Challenge

/-! Axiom audits for the sparse challenge families of the Akita opening: the units of short ring
elements (`Spec/Compression/Akita/ShortUnit`), the families with their norms, their size and
their unit differences (`ChallengeFamily`), the decoders from one slot and their slot
probability (`Extract/Draw`), the families of the schedule `key20.aks` (`Key20Challenge`), the
opening of this schedule with these decoders (`Key20Opening`), and the final verifier with these
decoders (`Export/BoundedDepth/FinalKey20Challenge`). -/

#audit_axioms NightstreamFPrime.Spec.Compression.Akita.X_pow_two_pow_sub_C_irreducible
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.exists_sq_eq_neg_one
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.not_square_of_sq_eq_neg_one
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.pair_eq_zero
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.eq_zero_of_factor_dvd
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.isUnit_of_short
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.magnitude_natCast
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.eq_or_eq_neg_magnitude
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.neg_natCast_ne
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.SparseFamily
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.peak
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.patterns
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.card_patterns
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.weight
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.member
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.pattern
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.weight_le
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.magnitude_coeff_member
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.pattern_member
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.member_mem
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.pattern_listed
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.card_sparseFamily
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.normOne_of_mem
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.normInf_of_mem
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.SparseFamily.sparse
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.SparseFamily.sparse_embed
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.SparseFamily.normInf_sub_le
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.SparseFamily.unitDifferences
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.mass_eq_card
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.slotChance_le_of_fibres
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.slotChance_le_pow
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.slotChance_le_of_injective
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.le_slotChance
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.draw
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.exists_of_draw
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.draw_mem
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.draw_injective
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.slotChance_draw_le
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.exists_pick
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.draw_isSome
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Key20.draw_units
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Key20.draw_sparse
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Key20.families_large
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Key20.slots_le
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Key20.slotChance_draw
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Key20.pow_lt_slotChance
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Key20.rootTwos
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Key20.rootMass
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Key20.rootPeak
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Key20.rootDecode
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Key20.rootDecode_units
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Key20.rootDecode_sparse
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Key20.rootDecode_isSome
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Key20.Tower
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Key20.Tower.accepts
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Key20.Tower.body
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Key20.Tower.Drawn
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Key20.Tower.suffix
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Key20.Tower.extract_sound
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Key20.Tower.extraction_le
#audit_axioms NightstreamFPrime.Export.BoundedDepth.FinalKey20.Opening.Drawn
#audit_axioms NightstreamFPrime.Export.BoundedDepth.FinalKey20.Opening.Drawn.sized
#audit_axioms NightstreamFPrime.Export.BoundedDepth.FinalKey20.verifier_sound_drawn
#audit_axioms NightstreamFPrime.Export.BoundedDepth.FinalKey20.history_sound_drawn
#audit_axioms NightstreamFPrime.Export.BoundedDepth.FinalKey20.history_knowledge_drawn
