import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Authority.PackedYZcol
import tests.Axioms.Support

/-! Fail-closed dependency gate for packed `yZcol` authority at the NIFS boundary. -/

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Authority.PackedYZcol.sourceBound_iff_packedYZcolBound' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Authority.PackedYZcol.sourceBound_iff_packedYZcolBound

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Authority.PackedYZcol.rawSourceFold_eq_canonicalParentAssignment' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Authority.PackedYZcol.rawSourceFold_eq_canonicalParentAssignment

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Authority.PackedYZcol.canonicalParentClaim_eq_sourceAggregate' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Authority.PackedYZcol.canonicalParentClaim_eq_sourceAggregate

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Authority.PackedYZcol.packedYZcolBound_or_mixingCollision_or_badRoot_or_parentProjectionMismatch' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Authority.PackedYZcol.packedYZcolBound_or_mixingCollision_or_badRoot_or_parentProjectionMismatch

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Authority.PackedYZcol.accepted_implies_refinement_or_yRingUnbound_or_mixingCollision_or_projectionBadRoot_or_parentProjectionMismatch_or_badEvent' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Authority.PackedYZcol.accepted_implies_refinement_or_yRingUnbound_or_mixingCollision_or_projectionBadRoot_or_parentProjectionMismatch_or_badEvent
