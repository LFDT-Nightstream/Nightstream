import Nightstream.SuperNeo.Folding.PiRLC.PaperCorrections
import tests.Axioms.Support

/-! Fail-closed trusted-dependency gate for the corrected ambient bound. -/

namespace tests.Axioms.PiRLCPaperCorrections

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.Norm
open Nightstream.SuperNeo.Folding.PiRLC.PaperCorrections

/-- Minimal concrete semantics that makes the shared `CE.Holds` norm field
observable at the Goldilocks midpoint. -/
def midpointSemantics : RelationSemantics Unit F Unit Unit Unit Unit where
  commit _ := ()
  projectPublicInput _ := ()
  normBounded bound assignment := centeredMagnitude assignment < bound
  ccsSatisfied _ _ := True
  evaluationPointValid _ _ := True
  evaluations _ _ _ := #[()]

def midpointStatement : CE.Instance Unit Unit Unit Unit Unit where
  constraintSystem := ()
  commitment := ()
  publicInput := ()
  point := ()
  evaluations := #[()]
  stage := .ambient

/-- Regression through the authority-bearing shared relation, not only through
the raw centered-magnitude helper. -/
theorem midpointResidue_sharedAmbientHolds :
    CE.Holds midpointSemantics productionGlobalParams midpointStatement
      midpointResidue := by
  refine ⟨?_, trivial, rfl⟩
  refine ⟨rfl, rfl, ?_⟩
  have bounded := all_centeredMagnitude_lt_correctedAmbientBound midpointResidue
  rw [← production_correctedAmbientBoundFor_eq] at bounded
  simpa [midpointSemantics, midpointStatement, NormStage.bound,
    correctedAmbientBoundFor] using bounded

end tests.Axioms.PiRLCPaperCorrections

/-- info: 'Nightstream.SuperNeo.Folding.PiRLC.PaperCorrections.midpointResidue_not_literalAmbientBounded' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiRLC.PaperCorrections.midpointResidue_not_literalAmbientBounded

/-- info: 'Nightstream.SuperNeo.Folding.PiRLC.PaperCorrections.all_centeredMagnitude_lt_correctedAmbientBound' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiRLC.PaperCorrections.all_centeredMagnitude_lt_correctedAmbientBound

/-- info: 'Nightstream.SuperNeo.Folding.PiRLC.PaperCorrections.production_correctedAmbientBoundFor_eq' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiRLC.PaperCorrections.production_correctedAmbientBoundFor_eq

/-- info: 'Nightstream.SuperNeo.Folding.PiRLC.PaperCorrections.ambientStageBound_eq_correctedAmbientBoundFor' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiRLC.PaperCorrections.ambientStageBound_eq_correctedAmbientBoundFor

/-- info: 'Nightstream.SuperNeo.Folding.PiRLC.PaperCorrections.correctedAmbientHolds_iff_ceHolds_of_ambient' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiRLC.PaperCorrections.correctedAmbientHolds_iff_ceHolds_of_ambient

/-- info: 'tests.Axioms.PiRLCPaperCorrections.midpointResidue_sharedAmbientHolds' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms tests.Axioms.PiRLCPaperCorrections.midpointResidue_sharedAmbientHolds
