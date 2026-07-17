import Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.PaddedIdentityEvaluation
import tests.Axioms.Support

/-!
Fail-closed dependency gate for the separate fixed-profile padded-identity CE
opening. These are model-level algebraic and CE-membership claims; they do not
assert production NIFS integration, Rust conformance, or row removal.
-/

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.PaddedIdentityEvaluation.paddedRowNumber_eq_iff' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.PaddedIdentityEvaluation.paddedRowNumber_eq_iff

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.PaddedIdentityEvaluation.decodeRow?_eq_some_iff' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.PaddedIdentityEvaluation.decodeRow?_eq_some_iff

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.PaddedIdentityEvaluation.paddedMatrixEntry_eq' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.PaddedIdentityEvaluation.paddedMatrixEntry_eq

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.PaddedIdentityEvaluation.rowRing_eq_expectedRowRing' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.PaddedIdentityEvaluation.rowRing_eq_expectedRowRing

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.PaddedIdentityEvaluation.matrixEvaluation_packedPoint_eq_packedYZcol' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.PaddedIdentityEvaluation.matrixEvaluation_packedPoint_eq_packedYZcol

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.PaddedIdentityEvaluation.claimedEvaluation_eq_packedYZcol_of_evaluationsBound' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.PaddedIdentityEvaluation.claimedEvaluation_eq_packedYZcol_of_evaluationsBound

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.PaddedIdentityEvaluation.claimedEvaluation_eq_packedYZcol_of_ceHolds' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.PaddedIdentityEvaluation.claimedEvaluation_eq_packedYZcol_of_ceHolds
