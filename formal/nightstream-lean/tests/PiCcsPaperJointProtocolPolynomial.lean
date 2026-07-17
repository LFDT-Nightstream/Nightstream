import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ProtocolPolynomial
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Necessity.NonlinearTerminal

open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

#check ProtocolPolynomial.VerifierInput
#check ProtocolPolynomial.VerifierInput.ext
#check ProtocolPolynomial.VerifierInput.sumcheckDegreeBound
#check ProtocolPolynomial.VerifierInput.sumcheckDegreeBound_eq_of_terms_eq
#check ProtocolPolynomial.Data.toVerifierInput
#check ProtocolPolynomial.Data.toVerifierInput_eq
#check ProtocolPolynomial.VerifierInput.initial
#check ProtocolPolynomial.verifierInput_initial_eq_joint_initial
#check ProtocolPolynomial.Data.toJointData
#check ProtocolPolynomial.messageAt
#check ProtocolPolynomial.terminalFromMessage
#check ProtocolPolynomial.qAtPoint_toCubePoint_eq_tableQ
#check ProtocolPolynomial.sumCompletions_polynomial_eq_summedQ
#check ProtocolPolynomial.canonicalGhosts_honest
#check ProtocolPolynomial.check_eq_true_iff_accepted
#check ProtocolPolynomial.check_implies_tableTruth_or_badEvent
#check ProtocolPolynomial.Necessity.NonlinearTerminal.residualTableTerminal_eq_two
#check ProtocolPolynomial.Necessity.NonlinearTerminal.protocolTerminal_eq_four
#check ProtocolPolynomial.Necessity.NonlinearTerminal.residualTableTerminal_ne_protocolTerminal

namespace Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ProtocolPolynomial.Tests

universe uField

/-- Regression: declared degree metadata cannot change the executable bound
when the actual sparse monomial syntax is unchanged. -/
theorem identicalTerms_have_identicalVerifierDegree
    {Field : Type uField}
    {shape : Shape}
    (left right : VerifierInput Field shape)
    (sameTerms : left.constraintPolynomial.terms =
      right.constraintPolynomial.terms) :
    left.sumcheckDegreeBound = right.sumcheckDegreeBound :=
  VerifierInput.sumcheckDegreeBound_eq_of_terms_eq left right sameTerms

end Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ProtocolPolynomial.Tests
