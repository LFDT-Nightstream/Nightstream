import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.PriorLink.ExactAuthority

/-! Focused interface gate for exact rich-carrier F-prime prior authority. -/

open Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics

#check PriorLink.Accepted
#check PriorLink.Accepted.previousPiDec
#check PriorLink.Accepted.current_eq_previous
#check PriorLink.Accepted.of_eq
#check PriorLink.Accepted.accepted_iff_eq
#check PriorLink.Accepted.currentCanonical
#check PriorLink.Accepted.of_resultTransition
#check PriorLink.Accepted.resultTransition
#check PriorLink.Accepted.accepted_iff_eq_of_resultTransition
#check PriorLink.slot_eq_or_familyDigest_failure
#check PriorLink.slot_eq_or_commitmentDigest_failure
#check PriorLink.slot_eq_or_commitmentDigest_failure_of_selectedNifs
#check PriorLink.slot_eq_or_canonicalParentDigest_failure
#check PriorLink.slot_eq_or_canonicalParentDigest_failure_of_openingSources
#check PriorLink.accepted_or_securityFailure
#check PriorLink.slot_eq_or_securityFailure

open Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.PriorLink.ExactAuthority

#check WithoutChildrenBinding
#check WithoutParentRecomposition
#check childrenBinding_necessary
#check parentRecomposition_necessary
