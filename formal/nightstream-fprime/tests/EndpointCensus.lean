import tests.AxiomAudit
import NightstreamFPrime
import NightstreamFPrime.Export.Stage1.SetupDistribution
import NightstreamFPrime.Export.Stage1.SetupBinding
import NightstreamFPrime.Export.Stage1.PiDECInputCheck
import NightstreamFPrime.Lifecycle.XOut
import NightstreamFPrime.Lifecycle.ProductionKey
import NightstreamFPrime.Spec.Folding.Nifs.PaperWeakSuffix
import Mathlib.Analysis.SpecificLimits.Basic
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Ring
import Mathlib.Algebra.BigOperators.Field
import Mathlib.Algebra.Order.BigOperators.Expect
import Mathlib.Logic.Equiv.Prod
import Mathlib.Data.Fintype.Option
import Mathlib.Probability.ProbabilityMassFunction.Basic
import Mathlib.Topology.Algebra.InfiniteSum.Real
import Mathlib.Topology.Algebra.InfiniteSum.Constructions
import Mathlib.Analysis.Normed.Group.InfiniteSum
import NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtractionWork
import Mathlib.Probability.ProbabilityMassFunction.Constructions
import Mathlib.Algebra.BigOperators.Ring.Finset
import NightstreamFPrime.Spec.Folding.PiRLC.CoordinateForkLaw
import Mathlib.Algebra.Polynomial.Eval.Defs
import NightstreamFPrime.Spec.Folding.Nifs.PaperStrongInterface
import Mathlib.Data.Real.Sqrt
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.IndependentExecution
import NightstreamFPrime.Lifecycle.PaperExtractionAlgebra
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.CausalExecution
import NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtraction
import NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.Binding
import NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.Norm.Product
import NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.RingFLaws
import NightstreamFPrime.Spec.Profile

/-! Every full name that the assurance surface and the trust boundary cite
exists, and every cited theorem passes the axiom audit. -/

#endpoint_census "ASSURANCE_SURFACE.md"
#endpoint_census "TRUST_BOUNDARY.md"
