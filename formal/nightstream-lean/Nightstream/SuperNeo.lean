import Nightstream.SuperNeo.Relations
import Nightstream.SuperNeo.CheckPlan
import Nightstream.SuperNeo.SumCheck
import Nightstream.SuperNeo.SumCheck.Polynomial
import Nightstream.SuperNeo.SumCheck.VerifierCertificate
import Nightstream.SuperNeo.Sampling.FirstAccepted
import Nightstream.SuperNeo.ProjectionCheck
import Nightstream.SuperNeo.Folding.BatchArity
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
import Nightstream.SuperNeo.Folding.PiCCS.PaperProduct
import Nightstream.SuperNeo.Folding.PiRLC
import Nightstream.SuperNeo.Folding.PiDEC
import Nightstream.SuperNeo.Folding.PiDEC.Necessity
import Nightstream.SuperNeo.Folding.Nifs.PaperProfile
import Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive
import Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionStrongSet
import Nightstream.SuperNeo.Concrete.Phi81Relation
import Nightstream.SuperNeo.Concrete.Parameters
import Nightstream.SuperNeo.Concrete.Phi81StrongSet

/-!
Curated public surface for the paper-level SuperNeo model.

The scalar-row model in `Concrete.Relation` is available through an explicit
import only. Its CE evaluator is not the paper Phi81 coefficient-matrix
construction, so this facade does not export it as an alternate paper relation.
-/
