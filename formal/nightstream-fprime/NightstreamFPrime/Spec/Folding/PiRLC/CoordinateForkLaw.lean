import Mathlib.Analysis.SpecificLimits.Basic
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Ring
import Mathlib.Algebra.BigOperators.Field
import Mathlib.Algebra.Order.BigOperators.Expect
import Mathlib.Logic.Equiv.Prod
import Mathlib.Data.Fintype.Option
import NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtraction
import Mathlib.Logic.Function.Basic

/-!
Connects the response-trace probability law to the actual typed PiRLC search.
The challenge carrier contains only verifier-valid scalars. Oracle failures
remain in the probability trace and are omitted only when passing candidate
assignments to the existing first-response selector.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Folding.PiRLC.CoordinateForkLaw

open NightstreamFPrime.Spec
open PaperForkExtraction

variable {Structure Assignment PublicInput Point Evaluation Commitment Scalar : Type*}
  {semantics : RelationSemantics
    Structure Assignment PublicInput Point Evaluation Commitment}
  {params : GlobalParams} {arity : BatchArity params}
  (algebra : Algebra Structure Assignment PublicInput Point Evaluation Commitment
    Scalar semantics params)

abbrev Challenge := {challenge : Scalar // algebra.challengeValid challenge}

def scalarVector (vector : Fin arity.total → Challenge algebra) : Fin arity.total → Scalar :=
  fun index => (vector index).val

end NightstreamFPrime.Spec.Folding.PiRLC.CoordinateForkLaw
