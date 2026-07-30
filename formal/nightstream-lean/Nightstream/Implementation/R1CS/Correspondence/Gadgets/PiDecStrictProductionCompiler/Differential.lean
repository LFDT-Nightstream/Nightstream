import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Nifs.PiDecCanonicalX
import Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Radix.UniformSignedDigits

/-!
Compact Rust/Lean differential execution at the strict `PiDEC` public-X
boundary.

Assurance tier: artifact-checked differential evidence for eleven bounded
cases over the fixed `54 x 5`, fourteen-child profile.

Owns: an independent Lean Boolean checker over typed parent/ordered-child
field values and paper-shape header data; exact agreement with Rust results
exported by the live compact fixture; honest and mutation cases.

Does not own: whole-`PiDEC` acceptance, commitment or evaluation values,
private-column decoding, delayed projection, or security primitives. The
Rust result bit is compared with independent semantics and is never used as
authority.

Emits constraints: no.
-/

namespace Nightstream.Implementation.R1CS.PiDecStrictProductionCompiler.Differential

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Radix
open Nightstream.Implementation.R1CS

abbrev Case :=
  Artifacts.FPrimeSelectiveFixedPoint.Nifs.PiDecCanonicalX.DifferentialCase

private abbrev cases :=
  Artifacts.FPrimeSelectiveFixedPoint.Nifs.PiDecCanonicalX.differentialCases

/-- Independent typed meaning of one compact cross-language case. The child
list is compared with the verifier-computed signed binary split; no generated
row or Rust acceptance bit occurs in this predicate. -/
def Accepted (case : Case) : Prop :=
  case.profileTag = 0 /\
  case.recursiveSelector = 1 /\
  case.publicColumn < 270 /\
  case.children.length = productionGlobalParams.k /\
  case.childEvaluationArities = List.replicate productionGlobalParams.k 13 /\
  case.children.map fieldOfNat =
    List.ofFn (splitScalar (fieldOfNat case.parent))

instance acceptedDecidable (case : Case) : Decidable (Accepted case) := by
  unfold Accepted
  infer_instance

/-- Executable independent Lean verifier used by the differential fixture. -/
def check (case : Case) : Bool :=
  decide (Accepted case)

theorem check_eq_true_iff (case : Case) :
    check case = true <-> Accepted case := by
  exact decide_eq_true_iff

/-- All generated Rust result bits agree with the independent Lean checker.

The executable certificate contains exactly eleven proof-free records. Each
record has at most fourteen child residues and fourteen arity naturals; no
row list, proof-carrying structure, or private assignment is evaluated. -/
theorem generated_cases_agree :
    (cases.all fun case =>
        decide (check case = case.rustAccepted)) = true := by
  decide

end Nightstream.Implementation.R1CS.PiDecStrictProductionCompiler.Differential
