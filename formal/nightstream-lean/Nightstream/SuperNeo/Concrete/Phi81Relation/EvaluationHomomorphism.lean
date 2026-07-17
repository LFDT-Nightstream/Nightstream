import Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.BaseLinear
import Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.CarrierAction
import Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.RingKAction
import Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.Embedding
import Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.RingFLaws
import Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.PiDEC
import Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.PiRLC
import Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.PiRLCFinite

/-!
Curated homomorphism results for the typed Phi81 evaluation map.

Protocol: carried CE evaluation in `Pi_DEC` and, eventually, `Pi_RLC`.
Phase: assignment combination through derived matrix evaluation.
Constraint family: semantic evaluation only; this file emits no rows.

Owns: the closed base-field-linear interface proved in `BaseLinear`; complete
carrier packing and canonical basis-kernel/product extension proved in
`CarrierAction`; the Boolean-MLE extension-ring action proved in
`RingKAction`; exact preservation of executable Phi81 multiplication by the
coefficientwise embedding proved in `Embedding`; symbolic executable Phi81
monomial normal forms and product-order commutation proved in `RingFLaws`;
one-source matrix-action orchestration proved in `PiRLC`; canonical finite
challenge combination and the exact `PiRLC.Algebra.evaluations_hom`-shaped
field proved in `PiRLCFinite`; and the production-parameter
`PiDEC.Algebra.evaluations_hom` discharge.

Does not own: separately exported global associativity/commutativity laws,
commitment or public-input homomorphisms, challenge validity, transcript
binding, norm growth, construction of the full `PiRLC.Algebra`, or end-to-end
acceptance.

Emits constraints: no.

Authority boundary: `BaseLinear` derives every lane from the canonical matrix
source and complete assignment. `RingKAction` now proves that Boolean MLE over
the 54 coefficient tables commutes with arbitrary fixed `RingK`
multiplication, including a coefficientwise embedded `RingF` challenge.
`CarrierAction` now proves that the canonical basis-defined coefficient kernel
extends to the executable `barBasis * block` product. `RingFLaws` proves from
symbolic normal forms and bilinearity the exact product-order identity needed
to reconcile the carrier and challenge actions. `PiRLC.productOrderLaw`
discharges the local premise for every challenge before the canonical
one-source matrix action is composed. `PiRLCFinite` then folds all sources in
one head-first challenge order and proves the exact evaluation field. No caller
supplies an algebra law or alternate evaluator.

| Stage path | Child owner | Mathematical guarantee | Excluded boundary |
|---|---|---|---|
| `nifs.pi_dec.verify.recomposition.evaluations` | `BaseLinear`, `PiDEC` | exact finite `F` combinations commute with every matrix and lane | Rust/R1CS refinement |
| `nifs.pi_dec.verify.algebra.evaluations_hom` | `PiDEC` | production `b = 2`, `k = 14` discharges the exact algebra field | Rust/R1CS refinement |
| `nifs.pi_rlc.verify.assignment_action` | `CarrierAction` | block action is fixed `ringFMul`; the basis-defined kernel extends to `barBasis * block` | Rust/R1CS refinement |
| `nifs.pi_rlc.verify.evaluation_action` | `RingKAction` | fixed-ring row action commutes with Boolean MLE | Rust/R1CS refinement |
| `nifs.pi_rlc.verify.evaluation_action.embedding` | `Embedding` | coefficientwise `RingF -> RingK` preserves executable multiplication | Rust/R1CS refinement |
| `nifs.pi_rlc.verify.evaluation_hom.ring_f.normal_form` | `RingFLaws` | executable multiplication of coefficient bases has the canonical Phi81 normal form | Rust/R1CS refinement |
| `nifs.pi_rlc.verify.evaluation_hom.product_order` | `RingFLaws` | reconciles `bar * (rho * z)` with `rho * (bar * z)` | Rust/R1CS refinement |
| `nifs.pi_rlc.verify.evaluation_hom.matrices` | `PiRLC` | one canonical source action commutes through every matrix evaluation | full algebra assembly and Rust/R1CS refinement |
| `nifs.pi_rlc.verify.evaluation_hom.finite` | `PiRLCFinite` | combines every complete assignment and evaluation under the same head-first challenges | full algebra assembly and Rust/R1CS refinement |
| `nifs.pi_rlc.verify.evaluation_hom.algebra` | `PiRLCFinite` | exact evaluation-homomorphism field signature | full algebra assembly and Rust/R1CS refinement |
| `nifs.pi_rlc.verify.algebra.remaining` | `PiRLCAlgebra` | public input, commitment, strong challenges, and norm growth belong to the complete algebra | owned by the sibling construction tree |
-/
