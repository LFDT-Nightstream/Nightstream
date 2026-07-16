import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.Ring.Defs
import Mathlib.Tactic.Ring

/-!
Owns: generic scalar-product substitution, mixed topological product/linear
SSA semantics with a unique executable decoder, the retained-identity rank
condition, and bounded carry algebra.

Does not own: extraction of the concrete Rust trace into this model, proof that
the production retained matrix has full column rank, trace-range/escape
validation, or the bridge from Rust `chunks(18)` to the bounded groups.

Emits constraints: no. It proves generic relations that a concrete trace
refinement may instantiate.

Authority boundary: source and direct relations use the same authoritative
operands and result; this file does not prove a production trace exposes them
without aliasing or escape.

| Predicate/theorem | Rust stage | Guarantee | Assumptions | Permits row removal? |
|---|---|---|---|---|
| `sourceProductSum_iff_direct`, `sourceBatch_iff_direct` | `identities.*` | Explicit scalar products are equivalent to direct substitution | Finite generic commutative semiring model | No — Rust refinement open |
| `mixedSsaExecution_iff_eq_decoder`, `mixedSsaExecution_unique` | `identities.* / selective lowering` | A topological mixed SSA trace has one executable reconstruction | Valid references and topological order | No — concrete trace refinement open |
| `retainedValues_unique_of_fullColumnRank` | `selective lowering` | Retained values are unique under the rank premise | Production matrix full-column-rank proof | No — Rust refinement open |
| `carryChain_zero_iff_direct`, `carry_value_unique` | `selective lowering` | Carry groups telescope to the direct product sum | Supplied bounded groups | No — concrete grouping refinement open |
| `boundedCarryEncoding_iff_direct` | `selective lowering` | At-most-18-term groups preserve the direct relation | Concrete `chunks(18)` bridge | No — Rust refinement open |

The mixed SSA and bounded-carry theorems close the model-level reconstruction
steps. A concrete `ProductSumBatchTrace` refinement must still instantiate the
model from validated rows and prove the production retained matrix full-rank.
These results also do not claim that a one-point Pi_RLC projection establishes
coefficient-wise ring equality; that security boundary is exact-or-bad-root.
-/

namespace SuperNeo.FPrimeRecursiveVerifier.PiRlcAlgebra.ProductSumRefinement

open scoped BigOperators

universe u v

/-- The source multiplication columns for one finite product sum. -/
def ProductDefinitions {Term : Type u} {K : Type v} [Mul K]
    (left right products : Term → K) : Prop :=
  ∀ term, products term = left term * right term

/-- Source form: explicit products followed by their weighted aggregate. -/
def SourceProductSum {Term : Type u} {K : Type v}
    [CommSemiring K] [Fintype Term]
    (coefficient left right : Term → K) (result : K) : Prop :=
  ∃ products : Term → K,
    ProductDefinitions left right products ∧
      result = ∑ term, coefficient term * products term

/-- Lowered form: substitute every uniquely determined product directly. -/
def DirectProductSum {Term : Type u} {K : Type v}
    [CommSemiring K] [Fintype Term]
    (coefficient left right : Term → K) (result : K) : Prop :=
  result = ∑ term, coefficient term * (left term * right term)

/-- Canonical values of all source product columns. -/
def reconstructedProducts {Term : Type u} {K : Type v} [Mul K]
    (left right : Term → K) : Term → K :=
  fun term => left term * right term

/-- Removing the explicit product columns preserves exactly the same relation. -/
theorem sourceProductSum_iff_direct
    {Term : Type u} {K : Type v}
    [CommSemiring K] [Fintype Term]
    (coefficient left right : Term → K) (result : K) :
    SourceProductSum coefficient left right result ↔
      DirectProductSum coefficient left right result := by
  classical
  constructor
  · rintro ⟨products, hProducts, hResult⟩
    unfold DirectProductSum
    calc
      result = ∑ term, coefficient term * products term := hResult
      _ = ∑ term, coefficient term * (left term * right term) := by
        apply Finset.sum_congr rfl
        intro term _
        rw [hProducts term]
  · intro hDirect
    refine ⟨reconstructedProducts left right, ?_, ?_⟩
    · intro term
      rfl
    · exact hDirect

/-- The product-definition rows admit only the canonical reconstruction. -/
theorem products_unique
    {Term : Type u} {K : Type v} [Mul K]
    (left right products : Term → K)
    (hProducts : ProductDefinitions left right products) :
    products = reconstructedProducts left right := by
  funext term
  simpa [reconstructedProducts] using hProducts term

/-- Pointwise source semantics for every identity in one traced batch. -/
def SourceBatch {Identity Term : Type u} {K : Type v}
    [CommSemiring K] [Fintype Term]
    (coefficient left right : Identity → Term → K)
    (result : Identity → K) : Prop :=
  ∀ identity,
    SourceProductSum (coefficient identity) (left identity) (right identity)
      (result identity)

/-- Pointwise lowered semantics for every identity in one traced batch. -/
def DirectBatch {Identity Term : Type u} {K : Type v}
    [CommSemiring K] [Fintype Term]
    (coefficient left right : Identity → Term → K)
    (result : Identity → K) : Prop :=
  ∀ identity,
    DirectProductSum (coefficient identity) (left identity) (right identity)
      (result identity)

/-- Exact batch lifting used by a validated `ProductSumBatchTrace`. -/
theorem sourceBatch_iff_direct
    {Identity Term : Type u} {K : Type v}
    [CommSemiring K] [Fintype Term]
    (coefficient left right : Identity → Term → K)
    (result : Identity → K) :
    SourceBatch coefficient left right result ↔
      DirectBatch coefficient left right result := by
  constructor
  · intro hSource identity
    exact (sourceProductSum_iff_direct
      (coefficient identity) (left identity) (right identity)
      (result identity)).mp (hSource identity)
  · intro hDirect identity
    exact (sourceProductSum_iff_direct
      (coefficient identity) (left identity) (right identity)
      (result identity)).mpr (hDirect identity)

/-! ### Mixed topological SSA and exact decoding -/

/-- A source used by a linear combination: authoritative input or earlier SSA value. -/
inductive SsaReference (External : Type u) where
  | external (index : External)
  | prior (index : Nat)

/-- One scaled source inside an R1CS linear combination. -/
structure SsaLinearTerm (External : Type u) (K : Type v) where
  source : SsaReference External
  coefficient : K

/-- Constant plus a finite list of scaled SSA sources. -/
structure SsaLinearForm (External : Type u) (K : Type v) where
  constant : K
  terms : List (SsaLinearTerm External K)

/-- The two exact fresh-column definitions accepted by the Rust parser. -/
inductive SsaInstruction (External : Type u) (K : Type v) where
  | product (left right : SsaLinearForm External K)
  | linear (value : SsaLinearForm External K)

/-- A prior reference is valid only when its target has already been defined. -/
def ssaReferenceWithin {External : Type u}
    (priorCount : Nat) : SsaReference External → Prop
  | .external _ => True
  | .prior index => index < priorCount

/-- Every source in a linear form is authoritative or topologically earlier. -/
def SsaLinearForm.ReferencesWithin
    {External : Type u} {K : Type v}
    (form : SsaLinearForm External K) (priorCount : Nat) : Prop :=
  ∀ term ∈ form.terms, ssaReferenceWithin priorCount term.source

/-- Topological validity of either exact instruction family. -/
def SsaInstruction.ReferencesWithin
    {External : Type u} {K : Type v}
    (instruction : SsaInstruction External K) (priorCount : Nat) : Prop :=
  match instruction with
  | .product left right =>
      left.ReferencesWithin priorCount ∧ right.ReferencesWithin priorCount
  | .linear value => value.ReferencesWithin priorCount

/-- Value of one authoritative or previously reconstructed source. -/
def ssaReferenceValue
    {External : Type u} {K : Type v} [Zero K]
    (external : External → K) (prior : List K) :
    SsaReference External → K
  | .external index => external index
  | .prior index => prior.getD index 0

/-- Evaluate one exact R1CS linear combination. -/
def evalSsaLinearForm
    {External : Type u} {K : Type v} [CommRing K]
    (external : External → K) (prior : List K)
    (form : SsaLinearForm External K) : K :=
  form.terms.foldl
    (fun sum term =>
      sum + term.coefficient * ssaReferenceValue external prior term.source)
    form.constant

/-- Evaluate one product or linear fresh-column definition. -/
def evalSsaInstruction
    {External : Type u} {K : Type v} [CommRing K]
    (external : External → K) (prior : List K) :
    SsaInstruction External K → K
  | .product left right =>
      evalSsaLinearForm external prior left *
        evalSsaLinearForm external prior right
  | .linear value => evalSsaLinearForm external prior value

/-- Every instruction is topological at its position in the fresh-column interval. -/
def SsaProgramTopological
    {External : Type u} {K : Type v} :
    Nat → List (SsaInstruction External K) → Prop
  | _, [] => True
  | priorCount, instruction :: rest =>
      instruction.ReferencesWithin priorCount ∧
        SsaProgramTopological (priorCount + 1) rest

/-- Executable reconstruction of every fresh column, in source-row order. -/
def decodeMixedSsaFrom
    {External : Type u} {K : Type v} [CommRing K]
    (external : External → K) :
    List K → List (SsaInstruction External K) → List K
  | prior, [] => prior
  | prior, instruction :: rest =>
      let value := evalSsaInstruction external prior instruction
      decodeMixedSsaFrom external (prior ++ [value]) rest

/-- Reconstruction appends exactly one value for every source row/column. -/
theorem decodeMixedSsaFrom_length
    {External : Type u} {K : Type v} [CommRing K]
    (external : External → K)
    (prior : List K)
    (program : List (SsaInstruction External K)) :
    (decodeMixedSsaFrom external prior program).length =
      prior.length + program.length := by
  induction program generalizing prior with
  | nil => simp [decodeMixedSsaFrom]
  | cons instruction rest inductionHypothesis =>
      simp only [decodeMixedSsaFrom, List.length_cons]
      rw [inductionHypothesis]
      simp only [List.length_append, List.length_singleton]
      omega

/-- Relational source semantics with explicit per-column defining equations. -/
inductive MixedSsaExecution
    {External : Type u} {K : Type v} [CommRing K]
    (external : External → K) :
    List K → List (SsaInstruction External K) → List K → Prop where
  | done (prior : List K) : MixedSsaExecution external prior [] prior
  | step
      (prior : List K)
      (instruction : SsaInstruction External K)
      (rest : List (SsaInstruction External K))
      (value : K)
      (output : List K)
      (valid : instruction.ReferencesWithin prior.length)
      (definition : value = evalSsaInstruction external prior instruction)
      (tail : MixedSsaExecution external (prior ++ [value]) rest output) :
      MixedSsaExecution external prior (instruction :: rest) output

/-- A topological mixed program always has the executable source execution. -/
theorem decodeMixedSsaFrom_executes
    {External : Type u} {K : Type v} [CommRing K]
    (external : External → K)
    (prior : List K)
    (program : List (SsaInstruction External K))
    (hTopology : SsaProgramTopological prior.length program) :
    MixedSsaExecution external prior program
      (decodeMixedSsaFrom external prior program) := by
  induction program generalizing prior with
  | nil => exact MixedSsaExecution.done prior
  | cons instruction rest inductionHypothesis =>
      rcases hTopology with ⟨hInstruction, hRest⟩
      let value := evalSsaInstruction external prior instruction
      refine MixedSsaExecution.step prior instruction rest value
        (decodeMixedSsaFrom external (prior ++ [value]) rest)
        hInstruction rfl ?_
      apply inductionHypothesis
      simpa [value] using hRest

/-- Every satisfying mixed SSA execution is the executable reconstruction. -/
theorem mixedSsaExecution_eq_decoder
    {External : Type u} {K : Type v} [CommRing K]
    {external : External → K}
    {prior : List K}
    {program : List (SsaInstruction External K)}
    {output : List K}
    (hExecution : MixedSsaExecution external prior program output) :
    output = decodeMixedSsaFrom external prior program := by
  induction hExecution with
  | done => rfl
  | step prior instruction rest value output _ hDefinition _ inductionHypothesis =>
      simp only [decodeMixedSsaFrom]
      rw [← hDefinition]
      exact inductionHypothesis

/-- Source satisfaction is equivalent to equality with the exact decoder. -/
theorem mixedSsaExecution_iff_eq_decoder
    {External : Type u} {K : Type v} [CommRing K]
    (external : External → K)
    (prior : List K)
    (program : List (SsaInstruction External K))
    (output : List K)
    (hTopology : SsaProgramTopological prior.length program) :
    MixedSsaExecution external prior program output ↔
      output = decodeMixedSsaFrom external prior program := by
  constructor
  · exact mixedSsaExecution_eq_decoder
  · intro hOutput
    rw [hOutput]
    exact decodeMixedSsaFrom_executes external prior program hTopology

/-- The defining rows admit at most one full fresh-column reconstruction. -/
theorem mixedSsaExecution_unique
    {External : Type u} {K : Type v} [CommRing K]
    {external : External → K}
    {prior : List K}
    {program : List (SsaInstruction External K)}
    {left right : List K}
    (hLeft : MixedSsaExecution external prior program left)
    (hRight : MixedSsaExecution external prior program right) :
    left = right :=
  (mixedSsaExecution_eq_decoder hLeft).trans
    (mixedSsaExecution_eq_decoder hRight).symm

/-! ### Retained-identity rank boundary -/

/-- Linear map from retained values to the retained coefficients of all identities. -/
def retainedIdentityMap
    {Identity Retained : Type u} {K : Type v}
    [CommSemiring K] [Fintype Retained]
    (matrix : Identity → Retained → K)
    (values : Retained → K) : Identity → K :=
  fun identity => ∑ retained, matrix identity retained * values retained

/-- Exact full-column-rank obligation checked by Rust Gaussian elimination. -/
def RetainedMatrixFullColumnRank
    {Identity Retained : Type u} {K : Type v}
    [CommSemiring K] [Fintype Retained]
    (matrix : Identity → Retained → K) : Prop :=
  Function.Injective (retainedIdentityMap matrix)

/-- Full column rank makes the retained identity boundary unique. -/
theorem retainedValues_unique_of_fullColumnRank
    {Identity Retained : Type u} {K : Type v}
    [CommSemiring K] [Fintype Retained]
    {matrix : Identity → Retained → K}
    (hRank : RetainedMatrixFullColumnRank matrix)
    {left right : Retained → K}
    (hIdentities : ∀ identity,
      retainedIdentityMap matrix left identity =
        retainedIdentityMap matrix right identity) :
    left = right :=
  hRank (funext hIdentities)

/--
Exact row semantics for splitting one long product sum into bounded groups.
`previous` is zero for the first row and the preceding canonical carry for
later rows. The final row targets `result` instead of allocating another carry.
-/
inductive CarryChain {K : Type v} [CommRing K] :
    K → List K → K → Prop where
  | last (previous group result : K)
      (row : group = result - previous) :
      CarryChain previous [group] result
  | next (previous group next : K) (rest : List K) (carry result : K)
      (row : group = carry - previous)
      (tail : CarryChain carry (next :: rest) result) :
      CarryChain previous (group :: next :: rest) result

/-- Every emitted carry row telescopes to the unsplit product-sum identity. -/
theorem carryChain_sound
    {K : Type v} [CommRing K]
    {previous result : K} {groups : List K}
    (hChain : CarryChain previous groups result) :
    result = previous + groups.sum := by
  induction hChain with
  | last previous group result hRow =>
      simp only [List.sum_cons, List.sum_nil, add_zero]
      rw [hRow]
      ring
  | next previous group next rest carry result hRow tail inductionHypothesis =>
      simp only [List.sum_cons]
      rw [inductionHypothesis, hRow]
      simp only [List.sum_cons]
      ring

/-- The direct identity reconstructs the unique prefix carry at every row. -/
theorem carryChain_complete
    {K : Type v} [CommRing K]
    (previous result head : K) (tail : List K)
    (hDirect : result = previous + (head :: tail).sum) :
    CarryChain previous (head :: tail) result := by
  induction tail generalizing previous head with
  | nil =>
      apply CarryChain.last
      simp only [List.sum_cons, List.sum_nil, add_zero] at hDirect
      rw [hDirect]
      ring
  | cons next rest inductionHypothesis =>
      let carry := previous + head
      apply CarryChain.next previous head next rest carry result
      · dsimp [carry]
        ring
      · apply inductionHypothesis carry next
        simp only [List.sum_cons] at hDirect ⊢
        dsimp [carry]
        rw [hDirect]
        ring

/-- Starting at zero, the bounded-row chain is exactly the direct sum. -/
theorem carryChain_zero_iff_direct
    {K : Type v} [CommRing K]
    (result head : K) (tail : List K) :
    CarryChain 0 (head :: tail) result ↔
      result = (head :: tail).sum := by
  constructor
  · intro hChain
    simpa using carryChain_sound hChain
  · intro hDirect
    apply carryChain_complete 0 result head tail
    simpa using hDirect

/-- A carry satisfying one row is forced to be the next prefix sum. -/
theorem carry_value_unique
    {K : Type v} [CommRing K]
    (previous group carry : K)
    (hRow : group = carry - previous) :
    carry = previous + group := by
  rw [hRow]
  ring

/-! ### Fixed arity-18 product groups -/

/-- Maximum number of products emitted in one selective CCS row. -/
def maxProductTerms : Nat := 18

theorem maxProductTerms_eq : maxProductTerms = 18 := rfl

/-- One scaled product in the direct identity. -/
structure BoundedProductTerm (K : Type v) where
  coefficient : K
  left : K
  right : K

def BoundedProductTerm.value
    {K : Type v} [Mul K] (term : BoundedProductTerm K) : K :=
  term.coefficient * term.left * term.right

/-- One nonempty selective row containing at most eighteen products. -/
structure BoundedProductGroup (K : Type v) where
  terms : List (BoundedProductTerm K)
  nonempty : terms ≠ []
  arity : terms.length ≤ maxProductTerms

def BoundedProductGroup.value
    {K : Type v} [CommRing K] (group : BoundedProductGroup K) : K :=
  (group.terms.map BoundedProductTerm.value).sum

/-- Ordered product values represented by a list of bounded groups. -/
def flattenedProductValues
    {K : Type v} [CommRing K]
    (groups : List (BoundedProductGroup K)) : List K :=
  groups.flatMap fun group => group.terms.map BoundedProductTerm.value

/-- The sum of group values is exactly the sum of the original ordered terms. -/
theorem boundedGroupValues_sum
    {K : Type v} [CommRing K]
    (groups : List (BoundedProductGroup K)) :
    (groups.map BoundedProductGroup.value).sum =
      (flattenedProductValues groups).sum := by
  induction groups with
  | nil => rfl
  | cons group rest inductionHypothesis =>
      simp [BoundedProductGroup.value, flattenedProductValues,
        inductionHypothesis]

/-- Carry-row relation for one nonempty list of arity-bounded product groups. -/
def BoundedCarryEncoding
    {K : Type v} [CommRing K]
    (result : K)
    (head : BoundedProductGroup K)
    (tail : List (BoundedProductGroup K)) : Prop :=
  CarryChain 0
    (BoundedProductGroup.value head ::
      tail.map BoundedProductGroup.value)
    result

/-- Arity-18 carry rows are sound and complete for the unsplit product sum. -/
theorem boundedCarryEncoding_iff_direct
    {K : Type v} [CommRing K]
    (result : K)
    (head : BoundedProductGroup K)
    (tail : List (BoundedProductGroup K)) :
    BoundedCarryEncoding result head tail ↔
      result = (flattenedProductValues (head :: tail)).sum := by
  unfold BoundedCarryEncoding
  rw [carryChain_zero_iff_direct]
  change result = ((head :: tail).map BoundedProductGroup.value).sum ↔ _
  rw [boundedGroupValues_sum]

end SuperNeo.FPrimeRecursiveVerifier.PiRlcAlgebra.ProductSumRefinement
