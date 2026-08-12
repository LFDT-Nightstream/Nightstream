import Nightstream.Implementation.NebulaV2.MemoryClaimRows
import Nightstream.Implementation.NebulaV2.ConditionalCarriedEqualityRows
import Nightstream.Implementation.R1CS.Canonical.KConcreteBridge
import Nightstream.Implementation.R1CS.Canonical.KMulChain

/-!
Contract: exact two-repetition terminal product-balance rows for Nebula V2.

Assurance tier: implementation model.

Owns two concrete extension-field products and one extension equality for each
of the two fixed repetitions. Satisfying rows derive
`h_is * h_ws = h_rs * h_fs` for the exact parsed claim products.

Does not own the phase Boolean row, product updates, full carry transitions,
absolute generated columns, or honest auxiliary-witness construction.

Emits constraints: yes.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.NebulaV2.MemoryProductBalanceRows

open Nightstream.Implementation.NebulaV2.MemoryClaimCodec
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS.Canonical.KHorner
open Nightstream.Implementation.R1CS.Canonical.KMul
open Nightstream.Protocol.NebulaV2.ProductState
open Nightstream.SuperNeo.Concrete

private theorem pair_ext
    {left right : Pair} (low : left.low = right.low)
    (high : left.high = right.high) : left = right := by
  cases left
  cases right
  simp_all

/-- The exact executable-field balance predicate used by the row layer. -/
def ConcreteBalanced (products : State K) : Prop :=
  ∀ repetition,
    K.mul (products repetition).initialSnapshot
        (products repetition).writes =
      K.mul (products repetition).reads
        (products repetition).finalSnapshot

def roleValue (products : Four K) : ProductRole → K
  | .initialSnapshot => products.initialSnapshot
  | .writes => products.writes
  | .reads => products.reads
  | .finalSnapshot => products.finalSnapshot

/-- Product coordinates reuse the exact typed columns owned by the complete
memory-claim validator. Each multiplication has its own three-column frame. -/
structure Layout where
  claim : MemoryClaimRows.Layout
  closePhaseColumn : Nat
  leftFrame : Fin 2 → KMul.Frame
  rightFrame : Fin 2 → KMul.Frame

def Layout.productColumn (layout : Layout) (repetition : Fin 2)
    (role : ProductRole) (limb : Fin 2) : Nat :=
  Relabel.column
    (layout.claim.fieldColumnMap (.product 1 repetition role limb))
    CanonicalU64.varCol

def Layout.productCarried (layout : Layout) (repetition : Fin 2)
    (role : ProductRole) : KMul.Carried where
  low := [(layout.productColumn repetition role 0, 1)]
  high := [(layout.productColumn repetition role 1, 1)]

def leftOutput (layout : Layout) (repetition : Fin 2) : KMul.Carried :=
  KMulChain.frameOutput (layout.leftFrame repetition)

def rightOutput (layout : Layout) (repetition : Fin 2) : KMul.Carried :=
  KMulChain.frameOutput (layout.rightFrame repetition)

def rowsFor (layout : Layout) (repetition : Fin 2) : List Row :=
  KMul.rows
      (layout.productCarried repetition .initialSnapshot)
      (layout.productCarried repetition .writes)
      (layout.leftFrame repetition) ++
    KMul.rows
      (layout.productCarried repetition .reads)
      (layout.productCarried repetition .finalSnapshot)
      (layout.rightFrame repetition) ++
    ConditionalCarriedEqualityRows.rows layout.closePhaseColumn
      (leftOutput layout repetition)
      (rightOutput layout repetition)

def rows (layout : Layout) : List Row :=
  (List.ofFn fun repetition : Fin 2 => rowsFor layout repetition).flatten

theorem rowsFor_length (layout : Layout) (repetition : Fin 2) :
    (rowsFor layout repetition).length = 8 := by
  simp [rowsFor, KMul.rows_length,
    ConditionalCarriedEqualityRows.rows_length]

theorem rows_length_exact (layout : Layout) :
    (rows layout).length = 16 := by
  simp [rows, rowsFor_length]

/-- The assignment gives each product coordinate its exact typed value. -/
def ProductsPlaced (layout : Layout) (assignment : Nat → Nat)
    (products : State K) : Prop :=
  ∀ repetition role limb,
    assignment (layout.productColumn repetition role limb) =
      kLimbValue (roleValue (products repetition) role) limb

/-- Parsed memory-claim columns place the exact after-state products used by
the balance block. -/
theorem productsPlaced_of_parsed_claim
    {layout : Layout} {assignment : Nat → Nat}
    {claim : MemoryClaimCodec.Claim}
    (parsed : MemoryClaimRows.ParsedColumnsMatch layout.claim assignment claim) :
    ProductsPlaced layout assignment claim.productsAfter := by
  intro repetition role limb
  have exactColumn := parsed.fields (.product 1 repetition role limb)
  calc
    assignment (layout.productColumn repetition role limb) =
        claim.fieldValue (.product 1 repetition role limb) := exactColumn
    _ = kLimbValue (roleValue (claim.productsAfter repetition) role) limb := by
      cases role <;> fin_cases limb <;> rfl

theorem carriedValue_eq_ofConcrete
    {layout : Layout} {assignment : Nat → Nat} {products : State K}
    (placed : ProductsPlaced layout assignment products)
    (repetition : Fin 2) (role : ProductRole) :
    carriedValue assignment (layout.productCarried repetition role) =
      KConcreteBridge.ofConcrete (roleValue (products repetition) role) := by
  unfold Layout.productCarried carriedValue KConcreteBridge.ofConcrete
  simp only [KMul.lcEval_singleton_col]
  rw [placed repetition role 0, placed repetition role 1]
  apply pair_ext
  · exact Nat.mod_eq_of_lt (roleValue (products repetition) role).c0.isLt
  · exact Nat.mod_eq_of_lt (roleValue (products repetition) role).c1.isLt

private theorem rowsFor_hold
    {layout : Layout} {assignment : Nat → Nat}
    (holds : Satisfies (rows layout) assignment)
    (repetition : Fin 2) :
    Satisfies (rowsFor layout repetition) assignment := by
  intro row member
  apply holds row
  rw [rows]
  apply List.mem_flatten.mpr
  exact ⟨rowsFor layout repetition, List.mem_ofFn.mpr ⟨repetition, rfl⟩,
    member⟩

private theorem left_rows_hold
    {layout : Layout} {assignment : Nat → Nat} {repetition : Fin 2}
    (holds : Satisfies (rowsFor layout repetition) assignment) :
    Satisfies
      (KMul.rows
        (layout.productCarried repetition .initialSnapshot)
        (layout.productCarried repetition .writes)
        (layout.leftFrame repetition)) assignment := by
  intro row member
  exact holds row (by simp [rowsFor, member])

private theorem right_rows_hold
    {layout : Layout} {assignment : Nat → Nat} {repetition : Fin 2}
    (holds : Satisfies (rowsFor layout repetition) assignment) :
    Satisfies
      (KMul.rows
        (layout.productCarried repetition .reads)
        (layout.productCarried repetition .finalSnapshot)
        (layout.rightFrame repetition)) assignment := by
  intro row member
  exact holds row (by simp [rowsFor, member])

private theorem equality_rows_hold
    {layout : Layout} {assignment : Nat → Nat} {repetition : Fin 2}
    (holds : Satisfies (rowsFor layout repetition) assignment) :
    Satisfies
      (ConditionalCarriedEqualityRows.rows layout.closePhaseColumn
        (leftOutput layout repetition)
        (rightOutput layout repetition)) assignment := by
  intro row member
  exact holds row (by simp [rowsFor, member])

theorem repetition_balanced_of_rows
    {layout : Layout} {assignment : Nat → Nat} {products : State K}
    (one : assignment 0 = 1)
    (phaseClosed : assignment layout.closePhaseColumn = 0)
    (placed : ProductsPlaced layout assignment products)
    (holds : Satisfies (rows layout) assignment)
    (repetition : Fin 2) :
    K.mul (products repetition).initialSnapshot
        (products repetition).writes =
      K.mul (products repetition).reads
        (products repetition).finalSnapshot := by
  have localHolds := rowsFor_hold holds repetition
  have leftSound := KMulChain.frameOutput_sound assignment
    (layout.productCarried repetition .initialSnapshot)
    (layout.productCarried repetition .writes)
    (layout.leftFrame repetition) (left_rows_hold localHolds)
  have rightSound := KMulChain.frameOutput_sound assignment
    (layout.productCarried repetition .reads)
    (layout.productCarried repetition .finalSnapshot)
    (layout.rightFrame repetition) (right_rows_hold localHolds)
  have rightLowCanonical : Program.CanonicalTerms
      (rightOutput layout repetition).low := by
    simp [rightOutput, KMulChain.frameOutput, KMul.outLow,
      Program.CanonicalTerms, goldilocksP]
  have rightHighCanonical : Program.CanonicalTerms
      (rightOutput layout repetition).high := by
    simp [rightOutput, KMulChain.frameOutput, KMul.outHigh,
      Program.CanonicalTerms, goldilocksP]
  have outputsEqual := ConditionalCarriedEqualityRows.rows_sound_closed one
    phaseClosed rightLowCanonical rightHighCanonical
    (equality_rows_hold localHolds)
  have initialExact :=
    carriedValue_eq_ofConcrete placed repetition .initialSnapshot
  have writesExact := carriedValue_eq_ofConcrete placed repetition .writes
  have readsExact := carriedValue_eq_ofConcrete placed repetition .reads
  have finalExact :=
    carriedValue_eq_ofConcrete placed repetition .finalSnapshot
  simp only [roleValue] at initialExact writesExact readsExact finalExact
  apply KConcreteBridge.ofConcrete_injective
  rw [KConcreteBridge.ofConcrete_mul, KConcreteBridge.ofConcrete_mul]
  rw [← initialExact, ← writesExact, ← readsExact, ← finalExact]
  exact leftSound.symm.trans (outputsEqual.trans rightSound)

/-- All 16 rows derive both fixed product-balance equations. -/
theorem concreteBalanced_of_rows
    {layout : Layout} {assignment : Nat → Nat} {products : State K}
    (one : assignment 0 = 1)
    (phaseClosed : assignment layout.closePhaseColumn = 0)
    (placed : ProductsPlaced layout assignment products)
    (holds : Satisfies (rows layout) assignment) :
    ConcreteBalanced products := by
  intro repetition
  exact repetition_balanced_of_rows one phaseClosed placed holds repetition

/-- Adapter from complete parsed claim columns to the exact terminal balance
predicate. -/
theorem parsed_claim_balanced_of_rows
    {layout : Layout} {assignment : Nat → Nat}
    {claim : MemoryClaimCodec.Claim}
    (one : assignment 0 = 1)
    (phaseClosed : assignment layout.closePhaseColumn = 0)
    (parsed : MemoryClaimRows.ParsedColumnsMatch layout.claim assignment claim)
    (holds : Satisfies (rows layout) assignment) :
    ConcreteBalanced claim.productsAfter :=
  concreteBalanced_of_rows one phaseClosed
    (productsPlaced_of_parsed_claim parsed) holds

end Nightstream.Implementation.NebulaV2.MemoryProductBalanceRows
