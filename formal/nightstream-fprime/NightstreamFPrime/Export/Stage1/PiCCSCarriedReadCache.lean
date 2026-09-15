import NightstreamFPrime.Export.Stage1.PiCCSCarriedRead
import NightstreamFPrime.Export.Stage1.PiCCSLinearRows
import Std.Data.HashMap.Lemmas

/-! Cache complete carried blocks computed from the original block function.
Repeated keys reuse the first computed block; misses use the original function. -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiCCSCarriedReadCache

open NightstreamFPrime.Spec
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Export.Stage1.PiRLCPartialTrace

/-- Compute each distinct requested block once. Every stored value comes from
`blocks`; the key list supplies no values or correctness claims. -/
def prepareCache (keys : List Nat) (blocks : Nat → Vector K ringDegree) :
    Std.HashMap Nat (Vector K ringDegree) :=
  keys.foldl (fun cache key =>
    if cache.contains key then cache else cache.insert key (blocks key)) ∅

/-- The original block function is evaluated only when the key is absent. -/
def cachedRead (cache : Std.HashMap Nat (Vector K ringDegree))
    (blocks : Nat → Vector K ringDegree) (key : Nat) : Vector K ringDegree :=
  match cache[key]? with
  | some value => value
  | none => blocks key

private theorem insert_read (cache : Std.HashMap Nat (Vector K ringDegree))
    (blocks : Nat → Vector K ringDegree)
    (valid : ∀ key, cachedRead cache blocks key = blocks key)
    (inserted : Nat) :
    ∀ key, cachedRead (cache.insert inserted (blocks inserted)) blocks key =
      blocks key := by
  intro key
  by_cases equal : inserted = key
  · subst key
    simp [cachedRead]
  · simpa [cachedRead, Std.HashMap.getElem?_insert, equal] using valid key

private theorem fold_read (keys : List Nat) (blocks : Nat → Vector K ringDegree)
    (cache : Std.HashMap Nat (Vector K ringDegree))
    (valid : ∀ key, cachedRead cache blocks key = blocks key) :
    ∀ key, cachedRead
      (keys.foldl (fun current inserted =>
        if current.contains inserted then current
        else current.insert inserted (blocks inserted)) cache) blocks key =
      blocks key := by
  induction keys generalizing cache with
  | nil => exact valid
  | cons inserted keys inductionHypothesis =>
      simp only [List.foldl_cons]
      apply inductionHypothesis
      by_cases present : cache.contains inserted
      · simpa only [if_pos present] using valid
      · simpa only [if_neg present] using insert_read cache blocks valid inserted

/-- Cache preparation preserves every original block read, including repeated
keys and keys absent from the preparation list. There is no support premise. -/
theorem cachedRead_prepareCache (keys : List Nat)
    (blocks : Nat → Vector K ringDegree) (key : Nat) :
    cachedRead (prepareCache keys blocks) blocks key = blocks key := by
  unfold prepareCache
  apply fold_read keys blocks ∅
  intro requested
  simp [cachedRead]

/-- Request blocks from the existing invocation inputs and retained cells.
Missing keys remain correct through the original-read fallback. -/
def interfaceKeys {columns : Nat} (interface : PoseidonSboxPlan.Interface columns) : List Nat :=
  interface.oneColumn.val / ringDegree ::
    ((List.ofFn interface.input ++ List.ofFn interface.sboxOutput ++ List.ofFn interface.output).flatMap
      fun form => form.entries.map (fun entry => entry.column.val / ringDegree))

/-- Share each requested original block across both scalar projections and
all 94 numeric rows of the same invocation. -/
def invocation {columns : Nat}
    (basis : FixedArray (Vector K ringDegree) ringDegree)
    (blocks : Nat → Vector K ringDegree) (interface : PoseidonSboxPlan.Interface columns) :
    Vector (Vector K Spec.ProductionRelation.matrixCount) 94 :=
  let cache := prepareCache (interfaceKeys interface) blocks
  PiCCSLinearRows.invocation (PiCCSCarriedRead.read basis (cachedRead cache blocks)) interface

/-- Caching changes no stored row or matrix port, for arbitrary original
blocks and interfaces. No assumption about the requested support is needed. -/
theorem invocation_eq {columns : Nat}
    (basis : FixedArray (Vector K ringDegree) ringDegree)
    (blocks : Nat → Vector K ringDegree) (interface : PoseidonSboxPlan.Interface columns) :
    invocation basis blocks interface =
      PiCCSLinearRows.invocation (PiCCSCarriedRead.read basis blocks) interface := by
  have reads : cachedRead (prepareCache (interfaceKeys interface) blocks) blocks = blocks :=
    funext (cachedRead_prepareCache (interfaceKeys interface) blocks)
  dsimp only [invocation]
  rw [reads]

end NightstreamFPrime.Export.Stage1.PiCCSCarriedReadCache
