import tests.AxiomAudit
import NightstreamFPrime.Spec.Compression.Akita.Replay.Stage3
import NightstreamFPrime.Spec.Compression.Akita.Replay.Stage3Table
import NightstreamFPrime.Spec.Compression.Akita.Replay.Stage3Match
import NightstreamFPrime.Spec.Compression.Akita.Key30Opening
import NightstreamFPrime.Spec.Compression.Akita.Replay.Prefix
import NightstreamFPrime.Spec.Compression.Akita.Replay.FusedJoin

/-! Axiom audits for Stage 3 of the replay of Akita openings: the weight of Stage 3 of the code
(`Replay/Stage3Weight.lean`) and the run of Stage 3 against `setupCheck`
(`Replay/Stage3.lean`), and the setup in pages of the code against the setup expansion of the
specification (`Replay/Prefix.lean`), and the weight of Stage 3 of the code against the weight
of `Batch.codeScan` (`Replay/Stage3Table.lean`, `Replay/Stage3Sum.lean`,
`Replay/Stage3Match.lean`), with the hypotheses of `stage3Weight_codeScan` at the root of
`key30.aks`. -/

#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.powerScaleExt_into
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.eqTable_size
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.eqTable_getD
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.eqIndex_mod
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.eqIndex_append_into
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.EqSplit.eval_into
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.stage3Run_runs
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.paged_apply
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.forIn_kept
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.kept_run
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.page_toList
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.kept_words
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.finEquiv_mk
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.page_expand
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.card_page
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.sum_paged
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.short_fraction
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.paged_le
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.stage3In
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.step_stage3_runs
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.setupAt
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.runAt
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.mle_spread
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.mle_run
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.sum_shared
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.deltas
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.deltas_append
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.imageRow_deltas
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.liftEntry_sub
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.liftWeight_sub
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.innerRows
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.outerRows
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.openRows
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.groupDeltas
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.group_deltas
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.rows_deltas
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.mle_spread_fin
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.codeWeight_sub
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.rows_mle
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.shared_zero
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.aeval_rep_zero
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.aeval_rep_mul_cast
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.scan_relocate
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.run_le
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.setupPair
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.setupPair_congr
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.setupPair_add
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.setupPair_sum
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.setupPair_mul_left
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.pair_sum
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.pair_sum'
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.inner_deltas
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.inner_pair
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.innerCode_slice
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.outer_deltas
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.outer_pair
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.open_deltas
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.open_pair
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.innerPart
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.outerPart
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.openPart
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.codeScan_mle
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.pick_of
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.stage3Weight_eq
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.placedFrom
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.placed_foldl
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.placed_eq
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.rowsBefore_zero
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.rowsBefore_succ
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.placedFrom_ofFn
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Ext.neg_into
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.sumBelow_into
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.range_getD
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.sum_range_if
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Side.of_ratio
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Side.of_proj
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Side.of_high
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.laneIn
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.lane_into
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.aIn
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.bIn
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.dIn
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.groupIn
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.ofNat_into
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.groupWeight_into
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.stage3Weight_into
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.eqAt_even
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.eqAt_odd
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.sum_pow_eqAt
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.powerScale_take_add
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.setupAt_split
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.laneIn_pow
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.runAt_lane
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.div_add_mul
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.log_le_of
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.pow_log_of
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.log_ge_of
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.offset_rowsBefore
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.div_add_le
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.last_slice
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.foldl_max_eq
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.foldl_min_pow
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.groupOf
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.ofBatch_groups
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.unitLen_eq
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.rows_eq
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.dCols_eq
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.zLen_eq
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.innerStart_eq
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.sliceStart_eq
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.groupSliceStart
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.maxSlice_eq
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.sliceStart_mono
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.min_pow
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.ofBatch_base
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.ofBatch_openRow
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.ofBatch_dColsAll
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.mainStart_order
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.responseCode_val
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.outerSetupCode_div
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.openSetupCode_val
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.modNat_one
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.inner_match
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.outer_match
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.openGroup
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.open_groups
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.open_match
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.stage3Weight_codeScan
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.stage3Run_codeScan

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Lib

/-- The hypotheses of `stage3Weight_codeScan` that are numbers hold at the root of each row of
`key30.aks`: `low` for the block `2^7` of `Key30.root_blocked`, `gadget` for the response base
`2^3`, and `innerFit`, `outerFit`, `openFit` for the `2^23` cells of the setup prefix. -/
theorem key30_root_numbers : ∀ r ∈ Key30.rows, 7 ≤ Key30.rootVars ∧
    (∀ g, ((Key30.root r).fold g).responseBase = 2 ^ 3) ∧
    (∀ g, (((Key30.root r).shape g).innerInstance false 0).size ≤ 2 ^ Key30.setupVars) ∧
    (∀ g, ((Key30.root r).shape g).outerInstance.size ≤ 2 ^ Key30.setupVars) ∧
    (Key30.root r).openInstance.size ≤ 2 ^ Key30.setupVars := by
  decide

/-- The hypotheses `innerPow`, `outerPow` and `openPow` of `stage3Weight_codeScan` hold at the root
of each row of `key30.aks`: the degrees `2^11`, `2^8`, `2^7` and `2^9` of `A` and `B` and `2^8`
of `D`. -/
theorem key30_root_pow (r : Key30.Row) :
    (∀ g, 2 ^ Nat.log 2 ((Key30.root r).shape g).innerDegree =
      ((Key30.root r).shape g).innerDegree) ∧
    (∀ g, 2 ^ Nat.log 2 ((Key30.root r).shape g).outerDegree =
      ((Key30.root r).shape g).outerDegree) ∧
    2 ^ Nat.log 2 (Key30.root r).openDegree = (Key30.root r).openDegree := by
  refine ⟨fun g => ?_, fun g => ?_, pow_log_of (k := 8) rfl⟩
  · fin_cases g
    · exact pow_log_of (k := 11) rfl
    · exact pow_log_of (k := 11) rfl
    · exact pow_log_of (k := 9) rfl
  · fin_cases g
    · exact pow_log_of (k := 8) rfl
    · exact pow_log_of (k := 7) rfl
    · exact pow_log_of (k := 8) rfl

/-- `Batch.low` of the root of each row of `key30.aks` is the `κ = 7` of `Key30.root_blocked`:
the plan `SetupPlan.ofBatch B (groupOrder 3) B.low` of the layout is the plan below. -/
theorem key30_root_low : ∀ r ∈ Key30.rows, (Key30.root r).low = 7 := by
  have least : ∀ r ∈ Key30.rows, (List.ofFn fun g => min (min ((Key30.root r).shape g).innerDegree
      ((Key30.root r).shape g).outerDegree) ((Key30.root r).carrierDegree g)).foldl min
        (Key30.root r).openDegree = 2 ^ 7 := by
    decide
  intro r member
  unfold Batch.low
  rw [least r member, Nat.log_pow (by norm_num)]

/-- `stage3Weight_codeScan` at the root of each row of `key30.aks`, for the plan of the code with
the block `2^7` and the response base `2^3`: every hypothesis is discharged. -/
example {E : Type} [Field E] [Fintype E] [Algebra Zq E] [Degree2 E] (r : Key30.Row)
    (member : r ∈ Key30.rows) (β : ∀ g, Module.Basis (Fin ((Key30.root r).planes g)) Zq E)
    (st : (Key30.root r).Statement E) (challenge : (Key30.root r).Challenge) (flat : Table)
    (α : Ext) (τ r2 ρ : List Ext) (τLength : τ.length = Key30.rootBatch)
    (r2Length : r2.length = Key30.rootVars) (ρLength : ρ.length = Key30.setupVars) :
    ((stage3Weight (SetupPlan.ofBatch (Key30.root r) (groupOrder 3) 7 3) α τ r2 ρ).into : E) =
      mle (ρ.map Ext.into) ((Batch.codeScan β st challenge Key30.rootVars
        (Key30.root_sized r member).1 8 Key30.quotientDigits₀ (Key30.root_sized r member).2.1
        Key30.rootBatch (Key30.root_sized r member).2.2.1 Key30.setupVars flat 7
        (Key30.root_blocked r member) (by decide)).weight α.into (τ.map Ext.into)
          (r2.map Ext.into)) := by
  have numbers := key30_root_numbers r member
  exact stage3Weight_codeScan β st challenge _ _ 8 _ _ _ _ _ flat 7 _ _ 3 numbers.2.1
    (key30_root_pow r).1 (key30_root_pow r).2.1 (key30_root_pow r).2.2 numbers.2.2.1
    numbers.2.2.2.1 numbers.2.2.2.2 α τ r2 ρ τLength r2Length ρLength

end NightstreamFPrime.Spec.Compression.Akita.Replay
