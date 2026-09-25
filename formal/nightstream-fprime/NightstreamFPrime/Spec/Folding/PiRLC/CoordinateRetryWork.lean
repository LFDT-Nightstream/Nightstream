import NightstreamFPrime.Spec.Folding.PiRLC.CoordinateChargedOracle

/-!
Operational work of the interactive coordinate retry loop. Its finite execution
keeps every abort and rejected response, stops at the first accepted response,
and adds the observed clock of each executed call. The expected work is the
sum of these charged transitions under the existing rejected-prefix law.

Averaging over the accepted base cancels the inverse line-acceptance rate.
Thus rare expensive calls are allowed: one coordinate's expected retry work
is at most one uniform full-vector call's mean work, without a worst-case cap.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Folding.PiRLC.CoordinateRetryWork

open scoped BigOperators
open CoordinateRetry CoordinateForkProbability CoordinateOracle
open CoordinateOracleStar CoordinateOracleCost CoordinateChargedOracle
open CoordinateChargedOracle.Law

variable {Index Challenge Assignment : Type*}

/-- A supplied response and its observed oracle/verifier clock. -/
abbrev Packet := (Challenge × Option Assignment) × Nat

structure Run where
  output : Option (Challenge × Option Assignment)
  calls : Nat
  work : Nat

variable [DecidableEq Index]

/-- Execute a finite prefix. An exhausted prefix supplies no accepted output;
its calls and work are still charged. -/
def search (check : (Index → Challenge) → Assignment → Bool)
    (coordinate : Index) (rest : {index // index ≠ coordinate} → Challenge) :
    List (Packet (Challenge := Challenge) (Assignment := Assignment)) →
      Run (Challenge := Challenge) (Assignment := Assignment)
  | [] => ⟨none, 0, 0⟩
  | packet :: following =>
      let step := queryStep check (callVector coordinate rest packet.1.1) packet.1.2 packet.2
      if step.1 then ⟨some packet.1, 1, step.2⟩ else
        let tail := search check coordinate rest following
        ⟨tail.output, tail.calls + 1, step.2 + tail.work⟩

/-- The stopped execution returns its exact accepted endpoint and charges all
earlier calls, including aborts. -/
theorem search_firstHit (check : (Index → Challenge) → Assignment → Bool)
    (coordinate : Index) (rest : {index // index ≠ coordinate} → Challenge)
    (before : List (Packet (Challenge := Challenge) (Assignment := Assignment)))
    (last : Packet (Challenge := Challenge) (Assignment := Assignment))
    (rejected : ∀ packet ∈ before, callAccepted check coordinate rest packet.1 = false)
    (accepted : callAccepted check coordinate rest last.1 = true) :
    search check coordinate rest (before ++ [last]) =
      ⟨some last.1, before.length + 1,
        (before.map fun packet => packet.2 + 1).sum + (last.2 + 1)⟩ := by
  induction before with
  | nil => simpa [search, queryStep, callAccepted] using accepted
  | cons packet before induction =>
      have headRejected := rejected packet (by simp)
      have tailRejected : ∀ value ∈ before,
          callAccepted check coordinate rest value.1 = false := by
        intro value member
        exact rejected value (by simp [member])
      simp only [List.cons_append, search, queryStep]
      change (if callAccepted check coordinate rest packet.1 then _ else _) = _
      rw [headRejected]
      simp only [Bool.false_eq_true, ↓reduceIte, induction tailRejected]
      simp [Nat.add_assoc, Nat.add_comm, Nat.add_left_comm]

variable [Fintype Index] [Fintype Challenge] [Nonempty Challenge] [Fintype Assignment]

omit [Fintype Index] [Nonempty Challenge] in
/-- A positive stopped trace runs to its exact endpoint and charges each
observed clock. The probability model and the finite driver use the same
abort-inclusive response sequence. -/
theorem positive_trace_search (law : Law Index Challenge Assignment)
    (check : (Index → Challenge) → Assignment → Bool)
    (coordinate : Index) (rest : {index // index ≠ coordinate} → Challenge)
    {rejections : Nat}
    (before : Fin rejections → Outcome (Challenge := Challenge) (Assignment := Assignment))
    (clocks : Fin rejections → Nat)
    (last : Outcome (Challenge := Challenge) (Assignment := Assignment)) (lastClock : Nat)
    (positive : 0 < traceMass law.oracle check coordinate rest before last) :
    search check coordinate rest
      (List.ofFn (fun position => (before position, clocks position)) ++ [(last, lastClock)]) =
      ⟨some last, rejections + 1,
        (List.ofFn (fun position => clocks position + 1)).sum + (lastClock + 1)⟩ := by
  have checks := traceMass_positive_implies_checks law.oracle check coordinate rest before last positive
  have rejected : ∀ packet ∈ List.ofFn (fun position => (before position, clocks position)),
      callAccepted check coordinate rest packet.1 = false := by
    intro packet member
    rcases List.mem_ofFn.mp member with ⟨position, rfl⟩
    exact checks.1 position
  simpa [List.map_ofFn, Function.comp_def] using
    search_firstHit check coordinate rest
      (List.ofFn (fun position => (before position, clocks position))) (last, lastClock)
      rejected checks.2

/-- Probability of reaching this next clocked transition after the concrete
rejected prefix. The accepted base is the same gate as `queryTailTerm`. -/
noncomputable def transitionMass (law : Law Index Challenge Assignment)
    (check : (Index → Challenge) → Assignment → Bool)
    (vector : Index → Challenge) (coordinate : Index) {rejections : Nat}
    (before : Fin rejections → Outcome (Challenge := Challenge) (Assignment := Assignment))
    (result : Outcome (Challenge := Challenge) (Assignment := Assignment)) (steps : Nat) : ℝ :=
  ((line law.oracle check).acceptance vector *
    ∏ position, rejectedCallMass law.oracle check coordinate
      (Equiv.funSplitAt coordinate Challenge vector).2 (before position)) *
    law.callMass coordinate (Equiv.funSplitAt coordinate Challenge vector).2 result steps

omit [Fintype Index] [Nonempty Challenge] in
/-- Clock erasure gives the same next-call trace law used for the invocation
tails. Time and response need not be independent within a call. -/
theorem transitionMass_hasSum (law : Law Index Challenge Assignment)
    (check : (Index → Challenge) → Assignment → Bool)
    (vector : Index → Challenge) (coordinate : Index) {rejections : Nat}
    (before : Fin rejections → Outcome (Challenge := Challenge) (Assignment := Assignment))
    (result : Outcome (Challenge := Challenge) (Assignment := Assignment)) :
    HasSum (transitionMass law check vector coordinate before result)
      (((line law.oracle check).acceptance vector *
        ∏ position, rejectedCallMass law.oracle check coordinate
          (Equiv.funSplitAt coordinate Challenge vector).2 (before position)) *
        CoordinateOracle.callMass law.oracle coordinate
          (Equiv.funSplitAt coordinate Challenge vector).2 result) :=
  (callMass_hasSum law coordinate _ result).mul_left _

omit [Fintype Index] [Nonempty Challenge] in
/-- On a stopping transition this is exactly the existing first-hit trace
mass, multiplied by the accepted-base gate. -/
theorem stopping_transition_hasSum (law : Law Index Challenge Assignment)
    (check : (Index → Challenge) → Assignment → Bool)
    (vector : Index → Challenge) (coordinate : Index) {rejections : Nat}
    (before : Fin rejections → Outcome (Challenge := Challenge) (Assignment := Assignment))
    (last : Outcome (Challenge := Challenge) (Assignment := Assignment))
    (accepted : callAccepted check coordinate
      (Equiv.funSplitAt coordinate Challenge vector).2 last = true) :
    HasSum (transitionMass law check vector coordinate before last)
      ((line law.oracle check).acceptance vector *
        traceMass law.oracle check coordinate
          (Equiv.funSplitAt coordinate Challenge vector).2 before last) := by
  simpa only [traceMass, acceptedCallMass, accepted, ↓reduceIte, mul_assoc] using
    transitionMass_hasSum law check vector coordinate before last

omit [Fintype Index] [Nonempty Challenge] in
private theorem transition_work_hasSum (law : Law Index Challenge Assignment)
    (check : (Index → Challenge) → Assignment → Bool)
    (vector : Index → Challenge) (coordinate : Index) {rejections : Nat}
    (before : Fin rejections → Outcome (Challenge := Challenge) (Assignment := Assignment)) :
    HasSum (fun steps => ∑ result,
      transitionMass law check vector coordinate before result steps *
        ((queryStep check
          (callVector coordinate (Equiv.funSplitAt coordinate Challenge vector).2 result.1)
          result.2 steps).2 : ℝ))
      (((line law.oracle check).acceptance vector *
        ∏ position, rejectedCallMass law.oracle check coordinate
          (Equiv.funSplitAt coordinate Challenge vector).2 (before position)) *
        law.lineWork coordinate (Equiv.funSplitAt coordinate Challenge vector).2) := by
  simpa only [transitionMass, Finset.mul_sum, mul_assoc] using
    (call_work_hasSum law check coordinate
      (Equiv.funSplitAt coordinate Challenge vector).2).mul_left
        ((line law.oracle check).acceptance vector *
          ∏ position, rejectedCallMass law.oracle check coordinate
            (Equiv.funSplitAt coordinate Challenge vector).2 (before position))

/-- Mean of the next executed clock, summed over the concrete rejected
prefixes and all clocked responses. -/
noncomputable def workAt (law : Law Index Challenge Assignment)
    (check : (Index → Challenge) → Assignment → Bool)
    (vector : Index → Challenge) (coordinate : Index) (rejections : Nat) : ℝ :=
  ∑ before : Fin rejections → Outcome (Challenge := Challenge) (Assignment := Assignment),
    ∑' steps, ∑ result,
      transitionMass law check vector coordinate before result steps *
        ((queryStep check
          (callVector coordinate (Equiv.funSplitAt coordinate Challenge vector).2 result.1)
          result.2 steps).2 : ℝ)

omit [Fintype Index] [Nonempty Challenge] in
/-- The charged execution law factors only between fresh calls. No factor
between the present call's response and its clock is used. -/
theorem workAt_eq (law : Law Index Challenge Assignment)
    (check : (Index → Challenge) → Assignment → Bool)
    (vector : Index → Challenge) (coordinate : Index) (rejections : Nat) :
    workAt law check vector coordinate rejections =
      queryTailTerm law.oracle check vector coordinate rejections *
        law.lineWork coordinate (Equiv.funSplitAt coordinate Challenge vector).2 := by
  unfold workAt
  simp_rw [(transition_work_hasSum law check vector coordinate _).tsum_eq]
  simp only [queryTailTerm, Finset.mul_sum, Finset.sum_mul, mul_assoc]

/-- The total cost of the entered retry transitions is their geometric sum.
The zero-success case is inherited from the proved invocation-tail theorem. -/
theorem workAt_hasSum (law : Law Index Challenge Assignment)
    (check : (Index → Challenge) → Assignment → Bool)
    (vector : Index → Challenge) (coordinate : Index) :
    HasSum (workAt law check vector coordinate)
      ((line law.oracle check).acceptance vector *
        conditionalCalls (line law.oracle check) vector coordinate *
        law.lineWork coordinate (Equiv.funSplitAt coordinate Challenge vector).2) := by
  change HasSum (fun rejections => workAt law check vector coordinate rejections) _
  simp_rw [workAt_eq]
  exact
    (queryTailTerm_hasSum law.oracle check vector coordinate).mul_right
      (law.lineWork coordinate (Equiv.funSplitAt coordinate Challenge vector).2)

omit [Nonempty Challenge] in
private theorem split_average (coordinate : Index) (function : (Index → Challenge) → ℝ) :
    (𝔼 vector, function vector) =
      𝔼 rest, 𝔼 challenge,
        function ((Equiv.funSplitAt coordinate Challenge).symm (challenge, rest)) := by
  let split := Equiv.funSplitAt coordinate Challenge
  calc
    (𝔼 vector, function vector) = 𝔼 pair, function (split.symm pair) :=
      Fintype.expect_equiv split function (fun pair => function (split.symm pair))
        (by intro vector; simp)
    _ = 𝔼 challenge, 𝔼 rest, function (split.symm (challenge, rest)) := by
      rw [← Finset.univ_product_univ, Finset.expect_product]
    _ = 𝔼 rest, 𝔼 challenge, function (split.symm (challenge, rest)) :=
      Finset.expect_comm _ _ _

omit [Nonempty Challenge] in
private theorem average_work (source : Line Challenge) (work : ℝ) (nonnegative : 0 ≤ work) :
    (𝔼 challenge, source.acceptance challenge * (1 / source.rate) * work) ≤ work := by
  rw [← Finset.expect_mul, ← Finset.expect_mul]
  have rate : (𝔼 challenge, source.acceptance challenge) = source.rate := by
    simp only [Fintype.expect_eq_sum_div_card, Line.rate, Line.weight, ← Finset.sum_div]
  rw [rate]
  by_cases zero : source.rate = 0
  · simpa [zero] using nonnegative
  · simp [zero]

omit [Nonempty Challenge] in
/-- Exact EPT cancellation: the cost may depend on all challenges and on the
response. Averaging the base acceptance cancels its line's retry rate. -/
theorem coordinate_work_bound (law : Law Index Challenge Assignment)
    (check : (Index → Challenge) → Assignment → Bool) (coordinate : Index) :
    (𝔼 vector, (line law.oracle check).acceptance vector *
      conditionalCalls (line law.oracle check) vector coordinate *
      law.lineWork coordinate (Equiv.funSplitAt coordinate Challenge vector).2) ≤
      𝔼 vector, law.meanWork vector := by
  rw [split_average coordinate, split_average coordinate (fun vector => law.meanWork vector)]
  apply Finset.expect_le_expect
  intro rest _
  have same :
      (𝔼 challenge,
        (line law.oracle check).acceptance
          ((Equiv.funSplitAt coordinate Challenge).symm (challenge, rest)) *
        conditionalCalls (line law.oracle check)
          ((Equiv.funSplitAt coordinate Challenge).symm (challenge, rest)) coordinate *
        law.lineWork coordinate
          (Equiv.funSplitAt coordinate Challenge
            ((Equiv.funSplitAt coordinate Challenge).symm (challenge, rest))).2) =
      𝔼 challenge, (coordinateLine (line law.oracle check) coordinate rest).acceptance challenge *
        (1 / (coordinateLine (line law.oracle check) coordinate rest).rate) *
        law.lineWork coordinate rest := by
    apply Finset.expect_congr rfl
    intro challenge _
    simp only [conditionalCalls, Equiv.apply_symm_apply, coordinateLine]
  rw [same]
  exact average_work _ _ (law.lineWork_nonnegative coordinate rest)

/-- Expected retry work at this depth across every source, after one accepted
base. Every rejected query has its own charged transition. -/
noncomputable def workTail (law : Law Index Challenge Assignment)
    (check : (Index → Challenge) → Assignment → Bool) (rejections : Nat) : ℝ :=
  ∑ coordinate, 𝔼 vector, workAt law check vector coordinate rejections

theorem workTail_hasSum (law : Law Index Challenge Assignment)
    (check : (Index → Challenge) → Assignment → Bool) :
    HasSum (workTail law check)
      (∑ coordinate, 𝔼 vector, (line law.oracle check).acceptance vector *
        conditionalCalls (line law.oracle check) vector coordinate *
        law.lineWork coordinate (Equiv.funSplitAt coordinate Challenge vector).2) := by
  apply hasSum_sum
  intro coordinate _
  have summed := hasSum_sum (s := Finset.univ)
    (fun vector _ => workAt_hasSum law check vector coordinate)
  simpa only [Fintype.expect_eq_sum_div_card] using
    summed.div_const (Fintype.card (Index → Challenge) : ℝ)

/-- Cost of the executed initial call and all entered retries, using their
joint clock law. This is not a calls-times-supplied-cost definition. -/
noncomputable def expectedQueryWork (law : Law Index Challenge Assignment)
    (check : (Index → Challenge) → Assignment → Bool) : ℝ :=
  (𝔼 vector, law.meanWork vector) + ∑' rejections, workTail law check rejections

/-- The paper's EPT closure with challenge-dependent and unbounded individual
call times. A polynomial bound on the exact uniform-call mean therefore gives
a polynomial bound on this same retry algorithm's expected query work. -/
theorem expectedQueryWork_bound (law : Law Index Challenge Assignment)
    (check : (Index → Challenge) → Assignment → Bool) :
    expectedQueryWork law check ≤
      ((Fintype.card Index : ℝ) + 1) * (𝔼 vector, law.meanWork vector) := by
  unfold expectedQueryWork
  rw [(workTail_hasSum law check).tsum_eq]
  have bound := Finset.sum_le_sum (s := Finset.univ)
    (fun coordinate _ => coordinate_work_bound law check coordinate)
  simp only [Finset.sum_const, Finset.card_univ, nsmul_eq_mul] at bound
  nlinarith

end NightstreamFPrime.Spec.Folding.PiRLC.CoordinateRetryWork
