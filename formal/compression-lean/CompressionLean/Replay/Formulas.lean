import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.Phi81CoefficientKernel
import CompressionLean.Replay.Data

/-!
Owns the public formulas the layer-1 verifier evaluates: the eq and
interpolation helpers of `circuit/algebra.rs`, the closed forms of `mle.rs`
(scaled eq point, interval and step functions, the identity, `χ_r`, the lane
split, Eval_K by long division), and the ring-row weights of `ring.rs`
(powers of `λ`, the batched target, the quotient lift, `τ`). Points are low
bit first.

Each formula computes Rust's value. Its loops are written as the recursion
of the model's counterpart, named in each docstring, so that `Bridge/` reads
them term by term; field arithmetic is exact, so the order of a sum or a
product does not change a value.

`τ` reads the `bar` transform from the F′ package's closed form
(`Phi81CoefficientKernel.nativeBarEntry`), not from Rust.
-/

namespace CompressionLean.Replay

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.Phi81CoefficientKernel

/-! ## Sums and products -/

/-- `Σ_{i < n} f i`. -/
def sumRange (n : Nat) (f : Nat → Ext) : Ext := (List.range n).foldl (fun total i => total + f i) 0

/-- `Π_{i < n} f i`. -/
def prodRange (n : Nat) (f : Nat → Ext) : Ext := (List.range n).foldl (fun total i => total * f i) 1

/-- `Π_{i < n} f i` in the base field. -/
def prodRangeGl (n : Nat) (f : Nat → Gl) : Gl := (List.range n).foldl (fun total i => total * f i) 1

/-- `Σ_{i < n} f i` in `K ⊗ Ext`. -/
def sumKx (n : Nat) (f : Nat → Kx) : Kx := (List.range n).foldl (fun total i => total.add (f i)) Kx.zero

/-! ## `circuit/algebra.rs` -/

def oneMinus (x : Ext) : Ext := 1 - x

/-- `eq(x, y) = 2·x·y + 1 - (x + y)` for one coordinate (`eq1`). -/
def eq1 (x y : Ext) : Ext := let both := x * y; both + both + 1 - (x + y)

/-- `Π_t eq(x_t, y_t)` over the common coordinates (the model's `eqEval`). -/
def eqEvalList : List Ext → List Ext → Ext
  | x :: xs, y :: ys => eq1 x y * eqEvalList xs ys
  | _, _ => 1

/-- `eq_eval`. -/
def eqEval (x y : Array Ext) : Ext := eqEvalList x.toList y.toList

/-- `eq(point, index)`, index bit `t` with coordinate `t` (the model's `eqAt`). -/
def eqAtList : List Ext → Nat → Ext
  | [], _ => 1
  | x :: rest, index => (if index % 2 = 1 then x else oneMinus x) * eqAtList rest (index / 2)

/-- `eq_at`. -/
def eqAt (point : Array Ext) (index : Nat) : Ext := eqAtList point.toList index

/-- `eq(point, i)` for every `i < 2^point.size` (`eq_table`). -/
def eqTable (point : Array Ext) : Array Ext := (Array.range (2 ^ point.size)).map (eqAt point)

def dot (x y : Array Ext) : Ext := sumRange (min x.size y.size) fun i => x[i]! * y[i]!

/-- The polynomial through `(i, values[i])` at `x` (`interpolate`). -/
def interpolate (values : Array Ext) (x : Ext) : Ext :=
  sumRange values.size fun i =>
    let numerator := prodRange values.size fun j => if i = j then 1 else x - Ext.ofGl (Gl.ofNat j)
    let denominator := prodRangeGl values.size fun j => if i = j then 1 else Gl.ofNat i - Gl.ofNat j
    values[i]! * numerator.scale denominator.inv

/-- `Σ_i c_i·x^i` by Horner (`horner`). -/
def horner (coefficients : Array Ext) (x : Ext) : Ext :=
  coefficients.foldr (fun c total => total * x + c) 0

/-- `generator^index` from the low `bits` bits of `index` (`power_from_bits`). -/
def powerFromBits (generator : Gl) (index : Nat) : Nat → Gl
  | 0 => 1
  | bits + 1 => (if index % 2 = 1 then generator else 1) * powerFromBits (generator * generator) (index / 2) bits

/-! ## `mle.rs` -/

/-- `κ = Π_t (1 + c^{2^t})` and `x_t = c^{2^t}/(1 + c^{2^t})`; rejects a pole
(`scaled_eq_point`, the model's `laneScale`, `lanePoint`, `NoLanePole`). -/
def scaledEqPoint (c : Ext) : Nat → Except String (Ext × Array Ext)
  | 0 => pure (1, #[])
  | bits + 1 => do
      let denominator := 1 + c
      if denominator.isZero then throw "a pole of the lane weights"
      let (kappa, rest) ← scaledEqPoint (c * c) bits
      pure (denominator * kappa, #[c * denominator.inv] ++ rest)

/-- The scan of `less_than` from coordinate `t - 1` down to `0` (the model's `lessThanScan`). -/
def lessThanScan (point : Array Ext) (bound : Nat) : Nat → Ext × Ext → Ext × Ext
  | 0, acc => acc
  | t + 1, (total, pre) =>
      if (bound / 2 ^ t) % 2 = 1 then lessThanScan point bound t (total + pre * oneMinus point[t]!, pre * point[t]!)
      else lessThanScan point bound t (total, pre * oneMinus point[t]!)

/-- `Σ_{x < bound} eq(point, x)` (`less_than`, the model's `lessThan`). -/
def lessThan (point : Array Ext) (bound : Nat) : Ext :=
  if bound / 2 ^ point.size ≠ 0 then 1 else (lessThanScan point bound point.size (0, 1)).1

/-- The stop of the first segment: the next segment's start, or `stop`. -/
def nextStart (stop : Nat) (rest : List (Nat × Ext)) : Nat := (rest.head?.map Prod.fst).getD stop

/-- The segments `(start, value)` from the one whose interval starts at `low`:
each value times `lessThan(next start) - low` (the model's `stepLoop`
without its running total). -/
def stepSum (point : Array Ext) (stop : Nat) : List (Nat × Ext) → Ext → Ext
  | [], _ => 0
  | (_, value) :: rest, low =>
      value * (lessThan point (nextStart stop rest) - low) + stepSum point stop rest (lessThan point (nextStart stop rest))

/-- Each segment's value on `[start, next start)`, the last one ending at
`stop` (`step`, the model's `stepValue`). -/
def step (point : Array Ext) (segments : List (Nat × Ext)) (stop : Nat) : Ext :=
  stepSum point stop segments (lessThan point (nextStart stop segments))

/-- `Σ_t 2^t·point_t` (`identity`, the model's `identityValue`). -/
def identityValue (point : Array Ext) : Ext :=
  point.foldr (fun x total => total + total + x) 0

/-- One factor per coordinate of `r`: `(1 - x) + r·(2x - 1)` at a point
coordinate, `1 - r` past the point (the model's `chiValue`). -/
def chiList : List (Gl × Gl) → List Ext → Kx
  | [], _ => Kx.one
  | r :: rs, x :: xs =>
      let scaled := (Kx.ofK r).scale (x + x - 1)
      Kx.mul { scaled with re := scaled.re + oneMinus x } (chiList rs xs)
  | r :: rs, [] => Kx.mul (Kx.ofK r).oneMinus (chiList rs [])

/-- `χ_r` at an `Ext` point of the low bits (`chi`). -/
def chi (r : Array (Gl × Gl)) (point : Array Ext) : Kx := chiList r.toList point.toList

/-- The head `H0` by distance `d` from the top lane (the model's `headFrom`). -/
def headFrom (tau : Array Ext) (length : Nat) (ratio ratioPower : Gl) : Nat → Ext
  | 0 => tau[D - 1]!
  | d + 1 =>
      let value := tau[D - 2 - d]! + (headFrom tau length ratio ratioPower d).scale ratio
      if D - 2 - d + length < D then value - tau[D - 2 - d + length]!.scale ratioPower else value

/-- `Σ_{i < n} τ_i·ratio^i`. -/
def prefixSum (tau : Array Ext) (ratio : Gl) (n : Nat) : Ext := sumRange n fun i => tau[i]!.scale (ratio.pow i)

/-- The lane split `[H0, H1]` of a run class (`lane_split`, the model's
`headValue`, `tailValue`). -/
def laneSplit (tau : Array Ext) (length : Nat) (ratio : Gl) : Array Ext × Array Ext :=
  ((Array.range D).map fun lane => headFrom tau length ratio (ratio.pow length) (D - 1 - lane),
    (Array.range D).map fun lane =>
      if lane + length ≥ D + 1 then (prefixSum tau ratio (lane + length - D)).scale (ratio.pow (D - lane)) else 0)

/-- One digit of the long division of `eval_k` (the model's `divStep`): from
remainder `rem` and flag `below`, digit `bit` gives the next remainder, flag
and quotient bit, or `none` when the quotient leaves the block bits or rises
above `blocks`. -/
def divStep (blocks blockBits t rem below bit : Nat) : Option (Nat × Nat × Nat) :=
  if blockBits ≤ t ∧ (if D ≤ 2 * rem + bit then 1 else 0) = 1 then none
  else if below = 1 ∨ (if D ≤ 2 * rem + bit then 1 else 0) < blocks / 2 ^ t % 2 then
    some (2 * rem + bit - D * (if D ≤ 2 * rem + bit then 1 else 0), 1, if D ≤ 2 * rem + bit then 1 else 0)
  else if (if D ≤ 2 * rem + bit then 1 else 0) = blocks / 2 ^ t % 2 then
    some (2 * rem + bit - D * (if D ≤ 2 * rem + bit then 1 else 0), 0, if D ≤ 2 * rem + bit then 1 else 0)
  else none

/-- The factor of a digit: `r_t` or `1 - r_t`, and below the block bits `s_t`
or `1 - s_t` by the quotient bit (the model's `divFactor`). -/
def divFactor (r : Array (Gl × Gl)) (sBlock : Array Ext) (t bit q : Nat) : Kx :=
  let factor := if bit = 1 then Kx.ofK r[t]! else (Kx.ofK r[t]!).oneMinus
  if t < sBlock.size then factor.scale (if q = 1 then sBlock[t]! else oneMinus sBlock[t]!) else factor

/-- What state `(rem, below)` sends to state `(rem', below')` through one digit. -/
def contribution (step : Option (Nat × Nat × Nat)) (rem' below' : Nat) (factor : Nat → Kx) (x : Kx) : Kx :=
  match step with
  | some (r', b', q) => if r' = rem' ∧ b' = below' then x.mul (factor q) else Kx.zero
  | none => Kx.zero

/-- The states `[remainder][below]` after the top `k` digits of `cb` (the
model's `evalKStates`); an absent state is zero. -/
def evalKStates (r : Array (Gl × Gl)) (sBlock : Array Ext) (blocks cb : Nat) : Nat → Array (Array Kx)
  | 0 => (Array.range D).map fun rem => (Array.range 2).map fun below =>
      if rem = 0 ∧ below = 0 then Kx.one else Kx.zero
  | k + 1 =>
      let prev := evalKStates r sBlock blocks cb k
      (Array.range D).map fun rem' => (Array.range 2).map fun below' =>
        sumKx D fun rem => sumKx 2 fun below => sumKx 2 fun bit =>
          contribution (divStep blocks sBlock.size (cb - 1 - k) rem below bit) rem' below'
            (divFactor r sBlock (cb - 1 - k) bit) (prev[rem]!)[below]!

/-- `Σ_{b<blocks} eq(s_block, b)·Σ_{l<D} τ_l·χ_r(D·b + l)` by long division of
the coordinate by `D`, high bit first (`eval_k`, the model's `evalKValue`). -/
def evalK (r : Array (Gl × Gl)) (sBlock : Array Ext) (tau : Array Ext) (blocks : Nat) : Kx :=
  let cb := Nat.log2 (D * 2 ^ sBlock.size).nextPowerOfTwo
  let states := evalKStates r sBlock blocks cb cb
  let total := sumKx D fun rem => ((states[rem]!)[1]!).scale tau[rem]!
  (r.extract cb r.size).foldl (fun total value => total.mul (Kx.ofK value).oneMinus) total

/-! ## `ring.rs` -/

/-- The row weights: row `i` has `λ^i` (`Mixing`). -/
structure Mixing where
  lam : Ext
  kappa : Nat
  matrices : Nat

namespace Mixing

def commitment (m : Mixing) (row : Nat) : Ext := m.lam.pow row
def evalK (m : Mixing) : Ext × Ext := (m.lam.pow m.kappa, m.lam.pow (m.kappa + 1))
def evalA (m : Mixing) (j : Nat) : Ext × Ext :=
  (m.lam.pow (m.kappa + 2 + 2 * j), m.lam.pow (m.kappa + 3 + 2 * j))
def «public» (m : Mixing) (block : Nat) : Ext := m.lam.pow (m.kappa + 2 + 2 * m.matrices + block)

end Mixing

/-- `weights.1·Re + weights.2·Im` of a `K` value (`project`). -/
def project (value : Gl × Gl) (weights : Ext × Ext) : Ext :=
  weights.1.scale value.1 + weights.2.scale value.2

/-- `Ȳ = Σ_rows λ^row·y_row` as a ring element (`targets`). -/
def targets (st : Statement) (m : Mixing) : Array Ext :=
  (Array.range D).map fun lane =>
    sumRange st.commitment.size (fun row => (m.commitment row).scale (st.commitment[row]!)[lane]!) +
      project st.evalK[lane]! m.evalK +
      sumRange st.evalA.size (fun j => project (st.evalA[j]!)[lane]! (m.evalA j)) +
      sumRange st.publicInput.size (fun block => (m.public block).scale (st.publicInput[block]!)[lane]!)

/-- `Ȳ(ζ) + Q(ζ)·Φ81(ζ)` with `Φ81 = X⁵⁴ + X²⁷ + 1` (`lifted_target`, the
model's `liftedTarget`). -/
def liftedTarget (target quotient : Array Ext) (zeta : Ext) : Ext :=
  let middle := zeta.pow 27
  let phi := middle * middle + middle + 1
  horner target zeta + horner quotient zeta * phi

/-- `bar(p, l)` from the F′ closed form: rows are output coefficients. -/
def bar (p l : Nat) : Gl :=
  if h : p < ringDegree ∧ l < ringDegree then Gl.ofNat (nativeBarEntry ⟨p, h.1⟩ ⟨l, h.2⟩).val else 0

/-- `τ_l = Σ_p bar(p, l)·ζ^p` (`tau`, the model's `laneWeight`). -/
def tau (zeta : Ext) : Array Ext :=
  (Array.range D).map fun lane => sumRange D fun p => (zeta.pow p).scale (bar p lane)

end CompressionLean.Replay
