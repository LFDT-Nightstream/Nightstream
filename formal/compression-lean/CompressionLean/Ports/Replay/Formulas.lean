import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.Phi81CoefficientKernel
import CompressionLean.Ports.Replay.Data

namespace CompressionLean.Replay

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.Phi81CoefficientKernel

def sumRange (n : Nat) (f : Nat → Ext) : Ext := (List.range n).foldl (fun total i => total + f i) 0

def prodRange (n : Nat) (f : Nat → Ext) : Ext := (List.range n).foldl (fun total i => total * f i) 1

def prodRangeGl (n : Nat) (f : Nat → Word) : Word := (List.range n).foldl (fun total i => total * f i) 1

def sumKx (n : Nat) (f : Nat → Kx) : Kx := (List.range n).foldl (fun total i => total.add (f i)) Kx.zero

def oneMinus (x : Ext) : Ext := 1 - x

def eq1 (x y : Ext) : Ext := let both := x * y; both + both + 1 - (x + y)

def eqEvalList : List Ext → List Ext → Ext
  | x :: xs, y :: ys => eq1 x y * eqEvalList xs ys
  | _, _ => 1

def eqEval (x y : Array Ext) : Ext := eqEvalList x.toList y.toList

def eqAtList : List Ext → Nat → Ext
  | [], _ => 1
  | x :: rest, index => (if index % 2 = 1 then x else oneMinus x) * eqAtList rest (index / 2)

def eqAt (point : Array Ext) (index : Nat) : Ext := eqAtList point.toList index

def eqTable (point : Array Ext) : Array Ext := (Array.range (2 ^ point.size)).map (eqAt point)

def dot (x y : Array Ext) : Ext := sumRange (min x.size y.size) fun i => x[i]! * y[i]!

def interpolate (values : Array Ext) (x : Ext) : Ext :=
  sumRange values.size fun i =>
    let numerator := prodRange values.size fun j => if i = j then 1 else x - Ext.ofGl (Word.ofNat j)
    let denominator := prodRangeGl values.size fun j => if i = j then 1 else Word.ofNat i - Word.ofNat j
    values[i]! * numerator.scale denominator.inv

def horner (coefficients : Array Ext) (x : Ext) : Ext :=
  coefficients.foldr (fun c total => total * x + c) 0

def powerFromBits (generator : Word) (index : Nat) : Nat → Word
  | 0 => 1
  | bits + 1 => (if index % 2 = 1 then generator else 1) * powerFromBits (generator * generator) (index / 2) bits

def scaledEqPoint (c : Ext) : Nat → Except String (Ext × Array Ext)
  | 0 => pure (1, #[])
  | bits + 1 => do
      let denominator := 1 + c
      if denominator.isZero then throw "a pole of the lane weights"
      let (kappa, rest) ← scaledEqPoint (c * c) bits
      pure (denominator * kappa, #[c * denominator.inv] ++ rest)

def lessThanScan (point : Array Ext) (bound : Nat) : Nat → Ext × Ext → Ext × Ext
  | 0, acc => acc
  | t + 1, (total, pre) =>
      if (bound / 2 ^ t) % 2 = 1 then lessThanScan point bound t (total + pre * oneMinus point[t]!, pre * point[t]!)
      else lessThanScan point bound t (total, pre * oneMinus point[t]!)

def lessThan (point : Array Ext) (bound : Nat) : Ext :=
  if bound / 2 ^ point.size ≠ 0 then 1 else (lessThanScan point bound point.size (0, 1)).1

def nextStart (stop : Nat) (rest : List (Nat × Ext)) : Nat := (rest.head?.map Prod.fst).getD stop

def stepSum (point : Array Ext) (stop : Nat) : List (Nat × Ext) → Ext → Ext
  | [], _ => 0
  | (_, value) :: rest, low =>
      value * (lessThan point (nextStart stop rest) - low) + stepSum point stop rest (lessThan point (nextStart stop rest))

def step (point : Array Ext) (segments : List (Nat × Ext)) (stop : Nat) : Ext :=
  stepSum point stop segments (lessThan point (nextStart stop segments))

def identityValue (point : Array Ext) : Ext :=
  point.foldr (fun x total => total + total + x) 0

def chiFactor (r : Word × Word) : Option Ext → Kx
  | some x => let scaled := (Kx.ofK r).scale (x + x - 1); { scaled with re := scaled.re + oneMinus x }
  | none => (Kx.ofK r).oneMinus

def chiList : List (Word × Word) → List Ext → Kx
  | [], _ => Kx.one
  | r :: rs, x :: xs => Kx.mul (chiFactor r (some x)) (chiList rs xs)
  | r :: rs, [] => Kx.mul (chiFactor r none) (chiList rs [])

def chi (r : Array (Word × Word)) (point : Array Ext) : Kx := chiList r.toList point.toList

def headFrom (tau : Array Ext) (length : Nat) (ratio ratioPower : Word) : Nat → Ext
  | 0 => tau[D - 1]!
  | d + 1 =>
      let value := tau[D - 2 - d]! + (headFrom tau length ratio ratioPower d).scale ratio
      if D - 2 - d + length < D then value - tau[D - 2 - d + length]!.scale ratioPower else value

def prefixSum (tau : Array Ext) (ratio : Word) (n : Nat) : Ext := sumRange n fun i => tau[i]!.scale (ratio.pow i)

def laneSplit (tau : Array Ext) (length : Nat) (ratio : Word) : Array Ext × Array Ext :=
  ((Array.range D).map fun lane => headFrom tau length ratio (ratio.pow length) (D - 1 - lane),
    (Array.range D).map fun lane =>
      if lane + length ≥ D + 1 then (prefixSum tau ratio (lane + length - D)).scale (ratio.pow (D - lane)) else 0)

def divStep (blocks blockBits t rem below bit : Nat) : Option (Nat × Nat × Nat) :=
  if blockBits ≤ t ∧ (if D ≤ 2 * rem + bit then 1 else 0) = 1 then none
  else if below = 1 ∨ (if D ≤ 2 * rem + bit then 1 else 0) < blocks / 2 ^ t % 2 then
    some (2 * rem + bit - D * (if D ≤ 2 * rem + bit then 1 else 0), 1, if D ≤ 2 * rem + bit then 1 else 0)
  else if (if D ≤ 2 * rem + bit then 1 else 0) = blocks / 2 ^ t % 2 then
    some (2 * rem + bit - D * (if D ≤ 2 * rem + bit then 1 else 0), 0, if D ≤ 2 * rem + bit then 1 else 0)
  else none

def divFactor (r : Array (Word × Word)) (sBlock : Array Ext) (t bit q : Nat) : Kx :=
  let factor := if bit = 1 then Kx.ofK r[t]! else (Kx.ofK r[t]!).oneMinus
  if t < sBlock.size then factor.scale (if q = 1 then sBlock[t]! else oneMinus sBlock[t]!) else factor

def contribution (step : Option (Nat × Nat × Nat)) (rem' below' : Nat) (factor : Nat → Kx) (x : Kx) : Kx :=
  match step with
  | some (r', b', q) => if r' = rem' ∧ b' = below' then x.mul (factor q) else Kx.zero
  | none => Kx.zero

def evalKStates (r : Array (Word × Word)) (sBlock : Array Ext) (blocks cb : Nat) : Nat → Array (Array Kx)
  | 0 => (Array.range D).map fun rem => (Array.range 2).map fun below =>
      if rem = 0 ∧ below = 0 then Kx.one else Kx.zero
  | k + 1 =>
      let prev := evalKStates r sBlock blocks cb k
      (Array.range D).map fun rem' => (Array.range 2).map fun below' =>
        sumKx D fun rem => sumKx 2 fun below => sumKx 2 fun bit =>
          contribution (divStep blocks sBlock.size (cb - 1 - k) rem below bit) rem' below'
            (divFactor r sBlock (cb - 1 - k) bit) (prev[rem]!)[below]!

def evalK (r : Array (Word × Word)) (sBlock : Array Ext) (tau : Array Ext) (blocks : Nat) : Kx :=
  let cb := Nat.log2 (D * 2 ^ sBlock.size).nextPowerOfTwo
  let states := evalKStates r sBlock blocks cb cb
  let total := sumKx D fun rem => ((states[rem]!)[1]!).scale tau[rem]!
  (r.extract cb r.size).foldl (fun total value => total.mul (Kx.ofK value).oneMinus) total

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

def project (value : Word × Word) (weights : Ext × Ext) : Ext :=
  weights.1.scale value.1 + weights.2.scale value.2

def targets (st : Statement) (m : Mixing) : Array Ext :=
  (Array.range D).map fun lane =>
    sumRange st.commitment.size (fun row => (m.commitment row).scale (st.commitment[row]!)[lane]!) +
      project st.evalK[lane]! m.evalK +
      sumRange st.evalA.size (fun j => project (st.evalA[j]!)[lane]! (m.evalA j)) +
      sumRange st.publicInput.size (fun block => (m.public block).scale (st.publicInput[block]!)[lane]!)

def liftedTarget (target quotient : Array Ext) (zeta : Ext) : Ext :=
  let middle := zeta.pow 27
  let phi := middle * middle + middle + 1
  horner target zeta + horner quotient zeta * phi

def bar (p l : Nat) : Word :=
  if h : p < ringDegree ∧ l < ringDegree then Word.ofNat (nativeBarEntry ⟨p, h.1⟩ ⟨l, h.2⟩).val else 0

def tau (zeta : Ext) : Array Ext :=
  (Array.range D).map fun lane => sumRange D fun p => (zeta.pow p).scale (bar p lane)

end CompressionLean.Replay
