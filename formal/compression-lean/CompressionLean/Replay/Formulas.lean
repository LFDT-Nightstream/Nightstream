import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.Phi81CoefficientKernel
import CompressionLean.Replay.Data

/-!
Owns the public formulas the layer-1 verifier evaluates, as Rust writes them:
the eq and interpolation helpers of `circuit/algebra.rs`, the closed forms of
`mle.rs` (scaled eq point, interval and step functions, the identity, `χ_r`,
the lane split, Eval_K by long division), and the ring-row weights of
`ring.rs` (powers of `λ`, the batched target, the quotient lift, `τ`).
Points are low bit first. The model's counterparts are named in each
docstring; `ClosedForms` and `EvalK` prove most of them equal to the tables.

`τ` reads the `bar` transform from the F′ package's closed form
(`Phi81CoefficientKernel.nativeBarEntry`), not from Rust.
-/

namespace CompressionLean.Replay

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.Phi81CoefficientKernel

/-! ## `circuit/algebra.rs` -/

def oneMinus (x : Ext) : Ext := 1 - x

/-- `eq(x, y) = 2·x·y + 1 - (x + y)` for one coordinate (`eq1`). -/
def eq1 (x y : Ext) : Ext := let both := x * y; both + both + 1 - (x + y)

/-- `Π_t eq(x_t, y_t)` (`eq_eval`, the model's `eqEval`). -/
def eqEval (x y : Array Ext) : Ext :=
  (x.zip y).foldl (fun total pair => total * eq1 pair.1 pair.2) 1

/-- `eq(point, i)` for every `i < 2^point.size`, index bit `t` with `point[t]`. -/
def eqTable (point : Array Ext) : Array Ext :=
  point.foldl (fun table x => table.map (· * oneMinus x) ++ table.map (· * x)) #[1]

/-- `eq(point, index)` (`eq_at`, the model's `eqAt`). -/
def eqAt (point : Array Ext) (index : Nat) : Ext :=
  (List.range point.size).foldl
    (fun total t => total * (if (index / 2 ^ t) % 2 = 1 then point[t]! else oneMinus point[t]!)) 1

def dot (x y : Array Ext) : Ext :=
  (x.zip y).foldl (fun total pair => total + pair.1 * pair.2) 0

/-- The polynomial through `(i, values[i])` at `x` (`interpolate`). -/
def interpolate (values : Array Ext) (x : Ext) : Ext :=
  (List.range values.size).foldl (fun total i =>
    let (numerator, denominator) := (List.range values.size).foldl (fun (acc : Ext × Gl) j =>
      if i = j then acc else (acc.1 * (x - Ext.ofGl (Gl.ofNat j)), acc.2 * (Gl.ofNat i - Gl.ofNat j))) (1, 1)
    total + values[i]! * numerator.scale denominator.inv) 0

/-- `Σ_i c_i·x^i` by Horner (`horner`). -/
def horner (coefficients : Array Ext) (x : Ext) : Ext :=
  coefficients.foldr (fun c total => total * x + c) 0

/-- `generator^index` from the low `bits` bits of `index` (`power_from_bits`). -/
def powerFromBits (generator : Gl) (index bits : Nat) : Gl :=
  ((List.range bits).foldl (fun (acc : Gl × Gl) i =>
    ((if (index / 2 ^ i) % 2 = 1 then acc.1 * acc.2 else acc.1), acc.2 * acc.2)) (1, generator)).1

/-! ## `mle.rs` -/

/-- `κ = Π_t (1 + c^{2^t})` and `x_t = c^{2^t}/(1 + c^{2^t})`; rejects a pole
(`scaled_eq_point`, the model's `laneScale`, `lanePoint`, `NoLanePole`). -/
def scaledEqPoint (c : Ext) (bits : Nat) : Except String (Ext × Array Ext) := do
  let mut kappa : Ext := 1
  let mut point := #[]
  let mut power := c
  for _ in [0:bits] do
    let denominator := 1 + power
    kappa := kappa * denominator
    if denominator.isZero then throw "a pole of the lane weights"
    point := point.push (power * denominator.inv)
    power := power * power
  return (kappa, point)

/-- `Σ_{x < bound} eq(point, x)` (`less_than`, the model's `lessThan`). -/
def lessThan (point : Array Ext) (bound : Nat) : Ext :=
  if bound / 2 ^ point.size ≠ 0 then 1
  else
    let scan := (List.range point.size).reverse.foldl (fun (acc : Ext × Ext) t =>
      let low := oneMinus point[t]!
      if (bound / 2 ^ t) % 2 = 1 then (acc.1 + acc.2 * low, acc.2 * point[t]!)
      else (acc.1, acc.2 * low)) (0, 1)
    scan.1

/-- `values[s]` on `[starts[s], starts[s + 1])`, the last segment ending at
`stop` (`step`, the model's `stepValue`). -/
def step (point : Array Ext) (starts : Array Nat) (values : Array Ext) (stop : Nat) : Ext :=
  let first := lessThan point (starts[0]?.getD stop)
  let scan := (List.range values.size).foldl (fun (acc : Ext × Ext) s =>
    let upper := lessThan point (starts[s + 1]?.getD stop)
    (acc.1 + values[s]! * (upper - acc.2), upper)) (0, first)
  scan.1

/-- `Σ_t 2^t·point_t` (`identity`, the model's `identityValue`). -/
def identityValue (point : Array Ext) : Ext :=
  point.foldr (fun x total => total + total + x) 0

/-- `χ_r` at an `Ext` point of the low bits; the higher coordinates of `r`
see bit 0 (`chi`, the model's `chiValue`). -/
def chi (r : Array (Gl × Gl)) (point : Array Ext) : Kx :=
  (List.range r.size).foldl (fun total t =>
    let value := Kx.ofK r[t]!
    let factor :=
      if h : t < point.size then
        let x := point[t]
        let scaled := value.scale (x + x - 1)
        { scaled with re := scaled.re + oneMinus x }
      else value.oneMinus
    total.mul factor) Kx.one

/-- The lane split `[H0, H1]` of a run class (`lane_split`, the model's
`headValue`, `tailValue`). -/
def laneSplit (tau : Array Ext) (length : Nat) (ratio : Gl) : Array Ext × Array Ext :=
  let ratioPower := ratio.pow length
  let head := (List.range (D - 1)).reverse.foldl (fun head lane =>
    let value := tau[lane]! + head[lane + 1]!.scale ratio
    let value := if lane + length < D then value - tau[lane + length]!.scale ratioPower else value
    head.set! lane value) (Array.replicate D tau[D - 1]!)
  let pre := (List.range D).foldl (fun (acc : Array Ext × Ext × Gl) k =>
    let total := acc.2.1 + tau[k]!.scale acc.2.2
    (acc.1.push total, total, acc.2.2 * ratio)) (#[], 0, 1)
  let tail := (Array.range D).map fun lane =>
    if lane + length ≥ D + 1 then pre.1[lane + length - (D + 1)]!.scale (ratio.pow (D - lane)) else 0
  (head, tail)

/-- One step of the long division of `eval_k`: the states `[remainder][below]`. -/
abbrev DivisionStates := Array (Option Kx × Option Kx)

def putState (states : DivisionStates) (remainder below : Nat) (value : Kx) : DivisionStates :=
  let slot := states[remainder]!
  let merge (old : Option Kx) : Option Kx := some (match old with | some e => e.add value | none => value)
  states.set! remainder (if below = 0 then (merge slot.1, slot.2) else (slot.1, merge slot.2))

/-- `Σ_{b<blocks} eq(s_block, b)·Σ_{l<D} τ_l·χ_r(D·b + l)` by long division of
the coordinate by `D`, high bit first (`eval_k`, the model's `evalKValue`). -/
def evalK (r : Array (Gl × Gl)) (sBlock : Array Ext) (tau : Array Ext) (blocks : Nat) : Kx :=
  let blockBits := sBlock.size
  let coordinateBits := Nat.log2 (D * 2 ^ blockBits).nextPowerOfTwo
  let start : DivisionStates := (Array.replicate D (none, none)).set! 0 (some Kx.one, none)
  let final := (List.range coordinateBits).reverse.foldl (fun states t =>
    let rT := Kx.ofK r[t]!
    let rFactor (bit : Nat) : Kx := if bit = 0 then rT.oneMinus else rT
    let boundBit := (blocks / 2 ^ t) % 2
    (List.range D).foldl (fun next remainder =>
      (List.range 2).foldl (fun next below =>
        match (if below = 0 then states[remainder]!.1 else states[remainder]!.2) with
        | none => next
        | some value =>
          (List.range 2).foldl (fun next bit =>
            let doubled := 2 * remainder + bit
            let quotient := if doubled ≥ D then 1 else 0
            if t ≥ blockBits ∧ quotient = 1 then next
            else
              let nextBelow : Option Nat :=
                if below = 1 then some 1
                else if quotient < boundBit then some 1
                else if quotient = boundBit then some 0
                else none
              match nextBelow with
              | none => next
              | some nb =>
                let factor := rFactor bit
                let factor :=
                  if t < blockBits then
                    factor.scale (if quotient = 1 then sBlock[t]! else oneMinus sBlock[t]!)
                  else factor
                putState next (doubled - D * quotient) nb (value.mul factor)) next) next)
      (Array.replicate D (none, none))) start
  let total := (List.range D).foldl (fun total remainder =>
    match final[remainder]!.2 with
    | some value => total.add (value.scale tau[remainder]!)
    | none => total) Kx.zero
  (r.extract coordinateBits r.size).foldl (fun total value => total.mul (Kx.ofK value).oneMinus) total

/-! ## `ring.rs` -/

/-- Powers of `λ` in row order (`Mixing`). -/
structure Mixing where
  powers : Array Ext
  kappa : Nat
  matrices : Nat

namespace Mixing

def new (lam : Ext) (rows kappa matrices : Nat) : Mixing :=
  ⟨(List.range (rows - 1)).foldl (fun powers _ => powers.push (powers.back! * lam)) #[1], kappa, matrices⟩

def commitment (m : Mixing) (row : Nat) : Ext := m.powers[row]!
def evalK (m : Mixing) : Ext × Ext := (m.powers[m.kappa]!, m.powers[m.kappa + 1]!)
def evalA (m : Mixing) (j : Nat) : Ext × Ext :=
  (m.powers[m.kappa + 2 + 2 * j]!, m.powers[m.kappa + 3 + 2 * j]!)
def «public» (m : Mixing) (block : Nat) : Ext := m.powers[m.kappa + 2 + 2 * m.matrices + block]!

end Mixing

/-- `weights.1·Re + weights.2·Im` of a `K` value (`project`). -/
def project (value : Gl × Gl) (weights : Ext × Ext) : Ext :=
  weights.1.scale value.1 + weights.2.scale value.2

/-- `Ȳ = Σ_rows λ^row·y_row` as a ring element (`targets`). -/
def targets (st : Statement) (m : Mixing) : Array Ext :=
  (Array.range D).map fun lane =>
    let commitment := (List.range st.commitment.size).foldl
      (fun total row => total + (m.commitment row).scale (st.commitment[row]!)[lane]!) 0
    let evalA := (List.range st.evalA.size).foldl
      (fun total j => total + project (st.evalA[j]!)[lane]! (m.evalA j)) 0
    let pub := (List.range st.publicInput.size).foldl
      (fun total block => total + (m.public block).scale (st.publicInput[block]!)[lane]!) 0
    commitment + project st.evalK[lane]! m.evalK + evalA + pub

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
  let powers := (List.range (D - 1)).foldl (fun powers _ => powers.push (powers.back! * zeta)) #[(1 : Ext)]
  (Array.range D).map fun lane =>
    (List.range D).foldl (fun total p => total + powers[p]!.scale (bar p lane)) 0

end CompressionLean.Replay
