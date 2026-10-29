import CompressionLean.Layer1.Model
import CompressionLean.Lib.IdentityTest

namespace CompressionLean.Layer1

open CompressionLean.Lib
open Polynomial

variable {F K : Type} [Field F] [Field K] [Algebra F K]

local notation "ι" => algebraMap F K

structure EarlyChallenges (K : Type) where
  beta : K
  lookup : K × K
  scatter : K × K

def earlyShape (s : Shape) : GkrShape where
  trees := 5
  depth t := if t = 0 then s.cubeVars else if t = 2 then s.rowVars
    else if t = 4 then s.slotVars + s.matrixVars else s.runVars
  factors t := if t = 2 then 1 else if t = 3 then 2 else if t = 4 then 1 else 0

def rowColumn (S : Setup F) (k : ℕ) : K := (S.runRow k : K)
def slotColumn (S : Setup F) (k : ℕ) : K := (S.runSlot k : K)
def coefColumn (S : Setup F) (k : ℕ) : K := ι (S.runCoef k)
def matrixColumn (S : Setup F) (k : ℕ) : K := (S.runMatrix k : K)
def blockColumn (S : Setup F) (σ : ℕ) : K := if σ < S.slots then (S.slotBlock σ : K) else 0

noncomputable def earlyBatch (s : Shape) (S : Setup F) (st : Statement F) (ch : EarlyChallenges K)
    (w : P0Tables F) : Batch K := fun t =>
  if t = 0 then ⟨[], fun x => ch.beta - ι (w.z x)⟩
  else if t = 1 then
    ⟨[], fun k => ch.lookup.1 - tuple ch.lookup.2 (rowColumn S k) (ι (w.e k).1) (ι (w.e k).2)⟩
  else if t = 2 then
    ⟨[fun i => ι (w.mult i)],
      fun i => ch.lookup.1 - tuple ch.lookup.2 (i : K) (ι (st.chi i).1) (ι (st.chi i).2)⟩
  else if t = 3 then
    ⟨[coefColumn S, fun k => ι (w.e k).1 + ch.scatter.2 * ι (w.e k).2],
      fun k => ch.scatter.1 - (slotColumn S k + (2 ^ s.slotVars : K) * matrixColumn S k)⟩
  else
    ⟨[fun x => if x / 2 ^ s.slotVars < s.matrices then
        ι (w.U (x / 2 ^ s.slotVars) (x % 2 ^ s.slotVars)).1 +
          ch.scatter.2 * ι (w.U (x / 2 ^ s.slotVars) (x % 2 ^ s.slotVars)).2 else 0],
      fun x => ch.scatter.1 - (x : K)⟩

structure EarlyValues (K : Type) where
  eRe : K
  eIm : K
  row : K
  pairs : ℕ → K × K
  block : K

structure EarlyOut (F K : Type) where
  candidates : List (P0Tables F)
  histogram : ℕ → F
  challenges : EarlyChallenges K
  gkr : GkrClaims K
  values : EarlyValues K

namespace EarlyOut

variable (s : Shape) (S : Setup F) (o : EarlyOut F K)

def runPoint : List K := o.gkr.leafPoint 3
def rowPoint : List K := o.gkr.leafPoint 2
def pairPoint : List K := o.gkr.leafPoint 4
def normPoint : List K := o.gkr.leafPoint 0

def zLeaf : K := o.challenges.beta - o.gkr.leafValues 0 0
def multValue : K := o.gkr.leafValues 2 0
def rowClaim : K := o.values.row
noncomputable def slotClaim : K :=
  o.challenges.scatter.1 - o.gkr.leafValues 3 2 -
    (2 ^ s.slotVars : K) * mle (o.runPoint) (matrixColumn S)
def coefClaim : K := o.gkr.leafValues 3 0
def embeddedBlock : K := o.values.block * ((o.runPoint.drop s.slotVars).map fun x => 1 - x).prod

end EarlyOut

noncomputable def tableSum (H : ℕ) (hist : ℕ → F) (beta : K) : K :=
  ∑ t ∈ Finset.range (2 * H + 1), ι (hist t) / (beta - slotValue H t)

def PairCheck (x y : K × K) : Prop := x.2 ≠ 0 ∧ y.2 ≠ 0 ∧ x.1 * y.2 = y.1 * x.2

def EarlyChecks (s : Shape) (st : Statement F) (o : EarlyOut F K) : Prop :=
  (∀ t, t < 2 * s.normBound + 1 → o.challenges.beta ≠ slotValue s.normBound t) ∧
  (o.gkr.roots 0).2 ≠ 0 ∧
  (o.gkr.roots 0).1 = (o.gkr.roots 0).2 * tableSum s.normBound o.histogram o.challenges.beta ∧
  PairCheck (o.gkr.roots 1) (o.gkr.roots 2) ∧ PairCheck (o.gkr.roots 3) (o.gkr.roots 4) ∧
  o.gkr.leafValues 1 0 =
    o.challenges.lookup.1 - tuple o.challenges.lookup.2 o.values.row o.values.eRe o.values.eIm ∧
  o.gkr.leafValues 2 1 =
    o.challenges.lookup.1 - tuple o.challenges.lookup.2 (mle o.rowPoint fun i => (i : K))
      (mle o.rowPoint fun i => ι (st.chi i).1) (mle o.rowPoint fun i => ι (st.chi i).2) ∧
  o.gkr.leafValues 3 1 = o.values.eRe + o.challenges.scatter.2 * o.values.eIm ∧
  o.gkr.leafValues 4 0 =
    ∑ j ∈ Finset.range s.matrices, eqAt (o.pairPoint.drop s.slotVars) j *
      ((o.values.pairs j).1 + o.challenges.scatter.2 * (o.values.pairs j).2) ∧
  o.gkr.leafValues 4 1 = o.challenges.scatter.1 - mle o.pairPoint fun x => (x : K)

instance : Nonempty (EarlyValues K) := ⟨⟨0, 0, 0, fun _ => (0, 0), 0⟩⟩

section Game

variable [Fintype K] [DecidableEq K]

noncomputable def normData (s : Shape) (hist : ℕ → F) (w : P0Tables F) : K[X] :=
  normPoly s.cubeVars s.normBound (fun x => ι (w.z x)) (fun t => ι (hist t))

def lookupData (s : Shape) (S : Setup F) (st : Statement F) (w : P0Tables F) : Lookup K where
  runs := 2 ^ s.runVars
  rows := 2 ^ s.rowVars
  run k := (rowColumn S k, ι (w.e k).1, ι (w.e k).2)
  table i := (ι (st.chi i).1, ι (st.chi i).2)
  mult i := ι (w.mult i)

def scatterData (s : Shape) (S : Setup F) (w : P0Tables F) : Scatter K where
  runs := 2 ^ s.runVars
  pairs := 2 ^ (s.slotVars + s.matrixVars)
  coef := coefColumn S
  e k := (ι (w.e k).1, ι (w.e k).2)
  tag k := slotColumn S k + (2 ^ s.slotVars : K) * matrixColumn S k
  U x := if x / 2 ^ s.slotVars < s.matrices then
      (ι (w.U (x / 2 ^ s.slotVars) (x % 2 ^ s.slotVars)).1, ι (w.U (x / 2 ^ s.slotVars) (x % 2 ^ s.slotVars)).2)
    else (0, 0)

noncomputable def blockTest (s : Shape) (S : Setup F) (w : P0Tables F) (point : List K) : Finset K :=
  if point.length < s.slotVars then
    foldBad (fun σ => ι (w.blockCopy σ) - blockColumn S σ) s.slotVars point
  else {1}

noncomputable def earlyExtra (s : Shape) (S : Setup F) (cands : List (P0Tables F)) (k : ℕ)
    (point : List K) : Finset K :=
  if k = s.runVars then Finset.univ.biUnion fun w : Fin cands.length => blockTest s S (cands.get w) point
  else ∅

noncomputable def earlyGame (s : Shape) (S : Setup F) (st : Statement F) (L0 : ℕ) :
    Game (Option (EarlyOut F K)) :=
  .recv {l : List (P0Tables F) // l.length ≤ L0} fun cands =>
  .recv (ℕ → F) fun hist =>
  .draw K (Finset.univ.biUnion fun w : Fin cands.1.length => zeros (normData s hist (cands.1.get w)))
    fun beta =>
  .draw (K × K) (Finset.univ.biUnion fun w : Fin cands.1.length => (lookupData s S st (cands.1.get w)).bad)
    fun lookup =>
  .draw (K × K) (Finset.univ.biUnion fun w : Fin cands.1.length => (scatterData s S (cands.1.get w)).bad)
    fun scatter =>
  Game.bind (gkr (earlyShape s) Finset.univ
      (fun w : Fin cands.1.length => earlyBatch s S st ⟨beta, lookup, scatter⟩ (cands.1.get w))
      (earlyExtra s S cands.1))
    fun result =>
      match result with
      | none => .pure none
      | some claims =>
          .recv (EarlyValues K) fun values =>
            have := Classical.propDecidable
            if EarlyChecks s st ⟨cands.1, hist, ⟨beta, lookup, scatter⟩, claims, values⟩ then
              .pure (some ⟨cands.1, hist, ⟨beta, lookup, scatter⟩, claims, values⟩)
            else .pure none

end Game

end CompressionLean.Layer1
