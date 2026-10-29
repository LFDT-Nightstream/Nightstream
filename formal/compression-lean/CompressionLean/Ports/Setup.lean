import CompressionLean.Layer1.Model
import CompressionLean.Fields
import NightstreamFPrime.Spec.AjtaiSetupV1
import Mathlib.Data.List.Sort

namespace CompressionLean.Setup

open NightstreamFPrime NightstreamFPrime.Spec

structure Push where
  row : ℕ
  matrix : ℕ
  start : ℕ
  len : ℕ
  initial : Gl
  ratio : Gl

namespace Push

def pad : Push := ⟨0, 0, 0, 0, 0, 0⟩

def ratioWord (p : Push) : ℕ := if p.len = 1 then 1 else p.ratio.val

def classCode (p : Push) : ℕ := p.len * 2 ^ 64 + p.ratioWord

end Push

def codeLength (code : ℕ) : ℕ := code / 2 ^ 64

def codeRatio (code : ℕ) : Gl := ((code % 2 ^ 64 : ℕ) : Gl)

def sortedSet (l : List ℕ) : List ℕ := l.dedup.insertionSort (· ≤ ·)

def runList (matrices : ℕ) (pushes : List Push) : List Push :=
  (List.range matrices).flatMap fun j => pushes.filter fun p => p.matrix = j

def classesOf (runs : List Push) : List ℕ := sortedSet (runs.map Push.classCode)

/-- Rust packs this key as `class << 40 | lane << 32 | block`; valid while blocks are below `2^32` and classes at most `2^24` (`from_rows` refuses wider keys). -/
def slotKey (classes : List ℕ) (p : Push) : ℕ :=
  classes.idxOf p.classCode * 2 ^ 40 + p.start % ringDegree * 2 ^ 32 + p.start / ringDegree

def keysOf (runs : List Push) : List ℕ := sortedSet (runs.map (slotKey (classesOf runs)))

def keyStart (w : ℕ) : ℕ := w % 2 ^ 32 * ringDegree + w / 2 ^ 32 % 2 ^ 8

def keyCode (classes : List ℕ) (w : ℕ) : ℕ := classes.getD (w / 2 ^ 40) 0

def seedKey (seed : List ℕ) (blocks i x : ℕ) : Gl :=
  if x % 64 < ringDegree ∧ x / 64 < blocks then (AjtaiSetupV1.wideCoefficientNat seed i (x / 64) (x % 64) : Gl)
  else 0

def setupOfRuns (seed : List ℕ) (blocks : ℕ) (runs : List Push) : Layer1.Setup Gl where
  key := seedKey seed blocks
  runRow k := (runs.getD k Push.pad).row
  runMatrix k := (runs.getD k Push.pad).matrix
  runSlot k := if k < runs.length then (keysOf runs).idxOf (slotKey (classesOf runs) (runs.getD k Push.pad)) else 0
  runCoef k := (runs.getD k Push.pad).initial
  runs := runs.length
  slotBlock σ := keyStart ((keysOf runs).getD σ 0) / ringDegree
  slotLane σ := keyStart ((keysOf runs).getD σ 0) % ringDegree
  slotLength σ := codeLength (keyCode (classesOf runs) ((keysOf runs).getD σ 0))
  slotRatio σ := codeRatio (keyCode (classesOf runs) ((keysOf runs).getD σ 0))
  slots := (keysOf runs).length

def Accepts (columns matrices : ℕ) (pushes : List Push) : Prop :=
  columns % ringDegree = 0 ∧ 0 < matrices ∧ 0 < columns / ringDegree ∧ columns / ringDegree ≤ 4708530 ∧
    ∀ p ∈ pushes, p.matrix < matrices ∧ p.len ≤ ringDegree ∧ p.start + p.len ≤ columns

instance (columns matrices : ℕ) (pushes : List Push) : Decidable (Accepts columns matrices pushes) := by
  unfold Accepts
  infer_instance

def build (seed : List ℕ) (columns matrices : ℕ) (pushes : List Push) : Option (Layer1.Setup Gl) :=
  if Accepts columns matrices pushes then some (setupOfRuns seed (columns / ringDegree) (runList matrices pushes))
  else none

end CompressionLean.Setup
