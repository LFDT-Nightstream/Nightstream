import CompressionLean.Replay.Field

namespace CompressionLean.Replay

open NightstreamFPrime.Export.NativePoseidon2

def rate : Nat := 12
def width : Nat := 16

def permute (s : Array Gl) : Array Gl :=
  let w (i : Nat) : Gl := s[i]!
  let out := State64.permute64
    { x0 := (w 0).word, x1 := (w 1).word, x2 := (w 2).word, x3 := (w 3).word,
      x4 := (w 4).word, x5 := (w 5).word, x6 := (w 6).word, x7 := (w 7).word,
      x8 := (w 8).word, x9 := (w 9).word, x10 := (w 10).word, x11 := (w 11).word,
      x12 := (w 12).word, x13 := (w 13).word, x14 := (w 14).word, x15 := (w 15).word,
      canonical := ⟨(w 0).canonical, (w 1).canonical, (w 2).canonical, (w 3).canonical,
        (w 4).canonical, (w 5).canonical, (w 6).canonical, (w 7).canonical,
        (w 8).canonical, (w 9).canonical, (w 10).canonical, (w 11).canonical,
        (w 12).canonical, (w 13).canonical, (w 14).canonical, (w 15).canonical⟩ }
  #[⟨out.x0, out.c0⟩, ⟨out.x1, out.c1⟩, ⟨out.x2, out.c2⟩, ⟨out.x3, out.c3⟩,
    ⟨out.x4, out.c4⟩, ⟨out.x5, out.c5⟩, ⟨out.x6, out.c6⟩, ⟨out.x7, out.c7⟩,
    ⟨out.x8, out.c8⟩, ⟨out.x9, out.c9⟩, ⟨out.x10, out.c10⟩, ⟨out.x11, out.c11⟩,
    ⟨out.x12, out.c12⟩, ⟨out.x13, out.c13⟩, ⟨out.x14, out.c14⟩, ⟨out.x15, out.c15⟩]

structure Duplex where
  state : Array Gl
  input : Array Gl
  output : Array Gl

namespace Duplex

def new : Duplex := ⟨Array.replicate width 0, #[], #[]⟩

def duplexing (d : Duplex) : Duplex :=
  let absorbed := d.input.size
  let state := (List.range absorbed).foldl (fun s i => s.set! i d.input[i]!) d.state
  let state :=
    if absorbed > 0 then
      let zeroed := (List.range (rate - absorbed)).foldl (fun s i => s.set! (absorbed + i) 0) state
      zeroed.set! rate (zeroed[rate]! + Gl.ofNat absorbed)
    else state
  let state := permute state
  ⟨state, #[], state.extract 0 rate⟩

def observe (d : Duplex) (value : Gl) : Duplex :=
  let d := { d with output := #[], input := d.input.push value }
  if d.input.size == rate then d.duplexing else d

def sample (d : Duplex) : Gl × Duplex :=
  let d := if !d.input.isEmpty || d.output.isEmpty then d.duplexing else d
  (d.output.back!, { d with output := d.output.pop })

end Duplex

inductive Draw where
  | ext (value : Ext)
  | pair (first second : Ext)
  | words (values : Array Gl)
  | sample (word : Gl)

structure Transcript where
  duplex : Duplex
  draws : Array Draw

abbrev ReplayM := ExceptT String (StateM Transcript)

def ensure {m : Type → Type} [Monad m] [MonadExcept String m] (ok : Bool) (what : String) : m Unit :=
  unless ok do throw what

def observeAll (values : Array Gl) : ReplayM Unit :=
  modify fun t => { t with duplex := values.foldl Duplex.observe t.duplex }

def observeExts (values : Array Ext) : ReplayM Unit :=
  observeAll (values.flatMap fun value => #[value.c0, value.c1, value.c2])

def observeExt (value : Ext) : ReplayM Unit := observeExts #[value]

def flatPairs (pairs : Array (Ext × Ext)) : Array Ext := pairs.flatMap fun pair => #[pair.1, pair.2]

def sampleWord : ReplayM Gl :=
  modifyGet fun t => let (word, duplex) := t.duplex.sample; (word, { t with duplex })

def record (draw : Draw) : ReplayM Unit :=
  modify fun t => { t with draws := t.draws.push draw }

def drawExt : ReplayM Ext := do
  let c0 ← sampleWord
  let c1 ← sampleWord
  let c2 ← sampleWord
  pure ⟨c0, c1, c2⟩

def sampleExt : ReplayM Ext := do
  let value ← drawExt
  record (.ext value)
  pure value

def samplePair : ReplayM (Ext × Ext) := do
  let first ← drawExt
  let second ← drawExt
  record (.pair first second)
  pure (first, second)

def sampleWords (count : Nat) : ReplayM (Array Gl) := do
  let values ← (List.range count).foldlM (fun values _ => do pure (values.push (← sampleWord))) #[]
  record (.words values)
  pure values

def hashLeaf (values : Array Gl) : Array Gl :=
  let chunks := (List.range ((values.size + rate - 1) / rate)).map fun c => values.extract (rate * c) (rate * c + rate)
  let state := chunks.foldl
    (fun state chunk => permute ((List.range chunk.size).foldl (fun s i => s.set! i chunk[i]!) state))
    (Array.replicate width 0)
  state.extract 0 4

def compress (left right : Array Gl) : Array Gl :=
  (permute (left ++ right ++ Array.replicate (width - 8) 0)).extract 0 4

def merkleRoot (leaf : Array Gl) (index : Nat) (path : Array (Array Gl)) : Array Gl :=
  (List.range path.size).foldl
    (fun node level =>
      if (index / 2 ^ level) % 2 = 0 then compress node path[level]! else compress path[level]! node)
    leaf

end CompressionLean.Replay
