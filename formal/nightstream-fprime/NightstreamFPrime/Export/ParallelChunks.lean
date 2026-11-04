set_option autoImplicit false

namespace NightstreamFPrime.Export.ParallelChunks

def workers : IO Nat := do
  return max 1 (((← IO.getEnv "LEAN_NUM_THREADS").bind String.toNat?).getD 1)

def start {Alpha : Type} (threads count : Nat) (run : Nat → IO Alpha) :
    IO (Array (Task (Except IO.Error (Alpha × Nat × Nat)))) := do
  let promises : Array (IO.Promise (Except IO.Error (Alpha × Nat × Nat))) ←
    (Array.range count).mapM fun _ => IO.Promise.new
  let next ← IO.mkRef 0
  let failed ← IO.mkRef false
  for _ in [:min threads count] do
    let _ ← IO.asTask (prio := Task.Priority.dedicated) do
      repeat
        let index ← next.modifyGet fun index => (index, index + 1)
        if within : index < promises.size then
          if ← failed.get then
            promises[index].resolve
              (.error (IO.userError s!"chunk {index} skipped after an earlier chunk failed"))
          else
            let started ← IO.monoNanosNow
            let result ← (run index).toBaseIO
            let finished ← IO.monoNanosNow
            if result matches .error _ then failed.set true
            promises[index].resolve (result.map fun value => (value, started, finished))
        else break
  return promises.map (·.result!)

def waitAll {Beta : Type} (tasks : Array (Task (Except IO.Error Beta))) : BaseIO Unit := do
  for task in tasks do
    let _ ← IO.wait task

def withScratch {Alpha Beta : Type} (threads count : Nat)
    (run : System.FilePath → Nat → IO Alpha)
    (consume : Array (Task (Except IO.Error (Alpha × Nat × Nat))) → IO Beta) : IO Beta :=
  IO.FS.withTempDir fun scratch => do
    let tasks ← start threads count (run scratch)
    try consume tasks finally waitAll tasks

end NightstreamFPrime.Export.ParallelChunks
