import NightstreamFPrime.Spec.Compression.Akita.Replay.Script

/-! The work bits of a grinding plan against the rule of the Lean model
(`Extract/Work.lean`): each run of a field-challenge site has `Extract.workBits loss` bits, the
work of the fold challenges of level `h` has `Extract.foldWorkBits loss setBits` bits with
`setBits` the least `Extract.setBits` of the families of the groups of the level, and a response
nonce or a run of fold challenges has no bits. `script` checks the site, the loss and the nonce
width of each run against the layout. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Akita

/-- The runs of the plan whose bits differ from the rule of the model, with the bits of the
rule. -/
def planDifferences (layout : Layout) (plan : Array Run) : List (Run × ℕ) :=
  let levels := layout.folds.map (·.groups) ++ [[layout.terminal.group]]
  let folds := plan.toList.filter (·.site.startsWith "FoldChallengeWork")
  let foldRuns := (folds.zip levels).filterMap fun (run, groups) =>
    let setBits := leastSetBits groups
    let bits := Extract.foldWorkBits run.loss setBits
    if run.setBits = setBits ∧ run.bits = bits then none else some (run, bits)
  let fieldRuns := plan.toList.filterMap fun run =>
    if run.site.startsWith "FoldChallengeWork" then none
    else if run.site.startsWith "FoldResponse" ∨ run.site.startsWith "FoldChallengeGroup" then
      if run.bits = 0 then none else some (run, 0)
    else
      let bits := Extract.workBits run.loss
      if run.bits = bits then none else some (run, bits)
  (if folds.length = levels.length then [] else [(default, folds.length)]) ++ foldRuns ++ fieldRuns

end NightstreamFPrime.Spec.Compression.Akita.Replay
