#!/usr/bin/env bash
# Translates crates/neo-verified to Lean with the toolchain in formal/aeneas/versions and
# writes formal/compression-lean/NeoVerified/. Fails when the proofs could not trust the
# result. Needs the pinned `charon` and `aeneas` on PATH and the pinned nightly in rustup.
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
pins=$root/formal/aeneas
crate=$root/crates/neo-verified
lean=$root/formal/compression-lean
out=$lean/NeoVerified
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

fail() { echo "aeneas_extract: $*" >&2; exit 1; }
pin() { sed -n "s/^$1=//p" "$pins/versions"; }
listed() { grep -v '^#' "$1" | sed '/^$/d' | sort; }

charon version | grep -qF "$(pin CHARON_VERSION) ($(pin CHARON_COMMIT))" ||
  fail "charon is not $(pin CHARON_VERSION) ($(pin CHARON_COMMIT))"
aeneas -version | grep -qxF "aeneas $(pin AENEAS_TAG)" || fail "aeneas is not $(pin AENEAS_TAG)"
rustup run "$(pin RUST_NIGHTLY)" rustc --version >/dev/null || fail "rustup has no $(pin RUST_NIGHTLY)"
grep -qF "rev = \"$(pin AENEAS_COMMIT)\"" "$lean/lakefile.toml" ||
  fail "the aeneas rev in formal/compression-lean/lakefile.toml is not $(pin AENEAS_COMMIT)"
for f in "$lean/lean-toolchain" "$root/formal/nightstream-fprime/lean-toolchain"; do
  [ "$(cat "$f")" = "$(pin LEAN_TOOLCHAIN)" ] || fail "${f#"$root"/} is not $(pin LEAN_TOOLCHAIN)"
done

(cd "$root" && cargo clippy -q -p neo-verified -- -D warnings) || fail "clippy rejects crates/neo-verified"
rules='\b(i8|i16|i32|i64|i128|isize|f32|f64|unsafe|size_of|HashMap|OnceLock|rayon|Fn|FnMut|FnOnce)\b'
rules+='|debug_assert|#\[cfg|\.expect\(|panic!|#\[(charon|aeneas)::|impl Drop|std::thread'
rules+="|\.(map|sum|product|any|all|fold|filter)\(|(break|continue) '"
if grep -rnE "$rules" "$crate/src" | grep -vE '^[^:]+:[0-9]+:[[:space:]]*//'; then
  fail "the lines above break a code rule of crates/neo-verified"
fi

(cd "$crate" && CARGO_TARGET_DIR="$root/target/charon" CARGO_PROFILE_DEV_OVERFLOW_CHECKS=true \
  charon cargo --preset=aeneas --sysroot "$(pin CHARON_SYSROOT)" --dest-file "$work/neo_verified.llbc") \
  >"$work/charon.log" 2>&1 || { cat "$work/charon.log" >&2; fail "charon failed"; }
if grep -E '^error' "$work/charon.log"; then fail "charon reported the errors above"; fi

aeneas -backend lean -split-files -abort-on-error -warnings-as-errors -dest "$work/lean" \
  "$work/neo_verified.llbc" >"$work/aeneas.log" 2>&1 || { cat "$work/aeneas.log" >&2; fail "aeneas failed"; }
if grep -E '\[(Error|Warn)' "$work/aeneas.log"; then fail "aeneas reported the lines above"; fi

written=$(cd "$work/lean" && ls | tr '\n' ' ')
# formal/nightstream-fprime/scripts/check-boundaries.sh rejects unscoped heartbeat and
# recursion overrides, and Aeneas writes one of each at the top of every file.
for file in "$work/lean"/*.lean; do
  sed -E -i.orig '/^\/- You can set the `max(Heartbeats|RecDepth)` value/d; /^set_option max(Heartbeats|RecDepth) [0-9]+$/d' "$file"
  rm -f "$file.orig"
done
[ "$written" = "Funs.lean Types.lean " ] || fail "aeneas wrote $written; any other file than Funs.lean and Types.lean holds hand-written models of items that it could not translate"
hatches='\b(sorry|admit|opaque|implemented_by|extern|skipKernelTC|native_decide)\b'
if grep -nE "$hatches" "$work/lean"/*.lean "$lean/CompressionLeanRust.lean" "$lean/CompressionLeanRust"/*.lean; then
  fail "the lines above hold an escape hatch"
fi
axioms=$(grep -hoE '^(private )?axiom [^ :]+' "$work/lean"/*.lean | awk '{print $NF}' | sort -u || true)
[ "$axioms" = "$(listed "$pins/axioms.txt")" ] || fail "the generated axioms differ from formal/aeneas/axioms.txt: $axioms"
models=$(grep -hoE '\b(core|alloc)\.[A-Za-z0-9_.]+|\b(U8|U16|U32|U64|U128|Usize|UScalar|Array|Slice|Vec)\.[a-z_]+' \
  "$work/lean/Funs.lean" | sort -u)
if [ "$models" != "$(listed "$pins/std-models.txt")" ]; then
  diff <(echo "$models") <(listed "$pins/std-models.txt") >&2 || true
  fail "the standard-library models (left) differ from formal/aeneas/std-models.txt (right)"
fi

# formal/nightstream-fprime/scripts/check-boundaries.sh rejects a Lean file of 1,500 lines or more.
# Aeneas writes the definitions in dependency order, so any doc comment at the top level starts a
# suffix that needs only the lines before it. Funs.lean is cut there into parts of about 1,200 lines;
# each part imports the one before, and the last keeps the name Funs.lean. Lake builds only the
# modules that a library names, so each part is a root of the NeoVerified library.
if grep -qE '^mutual' "$work/lean/Funs.lean"; then fail "Funs.lean has a mutual block, which the split cannot cut"; fi
mkdir "$work/split"
awk -v dir="$work/split" -v limit=1200 '
  { line[NR] = $0 }
  $0 == "namespace neo_verified" && !start { start = NR }
  $0 == "end neo_verified" { stop = NR }
  END {
    parts = 1; first[1] = start + 1
    for (i = start + 1; i < stop; i++)
      if (line[i] ~ /^\/-- / && start + i - first[parts] > limit) first[++parts] = i
    first[parts + 1] = stop
    for (p = 1; p <= parts; p++) {
      file = dir "/Funs" (p < parts ? p : "") ".lean"
      for (i = 1; i <= start; i++) {
        print line[i] > file
        if (p > 1 && line[i] == "public import NeoVerified.Types") print "public import NeoVerified.Funs" (p - 1) > file
      }
      for (i = first[p]; i < first[p + 1]; i++) print line[i] > file
      print "end neo_verified" > file
    }
  }' "$work/lean/Funs.lean"

for part in "$work/split"/*.lean; do
  part=$(basename "$part" .lean)
  grep -qF "\"NeoVerified.$part\"" "$lean/lakefile.toml" ||
    fail "the NeoVerified roots in formal/compression-lean/lakefile.toml do not name NeoVerified.$part"
done

mkdir -p "$out"
rm -f "$out"/Funs*.lean
cp "$work/lean/Types.lean" "$work/split/"*.lean "$out/"
echo "aeneas_extract: wrote ${out#"$root"/}"
