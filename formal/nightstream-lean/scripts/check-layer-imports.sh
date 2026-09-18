#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_ROOT="$ROOT/Nightstream"
failed=0

allowed_import() {
  local source_layer="$1"
  local target_layer="$2"

  case "$source_layer" in
    SuperNeo)
      [[ "$target_layer" == "SuperNeo" ]]
      ;;
    HyperNova)
      [[ "$target_layer" == "HyperNova" ]]
      ;;
    Protocol)
      [[ "$target_layer" =~ ^(SuperNeo|HyperNova|Protocol)$ ]]
      ;;
    Implementation)
      [[ "$target_layer" =~ ^(SuperNeo|HyperNova|Protocol|Implementation)$ ]]
      ;;
    Assurance)
      [[ "$target_layer" =~ ^(SuperNeo|HyperNova|Protocol|Implementation|Assurance)$ ]]
      ;;
    Checks)
      [[ "$target_layer" =~ ^(SuperNeo|HyperNova|Protocol|Implementation|Assurance|Checks|Umbrella)$ ]]
      ;;
    *)
      return 1
      ;;
  esac
}

while IFS= read -r -d '' file; do
  relative="${file#"$SOURCE_ROOT/"}"
  source_layer="${relative%%/*}"
  source_layer="${source_layer%.lean}"

  while IFS=: read -r line imported; do
    [[ -n "$imported" ]] || continue
    target="${imported#Nightstream.}"
    if [[ "$imported" == "Nightstream" ]]; then
      target_layer="Umbrella"
    else
      target_layer="${target%%.*}"
    fi
    if ! allowed_import "$source_layer" "$target_layer"; then
      printf '%s:%s: forbidden %s -> %s import: %s\n' \
        "$relative" "$line" "$source_layer" "$target_layer" "$imported" >&2
      failed=1
    fi
  done < <(
    awk '
      $1 == "import" {
        for (i = 2; i <= NF; i++) {
          if ($i == "Nightstream" || $i ~ /^Nightstream\./) {
            print NR ":" $i
          }
        }
      }
    ' "$file"
  )
done < <(find "$SOURCE_ROOT" -type f -name '*.lean' -print0)

if (( failed != 0 )); then
  echo "[layers] import direction check failed" >&2
  exit 1
fi

echo "[layers] import direction check passed"
