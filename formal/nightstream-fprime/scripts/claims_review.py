"""Write the claims review files (`scripts/validate.sh review`).

`tests/ClaimsReview.lean` lists the theorems of `NightstreamFPrime.Claims` and the Nightstream
names that their statements use. This script adds each hypothesis's class from the premise table
of `SECURITY_MODEL.md`, matched as `check_premises.py` matches it, and writes the review file and
its glossary to `docs/claims-review/` (not in git). Usage: `python3 -B scripts/claims_review.py
[output.md]`.
"""
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
ROOT = PKG.parents[1]
DATA = PKG / '.lake/claims-review.json'
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'docs/claims-review/nightstream-claims-review.md'
CAP = 40
sys.path.insert(0, str(PKG / 'scripts'))
from check_premises import table_rows, covers, premise_lines  # noqa: E402

ORDER = ['Foundations', 'Encodings', 'Fold', 'Step', 'Ivc', 'Nebula', 'Compression', 'Final']
data = json.loads(DATA.read_text())
model = (PKG / 'SECURITY_MODEL.md').read_text()
rows, _ = table_rows(model)
premises = premise_lines((PKG / 'artifacts/nightstream-fprime-premises-v1.tsv').read_text())
commit = subprocess.run(['git', 'rev-parse', '--short=9', 'HEAD'], cwd=ROOT,
                        capture_output=True, text=True).stdout.strip()


def source(module, start, stop):
    path = PKG / (module.replace('.', '/') + '.lean')
    lines = path.read_text().splitlines()
    first = start - 1
    if first > 0 and lines[first - 1].rstrip().endswith('-/'):
        while first > 0 and not lines[first].lstrip().startswith('/--'):
            first -= 1
    rel = path.relative_to(ROOT)
    return rel, first + 1, lines[first:stop]


def row_of(theorem, label, head):
    for index, (names, row_head, kind, _) in enumerate(rows):
        if any(covers(n, theorem, label) for n in names) and row_head in ('*', head):
            return index + 1, kind
    return None, 'none'


by_claim = defaultdict(list)
for theorem, label, head in premises:
    if label != '-':
        by_claim[theorem].append((label, head, *row_of(theorem, label, head)))

out = []
w = out.append
w(f'# Nightstream Lean claims: review file\n')
w(f'Generated from commit `{commit}` of https://github.com/LFDT-Nightstream/Nightstream by '
  '`scripts/validate.sh review`. Lean emits the claim list and the glossary names; nothing here '
  'is written by hand except this brief.\n')
w('## Brief for the reviewer\n')
w('Nightstream is a lattice-based folding prover over `F_q`, `q = 2^64 - 59`, ring '
  '`F_q[X]/(X^64 + 1)`. It uses SuperNeo v1.2 (ePrint 2026/242) for folding, HyperNova '
  '(ePrint 2023/573, with its errata Assumption 1) for recursion, Nebula for memory, and '
  'Akita (ePrint 2026/1983) for the compressed proof. Nightstream claims exactly the '
  'theorems below: each one restates a deeper Lean theorem with every hypothesis in its '
  'signature, and the Lean kernel checks it. Each hypothesis has a class:\n')
w('- `lean`: a named Lean theorem discharges it at the production values.\n'
  '- `game`: it defines the attack being measured (adversary class or accept condition).\n'
  '- `link`: Lean emits data and a Rust test checks it on the shipped packages.\n'
  '- `assumption`: an entry of the assumption list below.\n'
  '- `open`: nothing backs it yet.\n')
w('Read the papers next to this file. For each claim, report:\n')
w('1. Does it state the paper\'s theorem, or a weaker or stronger one? Name the paper section.\n'
  '2. Is a hypothesis missing, too strong, or impossible to meet for the production setup? '
  'A claim whose premises cannot all hold says nothing.\n'
  '3. Is the class of each premise right, given its evidence?\n'
  '4. Should the claim be added to, removed, or changed?\n')
w('Also list each theorem of the papers that has no claim here. Give one line per finding: '
  'the claim name, `add`, `remove` or `change`, the reason, and the paper section. You cannot '
  'run Lean, so mark each finding that needs a new proof. The glossary shows each name that '
  'the claim statements use, one level deep, with its file and line, in the second file '
  f'`{OUT.with_name(OUT.stem + "-glossary.md").name}`; ask for a deeper name when you need it.\n')

begin, end = model.index('<!-- assumptions:begin -->'), model.index('<!-- assumptions:end -->')
w('## The assumption list\n')
w(model[begin + len('<!-- assumptions:begin -->'):end].strip() + '\n')

claims = defaultdict(list)
for claim in data['claims']:
    loc = claim['location']
    claims[loc['module'].split('.')[-1]].append(claim)
for file in ORDER:
    group = sorted(claims[file], key=lambda c: c['location']['start'])
    path = PKG / f'NightstreamFPrime/Claims/{file}.lean'
    text = path.read_text()
    doc = text[text.index('/-!'):text.index('-/', text.index('/-!')) + 2]
    w(f'## Claims.{file} ({len(group)} claims)\n')
    w(f'Source: `{path.relative_to(ROOT)}`\n')
    w('```lean\n' + doc + '\n```\n')
    for claim in group:
        loc = claim['location']
        rel, first, lines = source(loc['module'], loc['start'], loc['stop'])
        name = claim['name'].removeprefix('NightstreamFPrime.')
        w(f'### `{name}`\n')
        w(f'`{rel}:{first}`\n')
        w('```lean\n' + '\n'.join(lines) + '\n```\n')
        lines_here = by_claim.get(name, [])
        if not lines_here:
            w('No hypothesis.\n')
            continue
        counts = defaultdict(int)
        for _, _, _, kind in lines_here:
            counts[kind] += 1
        w('Premises: ' + ', '.join(f'{k} {v}' for k, v in sorted(counts.items())) + '.\n')
        w('| Hypothesis | Head | Class | Row |\n|---|---|---|---|')
        for label, head, row, kind in lines_here:
            w(f'| `{label}` | `{head}` | {kind} | R{row} |')
        w('')

w('## Premise table rows\n')
w('Each hypothesis above points to one row. The evidence is from `formal/nightstream-fprime/SECURITY_MODEL.md`.\n')
for index, (names, head, kind, evidence) in enumerate(rows):
    w(f'**R{index + 1}** ({kind}, head `{head}`): ' + ', '.join(f'`{n}`' for n in names)
      + '\n\n' + evidence + '\n')

w('## Glossary\n')
w(f'Each Nightstream name that a claim statement uses, one level deep, grouped by module. '
  f'Entries longer than {CAP} lines are cut; the file and line show the rest.\n')
users = defaultdict(list)
for claim in data['claims']:
    for name in claim['uses']:
        users[name].append(claim['name'].removeprefix('NightstreamFPrime.Claims.'))
by_module = defaultdict(list)
for entry in data['glossary']:
    if 'start' in entry['location']:
        by_module[entry['location']['module']].append(entry)
for module in sorted(by_module):
    w(f'### {module.removeprefix("NightstreamFPrime.")}\n')
    for entry in sorted(by_module[module], key=lambda e: e['location']['start']):
        loc = entry['location']
        rel, first, lines = source(module, loc['start'], loc['stop'])
        cut = len(lines) > CAP
        shown = lines[:CAP] + ([f'-- ({len(lines) - CAP} more lines, see {rel}:{first})'] if cut else [])
        used_by = sorted(set(users[entry['name']]))
        w(f'#### `{entry["name"].removeprefix("NightstreamFPrime.")}` ({entry["kind"]})\n')
        w(f'`{rel}:{first}`. Used by {len(used_by)} claim(s): ' + ', '.join(used_by[:8])
          + (' and more' if len(used_by) > 8 else '') + '.\n')
        w('```lean\n' + '\n'.join(shown) + '\n```\n')

split = out.index('## Glossary\n')
glossary_out = OUT.with_name(OUT.stem + '-glossary.md')
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text('\n'.join(out[:split]) + '\n')
glossary_out.write_text(f'# Nightstream Lean claims: glossary\n\nGenerated from commit `{commit}`, '
                        f'with `{OUT.name}`.\n\n' + '\n'.join(out[split:]) + '\n')
for path in (OUT, glossary_out):
    print(f'{path}: {path.stat().st_size} bytes, {len(path.read_text().splitlines())} lines')
print(f'{len(data["claims"])} claims, {sum(len(v) for v in by_module.values())} glossary entries')
