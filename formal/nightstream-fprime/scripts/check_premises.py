#!/usr/bin/env python3
"""Fail when a hypothesis of a claim has no class in SECURITY_MODEL.md.

Lean writes one TSV file per claims module in PREMISES. Each line gives the theorem, the
hypothesis and the constant at the head of its type. A hypothesis is a
binder whose type is a proposition, or a proof field of a binder whose type is a structure of the
package or a function to one. The Lean build fails when a premise file is not current
(`#premise_census`).

SECURITY_MODEL.md holds one table between the premise markers. A row gives hypothesis names (a
name that ends in `.*` covers each field under it), the head (`*` for each head), the number of
lines of PREMISES that it covers, a class and the evidence of the class. A name `Scope/name`
covers only the claims under `Claims.Scope`:

  assumption  an assumption of the allowed list: the evidence cites its `[ID]`.
  lean        Lean proves it: the evidence names the Lean theorem or instance that discharges it.
              When that theorem is in PREMISES, it must not take the hypothesis itself.
  link        Rust meets it by Lean-emitted data: the evidence names the emitted file and the
              Rust test that compares it.
  game        it is the accept condition or the experiment of the theorem: the evidence cites the
              `SPEC.md` algorithm or section, or the adversary of SECURITY_MODEL.md.
  open        no evidence: the evidence starts with `Open:` and gives the reason.

Rules: each line of PREMISES has exactly one row, each name of a row has a line, each row covers
the number of lines that it gives, and each row has the evidence of its class. So a hypothesis
with a new name, or with the name of a row, fails until a person changes the table. Each theorem
of PREMISES also has a pin in SPEC.md, so a changed statement fails `#spec_census`.
"""
import re
import subprocess
from collections import Counter
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
ROOT = PACKAGE.parents[1]
PREMISES = PACKAGE / 'artifacts/nightstream-fprime-premises-v1'
BEGIN, END = '<!-- premises:begin -->', '<!-- premises:end -->'
CLASSES = ('assumption', 'lean', 'link', 'game', 'open')
TICKED = re.compile(r'`([^`]+)`')
LEAN_NAME = re.compile(r'(?:Claims|Export|Layout|Lifecycle|Spec)\.[\w.]+')
SPEC_PLACE = re.compile(r'`SPEC\.md` (Algorithm \d+|section \d+(?:\.\d+)?)')
ADVERSARY = re.compile(r'adversary (\d+)')


def read_premises():
    modules = PACKAGE / 'NightstreamFPrime/Claims'
    return ''.join((PREMISES / module.with_suffix('.tsv').name).read_text()
                   for module in sorted(modules.glob('*.lean')))


def premise_lines(text):
    lines = [tuple(line.split('\t')) for line in text.splitlines() if line]
    return [line for line in lines if len(line) == 3]


def table_rows(model):
    """(names, head, class, evidence, lines) of each row of the table, or an error."""
    if model.count(BEGIN) != 1 or model.count(END) != 1:
        return None, [f'the page needs one table between {BEGIN} and {END}']
    rows, errors = [], []
    for line in model[model.index(BEGIN):model.index(END)].splitlines():
        cells = [cell.strip() for cell in line.strip().strip('|').split('|')]
        if not line.startswith('|') or len(cells) != 5 or set(cells[0]) <= set('- ') \
                or cells[0] == 'Premise':
            continue
        names, head = TICKED.findall(cells[0]), cells[1].strip('`')
        if not names or not cells[2].isdigit() or cells[3] not in CLASSES:
            errors.append(f'row {cells[0]!r} needs a backticked name, a number of lines and a '
                          f'class of {CLASSES}')
            continue
        rows.append((names, head, cells[3], cells[4], int(cells[2])))
    return rows, errors


def covers(name, theorem, label):
    scope, _, name = name.rpartition('/')
    if scope and theorem != f'Claims.{scope}' and not theorem.startswith(f'Claims.{scope}.'):
        return False
    return label.startswith(name[:-1]) if name.endswith('.*') else label == name


def rust_tests():
    found = subprocess.run(['git', 'grep', '-h', '-o', '-E', 'fn [A-Za-z0-9_]+', '--',
                            'crates/*/tests/*.rs'],
                           cwd=ROOT, capture_output=True, text=True).stdout
    return {line.split()[-1] for line in found.splitlines()}


def evidence_errors(row, lines, context):
    names, head, kind, evidence, _ = row
    ticked = TICKED.findall(evidence)
    where = f'row {names[0]!r}'
    if kind == 'assumption':
        cited = re.findall(r'\[([A-Z][A-Z0-9-]+)\]', evidence)
        if not cited or any(c not in context['assumptions'] for c in cited):
            yield f'{where}: an assumption row cites an `[ID]` of the assumption list'
    elif kind == 'lean':
        cited = [token for token in ticked if LEAN_NAME.fullmatch(token)]
        if not cited:
            yield f'{where}: a lean row names the Lean theorem or instance that discharges it'
        for theorem, label, line_head in lines:
            if theorem in cited and any(covers(name, theorem, label) for name in names) \
                    and head in ('*', line_head):
                yield f'{where}: `{theorem}` takes `{label}` itself, so it does not discharge it'
    elif kind == 'link':
        files = [token for token in ticked if '/' in token and (ROOT / token).is_file()]
        tests = [token for token in ticked if token in context['tests']]
        if not files or not tests:
            yield f'{where}: a link row names an emitted file that exists and a Rust test'
    elif kind == 'game':
        places = SPEC_PLACE.findall(evidence)
        missing = [p for p in places if not re.search(
            rf'^#+ {re.escape(p.removeprefix("section "))}[. ]', context['spec'], re.M)]
        adversaries = [n for n in ADVERSARY.findall(evidence)
                       if not re.search(rf'^### {n}\. ', context['model'], re.M)]
        if not places and not ADVERSARY.search(evidence):
            yield f'{where}: a game row cites a `SPEC.md` algorithm or section, or an adversary'
        for place in missing + adversaries:
            yield f'{where}: {place!r} is not a heading'
    elif not evidence.startswith('Open:') or len(evidence) < 20:
        yield f'{where}: an open row starts with `Open:` and gives the reason'


def check(model, premises, context):
    rows, errors = table_rows(model)
    if rows is None:
        return errors, 0
    lines = premise_lines(premises)
    used, covered = set(), Counter()
    for theorem, label, head in lines:
        if label == '-':
            continue
        matches = [(index, name) for index, (names, row_head, *_) in enumerate(rows)
                   for name in names if covers(name, theorem, label) and row_head in ('*', head)]
        if len({index for index, _ in matches}) != 1:
            errors.append(f'`{theorem}` takes `{label}` (`{head}`), which has '
                          f'{len(matches)} rows in the premise table; it needs exactly one')
        used.update(matches)
        covered.update({index for index, _ in matches})
    for index, row in enumerate(rows):
        for name in row[0]:
            if (index, name) not in used:
                errors.append(f'row name `{name}` (`{row[1]}`) covers no hypothesis')
        if covered[index] != row[4]:
            errors.append(f'row {row[0][0]!r} gives {row[4]} lines, but it covers '
                          f'{covered[index]}: read each new line, then change the number')
        errors.extend(evidence_errors(row, lines, context))
    for theorem in sorted({line[0] for line in lines}):
        if not re.search(rf'`{re.escape(theorem)}` \(`[0-9a-f]{{16}}`\)', context['spec']):
            errors.append(f'`{theorem}` has no pin in SPEC.md')
    return errors, sum(1 for row in rows if row[2] == 'open')


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from check_assumptions import assumption_list
    model = (PACKAGE / 'SECURITY_MODEL.md').read_text()
    try:
        premises = read_premises()
    except FileNotFoundError as error:
        print(f'[premises] {error.filename} is missing: run scripts/validate.sh premises',
              file=sys.stderr)
        return 1
    context = {'assumptions': assumption_list(model)[1], 'tests': rust_tests(), 'model': model,
               'spec': (ROOT / 'SPEC.md').read_text()}
    errors, opened = check(model, premises, context)
    for error in errors:
        print(f'[premises] SECURITY_MODEL.md: {error}', file=sys.stderr)
    if errors:
        return 1
    theorems = {line[0] for line in premise_lines(premises)}
    print(f'[premises] each hypothesis of {len(theorems)} theorems has one class; '
          f'{opened} rows are open')
    return 0


if __name__ == '__main__':
    sys.exit(main())
