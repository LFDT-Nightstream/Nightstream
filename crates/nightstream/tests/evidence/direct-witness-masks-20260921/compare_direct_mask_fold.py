from pathlib import Path
import importlib.util
import json

repo = Path('/Users/nijaar/starstream/Nightstream')
root = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('compare', repo / 'crates/nightstream/tests/compare_recursive_outputs.py')
compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare)
cpu = root / 'second-fold-cpu/fold-2'
metal = root / 'direct-mask-second-metal/fold-2'
files = []
for child in range(16):
    name = f'digit-{child}.json'
    files.append(compare.compare_json(metal / name, cpu / name, f'complete child {child} matrix'))
for name in ('parent.json', 'nifs.json'):
    files.append(compare.compare_json(metal / name, cpu / name, f'complete {name}'))
a, b = metal / 'proof.native', cpu / 'proof.native'
compare.equal(a.read_bytes(), b.read_bytes(), 'every canonical proof byte')
files.append(compare.file_record(a, b, 'every canonical proof byte', 'exact bytes'))
receipt = {'outcome': 'passed', 'files': files, 'scope': 'All sixteen complete returned child matrices, all claims and openings, parent, transcript, identities, and canonical proof bytes. Same step-2 source files were compared in the PiCCS test.'}
with (root / 'direct-mask-second-output-comparison.json').open('x') as out:
    json.dump(receipt, out, indent=2)
    out.write('\n')
print('Complete second-fold CPU/Metal output comparison passed.', flush=True)
