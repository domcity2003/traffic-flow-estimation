"""Build frozen v4 state predictions; rank/blend chosen on January train truth."""
import argparse
import json
from pathlib import Path
from src.build_state import build

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--release',type=Path,default=Path('data/release'))
    p.add_argument('--config',type=Path,default=Path('config/state_v4.json'))
    p.add_argument('--output',type=Path,default=Path('outputs/v4/state.csv'))
    a = p.parse_args()
    weights=json.loads(Path('config/state_weights.json').read_text())
    rules=json.loads(a.config.read_text())
    print(f'Wrote {build(a.release,a.output,weights,rules):,} rows')
