"""V6: increase the bounded state correction from one half to three quarters."""
from pathlib import Path
from src.build_state_v5 import build

if __name__ == '__main__':
    build(output=Path('outputs/v6/state.csv'),blend=.75)
