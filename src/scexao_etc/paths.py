from pathlib import Path

rootdir = Path(__file__).parent
datadir = rootdir / "data"
datadir.mkdir(exist_ok=True)