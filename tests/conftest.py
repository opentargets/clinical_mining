import sys
from pathlib import Path

# Make scripts/ importable as a package during tests
sys.path.insert(0, str(Path(__file__).parent.parent))
