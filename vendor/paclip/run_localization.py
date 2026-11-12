"""Run this checkout even though its directory is named paclip原.

All absolute paclip imports must resolve here, never to the adjacent modified
project. Importing this module also establishes the same isolation for tests.
"""
from pathlib import Path
import importlib.util
import sys

ROOT = Path(__file__).resolve().parent
if 'paclip' in sys.modules:
    if Path(sys.modules['paclip'].__file__).resolve().parent != ROOT:
        raise RuntimeError('Another paclip checkout is already imported')
else:
    spec = importlib.util.spec_from_file_location(
        'paclip', ROOT / '__init__.py', submodule_search_locations=[str(ROOT)])
    module = importlib.util.module_from_spec(spec)
    sys.modules['paclip'] = module
    spec.loader.exec_module(module)

if __name__ == '__main__':
    from paclip.localization_v3.experiment import main
    main()
