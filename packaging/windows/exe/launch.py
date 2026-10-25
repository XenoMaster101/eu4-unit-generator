"""The frozen app's entry point (PyInstaller, eu4unitgen.spec): the same as ``python -m eu4unitgen.gui``."""
import sys

from eu4unitgen.gui.app import main

sys.exit(main())
