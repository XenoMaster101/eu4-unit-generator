# Release notes

Each version's section is the text of its GitHub release and what the app's Help → Check for updates shows.

## 1.0.0

The first public release.

**Installing:** download `eu4unitgen-1.0.0-setup.exe` and run it. The installer isn't code-signed, so Windows may show
"Windows protected your PC": click **More info → Run anyway**. (Signing costs a yearly certificate; the source is all
here, and `SHA256SUMS.txt` lets you check the download.) Everything goes into one folder you choose (default
`D:\EU4 Unit Generator`). On first start the app lists every extra download it needs (Blender, the AI models and
so on), with its size, licence and where it goes, before anything is downloaded.

**Updating:** from now on, Help → Check for updates installs a newer version in place: no uninstalling, and your
settings, units, library and downloads stay. Running a newer setup.exe over an installed copy does the same.

What's in it:
- **Make a part** from a picture (soldier, horse, gun) or **import a 3D model** (.glb or .fbx), with its licence and
  credit recorded; the standard and stylised (exaggerated, e.g. chibi) profiles.
- **Checks before showing:** arms down, limbs on the skeleton, weapons rigid and in hand, clipping, folds; up to
  three passes with fixes.
- **Build a unit:** infantry, cavalry and artillery, four tiers each; cavalry **on foot** (no horse, a base-game
  weapon per tier).
- **Add to mod:** a normal mod, an achievement-safe one, or a **culture pack** (a new graphical culture any nation
  can use); every export writes a `README.txt` saying what each file is for and how to install it.
- **The default library:** the bay horse, the English horse, the cataphract horse, the cataphract, the bombard, and
  the worked example **Byzantine cataphract cavalry**, all fitted to your game on first use.
- Help: the example, the tour, the picture guide, answers to common questions.

No Paradox files are in the download: the app reads your own EU4 install. The base game only, never DLC files.
