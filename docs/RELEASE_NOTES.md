# Release notes

Each version's section is the text of its GitHub release and what the app's Help → Check for updates shows.

## 1.0.5

- Check for updates: the update now keeps running after the app closes, writes its own log
  (settings\logs\update.log), and leaves the Start-menu and Desktop shortcuts alone for a copy Windows doesn't list
  as installed (a second copy). Tested end to end: downloaded from GitHub, checked, updated in place, restarted.

## 1.0.4

- **Smart App Control:** on PCs with it turned on, Windows blocks the unsigned setup.exe with no way past it. Use the
  zip instead: right-click it → Properties → tick **Unblock** → OK, unzip it, and double-click **Install EU4 Unit
  Generator** (it runs on the app's own Python, which is signed). It installs, or updates an installed copy in place.
- Help → Check for updates now updates from the zip the same way, so it works with Smart App Control on.

## 1.0.3

Setup fixes (found on a PC with another Blender installed and after an uninstall):
- **Blender:** setup took any Blender it found (e.g. 5.2), saved it in the settings and skipped downloading its
  own; io_pdx_mesh then failed ("No module named 'imp'"). The app now only uses Blender 4.2: its own copy, or a 4.2
  installed on the PC. Another version saved in the settings is ignored, and io_pdx_mesh waits for step 1.
- **Ubuntu:** showed as done whenever WSL was installed, even after an uninstall removed the app's Linux disk. It now
  checks the app's own Linux (EU4UnitGen) is there, and the AI steps never run in another Linux of yours.
- **The wizard's Close button** crashed ("'dict' object is not callable").
- All download addresses and pinned package versions checked live.

## 1.0.2

- Setup is in two groups: **Needed for everything** (Blender and io_pdx_mesh, about 640 MB) and **Optional: the AI
  pack** (WSL, Ubuntu, TRELLIS.2 and the AI models, about 40 GB, only to make parts from your own pictures), with one
  page saying what the AI pack is, why it needs Linux, what it needs from your PC, and that you can skip it.

## 1.0.1

- Reinstalling after an uninstall that kept your work: the installer refused the folder ("already has other files in
  it"). It now installs there and your work is kept.

## 1.0.0

The first public release.

**Installing:** download `eu4unitgen-1.0.0-setup.exe` and run it (blocked by Smart App Control? use the zip: see 1.0.4). The installer isn't code-signed, so Windows may show
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
