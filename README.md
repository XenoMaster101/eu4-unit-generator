# eu4unitgen

Makes Europa Universalis IV units from your own pictures and 3D models. Give it a picture of a soldier, a horse or a gun, or import a 3D model, and the app builds the part, fits it to the game's skeleton and animations, textures it, checks it, and puts it in your library. Build infantry, cavalry and artillery from those parts, four tiers each, and add them to a mod.

It never copies Paradox files: everything from the game is read from your own EU4 install, and never from DLC. GPL-3.0. Built for the *Hellenistic Revolution* total conversion; works for any mod.

## What you need

- Windows 10 or 11 (64-bit) and Europa Universalis IV (Steam).
- An NVIDIA graphics card (8 GB or more; tested on an RTX 3060 12 GB) and 32 GB of RAM: the AI that makes the 3D models runs on it.
- About 60 GB of free disk space on one drive: everything the app needs goes in one folder.
- An internet connection for the setup (about 40 GB of downloads, 1 to 2 hours) and a free Hugging Face account (for one AI model's licence).

Nothing else: the app brings its own Python, and its setup wizard fetches everything else for you, saying what each piece is for.

## Install

Download **`eu4unitgen-1.0.0-setup.exe`** from the [Releases page](https://github.com/XenoMaster101/eu4-unit-generator/releases) and run it.

**"Windows protected your PC"**: the installer isn't code-signed (a signing certificate costs a yearly fee), so Windows SmartScreen warns about it the first time. Click **More info**, then **Run anyway**. To check the download is the one published, compare its SHA-256 with `SHA256SUMS.txt` on the release page (in PowerShell: `Get-FileHash .\eu4unitgen-1.0.0-setup.exe`).

1. Read the licence, press **Next**, choose the folder (default `D:\EU4 Unit Generator`), keep or untick the shortcuts, and press **Install**. It takes a minute or two (about 260 MB to download), then the app starts.
2. The first time, the app opens its **setup wizard**. Its first page, **What gets downloaded, and why**, lists every download: what it is, why the app needs it, its size, its licence, where it comes from and where it goes. Then work down the steps; each has a button, a progress bar and plain instructions, and nothing is typed in a terminal:
   1. **Blender 4.2 LTS**: the 3D program the app works in, in the background (fitting, painting, checking, previews).
   2. **The io_pdx_mesh add-on**: lets Blender read the game's units and write EU4's mesh files.
   3. **Windows Subsystem for Linux**: TRELLIS.2, the AI that turns your picture into a 3D model, only runs on Linux; WSL is Windows' built-in way to run it. Windows asks for permission; restart afterwards and open the app again.
   4. **Ubuntu in WSL**: the Linux TRELLIS.2 runs in, kept in the app's folder; no Linux user or password to make.
   5. **TRELLIS.2**: downloads it and builds it for your graphics card (about 40 minutes).
   6. **The Windows AI environment**: runs the two cut-out AIs (your subject off the background; the helmet, shield and weapon as separate parts). About 10 minutes.
   7. **The AI models**: about 20 GB. One of them, DINOv3 (TRELLIS.2's "eyes"), is gated by Meta: the step's buttons open its licence page and the token page; accept the licence and paste a read token in the box (it's used for that download only and never saved).
3. On the **Setup** screen, check the **EU4 install** and the **EU4 mod folder** (usually found for you) and press **Save settings**.

**Smart App Control** (Windows 11) blocks unsigned installers outright, with no Run anyway. On those PCs use the zip (`eu4unitgen-<version>-windows.zip`): right-click it → **Properties** → tick **Unblock** → **OK**, unzip it, and double-click **Install EU4 Unit Generator**. It runs on the app's own Python, which is signed, and installs or updates in place exactly like setup.exe.

Everything lives in the folder you chose:

| Folder | What's in it |
|---|---|
| `app` | the program and its own Python |
| `tools` | Blender 4.2 and io_pdx_mesh |
| `ai` | the AI environment, models and weights (if you set them up) |
| `wsl` | the Linux (Ubuntu) disk TRELLIS.2 runs in (if you set it up) |
| `work` | your pictures, projects, result sheets and exported mods |
| `settings` | settings, logs and temporary files |

Outside that folder the app adds only its shortcuts and its entry in Windows' Apps list. Two Windows features are system-wide by nature: WSL itself, and `.wslconfig` in your user folder (WSL's memory setting; the wizard writes it only if you have none).

## Updating

**Help > Check for updates** looks for a newer release, shows what changed, and **Download and install** updates the app in place: it downloads the new version, checks it against the release's checksums, closes the app, updates it and starts it again (this works with Smart App Control on). No uninstalling: your settings, units, library, Blender and AI downloads stay. Running a newer setup.exe yourself does the same: it finds the installed copy and updates it.

## Uninstall

Windows **Settings > Apps > Installed apps > EU4 Unit Generator > Uninstall**. It removes the whole folder (the app, Blender, the AI parts, the Linux disk, the settings), the shortcuts, the Apps entry and the Linux distribution it installed. It asks first whether to delete your work (`work`); by default your work is kept.

## Using it

New to the app? **Help > Show me an example** walks through a finished cataphract with its real results, and **Help > Take the tour** plays the app's screens with captions. Every button has a tooltip.

The bar at the top of every screen: **Home · Build a unit · Make a part · Library · Setup · Help**.

### The library

The **Library** holds your parts (soldiers, horses, guns) and your units. It comes with a default library: the bay horse, the English horse, the cataphract horse, the cataphract (a soldier and rider), the bombard, and a finished example unit, **Byzantine cataphract cavalry**. Each is fitted to your game the first time you use it (a few minutes, no AI). Open the example to see a finished unit, change it, or add it to a mod.

### Make a part

- **From a picture** (needs the AI): a soldier from the front (arms a little out), a horse from the side, a gun from three-quarter. The guide beside the box shows good and bad examples, and the picture's checks appear at once. About 15 to 30 minutes; you can close the app while it runs.
- **From a 3D model** (no AI): import a `.glb` (preferred) or `.fbx`, with its licence (your own work, CC0, or CC BY with the credit to write into the mod). Choose the profile: **standard** (normal proportions) or **stylised** (exaggerated, e.g. a big chibi head: the head is kept whole and rigid, and the limbs are fitted to the skeleton).

Every part goes through **the checks before showing** it: arms down from the T-pose, limbs on the skeleton, weapons and shields rigid and in hand, clipping, folds. Up to three passes, with a fix tried between them; what's left is shown in plain words.

### Build a unit

Pick **infantry**, **cavalry** or **artillery**, then the parts for each tier (EU4 changes a country's units at military tech 1, 12, 21 and 26). Cavalry can be **on foot**: no horse, the soldier stands in as the cavalry with a base-game weapon you choose per tier (a yari, a matchlock, a halberd, a sabre, a musket...), and plays the matching infantry animations. Tiers sharing the same soldier share one model.

### Add to mod

From a unit's page, **Add to mod**, as one of:

- **Normal mod**: a new unit for the country you choose, with its stats, tech and colours. It changes the game's checksum, so no achievements while it's on.
- **Achievement-safe**: graphics only. The chosen graphical culture's unit at that tier wears yours. Achievements still work.
- **A culture pack**: a new graphical culture (e.g. `mynationgfx`) with the unit's whole set, for any nation you give it to (`graphical_culture = mynationgfx` in the country's file). Name the mod's folder and its launcher name.

Every export writes a **`README.txt`** into the mod listing each folder and file, what the game uses it for, and how to install it; the app shows the same text under **What's in it**. If Windows blocks writing to the mod folder (Controlled Folder Access), the app saves the mod elsewhere and tells you what to copy where. Then enable the mod in the EU4 launcher.

## A good reference picture

- **Horse**: a side view, head to tail in frame, all four legs visible and a little apart, armour and cloth as you want them in game.
- **Soldier**: a front view standing (an A-pose), the whole figure in frame, arms a little out from the body, weapons held clear of it.
- **Gun**: a three-quarter view, the whole gun and carriage in frame.
- A plain, even background; no other figures, props or text; at least about 768 pixels on the long side.

The app warns before a run if a picture looks wrong (a side view given as a soldier, arms held down at the sides, a busy background, a cropped subject).

## Known limits

- Plumes, crests and manes are solid shapes: EU4 has no see-through (cutout) shader for unit meshes.
- A soldier made from a picture is fitted to one pose; very loose cloth can fold in the animations (the checks say so).
- Ships, portraits and city models come later.

## For developers

See [docs/DEVELOPER.md](docs/DEVELOPER.md) (the pipeline, where each rule lives, the regression set, tests), [docs/SPEC.md](docs/SPEC.md) (the full specification and history) and [docs/RELEASE_NOTES.md](docs/RELEASE_NOTES.md). A release: `python packaging/windows/build_package.py`, then `python packaging/release.py`. Licences of every model and tool: the app's **About and licences** page, or SPEC §9.
