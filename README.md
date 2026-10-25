# eu4unitgen

Makes Europa Universalis IV units from your own pictures. Drop in an image of a horse or a rider, give it a name, press **Generate**, and the app builds the 3D model, fits it to the game's skeleton and animations, textures it, checks it, and adds it to your mod.

It never copies Paradox files: everything from the game is read from your own EU4 install. GPL-3.0. Built for the *Hellenistic Revolution* total conversion; works for any mod.

## What you need

- Windows 10 or 11 (64-bit) with an NVIDIA graphics card (8 GB or more; tested on an RTX 3060 12 GB) and 32 GB of RAM.
- About 60 GB of free disk space on one drive: everything the app needs goes in one folder.
- Europa Universalis IV installed (Steam).
- An internet connection for the setup (about 25 GB of downloads) and a free Hugging Face account (for one AI model's licence).

Nothing else: the app brings its own Python, and its setup wizard fetches Blender, Linux (WSL) and the AI models for you.

## Install

The download is one zip. Unzip it anywhere; the folder holds:

- **Install EU4 Unit Generator**: the one thing to click.
- **README.txt**: this text.
- **LICENSE.txt**: the GNU General Public License v3.
- **files**: everything the installer uses. Leave it next to the installer.

1. Double-click **Install EU4 Unit Generator**. If Windows asks whether to run it, choose **Run**.
2. The installer's window opens: read the licence, press **Next**, choose the folder (default `D:\EU4 Unit Generator`; it shows the free space), keep or untick the Start-menu and Desktop shortcuts, and press **Install**. It takes a minute or two (about 260 MB to download), then **Finish** starts the app.
3. The first time, the app opens its **setup wizard**. Work down the list; each step has a button, a progress bar and plain instructions, and nothing is typed in a terminal:
   1. **Blender 4.2 LTS**: downloaded into the app's folder (about 380 MB).
   2. **The io_pdx_mesh add-on**: lets Blender read and write EU4's mesh files.
   3. **Windows Subsystem for Linux**: Windows asks for permission; restart Windows afterwards and open the app again.
   4. **Ubuntu in WSL**: installed into the app's folder; no Linux user or password to make.
   5. **TRELLIS.2**: downloads it and builds its environment (about 40 minutes).
   6. **The Windows AI environment**: about 10 minutes.
   7. **The AI weights**: about 20 GB. One of them, DINOv3, needs you to accept its licence on Hugging Face first: the step's buttons open its page and the token page; paste a read token in the box (it's used for that download only and never saved).
4. On the **Setup** screen, check the **EU4 install** and the **EU4 mod folder** (usually found for you) and press **Save settings**.

Everything lives in the folder you chose:

| Folder | What's in it |
|---|---|
| `app` | the program and its own Python |
| `tools` | Blender 4.2 and io_pdx_mesh |
| `ai` | the AI environment, models and weights |
| `wsl` | the Linux (Ubuntu) disk TRELLIS.2 runs in |
| `work` | your references, projects, result sheets and exported mods |
| `settings` | settings, logs and temporary files |

Outside that folder the app adds only its shortcuts and its entry in Windows' Apps list. Two Windows features are system-wide by nature: WSL itself, and `.wslconfig` in your user folder (WSL's memory setting; the wizard writes it only if you have none).

There is also **eu4unitgen-setup.exe**, a classic installer that does the same. It isn't code-signed, so on a PC with Smart App Control turned on Windows refuses to run it; use the zip there.

## Uninstall

Windows **Settings > Apps > Installed apps > EU4 Unit Generator > Uninstall**. It removes the whole folder (the app, Blender, the AI parts, the Linux disk, the settings), the shortcuts, the Apps entry and the Linux distribution it installed. It asks first whether to delete your work (`work`); by default your work is kept.

## Making a unit

1. Open **EU4 Unit Generator**. Press **Input guide** once to see what makes a good reference (below).
2. Drop your image on the box, type a **name** (lower case, e.g. `cataphract_horse`) and the **rights** statement: who made the image and that you may use it.
3. Choose **Horse** or **Rider**, and the **unit slot** (cavalry tier 1-4). For a rider, choose the horse it rides (one of your finished horses), so it gets the same colours and its legs clear that horse.
4. Press **Generate**. The app first checks free RAM, that EU4 is closed, WSL, the weights and disk space, and tells you how to fix anything missing. A run takes about 15 minutes for a horse and 20 for a rider. You can close the app; the run carries on, and a stopped run continues where it left off when you press **Resume**. The 3D steps wait while EU4 is open.
5. When it's done you get the **result sheet** (your image next to the unit in game, close up and at map zoom against vanilla) and a **quality report** in plain words: outline against the reference, how it bends in the animations compared with vanilla, which parts were kept or rebuilt, texture use, and brightness at map zoom.
6. **Edits**, without code: plume or head tassels on or off, an emblem (choose one or upload a white-on-black mask), cloth colour, cloth as the country colour, brightness, shield side, and hem length. Each edit rebuilds only what it needs (seconds for brightness or colour, a few minutes for a rebuild); the 3D model is never remade.
7. **Add to mod**:
   - **Normal mod**: a mod of its own named after the unit (e.g. *Cataphract units*), for the country you choose, with its stats, tech and colours. It changes the game's checksum, so no achievements while it's on.
   - **Achievement-safe**: graphics only. The chosen graphical culture's cavalry at that tier wears your unit (every country of that culture without its own sprite pack shows it). Nothing in the folders the game's checksum covers, so achievements still work.

   If Windows blocks writing to the mod folder (Controlled Folder Access), the app stages the mod instead and tells you exactly what to copy where. Then enable the mod in the EU4 launcher.

## A good reference

- **Horse**: a side view, head to tail in frame, all four legs visible and a little apart, armour and cloth as you want them in game.
- **Rider or soldier**: a front view standing, the whole figure in frame, arms a little out from the body, weapons held clear of it (lance upright beside the body, shield out to the side).
- A plain, even background; no other figures, props or text.
- At least about 768 pixels on the long side.
- Your own image, or one you have the right to use.

The app warns before a run if an image looks wrong (a side view given as a rider, a busy background, a cropped subject).

## Known limits

- Plumes, crests and manes are solid shapes: EU4 has no see-through (cutout) shader for unit meshes.
- Only cavalry so far; foot soldiers and other unit types are on the roadmap.
- The rider's arms are fitted to one pose; the weapon is in one hand and the shield on the other forearm.

## For developers

See [docs/DEVELOPER.md](docs/DEVELOPER.md) (the pipeline, where each rule lives, the regression set, tests) and [docs/SPEC.md](docs/SPEC.md) (the full specification and history). Licences of every model and tool: the app's **About and licences** page, or SPEC §9.
