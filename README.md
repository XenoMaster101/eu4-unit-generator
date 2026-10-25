# eu4unitgen

Makes Europa Universalis IV units from your own pictures. Drop in an image of a horse or a rider, give it a name, press **Generate**, and the app builds the 3D model, fits it to the game's skeleton and animations, textures it, checks it, and adds it to your mod.

It never copies Paradox files: everything from the game is read from your own EU4 install. GPL-3.0. Built for the *Hellenistic Revolution* total conversion; works for any mod.

## What you need

- Windows 10 or 11 (64-bit) with an NVIDIA graphics card (8 GB or more; tested on an RTX 3060 12 GB) and 32 GB of RAM.
- About 60 GB of free disk space (the AI weights and the WSL environment).
- Europa Universalis IV installed (Steam).
- [Blender 4.2 LTS](https://www.blender.org/download/lts/4-2/) (the free 3D program; the app runs it in the background).

## Install

1. Unzip `eu4unitgen-<version>-windows.zip` anywhere.
2. Double-click **Install.cmd**. It finds Python 3.11 or newer (or offers to install Python 3.12 for you), installs the app in `%LOCALAPPDATA%\eu4unitgen\app`, and puts **EU4 Unit Generator** on your Desktop and in the Start menu.
3. The app opens on its **Setup** screen the first time. Work down the list; each step opens its own window and says when it's done, then press **Check again**:
   - **Install WSL**: asks for administrator rights; Windows restarts afterwards.
   - **Install Ubuntu**: a window asks you to make up a Linux user name and password.
   - **Give WSL enough memory**.
   - **Get TRELLIS.2**.
   - **Set up TRELLIS.2 in WSL**: about 40 minutes; it asks for your Linux password once.
   - **Set up the Windows AI environment**: about 10 minutes.
   - **Download the AI weights**: about 20 GB. One of them, DINOv3, needs you to accept its licence on Hugging Face first. The screen has the link; then paste a Hugging Face read token in the box.
   - **Install the io_pdx_mesh add-on** (after Blender is installed).
4. At the top of Setup, set the **EU4 install** folder and your **EU4 mod folder** (usually found for you). Press **Save settings**.

Nothing is typed in a terminal except the Linux password Ubuntu asks for.

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
