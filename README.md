# eu4unitgen

Generates Europa Universalis IV land-unit sets (infantry, cavalry, artillery × 4 sprite tiers) for a nation: entities, sprite packs, unit files, localisation and the tech/country lines to merge, validated against your own EU4 install. Built for the *Hellenistic Revolution* total conversion. GPL-3.0.

It never copies Paradox assets: everything is read from your EU4 install at runtime. Details are in [docs/SPEC.md](docs/SPEC.md).

## Install

Python 3.11+ on Windows or Linux.

```
python -m venv .venv
.venv\Scripts\activate          # Linux: source .venv/bin/activate
pip install -e .[dev]
```

## How to run the app

```
python -m eu4unitgen.gui
```

(After `pip install -e .` there is also an `eu4unitgen-gui` command.)

The wizard has six screens, left to right:

1. **Settings**: your EU4 install folder (pre-filled from the `EU4_PATH` environment variable), the output folder, optionally your mod folder for collision checks, the test-mod staging folder, and Blender (pinned to 4.2.23 LTS with io_pdx_mesh 0.91; used from Phase 2 on, only as a background process). Press **Load EU4 install**, then tick the DLC you own.
2. **Unit set**: tag(s), set name, graphical culture (pick one or type a new one such as `hellenicgfx`), the vanilla culture to copy units from, adjective, colour.
3. **Textures** (optional): a texture recipe, one slot at a time (tier 1–4 × infantry/horse/cannon; infantry also dresses the rider and the gun crew). Protect skin/metal, recolour from a reference image's palette, recolour cloth, overlay a pattern, paint country-colour masks, set gloss. **Preview slot** shows the base and the result tinted the way the game tints them; **Export painting guide** writes PNGs to paint your own masks over. Tick "Use a texture recipe" so Validate/Generate/Test mod use it; Save/Load keeps it as a `.json` file.
4. **Validate**: runs every validator rule and lists errors, warnings and notes.
5. **Generate**: writes the drop-in package to `<output>/<set>/` and shows its files and `INTEGRATION.md` (the lines you merge by hand). **Open output folder** opens it in Explorer/your file manager.
6. **Test mod**: builds a throwaway test mod (or the two-launch in-game test suite, or the texture test) into the staging folder, default `build\test_mod`. **Open staging folder**, then copy its contents into your EU4 `mod` folder.

Settings are remembered between runs. Nothing is ever written into the EU4 install or your mod folder.

## Command line

The same steps without the GUI:

```
python -m eu4unitgen.cli generate --tag MAC --gfx hellenicgfx      # package to output/mac/
python -m eu4unitgen.cli pick-test-tag                              # tags no sprite pack touches
python -m eu4unitgen.cli test-mod --tag HDA --gfx hellenicgfx --stage-dir build/test_mod
python -m eu4unitgen.cli test-mod --textures --tag HDA --gfx hellenicgfx --set macedonian_test --test-sprite-techs 2 3 4 --stage-dir build/test_mod
```

`--texture-recipe recipe.json` (on `generate` and `test-mod`) retextures the set; the recipe format is described in `src/eu4unitgen/core/recipe.py`. `--textures` builds the texture test (Phase 2): tier 1 retextured through a `meshsettings` texture override, with a checklist. In the GUI: Test mod → "Texture test".

Add `--help` to any command for its options.

## Tests

```
pytest
```

Set `EU4_PATH` to your install to also run the checks against real game files.
