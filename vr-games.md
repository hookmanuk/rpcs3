# VR games

This is a VR build of the RPCS3 PlayStation 3 emulator: selected games are rendered in stereo 3D in a
PC VR headset, with head tracking. Below are the games with a VR profile, from most to least playable.
The VR profiles (`vr_profiles/`) and game patches (`patches/`) come with the release, so there is nothing
extra to download.

## Installing

You need:

- Windows 10 or 11, 64-bit, and a graphics card with Vulkan support.
- A PC VR headset with an OpenXR runtime, such as SteamVR.
- The [Microsoft Visual C++ Redistributable (x64)](https://aka.ms/vs/17/release/vc_redist.x64.exe), if it
  is not installed already.
- The PS3 system software: download the official update file `PS3UPDAT.PUP` from
  [PlayStation's PS3 system software page](https://www.playstation.com/support/hardware/ps3/system-software/).
- Your own copies of the games, dumped from discs you own.

Then:

1. [Download and extract the zip](https://github.com/hookmanuk/rpcs3/releases/latest) to a folder you can write to (not `Program Files`). RPCS3 keeps its settings, installed
   firmware and game data in that folder.
2. Run `rpcs3.exe`. Install the firmware with `File > Install Firmware` and pick `PS3UPDAT.PUP`.
3. Add your games with `File > Add Games` (a folder of disc dumps) or boot one with `File > Boot Game`.
   Optionally install game updates with `File > Install Packages/Raps/Edats`.

This build has no automatic updater (the standard RPCS3 updater would replace it with a build without
VR): get new VR builds from the same place as this one.

## Running a game in VR

1. Start your OpenXR runtime before launching a game.
2. Launch a game from the list below as normal. VR switches on by itself for a game with a VR profile
   (`Configuration > Video > VR > Enabled`, on by default). Games without a profile play normally.
3. In-game, the home menu (PS button or Start+Select) has a **VR** tab: HUD size and position, world scale.

---

## Playable

| Game | ID | Framerate |
|---|---|---|
| WipEout HD Fury | BCES00664 | Headset refresh rate |
| Pure | BLUS30182 | Headset refresh rate |
| ICO | BCUS98259 | 30 FPS (Needs Driver Smoothing) |

### 1. WipEout HD Fury (BCES00664): headset refresh rate

- **Requires the 2.51 update.** The disc version (2.00) has no stereoscopic code. Install the updates
  2.10, 2.30, 2.50 and 2.51 in order (`File > Install Packages/Raps/Edats`).
- **Frame rate:** follows the headset (90 Hz = 90 FPS) at real-time speed. No patch needed.
- **Recommended settings** (for 90 FPS): Video > `Relaxed ZCULL Sync` on, `Accurate ZCULL stats` off,
  `Shader Precision` Low; CPU > `Thread Scheduler` RPCS3 Scheduler. `Resolution Scale` around 300-400%:
  every render target exists once per eye, so at very high scales (750%) even a 32 GB card runs out of video
  memory and the game slows to a crawl.

### 2. Pure (BLUS30182): headset refresh rate

- **Frame rate:** follows the headset at real-time speed. The patch "Unlocked frame rate (follows Vblank
  Rate)" is on by default: races run at the display rate instead of 30. The VR profile tells the game the
  current rate every frame, so its clock stays correct at any refresh rate.
- **Recommended settings:** raise `Resolution Scale` as far as your GPU allows.

### 3. ICO (BCUS98259, ICO & Shadow of the Colossus Collection): 30 FPS

- **Frame rate:** 30 FPS, the game's own rate. Its logic is tied to the display clock, so it keeps a 60 Hz
  clock in VR (Match Headset Refresh Rate is not offered). Reprojection Margin Auto renders 10 degrees
  beyond the view so head turns between frames show no black borders.
- **Patches on by default:**
  - *Disable MLAA*: required; with MLAA only one eye is drawn correctly.
  - *Full Pixel Mode always on*: without it the picture is zoomed ~16% and the world swims on head turns.
    The in-game option always shows ON.
  - *Wider view (VR culling)*, scale 20: the game only draws what its own camera sees, so looking around
    showed the sky through missing walls. Lower the scale if frame rate suffers. Without VR this makes the
    TV picture a wide-angle view: set 1.0 to play flat.
- **Recommended settings:** raise `Resolution Scale`. For savestates, turn on Advanced >
  `SPU Compatible Savestates Mode`. Patches are applied at boot, not when loading a savestate.
- Splash screens and videos are shown on a flat screen in front of you.
- **Known issues:** flames can fade oddly right next to walls; some distant objects may still pop in.

### 4. Shadow of the Colossus (BCUS98259, ICO & Shadow of the Colossus Collection): headset refresh rate

New in this release: checked on the desktop (both eyes, head-turn tests, game speed), not yet played
through in a headset.

- **Frame rate:** follows the headset (90 Hz = 90 FPS) at real-time speed: the game times itself from its
  frame rate, and the VR profile gives it the headset's rate every frame. On a TV it runs at 60 FPS.
- **Patches on by default:**
  - *Disable MLAA*: required, as for ICO.
  - *Full Pixel Mode always on*: without it the picture is zoomed ~19% and the world swims on head turns.
  - *Frame rate follows Vblank Rate*: a frame on every display refresh instead of every second one.
  - *Wider view (VR culling)*, scale 3: the game only draws a narrow 44-degree view, so in VR everything
    around it was bright fog. Scale 3 draws about 150 x 130 degrees. Without VR this makes the TV picture a
    wide-angle view: set 1.0 to play flat.
- **Recommended settings:** raise `Resolution Scale`.

---


Defaults, all changeable per game:

| Setting | Default | What it does |
|---|---|---|
| Match Headset Refresh Rate | on | Runs the PS3's display clock at the headset's refresh rate, so games that allow it render a new frame for every headset refresh (90, 120 Hz...). Only offered for games that are not frame-capped (WipEout, Pure, Shadow of the Colossus); capped games such as Ico keep a 60 Hz clock. |
| HUD Fixed In Front | on | HUD and menus stay in front of you instead of following your head. |
| Reprojection Margin | Auto | Renders beyond the edges of the view so the headset can turn older frames without black borders. Auto: 10 degrees for frame-capped games (Ico), 0 otherwise. |
| Game patches marked "on by default" | on | Switch them off in `Manage > Game Patches` if you want to. |

---

## Not yet playable (not included in the release)

These have VR profiles that work on the desktop, but have not been played through in a headset, and some
have known problems. Their profiles and patches are **not shipped**: they are kept in `vr-non-working/` in
the source repository, with instructions for trying them.

| Game | ID | State |
|---|---|---|
| Blur | BLUS30295 | 45-50 FPS in stereo. Needs `Write Color Buffers` on (black screen otherwise). Car shadow may lag on big head turns; mirror motion blur in one eye. |
| Need for Speed Most Wanted | BLUS31010 | Frame-rate patches are fixed per rate (60/72/80/90/120): pick the one matching your headset and set Vblank Rate to match. In-race speed at 90 FPS not yet confirmed. |
| inFamous 2 | BCUS98125 | 52-72 FPS in stereo. A layer processed on the SPUs is only correct in the left eye (slight ghost in the right). |
| Metal Gear Solid 4 | BLUS30109 | ~35 FPS in stereo in Act 1. Once hung ~8 minutes into Act 1. Set `LLVM Precompilation` off (first boot otherwise compiles for 20+ minutes). |
| inFamous | BCUS98119 | ~25 FPS in stereo (13,500 draws per frame): too slow for VR. |
| Split/Second | BLUS30300 | Profile made by the automatic generator; frame-rate patch needs Vblank Rate and its Refresh Rate to match. Not tested in a headset. |
| God of War III | BCUS98111 | Early experimental profile. Not tested in a headset. |

---

RPCS3 is licensed under the GNU GPL v2 (see `LICENSE`). Source code for this VR build:
<https://github.com/hookmanuk/rpcs3/tree/openxr>.
