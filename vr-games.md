# VR games

Games with a VR profile in this build, from most to least playable. The VR profiles (`vr_profiles/`)
and game patches (`patches/`) ship with the release, so there is nothing extra to download.

## Getting started

1. Start SteamVR (or another OpenXR runtime) before launching a game.
2. Launch a game from the list below as normal. VR switches on by itself for a game with a VR profile
   (`Configuration > Video > VR > Enabled`, on by default). Games without a profile play normally.
3. In-game, the home menu (PS button) has a **VR** tab: HUD size and position, world scale, and the
   options below.

Defaults, all changeable per game:

| Setting | Default | What it does |
|---|---|---|
| Match Headset Refresh Rate | on | Runs the PS3's display clock at the headset's refresh rate, so games that allow it render a new frame for every headset refresh (90, 120 Hz...). Only offered for games that are not frame-capped (WipEout, Pure); capped games such as Ico keep a 60 Hz clock. |
| HUD Fixed In Front | on | HUD and menus stay in front of you instead of following your head. |
| Reprojection Margin | Auto | Renders beyond the edges of the view so the headset can turn older frames without black borders. Auto: 10 degrees for frame-capped games (Ico), 0 otherwise. |
| Game patches marked "on by default" | on | Switch them off in `Manage > Game Patches` if you want to. |

Splash screens, videos and menus without 3D are shown on a flat screen in front of you.

---

## Playable

### 1. WipEout HD Fury (BCES00664): headset refresh rate

- **Requires the 2.51 update.** The disc version (2.00) has no stereoscopic code. Install the updates
  2.10, 2.30, 2.50 and 2.51 in order (`File > Install Packages/Raps/Edats`).
- **Frame rate:** follows the headset (90 Hz = 90 FPS) at real-time speed. No patch needed.
- **Recommended settings** (for 90 FPS): Video > `Relaxed ZCULL Sync` on, `Accurate ZCULL stats` off,
  `Shader Precision` Low; CPU > `Thread Scheduler` RPCS3 Scheduler. Raise `Resolution Scale` as far as
  your GPU holds the headset's frame rate.

### 2. Pure (BLUS30182): headset refresh rate

- **Frame rate:** follows the headset at real-time speed. The patch "Unlocked frame rate (follows Vblank
  Rate)" is on by default: races run at the display rate instead of 30. The VR profile tells the game the
  current rate every frame, so its clock stays correct at any refresh rate.
- **Recommended settings:** raise `Resolution Scale` as far as your GPU allows.

### 3. ICO (BCUS98259, ICO & Shadow of the Colossus Collection): 30 FPS

Only ICO has a profile; Shadow of the Colossus does not.

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
- **Known issues:** flames can fade oddly right next to walls; some distant objects may still pop in.

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
