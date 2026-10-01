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
   (**Enable VR Support** in the game's custom configuration, on by default). Games without a profile play
   normally.
3. In-game, the home menu (PS button or Start+Select) has a **VR** tab: HUD size and position, world scale.

To raise the resolution or change the VR settings for a game, see [vr-settings.md](vr-settings.md).

---

## Playable

| Game | ID | Suggested Framerate |
|---|---|---|
| WipEout HD Fury | BCES00664 | Headset refresh rate |
| Pure | BLUS30182 | Headset refresh rate |
| ICO | BCUS98259 | 30 FPS (Needs Driver Smoothing) |
| Shadow of the Colossus | BCUS98259 | 60 FPS (no maximum) |
| Ridge Racer 7 | BCAS20001 | Headset refresh rate |
| Demon's Souls | BLUS30443 | Headset refresh rate |
| Bayonetta | BLUS30367 | Headset refresh rate |
| Killzone HD | BCES01743 | 90 FPS |

### 1. WipEout HD Fury (BCES00664): headset refresh rate

- **Frame rate:** the headset's refresh rate by default (90 Hz = 90 FPS), at real-time speed at any rate. No
  patch needed. A slower PC can choose a lower rate with the VR **Frame Rate** setting.
- **Recommended settings** (for 90 FPS): Video > `Relaxed ZCULL Sync` on, `Accurate ZCULL stats` off,
  `Shader Precision` Low; CPU > `Thread Scheduler` RPCS3 Scheduler. `Resolution Scale` around 300-400%:
  every render target exists once per eye, so at very high scales (750%) even a 32 GB card runs out of video
  memory and the game slows to a crawl.

### 2. Pure (BLUS30182): headset refresh rate

- **Frame rate:** the headset's refresh rate by default; a slower PC can choose a lower rate with the VR
  **Frame Rate** setting. The patch "Unlocked frame rate (follows Vblank Rate)" is on by default: races run at
  the display rate instead of 30. The VR profile tells the game the current rate every frame, so its clock
  stays correct at any rate.
- **Recommended settings:** raise `Resolution Scale` as far as your GPU allows.

### 3. ICO (BCUS98259, ICO & Shadow of the Colossus Collection): 30 FPS

- **Frame rate:** 30 FPS, the game's own rate and its maximum (its logic is tied to the display clock, so it
  keeps a 60 Hz clock). Reprojection Margin Auto renders 10 degrees beyond the view so head turns between
  frames show no black borders.
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
  On the first run of a new install the menus and HUD may not show correctly (not scaled into the HUD box);
  they are fine from the second launch.

### 4. Shadow of the Colossus (BCUS98259, ICO & Shadow of the Colossus Collection): 60 FPS

- **Frame rate:** 60 FPS by default, as the community 60 FPS patch: the patch *Frame rate follows Vblank Rate*
  draws a frame on every vblank instead of every second one, and the VR profile runs the vblank at the chosen
  frame rate (ICO in the same collection keeps its 30 FPS). Faster PCs can choose more with the VR **Frame
  Rate** setting; if frames run late the game can drop objects for a frame (see Known issues). A headset
  running at the same rate or a multiple (60 FPS: 60 or 120 Hz) is smoothest.
- **Patches on by default:**
  - *Disable MLAA*: required, as for ICO.
  - *Full Pixel Mode always on*: without it the picture is zoomed ~19% and the world swims on head turns.
  - *Disable Mesh Trimming*: the game drops small triangles for the PS3's resolution; at higher resolution
    scales and in VR that shows as missing pieces of stairs, legs and other thin geometry.
  - *Wider view (VR culling)*, scale 3: the game only draws a narrow 44-degree view, so in VR everything
    around it was fog. Scale 3 draws about 150 x 130 degrees. Without VR this makes the TV picture a
    wide-angle view: set 1.0 to play flat.
  - *Frame rate follows Vblank Rate*: 60 FPS (above).
  - *Disable Motion Blur* and *Disable Bloom*: motion blur smears the view on head turns, and both cost GPU
    time at high resolution scales.
- **Recommended settings:** `Resolution Scale` 300-400% (on an RTX 5090, 400% holds 60 FPS).
- **Known issues:** at the wider view's edge, big head turns can show fog beyond the drawn area. If frames
  run late (resolution scale too high for the GPU), the game drops objects for a frame: flashing holes. Lower
  the resolution scale.

### 5. Ridge Racer 7 (BCAS20001, Asian release, English/Japanese): headset refresh rate

- **Frame rate:** the headset's refresh rate by default; a slower PC can choose a lower rate with the VR
  **Frame Rate** setting. The game counts time in frames (at 90 Hz it ran 1.5x fast), so the patch *Frame rate
  follows VR* is on by default: the VR profile gives it the current frame rate, and races, physics and lap
  times run at real speed at any rate.
- **Recommended settings:** `Resolution Scale` 300-400% (on an RTX 5090, 400% holds 90 FPS on the starting
  grid).
- **Known issues:** emulation can occasionally stop when the main-menu video starts; restart the game if it
  does. Don't turn on `Log shader programs` for this game: it makes this happen every time.

### 6. Demon's Souls (BLUS30443, US release): headset refresh rate

- **Frame rate:** the headset's refresh rate by default; a slower PC can choose a lower rate with the VR
  **Frame Rate** setting. The patch *Unlocked frame rate (follows Vblank Rate)* is on by default (the community
  Unlock FPS patch): game time follows the real clock, so running and rolling keep real speed at 90 FPS.
  Below 20 FPS the game slows down.
- **Patches on by default:** *Disable Motion Blur* (it followed the game camera, not your head) and *Wider
  view (VR culling)*, which draws more of the world around you (130 degrees by default; lower it if frame
  rate suffers, or set 43 to play without VR). The VR profile also turns off the game's depth of field, which
  blurred the floor and the edges of the view.
- **Cutscenes:** in-engine cutscenes play at the full frame rate: the patch *Smooth cutscenes* (on by default) interpolates their camera and object tracks, which the game steps at 30 per second. Pre-rendered videos (30 FPS) play on a fixed screen in front of you, placed by the HUD settings. Patches apply when the game boots, not when a savestate is loaded.

### 7. Bayonetta (BLUS30367, US release): headset refresh rate

- **Frame rate:** the headset's refresh rate by default; a slower PC can choose a lower rate with the VR
  **Frame Rate** setting. The patch *Unlocked frame rate (real-time above 60 FPS)* is on by default: the game
  measures real time but never let a frame count for less than a 60 FPS frame, so at 90 FPS it ran 1.5x fast.
  With the patch, combat, movement and timers run at real speed at any rate up to 240 FPS. Patches apply when
  the game boots, not when a savestate is loaded.
- **Recommended settings:** Video > `Multithreaded RSX` on (stereo doubles the emulator's per-draw work);
  `Resolution Scale` as high as your GPU allows while holding the frame rate.
- **Known issues:** some menus show visual glitches in VR. The pink magic wisps around Bayonetta are drawn by
  the game on the screen, not in the world, so they sit in the HUD box instead of on her.

### 8. Killzone HD (BCES01743, European disc): 90 FPS

- **Required settings:** Video > `Write Color Buffers` **and** `Read Color Buffers` both on (right-click the game
  > Create Custom Configuration). Without them the 3D world is black in levels (menus and HUD still show).
- **Frame rate:** the patch *Frame rate 90 FPS (set Vblank Rate 90)* is on by default and the game runs at 90 FPS
  in the headset at real-time speed. For another rate, turn it off and turn on the 60, 72 or 120 FPS entry in
  `Manage > Game Patches`, and set the VR **Frame Rate** to match. Intro and menu videos play at 60 FPS (the
  game's video player stalls above that); gameplay returns to 90. Patches apply when the game boots.
- **In the headset:** the film grain is off (to turn it back on, set `"hidden": false` in the "Film grain" entry
  of `bin/vr_profiles/BCES01743.json`). The aiming reticule is shown at 35% of its size (the two `scale` values
  in the same file). The HUD sits at 4 m by default; change it with **HUD Depth** in the home menu's VR tab.
- **Recommended settings:** `Resolution Scale` 300% held 88-90 FPS in testing.

---


Defaults, all changeable per game:

| Setting | Default | What it does |
|---|---|---|
| Frame Rate | the game's own (table above) | The game's frame rate in the headset: 30, 45, 60, 72, 75, 80, 90, 120, 144 or Unlimited (the headset's refresh rate). Until you choose one, each game runs its own default (ICO and Shadow of the Colossus share a configuration but keep their own). Only rates up to what the game works at are offered (ICO: 30). The PS3's display clock follows it; the Vblank Rate setting applies again without a headset. |
| HUD Fixed In Front | on | HUD and menus stay in front of you instead of following your head. |
| HUD Depth | Auto | How far away the HUD and menus appear (1-10 m). Auto: the game's own distance from its VR profile (Killzone HD: 4 m), otherwise 2 m. |
| Reprojection Margin | Auto | Renders beyond the edges of the view so the headset can turn older frames without black borders. Auto: 10 degrees when the game runs below the headset's refresh rate, 0 otherwise. |
| Game patches marked "on by default" | on | Switch them off in `Manage > Game Patches` if you want to. |

---

## Frame generation (OFXR Bridge)

[OFXR Bridge](https://github.com/djules75/OFXR-Bridge) 0.2.7 or later works with this build: turn on its
**Vulkan support** option. It adds one generated frame after every game frame, so the game must run at exactly
half the headset's refresh rate or frames are dropped unevenly and turns judder. Set the VR **Frame Rate** to
half the refresh: **45 FPS** on a 90 Hz headset, 60 FPS on 120 Hz.

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
| MotorStorm: Pacific Rift | BCUS98155 | Frame-rate patch reaches 90 FPS flat at real-time speed, but stereo drops to ~50-70 FPS at a race start with the pack in view (the emulator's per-draw work; needs multiview). Needs `Write Color Buffers` on. |
| Dragon's Dogma: Dark Arisen | BLUS31155 | Needs the 01.02 update. Enable the community patch *Unlock FPS*; the VR build's patch *Full screen (no letterbox)* is on by default. 90 FPS in stereo in the prologue; the open world is untested. NPC name tags stay in the HUD box. Not tested in a headset. |
| Ratchet & Clank Collection | BCUS98282 | Ratchet & Clank 1 only: runs at its native 60 FPS (the game speeds up at higher rates), the headset reprojects. Pause-menu frames turn with your head. Not tested in a headset. |
| Tales of Xillia | BLUS31006 | Enable the community patch *60 FPS*; the VR build's *Frame rate follows VR* (on by default) keeps the game at real-time speed at the headset rate (walking speed measured equal at 60 and 90). Field exploration checked on the desktop only; battles not checked. |
| The Darkness | BLUS30035 | Not usable yet: in VR the lighting breaks (red light over surfaces, dark bands). The community patch *60 FPS* works and keeps real-time speed. |
| Gran Turismo 5 | BCUS98114 | Work in progress: some menu clipping when you lean back, frame drops at the race start. Frame-rate patch *Frame rate follows VR*. |

---

RPCS3 is licensed under the GNU GPL v2 (see `LICENSE`). Source code for this VR build:
<https://github.com/hookmanuk/rpcs3/tree/openxr>.
