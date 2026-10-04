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

**Sustained in VR** is the highest headset refresh rate (72, 90 or 120 Hz) the game held in a test scene with
under 1% missed frames, rendered at **4K per eye (Resolution Scale 300%) on a Ryzen 7 9800X3D with an RTX 5090**.
72 Hz is the minimum we count as fully VR compatible. Slower PCs or other settings will differ; "-" means not
measured yet.

| Game | ID | Suggested Framerate | Sustained in VR |
|---|---|---|---|
| WipEout HD Fury | BCES00664 | Headset refresh rate | 90 Hz |
| Pure | BLUS30182 | Headset refresh rate | 90 Hz |
| ICO | BCUS98259 | 30 FPS (Needs Driver Smoothing) | 30 FPS (the game's limit) |
| Shadow of the Colossus | BCUS98259 | 60 FPS (no maximum) | 90 Hz |
| Ridge Racer 7 | BCAS20001 | Headset refresh rate | 90 Hz |
| Demon's Souls | BLUS30443 | Headset refresh rate | 120 Hz |
| Bayonetta | BLUS30367 | Headset refresh rate | 120 Hz |
| Killzone HD | BCES01743 | 90 FPS | 90 Hz |
| Tales of Xillia | BLUS31006 | Headset refresh rate | 120 Hz |
| Super Stardust HD | NPUA80068 | Headset refresh rate | 120 Hz |
| God of War | BCES00800 | Headset refresh rate | 120 Hz |
| God of War II | BCES00800 | Headset refresh rate | 120 Hz |
| Dante's Inferno | BLUS30405 | Headset refresh rate | 120 Hz |
| Asura's Wrath | BLUS30721 | Headset refresh rate | 120 Hz |

### 1. WipEout HD Fury (BCES00664): headset refresh rate

- **Frame rate:** the headset's refresh rate by default (90 Hz = 90 FPS), at real-time speed at any rate. No
  patch needed. A slower PC can choose a lower rate with the VR **Frame Rate** setting.
- **In the headset:** the VR build's patch *Wider view (VR culling)* (on by default) widens the race camera's view
  so the track above and beside you is drawn when you look around (without it the track overhead ends in sky).
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

### 9. Tales of Xillia (BLUS31006, US disc 01.00): headset refresh rate

- **Frame rate:** the patches *60 FPS (VR)* (the community *60 FPS*, bundled) and *Frame rate follows VR* are on
  by default; there is nothing to download. The game advances one 60 FPS step per frame, so above 60 FPS it ran
  fast; with the patches it runs at real speed at the headset rate (walking speed measured the same at 60 and
  90). Turning on the community *60 FPS* as well does no harm. Patches apply when the game boots, not when a
  savestate is loaded.
- **In the headset:** menus show on a flat screen in front of you.
- **Recommended settings:** `Resolution Scale` 300% held 120 FPS in testing (the first field).

### 10. Super Stardust HD (NPUA80068, PSN): headset refresh rate

- A PSN game: install it with `File > Install Packages/Raps/Edats`, with its licence (`.rap`).
- **Frame rate:** the headset's refresh rate by default, at real-time speed at any rate. No patch needed.
- **In the headset:** the planet floats in front of you like a tabletop model; **World Scale** in the home
  menu's VR tab changes its size. The title, menus, demo and game over screens show on a flat screen in front
  of you; play and the pause menu are in 3D.
- **Recommended settings:** `Resolution Scale` 300% held 120 FPS in testing.

### 11. God of War (BCES00800, God of War Collection, UK disc 01.00): headset refresh rate

- Make sure your game configuration Default Resolution is 720p, 1080p will not work: Right-click God of War Collection → Change Custom Configuration → GPU → set Resolution to 1280x720.
- Start it from the collection's game selector. The selector and its intro show on a flat screen in front of you.
- **Frame rate:** the headset's refresh rate, at real-time speed (the VR profile sets the game's frame rate). The
  VR build's patch *QTE button mashing at any frame rate (VR)* (on by default) keeps the button-mashing QTEs (the
  Hydra and others) winnable above 60 FPS. Patches apply when the game boots.
- **In the headset:** the menus, the pause menu and the Power Up screen show on a flat screen in front of you.
  The game draws its scene inside a black border; the headset shows only the scene.
- **Recommended settings:** `Resolution Scale` 300% held 120 FPS in testing.
- **Known issues:** once in about eight boots it crashed when switching from the selector into the game: start it
  again. Loading a savestate and then restarting from a checkpoint can stop the game: boot it fresh instead.

### 12. God of War II (BCES00800, God of War Collection, UK disc 01.00): headset refresh rate

- Make sure your game configuration Default Resolution is 720p, 1080p will not work: Right-click God of War Collection → Change Custom Configuration → GPU → set Resolution to 1280x720.
- Start it from the collection's game selector, as God of War.
- **Frame rate:** the headset's refresh rate, at real-time speed (the VR profile sets the game's frame rate). No
  patch needed: its QTEs already run in real time at any rate.
- **In the headset:** as God of War: menus, pause and Power Up on a flat screen, and only the scene inside the
  game's black border.
- **Recommended settings:** `Resolution Scale` 300% held 120 FPS in testing.
- **Known issues:** as God of War (the selector switch, a checkpoint restart after a savestate).

### 13. Dante's Inferno (BLUS30405, US disc 01.00): headset refresh rate

- **Frame rate:** the headset's refresh rate, at real-time speed. No patch needed: the VR profile sets the game's
  frame interval.
- **VR patches (on by default):** *Wider view (VR culling)* widens the game camera's view so the scenery at the
  edges of the headset view is drawn (Scale 2.75: 170 degrees; lower it in `Manage > Game Patches` if you see
  problems); *Disable camera shake (VR)* removes the screen shake on hits. Patches apply when the game boots.
- **In the headset:** the splash screens, menus and the intro movie show on a flat screen in front of you.
- **Recommended settings:** `Resolution Scale` 300% held 120 FPS in testing.
- **Known issues:** the sun's lens flare does not show with the wide view; the sky is a flat backdrop.

### 14. Asura's Wrath (BLUS30721, US disc 01.00): headset refresh rate

- **Frame rate:** the patches *Unlock FPS (VR)*, *Disable Motion Blur (VR)* and *Disable Depth of Field (VR)* (the
  community patches, bundled) are on by default; there is nothing to download. The game runs at real-time speed
  at the headset rate, without the blur effects that look wrong in a headset. Button-mashing QTEs may be harder
  at higher rates. Turning on the community versions as well does no harm.
- **VR patch (on by default):** *Wider view culling (VR)* stops the game hiding objects outside its narrow cutscene
  cameras (Culling 180 degrees; "The game's own view" turns it off). It also applies when a savestate is loaded.
- **In the headset:** cutscenes show without the letterbox bars; the intro video and the TV screens are flat
  screens in the world.
- **Recommended settings:** `Resolution Scale` 300% held 120 FPS in testing (Episode 1 space battle).
- **Known issues:** in the palace, the golden disc above the throne can turn black depending on where you look.

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

Sustained in VR as above (4K per eye, Ryzen 7 9800X3D + RTX 5090).

| Game | ID | Sustained in VR | State |
|---|---|---|---|
| Blur | BLUS30295 | - | 45-50 FPS in stereo. Needs `Write Color Buffers` on (black screen otherwise). Car shadow may lag on big head turns; mirror motion blur in one eye. |
| Need for Speed Most Wanted | BLUS31010 | - | Frame-rate patches are fixed per rate (60/72/80/90/120): pick the one matching your headset and set Vblank Rate to match. In-race speed at 90 FPS not yet confirmed. |
| inFamous 2 | BCUS98125 | - | 52-72 FPS in stereo. A layer processed on the SPUs is only correct in the left eye (slight ghost in the right). |
| Metal Gear Solid 4 | BLUS30109 | - | ~35 FPS in stereo in Act 1. Once hung ~8 minutes into Act 1. Set `LLVM Precompilation` off (first boot otherwise compiles for 20+ minutes). |
| inFamous | BCUS98119 | - | ~25 FPS in stereo (13,500 draws per frame): too slow for VR. |
| Split/Second | BLUS30300 | - | Profile made by the automatic generator; frame-rate patch needs Vblank Rate and its Refresh Rate to match. Not tested in a headset. |
| God of War III | BCUS98111 | - | Early experimental profile. Not tested in a headset. |
| MotorStorm: Pacific Rift | BCUS98155 | - | Frame-rate patch reaches 90 FPS flat at real-time speed, but stereo drops to ~50-70 FPS at a race start with the pack in view (the emulator's per-draw work; needs multiview). Needs `Write Color Buffers` on. |
| Dragon's Dogma: Dark Arisen | BLUS31155 | below 72 (~68, prologue) | Needs the 01.02 update. Enable the community patch *Unlock FPS*; the VR build's patch *Full screen (no letterbox)* is on by default. Only the prologue has been tested; the open world is untested. Graphics errors in the headset: some text is cut off and a phantom layer appears when you turn your head. |
| Ratchet & Clank Collection | BCUS98282 | below 72 in heavy scenes (R&C 1 ~67, R&C 3 ~70) | All three games run at real-time speed at the headset rate (the profiles set their frame-time values; no patch). Heavy scenes do not hold 72 Hz at 4K per eye yet. Ratchet & Clank 1's pause-menu frames turn with your head. Not tested in a headset. |
| The Darkness | BLUS30035 | below 72 (~53 in gameplay) | Enable the community patch *60 FPS* (keeps real-time speed). The broken lighting in stereo is fixed (checked in the opening, on the desktop and in a simulated headset view). Too slow at 4K per eye; try a lower resolution scale. Not tested in a headset. |
| Dynasty Warriors 6 Empires | BLUS30306 | 60 (frame-locked) | Runs at 60 FPS (the game speeds up above 60); the headset reprojects. Battles checked on the desktop only. |
| Jak and Daxter Collection | BCUS98281 | Jak 1, Jak II: 60 (frame-locked); Jak 3 ~57 | Jak 1 and Jak II step a fixed 1/60 s per frame (they run fast above 60), so their profiles hold them at 60 FPS and the headset reprojects. Jak 1's lens flares are hidden in VR. Jak 3 is too slow (~57 FPS at 72 Hz) and has no profile. Checked on the desktop and in a simulated headset only. |
| Anarchy Reigns | BLUS30632 | 90 Hz | Patch *Frame rate follows VR* (fork, `BLUS30632_patch.yml`, replaces the community *60 FPS*: turn that off): real-time at 90 FPS (Vblank 180, the profile's `vblanks_per_frame 2`). Training > Practice tested on the desktop: stereo, rotation audit and the boxed HUD are right. Campaign and online modes not checked; not tested in a headset. |
| Gran Turismo 5 | BCUS98114 | 90 Hz (race start) | Work in progress: some menu clipping when you lean back; frame drops at some race starts. Frame-rate patch *Frame rate follows VR*. |
| Puppeteer | BCUS98227 | 72 Hz | No patch needed: runs at real-time speed at the headset rate (the profile sets the game's frame time). The game post-processes each frame on the SPUs, which only ever sees the left eye: in VR the profile reads the frame from before that step, so its anti-aliasing is skipped. The stage looks small in the headset (the game's narrow theatre camera). Not tested in a headset. |
| Killzone 2 | BCUS98116 | - | Needs `Write Color Buffers` and `Read Color Buffers` on. Runs at 45 FPS by default (stereo reaches about 60). Not tested in a headset. |
| Kingdom Hearts HD 1.5 ReMIX | BLUS31212 | 120 Hz | Kingdom Hearts Final Mix only. The VR build's patch *Unlocked frame rate (VR)* (on by default) keeps the game at real-time speed at the headset rate. Not tested in a headset. |
| Kingdom Hearts HD 2.5 ReMIX | BLUS31460 | 72 Hz | Kingdom Hearts II Final Mix only. The VR build's patch *Unlocked frame rate (VR)* (on by default) keeps the game at real-time speed at the headset rate. Not tested in a headset. |
| Sonic & All-Stars Racing Transformed | BLUS30839 | 90 Hz (120 with Wider view off) | The VR build's patches *Unlocked frame rate (follows Vblank Rate)* (real-time at any rate) and *Wider view (VR culling)* (Scale 3.0: the track and sky fill the headset's view) are on by default. Distant soft shadows can differ between the eyes. Not tested in a headset. |
| Dynasty Warriors: GUNDAM | BLUS30058 | 120 Hz | The VR build's patches *Frame rate follows VR* (keeps battles at real-time speed above 60) and *Wider view (VR culling)* are on by default. Official Mode battles checked; space missions and cutscenes not. Not tested in a headset. |
| X-Men Origins: Wolverine | BLUS30268 | 72 Hz | The VR build's patch *Unlocked frame rate (VR)* (on by default) removes the game's 62 FPS cap; the profile runs it at 72 (Vblank 144). For flat 60 FPS also enable *Present every vblank (flat 60 FPS)*. Some surfaces are shaded differently in each eye, and the sky can end at the edge of the game's view. Not tested in a headset. |

---

RPCS3 is licensed under the GNU GPL v2 (see `LICENSE`). Source code for this VR build:
<https://github.com/hookmanuk/rpcs3/tree/openxr>.
