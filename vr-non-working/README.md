# VR profiles and patches that are not shipped

Work in progress for games that are not yet playable in VR (see `vr-games.md`). They live outside `bin/`,
so release packages do not include them.

To try one, copy its profile to `bin/vr_profiles/` and its patch file (if any) to `bin/patches/`.
Frame-rate patches here are not marked "Enabled By Default"; enable them in `Manage > Game Patches`.

| Game | ID | Profile | Patch |
|---|---|---|---|
| God of War III | BCUS98111 | `vr_profiles/BCUS98111.json` | |
| Gran Turismo 5 | BCUS98114 | `vr_profiles/BCUS98114.json` | `patches/BCUS98114_patch.yml` |
| inFamous | BCUS98119 | `vr_profiles/BCUS98119.json` | |
| inFamous 2 | BCUS98125 | `vr_profiles/BCUS98125.json` | |
| Metal Gear Solid 4 | BLUS30109 | `vr_profiles/BLUS30109.json` | `patches/BLUS30109_patch.yml` |
| Blur | BLUS30295 | `vr_profiles/BLUS30295.json` | `patches/BLUS30295_patch.yml` |
| Split/Second | BLUS30300 | `vr_profiles/BLUS30300.json` | `patches/BLUS30300_patch.yml` |
| Need for Speed Most Wanted | BLUS31010 | `vr_profiles/BLUS31010.json` | `patches/BLUS31010_patch.yml` |
| Dragon's Dogma: Dark Arisen (update 01.02) | BLUS31155 | `vr_profiles/BLUS31155.json` | `patches/BLUS31155_patch.yml` (full screen, on by default); also enable the community *Unlock FPS* |
| Ratchet & Clank Collection (R&C 1 tested) | BCUS98282 | `vr_profiles/BCUS98282.json` | |
| The Darkness | BLUS30035 | `vr_profiles/BLUS30035.json` (stereo lighting broken) | community *60 FPS* patch |
| Dynasty Warriors 6 Empires | BLUS30306 | `vr_profiles/BLUS30306.json` (60 FPS) | |
| Puppeteer | BCUS98227 | `vr_profiles/BCUS98227.json` (90 FPS real-time; stereo via `texture_redirects`) | |
| Jak and Daxter Collection (Jak 1, Jak II) | BCUS98281 | `vr_profiles/BCUS98281.json`, `BCUS98281.jak1.json`, `BCUS98281.jak2.json` | |
| Anarchy Reigns | BLUS30632 | `vr_profiles/BLUS30632.json` | `patches/BLUS30632_patch.yml` (*Frame rate follows VR*, on by default; turn off the community *60 FPS*) |
| MotorStorm: Pacific Rift | BCUS98155 | `vr_profiles/BCUS98155.json` | `patches/BCUS98155_patch.yml` |
| Killzone 2 | BCUS98116 | `vr_profiles/BCUS98116.json` (needs Write and Read Color Buffers) | |
| Kingdom Hearts HD 1.5 ReMIX (KH Final Mix) | BLUS31212 | `vr_profiles/BLUS31212.json` (launcher), `BLUS31212.kingdom.json` | `patches/BLUS31212_patch.yml` (*Unlocked frame rate (VR)*, on by default) |
| Kingdom Hearts HD 2.5 ReMIX (KH II Final Mix) | BLUS31460 | `vr_profiles/BLUS31460.json` (launcher), `BLUS31460.kingdom2.json` | `patches/BLUS31460_patch.yml` (*Unlocked frame rate (VR)*, on by default) |

Notes and evidence for each game are in the separate plans repository (`plans/profiles/<ID>-notes.md`).
