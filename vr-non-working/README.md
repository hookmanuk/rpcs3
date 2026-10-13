# VR profiles and patches that are not shipped

Work in progress for games that are not yet playable in VR (see `vr-games.md`). They live outside `bin/`,
so release packages do not include them.

To try one, copy its profile to `bin/vr_profiles/` and its patch file (if any) to `bin/patches/`.
Frame-rate patches here are not marked "Enabled By Default"; enable them in `Manage > Game Patches`.

| Game | ID | Profile | Patch |
|---|---|---|---|
| God of War III | BCUS98111 | `vr_profiles/BCUS98111.json` | |
| inFamous | BCUS98119 | `vr_profiles/BCUS98119.json` | |
| inFamous 2 | BCUS98125 | `vr_profiles/BCUS98125.json` | |
| Metal Gear Solid 4 | BLUS30109 | `vr_profiles/BLUS30109.json` | `patches/BLUS30109_patch.yml` |
| Blur | BLUS30295 | `vr_profiles/BLUS30295.json` | `patches/BLUS30295_patch.yml` |
| Split/Second | BLUS30300 | `vr_profiles/BLUS30300.json` | `patches/BLUS30300_patch.yml` |
| Need for Speed Most Wanted | BLUS31010 | `vr_profiles/BLUS31010.json` | `patches/BLUS31010_patch.yml` |
| Dragon's Dogma: Dark Arisen (update 01.02) | BLUS31155 | `vr_profiles/BLUS31155.json` | `patches/BLUS31155_patch.yml` (full screen, on by default); also enable the community *Unlock FPS* |
| The Darkness | BLUS30035 | `vr_profiles/BLUS30035.json` (stereo lighting broken) | community *60 FPS* patch |
| Dynasty Warriors 6 Empires | BLUS30306 | `vr_profiles/BLUS30306.json` (60 FPS) | |
| Puppeteer | BCUS98227 | `vr_profiles/BCUS98227.json` (90 FPS real-time; stereo via `texture_redirects`) | |
| Jak and Daxter Collection (Jak 1, Jak II) | BCUS98281 | `vr_profiles/BCUS98281.json`, `BCUS98281.jak1.json` (60 FPS), `BCUS98281.jak2.json` | `patches/BCUS98281_patch.yml` (*Frame rate follows VR (Jak II)*, on by default) |
| Anarchy Reigns | BLUS30632 | `vr_profiles/BLUS30632.json` | `patches/BLUS30632_patch.yml` (*Frame rate follows VR*, on by default; turn off the community *60 FPS*) |
| Killzone 2 | BCUS98116 | `vr_profiles/BCUS98116.json` (needs Write and Read Color Buffers) | |
| Kingdom Hearts HD 1.5 ReMIX (KH Final Mix) | BLUS31212 | `vr_profiles/BLUS31212.json` (launcher), `BLUS31212.kingdom.json` | `patches/BLUS31212_patch.yml` (*Unlocked frame rate (VR)*, on by default) |
| Kingdom Hearts HD 2.5 ReMIX (KH II Final Mix) | BLUS31460 | `vr_profiles/BLUS31460.json` (launcher), `BLUS31460.kingdom2.json` | `patches/BLUS31460_patch.yml` (*Unlocked frame rate (VR)*, on by default) |
| Dynasty Warriors: GUNDAM | BLUS30058 | `vr_profiles/BLUS30058.json` | `patches/BLUS30058_patch.yml` (*Frame rate follows VR*, *Wider view (VR culling)*, both on by default) |
| X-Men Origins: Wolverine | BLUS30268 | `vr_profiles/BLUS30268.json` (72 FPS: Vblank 144) | `patches/BLUS30268_patch.yml` (*Unlocked frame rate (VR)* on by default; *Present every vblank (flat 60 FPS)* off) |
| Dragon Age II | BLUS30645 | `vr_profiles/BLUS30645.json` (just under 72 FPS in busy fights) | |
| MX vs ATV Reflex | BLUS30321 | `vr_profiles/BLUS30321.json` (generated, 30 FPS; not checked) | |
| Prince of Persia | BLUS30214 | `vr_profiles/BLUS30214.json` (90 FPS at 300%) | `patches/BLUS30214_patch.yml` (*Unlocked frame rate (follows Vblank Rate)*, on by default) |
| NiGHTS into dreams... | NPUB30798 | `vr_profiles/NPUB30798.json` (hand-made; 30 FPS, no frame-rate patch yet) | |

Notes and evidence for each game are in the separate plans repository (`plans/profiles/<ID>-notes.md`).
