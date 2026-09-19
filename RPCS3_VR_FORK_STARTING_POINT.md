# RPCS3 Native VR Fork — Starting Point

## Purpose

This document captures the current technical direction for adding **native, geometry-correct stereoscopic 6DoF VR** to an RPCS3 fork.

The goal is **not** a virtual cinema screen, post-process depth reconstruction, alternate-eye rendering, or heuristic screen-space stereo. The target is genuine per-eye rendering from a camera transform sufficiently upstream that head rotation and translation produce correct geometry, parallax, and newly exposed scene content.

The initial reference title is **WipEout HD Fury (BCES00664), updated to v2.51**. Its native PS3 stereoscopic mode is working in RPCS3 and provides a valuable known-good stereo reference.

---

## 1. Core requirements

The project should ultimately provide:

- Genuine geometry-correct left/right views.
- Full **6DoF** head tracking: yaw, pitch, roll, X, Y, Z.
- OpenXR presentation.
- Both eyes generated from the **same guest frame**.
- Normal PS3/Cell simulation executed once per guest frame.
- A path that works with games that did not originally support stereoscopic 3D.
- Deterministic per-game profiles where necessary.
- Explicit handling/classification of world geometry, HUD, sky, shadows, render-to-texture passes, post-processing, etc.
- Preservation of the game's own camera motion, with the HMD pose composed locally on top.

The project should avoid relying on:

- Screen-space depth reconstruction as the primary stereo mechanism.
- Alternate-eye rendering (AER).
- Running the entire emulated PS3 twice for two eyes.
- Assuming that a final `gl_Position` offset alone is sufficient for 6DoF.
- Assuming that every draw using a matrix is a world-camera draw.

---

## 2. Why RPCS3 / PS3 is a promising target

Compared with PS2 emulation, PS3 graphics are much closer to a conventional programmable GPU pipeline. RPCS3 receives RSX state, translates RSX vertex programs to host shaders, maintains transform constants, and eventually issues host Vulkan draws.

The important conceptual path is approximately:

```text
PS3 game / Cell
    |
    v
RSX commands
    |
    v
RPCS3 RSX state
    |- vertex/transform program
    |- transform constants
    |- vertex arrays
    |- draw state
    v
VKGSRender / Vulkan backend
    |
    |- translated vertex shader
    |- translated fragment shader
    |- uploaded transform constants
    |- Vulkan pipeline/draw
    v
host render target / presentation
```

The exact current source paths and function names must be verified against the checked-out RPCS3 revision rather than assumed from this document. Areas known to be relevant include RPCS3's RSX method/state code and Vulkan renderer, including `Emu/RSX/rsx_methods.h` and `Emu/RSX/VK/VKGSRender.cpp` in historical/current layouts.

A particularly interesting RSX resource is the transform-constant bank (historically exposed as roughly 512 `vec4` constants). A game may place camera/view/projection matrices in title-specific constant slots. This makes RPCS3 a good candidate for a per-title/profile-driven camera interception system.

---

## 3. WipEout HD as the first reference title

Initial title:

```text
WipEout HD Fury
Title ID: BCES00664
Version: 2.51
```

RPCS3's per-game configuration can expose the PS3's native stereoscopic support. For WipEout, the working configuration is based on:

```text
Default Resolution: 1280x720
Enable 3D Support: enabled
3D Display Mode: Side-by-side
```

After updating the title, WipEout detects the emulated 3D-capable display and produces genuine stereoscopic output.

This is useful for much more than simply viewing the game in 3D. WipEout gives us a **ground-truth oracle**:

1. The game already knows how to produce correct left and right cameras.
2. RPCS3 receives the RSX state generated for both views.
3. We can capture and compare the two eye paths.
4. Differences in vertex programs, constants, render targets and draw order can help identify the actual camera transformation.
5. Our later injected stereo can be compared against the game's official implementation.

### Native stereo frame-rate limitation

WipEout's original PS3 stereoscopic mode reduces the title from its normal ~60 fps mode to approximately **30 fps** while producing two views. This made sense on original RSX hardware because stereo substantially increased rendering work.

We do **not** want this native 30 fps mode to become the final RPCS3 VR implementation. It is primarily a reverse-engineering and validation tool.

The desired later experiment is:

```text
WipEout normal 2D mode @ 60 fps
            |
            v
      one RSX workload
            |
       RPCS3 intercept
         /       \
        /         \
   left camera   right camera
        \         /
         \       /
          OpenXR
```

If successful, this would bypass WipEout's original 30 fps PS3 stereo compromise.

---

## 4. HindsightVR as a reference architecture

HindsightVR is highly relevant:

- Repository: https://gitlab.com/aknumbers/hindsightvr

It should be checked out alongside RPCS3 and studied before inventing our own camera-discovery and XR architecture.

HindsightVR demonstrates several concepts directly applicable to this project:

- Emulator-independent XR/runtime core plus emulator-specific adapters.
- Per-game camera/profile data rather than hard-coding every game into C++.
- Camera hunting by inspecting game-selected shader constants.
- Candidate ranking and deliberate perturbation to prove which constants affect the real world camera.
- Draw classification so HUD, shadows, light-space matrices and other false positives are not treated as the camera.
- Genuine stereo generated from one guest frame.
- Rejection of alternate-eye rendering due to temporal mismatch between eyes.
- Vulkan multiview as a possible optimized stereo implementation.
- Awareness of deferred rendering, resolve passes, render-to-texture and other cases where correct stereo geometry can later be flattened back into mono.
- OpenXR predicted pose handling and a separation between emulator integration and XR presentation.

Hindsight's Tier-2 style camera discovery is particularly relevant to RPCS3: games may choose their own vertex shader constant locations, so a title profile identifies the shader/draw and constant range containing the camera transform.

Useful Hindsight areas to study include its primer/documentation, Xenia and xemu integrations, profile system, camera telemetry/hunting tools, draw classification, OpenXR core and Vulkan multiview implementation.

Do **not** assume undocumented/private Hindsight adapters are RPCS3. Only use publicly verifiable implementation details.

---

## 5. Proposed RPCS3 camera-discovery methodology

The first development tool should be an **RSX Stereo / Camera Inspector**, not OpenXR.

For relevant draws, capture enough information to correlate and compare them, such as:

```text
frame number
pass/draw index
vertex-program identity/hash
fragment-program identity/hash (useful for classification)
render-target identity
viewport/scissor where useful
RSX stereo/eye state if available
transform constant bank or selected changed ranges
other state required to correlate L/R draws
```

### WipEout comparison modes

Capture equivalent scenes in at least:

```text
A. normal 2D @ ~60 fps
B. native stereo left eye @ ~30 fps
C. native stereo right eye @ ~30 fps
```

The ideal discovery is something structurally like:

```text
vertex program hash: H
camera constants: c[N..N+3]

2D:
    M

native stereo left:
    M_left

native stereo right:
    M_right
```

The exact matrix representation/order cannot be assumed. It must be inferred and experimentally verified.

### Candidate ranking

Useful signals include:

- Constant ranges that differ predictably between native L/R eyes.
- Matrices that are stable for world draws but not HUD draws.
- Values that change with the game's camera orientation/position.
- Reuse across many world-geometry draws.
- Association with the same vertex-program hash or family of hashes.

Beware false positives such as:

- shadow/light-space matrices;
- reflection cameras;
- projection matrices for RTT passes;
- HUD transforms;
- fullscreen/post-process draws;
- sky transforms;
- model/world matrices.

### Perturbation proof

Once a candidate is identified, deliberately perturb it.

We need to prove independent control over:

```text
X translation
Y translation
Z translation
yaw
pitch
roll
```

The key test is **translation**.

If moving the injected camera sideways merely shifts the completed image, the interception point is wrong. Correct translation should produce motion parallax and reveal/occlude geometry according to the new viewpoint.

A successful six-axis manual camera test is the primary go/no-go gate before OpenXR work begins.

---

## 6. Camera composition

The VR camera should not replace the game's camera. Conceptually it should compose transforms approximately as:

```text
final camera = game camera * HMD local pose * eye pose
```

The exact multiplication order and coordinate conversions depend on RPCS3/game matrix conventions and must be derived experimentally.

This preserves:

- vehicle/cockpit motion;
- scripted cameras;
- game camera animation;
- banking;
- camera shake where desired;

while adding the player's local tracked head motion.

World scale must be configurable per game. OpenXR reports physical poses in metres, while games use arbitrary world units. WipEout's native eye separation may provide a useful initial clue for deriving game-units-per-metre, but native stereo separation may be artistically tuned and should not automatically be treated as physically calibrated IPD.

---

## 7. Stereo rendering strategy

### Prototype: dual host draws

The simplest first proof is likely to issue/replay the relevant host GPU work for both eyes with different camera constants.

Conceptually:

```text
one guest draw
    |
    +--> host draw: left camera
    |
    +--> host draw: right camera
```

The Cell/PPU/SPU side should not execute twice.

This is easier to instrument and reason about than immediately implementing multiview.

### Later optimization: Vulkan multiview

Hindsight demonstrates that Vulkan multiview can be a strong final architecture. Conceptually:

```text
              guest draw
                  |
          translated shader
                  |
          +-------+-------+
          |               |
       view 0           view 1
          |               |
   left constants    right constants
          |               |
          +-------+-------+
                  |
          layered target
                  |
               OpenXR
```

A shader could select eye-specific camera data using the Vulkan view index. This potentially avoids duplicating significant host-side draw submission work.

Do not optimize for multiview until the dual-eye camera proof works.

---

## 8. Render-to-texture and multipass complications

Duplicating a world draw does **not** automatically mean a game becomes correctly stereoscopic.

PS3 games can use pipelines such as:

```text
world geometry
     |
     v
G-buffer / intermediate render targets
     |
     v
lighting / resolves
     |
     v
post-processing
     |
     v
final framebuffer
```

Stereo information can be lost or incorrectly combined later if intermediate resources remain mono.

Potential categories include:

- direct world geometry;
- G-buffer passes;
- depth prepasses;
- shadow maps;
- reflection/refraction passes;
- sky;
- particle/effect passes;
- HUD/UI;
- fullscreen post-processing;
- final compositing.

The eventual per-game profile may therefore need more than a camera constant address. It may include shader hashes, render-target rules and pass classifications.

Hindsight's Xenia work is particularly relevant here because it encountered deferred-rendering/resolve issues where correct stereo geometry could be flattened later in the pipeline.

---

## 9. HUD and 2D elements

HUDs should not automatically inherit the world camera transform.

Possible policies include:

```text
WORLD       -> stereo + full tracked camera
SKY         -> rotation, special translation policy
HUD         -> mono/head-locked or configurable depth plane
POSTPROCESS -> per-eye or carefully classified
SHADOW      -> generally not treated as eye camera
REFLECTION  -> separate camera/pass rules
```

These policies should be profile-driven where needed.

A central reason for rejecting generic post-process stereo reconstruction is that HUD, overlays and effects otherwise become difficult to distinguish reliably from world geometry.

---

## 10. CPU-side culling is a hard limitation

Even with perfect camera injection, RPCS3 cannot render geometry that the game never submitted to RSX.

Large head translations may therefore expose:

- missing objects;
- aggressive frustum-culling boundaries;
- LOD errors;
- missing off-camera scenery.

The initial 6DoF target should therefore be **normal seated head movement**, roughly on the order of tens of centimetres rather than room-scale locomotion.

Do not interpret this as a reason to implement only 3DoF. Translation is important for presence and must work from the first meaningful camera prototype.

---

## 11. Frame-rate mismatch and OpenXR cadence

The emulator/game and the headset do not need to run at the same rate.

There are conceptually three clocks:

```text
guest simulation       stereo rendering       HMD compositor
    30/60 Hz      ->       30/60 Hz       ->      90/120 Hz
```

The fundamental initial rule is:

> **One guest frame produces both eyes.**

Never produce left from guest frame N and right from guest frame N+1.

If the game runs at 60 fps and the HMD at 90 Hz, OpenXR/compositor reprojection can reuse/reproject frames for display refreshes where no new guest frame exists.

### World motion versus head motion

Game/world motion remains tied to the guest cadence:

```text
vehicles
animations
physics
scripted camera movement
```

Head movement ideally feels closer to HMD cadence:

```text
yaw/pitch/roll
X/Y/Z head movement
```

Compositor reprojection is particularly effective for orientation. Translational reprojection is harder because a flat image does not contain geometry newly revealed by head translation.

---

## 12. Possible future RSX replay / late camera rendering

A longer-term research direction is to decouple the camera-render cadence further from Cell simulation.

Potential architecture:

```text
Cell simulation
    60 Hz
      |
      v
RSX workload N
      |
      +--> render stereo at HMD pose t0
      |
      +--> retain/replay relevant rendering
               |
               +--> render at newer pose t1
               +--> render at newer pose t2

OpenXR/HMD @ 90/120 Hz
```

If safe, this could provide:

```text
world simulation:       60 Hz
world animation:        60 Hz
geometry head response: 90/120 Hz
HMD refresh:            90/120 Hz
```

This is **not yet proven** and should not be a dependency of the initial project.

Difficulties include:

- dynamic vertex/index buffers;
- render-to-texture dependencies;
- queries;
- temporal effects;
- render-target lifetime;
- state mutation between draws/frames;
- synchronization;
- game-generated camera-dependent effects.

However, the initial renderer architecture should avoid unnecessarily making later RSX workload replay impossible.

---

## 13. OpenXR integration

OpenXR should come **after** camera manipulation is proven.

The eventual OpenXR layer should provide:

- predicted display timing;
- predicted per-eye poses;
- eye orientation and position;
- swapchains/layer submission;
- headset runtime independence where practical.

Do not manually invent IPD as the final mechanism when OpenXR already supplies per-eye view poses.

The intended path is roughly:

```text
OpenXR predicted XrView poses
             |
             v
coordinate/world-scale conversion
             |
             v
game camera + HMD pose + eye pose
             |
             v
RSX/host camera injection
             |
             v
left/right render targets
             |
             v
OpenXR swapchain/layer submission
```

Hindsight's runtime/core separation should be studied for reusable design ideas and possibly reusable code, subject to compatibility, licensing and platform requirements.

---

## 14. Development phases and gates

### Phase 0 — Source investigation

Before changing architecture, trace the current RPCS3 implementation and document exact current files/functions for:

1. PS3 native stereoscopic-3D configuration/state.
2. RSX native stereo rendering/framebuffer handling.
3. Vulkan SBS/3D presentation.
4. RSX vertex-program translation.
5. Transform-constant state and upload.
6. Draw/shader identity/hash generation.
7. Render-target identity/lifetime.

Also inspect Hindsight's equivalent mechanisms.

**Deliverable:** source-level architecture report and minimal instrumentation patch plan.

### Phase 1 — RSX Stereo Inspector

Instrument WipEout's native stereo mode.

Capture/diff:

- draw sequence;
- vertex shader/program hash;
- transform constants;
- render targets;
- eye/stereo state;
- relevant pipeline classification information.

**Deliverable:** candidate WipEout camera transform(s).

### Phase 2 — Manual 6DoF camera injection

Add developer controls for:

```text
X Y Z
Yaw Pitch Roll
```

Test in a representative world scene.

**Pass condition:** translation creates correct geometry parallax and newly exposed geometry.

**Fail condition:** only image-space shifting/warping occurs, or no sufficiently upstream transform can be controlled.

### Phase 3 — Emulator-generated stereo at normal game rate

Disable WipEout's native PS3 stereo mode and return to normal ~60 fps rendering.

Generate left/right viewpoints inside RPCS3 from each normal guest frame.

**Pass condition:** geometry-correct stereo comparable to native WipEout stereo while retaining normal game cadence.

This is the project's major feasibility milestone.

### Phase 4 — OpenXR

Replace manual camera offsets with OpenXR predicted per-eye poses.

Start with seated-scale 6DoF.

### Phase 5 — Classification/profile system

Move WipEout-specific discoveries into deterministic profile data where practical:

```text
title/version
shader hashes
camera constant locations
world scale
pass classifications
HUD policy
special render-target rules
known limitations
```

### Phase 6 — Generalize to non-native-stereo games

Use Hindsight-style camera hunting to discover transforms for titles without PS3 stereo support.

### Phase 7 — Optimization

Investigate:

- Vulkan multiview;
- reduced duplicate submission;
- late pose updates;
- RSX workload replay;
- higher-frequency head-camera rendering.

---

## 15. Immediate source-investigation brief

A coding agent joining the project should initially receive the following task:

> We are developing native VR support for an RPCS3 fork. Do not modify code yet.
>
> Our first test game is WipEout HD Fury BCES00664 v2.51. RPCS3's existing Enable 3D Support feature works and the game successfully produces its native PS3 stereoscopic output. This is our ground-truth stereo reference.
>
> The project goal is genuine geometry-correct stereoscopic **6DoF** VR through OpenXR, not a virtual screen, alternate-eye rendering or post-process depth reconstruction.
>
> First inspect the checked-out RPCS3 source and produce an exact source-level trace of:
>
> - PS3 stereoscopic-3D configuration/state;
> - RSX stereo rendering and framebuffer handling;
> - Vulkan stereo/SBS presentation;
> - RSX vertex-program translation;
> - RSX transform constants;
> - where transform constants are uploaded/bound for each Vulkan draw;
> - how draws and vertex programs can be uniquely identified/hashes obtained;
> - how render targets can be identified for pass classification.
>
> Also inspect the checked-out HindsightVR repository (`https://gitlab.com/aknumbers/hindsightvr`). Concentrate on its Tier-2 camera hunting, Xenia/xemu adapters, constant capture and perturbation tooling, draw classification, profile system, multiview approach and OpenXR/runtime architecture. Reuse methodology where appropriate rather than blindly porting code.
>
> Do **not** implement OpenXR yet and do not make architectural changes yet.
>
> Our first implementation milestone is an **RSX Stereo Inspector** for WipEout. For every relevant draw, capture enough information to compare native stereo left and right, including vertex-program identity/hash, transform constants, render-target identity, draw index/order, and any RSX stereo/eye state available.
>
> We want to discover which vertex program and constants encode WipEout's actual camera transformation by comparing the game's known-good native left and right views.
>
> Before writing code, report:
>
> 1. exact files/classes/functions involved;
> 2. call/data path from RSX command processing to Vulkan draw;
> 3. how RPCS3 currently implements native PS3 stereoscopic output;
> 4. smallest safe instrumentation points;
> 5. relevant Hindsight techniques/code that map onto RPCS3;
> 6. a minimal patch plan;
> 7. any assumptions in this document that the current source disproves.
>
> Cite file paths and function names from the checked-out revisions. Do not assume architecture without verifying it in source.

---

## 16. Suggested repository layout

For development/reference work, keeping the repositories adjacent is convenient:

```text
F:\rpsc3\source\
    rpcs3\
    hindsightvr\
```

Suggested RPCS3 branch name:

```text
vr-prototype
```

The first commit should ideally contain only instrumentation / RSX stereo-camera inspection rather than OpenXR integration.

---

## 17. Key engineering principles

1. **Prove the camera before building XR plumbing.**
2. **Translation is the decisive test for true 6DoF.**
3. **Both eyes must represent the same guest instant.**
4. **Execute PS3 simulation once; duplicate/expand rendering on the host side.**
5. **Use WipEout native stereo as an oracle, not as the final rendering mode.**
6. **Prefer deterministic title profiles over runtime guesses.**
7. **Classify draws; not every matrix is a camera.**
8. **Expect multipass/deferred rendering to complicate stereo.**
9. **Do not let HUD and post-processing accidentally inherit world-camera behaviour.**
10. **Design for seated 6DoF first; CPU culling limits large translation.**
11. **Let OpenXR provide predicted per-eye poses.**
12. **Keep Vulkan multiview and RSX replay as later optimizations, not prototype dependencies.**
13. **Verify every RPCS3/Hindsight source assumption against the exact checked-out revision.**

---

## 18. Definition of the first major success

The project becomes substantially de-risked when the following demonstration exists:

```text
WipEout HD normal non-3D mode
        ~60 fps guest execution
                 |
                 v
       one PS3/RSX frame
                 |
                 v
       RPCS3 camera interception
              /     \
             /       \
       left view    right view
             \       /
              \     /
        geometry-correct stereo
                 |
                 v
       manual tracked-style 6DoF
```

The demonstration must show that X/Y/Z translation changes perspective and visibility correctly, not merely image position.

Once that works, integrating OpenXR is an engineering task built on a validated rendering mechanism rather than a speculative architecture.

---

## References

- RPCS3 source: https://github.com/RPCS3/rpcs3
- HindsightVR: https://gitlab.com/aknumbers/hindsightvr

These repositories evolve. File paths, APIs and implementation details in this document are starting hypotheses/context and must be checked against the revisions used by the fork.
