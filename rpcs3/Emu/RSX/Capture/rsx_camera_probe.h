#pragma once

// RSX camera probe (VR fork, Gate 4)
//
// Causal test rig for the camera seam that the Gate 3 inspector found by
// correlation. It perturbs ONLY the transient per-draw copy of the vertex
// transform constants; rsx::method_registers.transform_constants is never
// touched, so guest simulation, FIFO progression and savestates are unaffected
// and every effect is reversible by simply not setting the environment variable.
//
// Enable:  RPCS3_VR_PROBE="<key>=<value>,..."        one fixed probe, read once
//          RPCS3_VR_PROBE_FILE="<path>"              re-read at every frame end,
//                                                    so a whole probe series can
//                                                    run against ONE booted scene
//                                                    (removes scene variation as
//                                                    a confound). Empty/missing
//                                                    file => no perturbation.
//
//   base=<n>        base slot of the 4x4 world->clip matrix (c[base..base+3]);
//                   c[base+4..] is tried next. Default: the profile's camera blocks
//   cam=<n>         slot holding camera world position (w==1), used as the
//                   pivot for rotations. Default: the profile's camera position
//
//   yaw=<deg>       rotate the world about the camera pivot (world Y)
//   pitch=<deg>     ... world X
//   roll=<deg>      ... world Z
//   tx,ty,tz=<f>    move the camera by this world-space offset
//
//   eye=<f>         move the camera along its OWN right axis, derived from the
//                   matrix itself. This is the probe that replicates what the
//                   game does natively between its two eyes.
//
//   slot=<n>,comp=<0..3>,add=<f>
//                   raw single-component add. Used for positive controls (a
//                   slot we believe is camera state) and negative controls (a
//                   slot we believe is not).
//
//   stereo=<sep>    apply a clip-space stereo shear to the camera
//   conv=<c>        block:  clip.x += sep * (clip.w - conv).
//                   This is the exact per-eye transform the game applies in its
//                   native 3D mode (fitted over 641 draws, residual < 4e-6). It is
//                   a post-multiply in clip space, so it is independent of any
//                   world/object matrix folded into the block.
//
//   Draw classifier (always on for matrix probes). A draw is perturbed only if
//     1. its render target has the output aspect ratio (a camera view), and
//     2. it carries a PERSPECTIVE camera block (column 3 != (0,0,0,1)),
//        auto-selected per program: c[256..259] if that is perspective
//        (single-matrix route), else c[260..263] (two-matrix route, where
//        c[256..259] is an affine world matrix).
//   Scored against the native-3D oracle's own per-eye labels this is exact:
//   641 true positives, 0 false positives, 0 false negatives. Rule 2 alone
//   excludes the HUD (orthographic) but NOT the 512x256 cascade passes, which
//   carry a perspective c[260] block the game keeps shared between eyes; rule 1
//   is what excludes them (58 draws).
//
//   reqcam=0|1      additionally require the program to read the camera slot
//                   (default 0; superseded by the perspective test, which is
//                   strictly more precise - kept for experiments).
//
//   layout=columns  treat camera blocks as DP4 rows, clip[i] = dot(c[base+i], v)
//                   (the profile's matrix_layout "column_vectors"), for an
//                   unprofiled game
//
//   title=<id>      only act on this title id (default: any title with a profile)
//
//   render=1         enable the Gate 5 renderer interface. The renderer calls
//                    apply_render_eye() twice on two *new* host constant
//                    allocations; it never reuses this probe's in-place
//                    experiment path. eye_sign is -1 for the native left eye
//                    and +1 for the native right eye. This applies both
//                    camera policies measured in Gate 4: the per-eye clip-X
//                    shear and the c[465] camera-position offset.
//
// Convention: RSX vertex programs emit clip position as
//     o = v.x*c[base] + v.y*c[base+1] + v.z*c[base+2] + c[base+3]
// i.e. a row-vector convention where c[base..base+3] are the ROWS of the
// world->clip matrix M, and o = v * M. A camera delta D is therefore applied as
// M' = D * M. Column 0 of M's upper 3x3 is the world-space direction that maps
// to clip X, which is the camera right axis.

#include <cmath>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <array>
#include <vector>

#include "util/types.hpp"
#include "util/atomic.hpp"

namespace rsx::vr
{
	// A profile fragment_constant_overrides entry (see title_profile).
	struct fragment_constant_override
	{
		u64 program = 0; // vertex program ucode hash
		u32 constant = 0; // index into the fragment program's constants (_fetch_constant(n))
		std::array<f32, 4> value{};
	};

	// A title's VR profile, bin/vr_profiles/<TITLE_ID>.json (schema 1). Every
	// game-specific value the stereo renderer uses comes from here.
	struct title_profile
	{
		std::string title_id;
		std::string app_version;             // expected game version; a mismatch is logged
		std::string name;                    // the game's name, for the settings (optional)

		// 4-slot camera matrices, tried in order; the first perspective one is
		// the draw's camera.
		std::vector<u32> camera_blocks;
		// Per camera block: its 4 slots when the profile lists them explicitly
		// ([0, 1, 2, 7]: MGS4 draws camera-relative with the translation row in c[7]),
		// otherwise all umax (contiguous from the base).
		std::vector<std::array<u32, 4>> camera_block_slots;
		// false (row_vectors): clip = v.x*c[b] + ... + c[b+3], one slot per matrix row.
		// true (column_vectors): clip[i] = dot(c[b+i], v), the transpose (PSGL/Cg DP4).
		bool column_vectors = false;
		// column_vectors_xyw: a DP4 camera of three slots, clip x, y and w at base,
		// base+1, base+2; the shader derives z from w (NFS Most Wanted: c[212..214],
		// z parameters in c[215]). Implies column_vectors.
		bool xyw_rows = false;
		// Blocks (camera or HUD) stored in the row_vectors layout whatever matrix_layout says:
		// Bayonetta's scene is column_vectors c[8..11], its sprites row_vectors c[24..27].
		std::vector<u32> row_vector_blocks;
		// Blocks holding another view of the same camera: Bayonetta's motion-vector pass keeps the
		// previous frame's view-projection in c[36..39] beside the camera c[8..11]. They get the camera
		// block's clip-space eye transform (X = M^-1 * M_eye), so velocities stay the game's instead of
		// the head rotation, eye offset and FOV change (characters blurred in the headset).
		std::vector<u32> linked_camera_blocks;
		// A camera block must be rigid: its clip x, y and w directions mutually
		// orthogonal. Rejects unrelated data that happens to sit in a listed block.
		bool require_rigid_camera = false;
		// Camera blocks accepted without the rigid test: a program whose block folds in an
		// object's non-uniform scale (Demon's Souls' fog gate distortion layer, c[4..7]:
		// clip y and w nearly parallel). The eye transform handles non-rigid blocks.
		std::vector<u32> nonrigid_camera_blocks;
		// A camera block must also project square pixels at the output aspect
		// (|clip y| / |clip x| within 10% of it). For engines whose camera sits at a
		// varying base after a varying number of object-matrix slots (inFamous 1/2),
		// where the profile lists overlapping bases and a HUD block can pass the
		// perspective and rigid tests.
		bool require_camera_aspect = false;
		// Programs that index their constants (bone palettes) get the whole bank, so any slot of a camera
		// block is found there: bones at c[3] in Dragon's Dogma's skinned programs looked like its static
		// camera c[3]. true: in those programs a camera block counts only if the program reads its slots
		// directly (the program's constant_ids), as the profile generator samples them.
		bool camera_slots_read_directly = false;
		// Vertex program ucode hashes whose draws keep the game camera in both eyes (no head transform or
		// stereo). For invisible helper passes that break when moved (The Darkness: scaled unit boxes drawn
		// without colour). Probe gamecam= tries candidates live.
		std::vector<u64> game_camera_programs;
		f32 output_aspect_tolerance = 0.f;  // camera views share the output aspect
		// Aspect of the render targets that hold camera views, when it is not the
		// output's: MGS4 renders its scene anamorphically into 1024x768 and stretches
		// it to 16:9. 0 = the output aspect.
		f32 camera_target_aspect = 0.f;
		// A render target of this size holds a camera view.
		bool is_view_target(u32 width, u32 height, f32 output_aspect) const
		{
			const f32 aspect = camera_target_aspect > 0.f ? camera_target_aspect : output_aspect;
			return width && height && std::fabs((static_cast<f32>(width) / height) / aspect - 1.f) <= output_aspect_tolerance;
		}
		// Render targets exactly this wide (guest pixels) keep the game's camera in
		// both eyes: views that are not the player's, e.g. Blur's rear-view mirror.
		std::vector<u32> game_camera_target_widths;

		u32 camera_position_slot = umax;     // umax: the game has none
		f32 eye_baseline = 0.f;              // native eye distance, world units

		// clip.x += sep * (clip.w - conv), sep = -/+ per_eye_separation.
		struct stereo_rule
		{
			u32 output_width_divisor = 1;    // applies to targets output_width / divisor wide
			f32 per_eye_separation = 0.f;
			f32 convergence = 0.f;
		};
		stereo_rule stereo;                          // default
		std::vector<stereo_rule> stereo_by_target_width;

		// Headset eyes offset by eye_baseline in world units, taken from each camera
		// matrix's own scale, instead of the stereo rule's fixed clip-space shift. The
		// shift stands for eye_baseline only at the projection it was measured with: a
		// patch that widens the game's view (Shadow of the Colossus's Wider view) makes
		// the same shift a much wider eye distance, so the world looked tiny.
		// stereo.eye_offset: "baseline".
		bool stereo_eye_offset_from_baseline = false;

		u32 screen_space_block = umax;       // orthographic block => HUD/menu box
		// The HUD block is read in rows whatever matrix_layout says (orthographic_block_layout "row_vectors"),
		// for a HUD that stores its pixel matrix in the other layout from the scene's camera.
		bool screen_space_block_rows = false;
		bool screen_space_bare_projection = false;
		// A projection with no view rotation and only a translation along the view
		// axis (W = z + d, d != 0): geometry the game draws in its camera's space at
		// a fixed depth, e.g. Blur's 3D HUD. Screen space too.
		bool screen_space_depth_offset_projection = false;
		// A bare projection (as above) whose aspect is not the output's: God of War HD draws
		// its HUD with the scene's slots and a 4:3 projection, and its scene with a 16:9 one
		// (both bare: the view is applied to the vertices). The HUD goes into the box.
		bool screen_space_offaspect_projection = false;
		// A draw without depth test whose camera block has no translation at all (the
		// view rotation and projection only, clip w = view z) is a full-screen pass
		// that builds view rays from it, e.g. inFamous 2's final composite. Left as the
		// game drew it: the eye shear would shift the rays and smear the image edge.
		// (inFamous 2 also renders world geometry camera-relative with such a block,
		// but always with depth test.)
		bool screen_space_rotation_only_passthrough = false;
		// HUD and menus drawn with no matrix at all (positions come out of the vertex
		// shader in screen space; ICO's pause menu): a draw that uses no camera block,
		// samples only ordinary textures and targets a buffer no camera draw wrote this
		// frame goes into the HUD box through the vertex context's viewport matrix.
		bool screen_space_passthrough_hud = false;
		// Orthographic-block draws that sample a colour render target are full-screen
		// passes (post-processing), not HUD: they stay as drawn. Demon's Souls draws its
		// HUD and its post passes with the same c[0]. Off by default (WipEout's and Pure's
		// passes never read their HUD block); the generator sets it when passes read it.
		bool screen_space_hud_skips_passes = false;
		// Only draws into a display buffer take the HUD box, and with hud_skips_passes only draws that
		// sample the scene or a display buffer are passes. Gran Turismo 5 rasterises its glyphs into a
		// 2048x1080 font atlas with the HUD's matrix (boxed, all text broke) and draws text from it.
		bool screen_space_hud_display_buffers_only = false;
		// A frame whose first write to a display buffer is a HUD-box draw: clear the shown region first (Killzone HD).
		bool screen_space_clear_outside_box = false;
		// The HUD box is applied to the vertex shader's output position instead of to the orthographic
		// block, so varyings the shader derives from the projected position (Gran Turismo 5's text clip
		// masks) stay with the geometry.
		bool screen_space_hud_box_after_shader = false;
		// Vertex programs (ucode hashes) whose positions come out already projected by
		// the game's camera (ICO's flames and glows: GS-style sprites, NDC with w = 1).
		// They get the latest camera draw's eye transform, B^-1 * B_eye, after the program.
		std::vector<u64> screen_space_preprojected_programs;

		// Vertex programs (ucode hashes) whose matrix-less draws are HUD even into a
		// target that camera draws also wrote (Shadow of the Colossus draws its title,
		// menu and font glyphs straight into the scene's final image). The HUD box's other
		// checks (full-frame target, ordinary textures only) still apply.
		std::vector<u64> screen_space_hud_programs;
		// Draws never boxed, by vertex program ucode hash and the size of texture 0: full-screen overlays drawn with the
		// HUD matrix and the HUD's own shaders (Killzone HD's film grain, 40 tiles of a 128x128 noise texture).
		struct unboxed_draw { u64 program = 0; u16 width = 0; u16 height = 0; };
		std::vector<unboxed_draw> screen_space_unboxed_draws;
		// Draws skipped entirely while VR is enabled, same match as unboxed_draws; each entry has a name and an
		// on/off flag ("hidden") so a player can switch an effect back on (Killzone HD: film grain).
		std::vector<unboxed_draw> hidden_draws;
		// HUD draws resized about the game screen's centre before the HUD box (same match as unboxed_draws).
		// Killzone HD's aiming reticule filled much of the headset view: 0.25.
		struct scaled_draw { u64 program = 0; u16 width = 0; u16 height = 0; f32 scale = 1.f; };
		std::vector<scaled_draw> screen_space_scaled_draws;
		// Draws whose orthographic block maps output pixels 1:1 are screen fills, not HUD: Gran Turismo 5
		// lays its HUD out in 1920x1080 units and clears and fades the screen in 1280x720 pixels. Boxed,
		// the fills left trails around the box and broke the menu text.
		bool screen_space_output_pixel_draws_not_hud = false;
		// Camera draws through a viewport smaller than their view target go into the HUD box with the
		// game's camera (Gran Turismo 5's rear-view mirror). Needs hud_box_after_shader.
		bool screen_space_subviewport_cameras_in_box = false;
		// Vertex program ucode hashes whose camera (perspective) draws are part of the 2D screen: they go into
		// the HUD box with the game's camera, after the shader. Ratchet & Clank's pause-menu panels are 3D
		// meshes drawn with their own perspective camera.
		std::vector<u64> screen_space_boxed_camera_programs;

		// Vertex constants holding texture-coordinate offsets (a pass's filter taps), divided
		// by the resolution scale so the filter keeps its footprint in rendered pixels. Ridge
		// Racer 7's scene resolve averages three taps about a quarter of a native pixel apart;
		// unscaled, at 600% that smeared every edge of the scene over about three pixels.
		struct scaled_constants
		{
			u64 program = 0; // vertex program ucode hash
			std::vector<u16> constant_slots; // ("slots" in the file; a Qt macro in C++)
		};
		std::vector<scaled_constants> resolution_scaled_constants;
		// Fragment constants replaced in the draws of one vertex program: turns off a post
		// effect by its strength constant. Demon's Souls' depth-of-field composite blends in its
		// blurred copy by fc[0]; the CoC it scales includes a blur growing from the screen centre,
		// which in the headset's wider view covered the floor and the top of the view.
		std::vector<fragment_constant_override> fragment_constant_overrides;
		// HUD-box draws in fixed mode keep the game's depth (z scaled by w'/w): the box
		// changes w with the head pose, which reordered Shadow of the Colossus's depth-tested
		// menu layers. Off by default: it broke ICO's HUD box.
		bool screen_space_hud_keep_depth = false;
		// Frames without any camera draw (Ico's splash screens and videos) are shown as the
		// fixed screen instead of over the whole view. Off by default: games whose pause
		// freezes the 3D (Pure, WipEout) would show the paused frame as a screen, HUD twice.
		bool screen_space_frames_without_3d_as_screen = false;

		f32 reference_screen_width = 0.f;    // metres; 0 = no Fixed Screen depth scaling

		// Frame rate in VR (with the default patches). max_fps: the most the game works at
		// (0 = no maximum; ICO 30); the VR "Frame Rate" setting offers nothing above it.
		// default_fps: what "Default" runs, a rate current hardware reaches (0 = the
		// headset's refresh rate; unset = max_fps, or 60 without one). vblanks_per_frame:
		// vblanks per game frame (ICO 2), so the vblank runs at frame rate x this. Games of
		// one collection share a configuration, so each profile sets its own default.
		u32 max_fps = 0;
		u32 default_fps = umax;
		u32 vblanks_per_frame = 1;
		// While the game has a video decoder open, the emulated vblank runs at most this fast (0 = no cap).
		// Killzone HD's movie player (libsail) stops taking frames above 60 Hz: the intro stayed black.
		u32 video_vblank_rate = 0;
		// In stereo, memory copies (NV0039) into a display buffer are skipped. Killzone HD saves each finished frame to
		// main memory and copies another memory image back into the display buffer; with Read Color Buffers the left
		// eye was then reloaded from memory at 1x while the host-only right eye kept the scaled frame.
		bool keep_rendered_display_buffers = false;
		// Distance of the HUD box in metres when the HUD Depth setting is Auto (0 = the 2 m default).
		// Killzone HD: 4 (its aiming reticule sat too close at 2 m).
		f32 hud_depth = 0.f;

		// The game builds effects across frames from full-screen buffers (Ico's glow and
		// previous-frame blend): with the head moving between frames, older-pose buffers are
		// shifted to the current pose (blend targets) and re-projected when read (feedback
		// textures). Off by default: in other games it moved buffers that are not such effects
		// (Pure: flashes, a bright square under the bike, and the paused frame floating in space).
		bool reproject_older_frames = false;

		// Guest floats holding the game's idea of the display refresh rate (Pure: PSGL
		// device+0x14, game time = vblank count / it). Written every frame with the
		// effective vblank rate, so the game keeps real-time speed at any vblank rate.
		// "[0x1050300]+0x14" = the pointer at 0x1050300, plus 0x14; "0xd2f4dc" = that address.
		struct guest_address
		{
			u32 address = 0;
			bool deref = false;
			u32 offset = 0;
		};
		std::vector<guest_address> game_refresh_rate_f32;
		// Guest floats holding the game's time step for one frame (Ridge Racer 7: its
		// 1/60 constants) and u32s holding its milliseconds per frame (Ridge Racer 7's race
		// timer, read from a word its VR patch sets up): written every frame with the running
		// game frame rate (effective vblank rate / vblanks_per_frame), so a frame-locked game
		// keeps real-time speed at any VR frame rate. Same address forms as above.
		std::vector<guest_address> game_frame_time_f32;
		// The frame time squared and cubed (Ratchet & Clank keeps 1/60, 1/3600 and 1/216000 side by side).
		std::vector<guest_address> game_frame_time_sq_f32;
		std::vector<guest_address> game_frame_time_cube_f32;
		std::vector<guest_address> game_frame_ms_u32;
		// u32s holding the game frame rate itself (Ridge Racer 7's VR patch advances its
		// 60 Hz frame counters by 60/fps per frame from it).
		std::vector<guest_address> game_fps_u32;

		// The game composites the previous frame's scene (ICO: left over from SPU
		// MLAA) with effects built from the current one (bloom). Each frame carries
		// its own head rotation, so the two disagree when the head turns. Copies
		// (blits) of a scene target drawn in an earlier frame read the matching
		// target drawn in this frame instead.
		bool current_frame_copies = false;

		// Camera draws into off-aspect targets with the player's projection (output aspect) follow
		// the head like the scene, keeping their viewport: views the scene samples at its own screen
		// position (Ridge Racer 7's road reflection tiles, 128x128 in a 512x128 target). Off by
		// default: other off-aspect targets keep the game camera in both eyes.
		bool offaspect_player_views = false;

		// Depth-tested draws into the scene that no camera block covers (the matrix folded
		// with an object's so it is not rigid, in another slot or layout, or skinned from the
		// whole constant bank) take the latest camera draw's eye transform, B^-1 * B_eye,
		// after their own program: the object part of B cancels, so any draw through the same
		// camera lands where the eye sees it. Draws that sample a colour render target
		// (post-processing) and the HUD are left alone. Off by default (experimental); the
		// profile generator sets it when camera blocks cover few scene draws (Demon's Souls).
		bool clip_space_scene_draws = false;

		// Guest memory the game copies its depth buffer into for its own occlusion culling
		// on the CPU/SPUs (Shadow of the Colossus, every frame). In stereo that depth is the
		// eye's, not the game camera's, so the culling hides visible objects (flashing holes,
		// popping), and the read waited for almost the whole stereo scene. Reads there get
		// far depth everywhere at once: nothing is culled by occlusion. "0xADDR:0xSIZE" each.
		std::vector<std::pair<u32, u32>> occlusion_depth_readback;

		// The stereo rule for a render target this wide.
		const stereo_rule& stereo_for(u32 target_width, u32 output_width) const;
	};

	// Load and validate bin/vr_profiles/<title_id>.json. Null (with the reason
	// logged) when the file is missing or invalid.
	// The running executable's file name, lower case, without extension ("shadow" for shadow.self).
	std::string running_executable_name();

	// vr_profiles/<TITLE_ID>.<executable>.json if present (one game of a collection), else <TITLE_ID>.json.
	std::shared_ptr<const title_profile> load_title_profile(std::string_view title_id, std::string_view executable = {});

	// map_vr_preprojected's program for a clip_space_scene_draws draw (no listed program).
	constexpr u64 scene_draw_program = umax;

	// True if this title id has a valid VR profile. Only profiled titles can be
	// rendered in stereo, so the VR options are offered for those alone.
	bool title_has_profile(std::string_view title_id);

	// The headset's display refresh rate (Hz, rounded) while an OpenXR session
	// runs; 0 clears it.
	void set_headset_refresh_rate(u32 hz);

	// A headset session is running (the VR frame rate applies).
	void set_headset_active(bool active);

	// The VR "Frame Rate" option at this index (vr_frame_rate): its frame rate, 0 for
	// Unlimited, umax for Default.
	u32 frame_rate_option_fps(u32 option);
	// Whether a game with this max_fps lists the option (Default always).
	bool frame_rate_option_allowed(u32 option, u32 max_fps);
	// The highest max_fps among the title's profiles (0 = no maximum), for the settings
	// dialog, which cannot tell which game of a collection will run.
	u32 title_max_fps(std::string_view title_id);
	// The distinct default_fps of the title's profiles, ascending (0 = the headset's rate).
	std::vector<u32> title_default_fps(std::string_view title_id);
	// Each game of the title (its profile name, else its executable) with its default and
	// maximum frame rates, for the settings dialog.
	struct title_game_frame_rate
	{
		std::string name;
		u32 default_fps = 0;
		u32 max_fps = 0;
	};
	std::vector<title_game_frame_rate> title_frame_rates(std::string_view title_id);
	// The Frame Rate option for this rate (0 = Unlimited), umax if none.
	u32 frame_rate_option_for_fps(u32 fps);

	// The running game's frame rate in VR (0 = the headset's refresh rate).
	u32 effective_frame_rate();

	// The vblank rate to emulate: while a headset runs, the VR frame rate times the
	// profile's vblanks_per_frame (Unlimited: the headset's refresh rate); otherwise the
	// configured Vblank Rate (which is never modified).
	u64 effective_vblank_rate();

	// Writes the effective vblank rate to the profile's game_refresh_rate_f32 targets.
	// Called once per frame by the RSX thread.
	void update_game_refresh_rate();

	// Reprojection Margin in degrees: the configured value, or for "Auto" (-1) 10 degrees
	// when the game runs below the headset's refresh rate and 0 when it does not.
	u32 effective_reprojection_margin();
	// HUD box distance in metres: the HUD Depth setting, or with Auto the VR profile's hud_depth, else 2 m.
	f32 effective_hud_depth();

	// True while stereo is rendered and [start, end] overlaps the profile's occlusion_depth_readback.
	bool occlusion_depth_readback(u32 start, u32 end);

	// While alive, full-bank constant uploads (programs with indexed constants) expose only the slots
	// in `ids` (the program's directly read constants) to camera block matching on this thread. Null or
	// empty ids: no restriction. VKGSRender sets it per draw for profiles with camera_slots_read_directly.
	struct full_bank_direct_slots
	{
		explicit full_bank_direct_slots(const std::vector<u16>* ids);
		~full_bank_direct_slots();
		full_bank_direct_slots(const full_bank_direct_slots&) = delete;
		full_bank_direct_slots& operator=(const full_bank_direct_slots&) = delete;
	};

	class camera_probe
	{
	public:
		static camera_probe& get();

		// Hot-path gate. False unless a perturbation is currently configured.
		bool enabled() const { return m_active.load(); }
		bool render_enabled() const;
		// Probe "hide=<hash>[@<target>][+...]": vertex programs (ucode hashes) whose draws are skipped, optionally
		// only into one colour target (hex address), to find which program draws an artefact. Empty unless set.
		const std::vector<std::pair<u64, u32>>& hidden_programs() const { return m_hidden_programs; }
		// Probe gamecam=<hash>[+<hash>...]: these vertex programs keep the game camera in the right eye (development).
		const std::vector<u64>& game_camera_programs() const { return m_game_camera_programs; }
		// gamecam=<hash>@nocolor: only that program's draws with every colour write off (depth/stencil passes).
		const std::vector<u64>& game_camera_nocolor_programs() const { return m_game_camera_nocolor_programs; }
		// Probe "dev=<bits>": renderer switches for live A/B measurements (see their users). 0 unless set.
		u32 dev_flags() const { return m_dev_flags; }
		// Probe "unboxfp=<id>[+<id>...]": fragment program session ids (inspector fp_session_id) left out of the HUD box.
		bool unboxed_fragment_program(u32 id) const { return std::find(m_unbox_fp.begin(), m_unbox_fp.end(), id) != m_unbox_fp.end(); }
		// The draw about to be bound samples a colour render target (post-processing).
		void set_draw_samples_colour_target(bool v) const { m_draw_samples_colour_target = v; }
		void set_draw_into_display_buffer(bool v) const { m_draw_into_display_buffer = v; }
		// The vertex program ucode hash of the draw about to be bound (only set while the profile needs it).
		void set_draw_program(u64 v) const { m_draw_program = v; }
		// hud_box_after_shader: the last bound draw is a HUD draw whose box the renderer applies.
		bool hud_env_requested() const { return m_hud_env_request; }
		// subviewport_cameras_in_box: a scissored clear through a sub-viewport of a view target (the
		// rear-view mirror's) moves into the box with the draws. rect: host pixels x1, y1, x2, y2, in
		// the left eye; right_rect receives the right eye's (shifted by the HUD parallax).
		bool map_subviewport_clear(f32 host_scale, u32 surface_w, u32 surface_h, f32 host_width, f32 host_height, f32 rect[4], f32 right_rect[4]) const;
		void clear_hud_env_request() const { m_hud_env_request = false; }
		// The draw about to be bound has depth test enabled.
		void set_draw_depth_test(bool v) const { m_draw_depth_test = v; }
		// Profile screen_space.scaled_draws: this draw's size factor in the HUD box (1 = as drawn).
		void set_draw_hud_scale(f32 v) const { m_draw_hud_scale = v; }
		// HUD box scissor: map_vr_screen_box records its transform; the renderer maps the game's
		// scissor (host pixels, window y down) through it so HUD clipping lands in the box, and
		// anything the game parked outside its screen stays clipped. Call clear before each eye.
		void clear_box_mapped() const { m_box_mapped = false; }
		// The draw just bound (since clear_box_mapped) was mapped into the HUD box through its constants.
		bool box_mapped() const { return m_box_mapped; }
		bool map_box_scissor(f32 host_scale_x, f32 host_scale_y, f32 host_width, f32 host_height, f32 rect[4]) const;
		// The profile's clip_space_scene_draws, unless the probe file overrides it (scene=0/1).
		bool scene_draws_by_clip_space() const;

		// Load the title's profile again on next use (a generated one was just
		// written). Only safe while no profile is loaded: the old one is freed.
		void reload_profile();

		// Frame boundary: re-read RPCS3_VR_PROBE_FILE if it changed.
		void poll();

		// Perturb a freshly filled transient constant buffer in place.
		// reloc_table_data/size describe the original guest index of each 16-byte
		// slot in the buffer; an empty table means the buffer holds the full
		// 468-slot bank at its natural indices.
		// guest_constants points at the pristine 512x u32[4] guest bank.
		// surface_w/h: the draw's render-target size, used by the draw classifier.
		void apply(void* buffer, const u16* reloc_table_data, usz reloc_table_size,
			const void* guest_constants, u16 surface_w, u16 surface_h) const;

		// Apply the title profile's native-eye transform to a freshly cloned
		// constant buffer. Returns true only when this is a perspective,
		// output-aspect world draw and its camera matrix was sheared. The camera
		// position slot is adjusted independently whenever the draw carries both
		// it and a usable perspective camera block (including the shared shadow route).
		bool apply_render_eye(void* buffer, const u16* reloc_table_data, usz reloc_table_size,
			u16 surface_w, u16 surface_h, f32 eye_sign) const;

		// Human-readable description of the active probe, for logging.
		const std::string& description() const { return m_description; }

		// The running title's VR profile, loaded on first use for each title.
		// Null when the title has none.
		const title_profile* profile() const;
		const title_profile* profile_slow() const;

		// Gate 6 headset view. quat_xyzw is the OpenXR head orientation (LOCAL
		// space) the next frame is rendered with. While set, output-aspect camera
		// draws are rotated by it and the stereo becomes parallel (translation
		// term only). eye_scale multiplies the game's eye separation, fov_scale
		// widens the rendered field of view. flip_y if NDC +Y is screen-down.
		// position_xyz is the head position in LOCAL space (metres); camera_depth is a
		// constant forward offset of the viewpoint (metres, + is forward). Both are
		// converted to game units using the game's own eye separation against ipd.
		void set_vr_view(const f32 quat_xyzw[4], const f32 position_xyz[3], f32 eye_scale,
			f32 fov_scale, bool flip_y, f32 ipd, f32 camera_depth);
		void clear_vr_view();

		// Native (non-headset-view) stereo: scale the game's eye separation, for
		// showing its stereo on a screen larger than the one it was tuned for.
		void set_screen_stereo_scale(f32 scale);

		// Render each eye into exactly this frustum instead of the game's own:
		// tangents (left, right, up, down) per eye, left/down negative, as
		// reported by the headset. Pass nullptr to keep the game's FOV.
		// Screen-space draws (HUD, menus) become a fixed output-aspect box fitted
		// inside the headset view, times hud_scale: head-locked, or with hud_fixed
		// anchored straight ahead in LOCAL space so the head can turn away from it.
		// hud_offset_x/y move the box centre by that fraction of the central view's
		// half-width/half-height (+ is right/up). hud_depth (metres, 0 = infinity)
		// places it at that stereo distance for eyes ipd metres apart.
		// tangents are the rendered eye frustums (the visible ones plus the
		// reprojection margin); the HUD box is sized from the visible ones.
		void set_vr_eye_fov(const f32 (*tangents)[4], const f32 (*visible)[4], f32 hud_scale, bool hud_fixed,
			f32 hud_offset_x, f32 hud_offset_y, f32 hud_depth, f32 ipd);

		// tan of the rendered half-angles, measured from the camera draws
		// (including fov_scale). False until a rigid camera block has been seen.
		// Also the readiness test for the headset-FOV remap.
		bool get_vr_fov(f32& tan_half_x, f32& tan_half_y) const;
		bool vr_hud_fixed() const { return m_vr_hud_fixed; }

	private:
		camera_probe();
		void parse(const std::string& cfg);
		void reset_params();
		void apply_vr_rotation(f32* const rows[4], const std::array<f32, 9>& R, const std::array<f32, 3>& head) const;
		void apply_vr_screen_space(const title_profile& profile, void* buffer, const u16* reloc_table_data, usz reloc_table_size,
			u16 surface_w, u16 surface_h, f32 eye_sign) const;
		void map_vr_screen_box(f32* const rows[4], f32 eye_sign, f32 aspect) const;

		bool m_enabled = false;          // subsystem on (default render path or probe config)
		atomic_t<bool> m_active{false};  // a perturbation is configured right now

		std::string m_config_path;
		u64 m_config_stamp = 0;
		// Live reload: the running title's profile files (time and size), checked twice a second.
		u64 m_profile_file_stamp = 0;
		u64 m_profile_file_check_ms = 0;

		std::string m_description;
		std::string m_title; // probe override (title=); empty = the running title

		// Cached profile of the running title (see profile()).
		mutable std::mutex m_profile_mutex;
		mutable std::string m_profile_title;
		mutable std::string m_profile_title_id; // what the cached key was built from (per-draw fast path)
		mutable std::string m_profile_boot;
		// Per-frame fast path: profile() revalidates at most once per frame (poll() clears the flag);
		// it was called several times per draw, each taking the mutex and comparing strings.
		mutable atomic_t<bool> m_profile_fast_valid{false};
		mutable atomic_t<const title_profile*> m_profile_fast{nullptr};
		mutable std::shared_ptr<const title_profile> m_profile;

		// Probe overrides (base=, cam=); umax = the profile's.
		u32 m_base = umax;
		u32 m_cam_slot = umax;

		f32 m_yaw = 0.f, m_pitch = 0.f, m_roll = 0.f;
		f32 m_tx = 0.f, m_ty = 0.f, m_tz = 0.f;
		f32 m_eye = 0.f;
		f32 m_stereo_sep = 0.f;
		f32 m_stereo_conv = 0.f;
		bool m_have_stereo = false;
		bool m_render_enabled = false;
		std::vector<std::pair<u64, u32>> m_hidden_programs;
		std::vector<u64> m_game_camera_programs;
		std::vector<u64> m_game_camera_nocolor_programs;
		u32 m_dev_flags = 0;
		std::vector<u32> m_unbox_fp;
		s32 m_scene_override = -1;           // probe file scene=0/1; -1 = the profile's
		mutable bool m_draw_samples_colour_target = false;
		mutable bool m_draw_into_display_buffer = true;
		mutable u64 m_draw_program = 0;
		mutable bool m_hud_env_request = false;
		mutable bool m_draw_depth_test = true;
		mutable f32 m_draw_hud_scale = 1.f;
		mutable bool m_box_mapped = false;
		mutable bool m_box_identity_pass = false;
		mutable f32 m_box_map[4][4] = {};

		// c[465] is global camera state. Shadow cascades carry a perspective
		// c[260] block whose clip-X axis is *not* camera right, so retain the
		// most recent output-aspect camera-right axis for their c[465] policy.
		// RSX draw processing is serialized on the renderer thread.
		mutable std::array<f32, 3> m_render_camera_right{};
		mutable bool m_render_camera_right_valid = false;

		// Gate 6 headset view (see set_vr_view). m_vr_rot is the head rotation
		// transposed, expressed in the draw's clip basis (NDC x, NDC y, clip w).
		bool m_vr_view = false;
		std::array<f32, 9> m_vr_rot{};
		f32 m_vr_eye_scale = 1.f;
		f32 m_vr_fov_scale = 1.f;
		bool m_vr_flip_y = false;
		bool m_vr_hmd_fov = false;
		f32 m_vr_hud_scale = 1.f;
		bool m_vr_hud_fixed = false;
		f32 m_screen_stereo_scale = 1.f;
		f32 m_vr_hud_offset_x = 0.f;
		f32 m_vr_hud_offset_y = 0.f;
		// Head translation for this frame, in the clip basis (right, up or down, forward):
		// m_vr_head_units in game units for camera draws, m_vr_head_m in metres for the
		// fixed screen-space box, which lives at a known distance.
		std::array<f32, 3> m_vr_head_units{};
		std::array<f32, 3> m_vr_head_m{};
		f32 m_vr_hud_depth = 0.f;   // metres
		f32 m_vr_hud_parallax = 0.f; // ipd / (2 * depth): per-eye view-space x shift at unit forward distance
		f32 m_vr_eye_fov[2][4]{};         // rendered
		f32 m_vr_eye_fov_visible[2][4]{}; // shown by the headset
		// Projection x/y scales relative to w, from the latest rigid camera block.
		mutable f32 m_vr_proj_x = 0.f;
		mutable f32 m_vr_proj_y = 0.f;
		mutable bool m_vr_proj_valid = false;
		// The latest output-aspect camera block per eye, as the game wrote it (B) and as
		// drawn for that eye (B_eye), for pre-projected draws (see map_vr_preprojected).
		mutable f32 m_vr_last_block[2][4][4]{};
		mutable f32 m_vr_last_eye_block[2][4][4]{};
		mutable bool m_vr_last_block_valid[2]{};
		void store_eye_block(f32 eye_sign, const f32 (&game)[4][4], f32* const rows[4]) const;
		void apply_linked_camera_blocks(const title_profile& profile, void* buffer, const u16* reloc, usz reloc_size,
			const f32 (&game)[4][4], f32* const rows[4]) const;

		// Rotation-invariance audit, RPCS3_VR_AUDIT=<degrees> (desktop only, no
		// headset): both eyes share one eye position and the right eye is yawed by
		// that angle. Under a pure rotation no world point changes colour, only
		// position, by a known homography, so warping the left eye by it must
		// reproduce the right; whatever does not was derived from camera
		// orientation the renderer leaves untransformed. tools/rotation_audit.py.
		f32 m_audit_yaw_deg = 0.f;
		std::array<f32, 9> m_audit_rot{};
		mutable bool m_audit_logged = false;
		// RPCS3_VR_AUDIT_FOV=<tan>: also remap both audit eyes onto a symmetric
		// frustum of that half-angle tangent, exercising the headset FOV remap.
		f32 m_audit_fov_tan = 0.f;

		// keep_viewport: the draw keeps its viewport (an off-aspect target sampled at screen position).
		void remap_to_eye_fov(f32* const rows[4], const f32* tangents, f32 A, f32 B, bool keep_viewport = false) const;
		// Undo the draw's viewport scale/offset where they differ from the render
		// target's (ICO sets a 1360x768 viewport on a 1216x688 target), so clip space
		// lands on the target exactly as the headset frustum mapping assumes.
		void undo_viewport(f32* const rows[4], bool displayed_region = false, bool inverse = false) const;

	public:
		// The HUD-box mapping as a clip-space matrix (row-vector: clip' = clip * m) for a
		// draw of the profile's passthrough HUD. False unless the headset view is active.
		bool map_vr_passthrough_hud(f32 m[4][4], f32 eye_sign, f32 aspect) const;
		// For a draw of a profile pre-projected program: the clip-space matrix
		// (row-vector) taking the game's clip space to this eye's, B^-1 * B_eye of the
		// latest camera draw. False when the program is not listed or no camera draw
		// was transformed yet.
		bool map_vr_preprojected(f32 m[4][4], f32 eye_sign, u64 program_hash) const;
	private:

		bool m_have_xform = false;
		bool m_require_cam = false;
		bool m_column_vectors = false;  // layout=columns (probe only; profiles set their own)

		// One-frame classifier report, logged on the first full frame after arming.
		mutable atomic_t<u32> m_stat_perturbed{0};
		mutable atomic_t<u32> m_stat_rejected_aspect{0};
		mutable atomic_t<u32> m_stat_rejected_no_perspective{0};
		bool m_report_pending = false;

		// Raw single-component probe
		bool m_have_raw = false;
		u32 m_raw_slot = 0;
		u32 m_raw_comp = 0;
		f32 m_raw_add = 0.f;
	};
}
