# Accuracy ledger

A ledger, not a journal: rows are edited in place. Each row is one place where the Vulkan
platform's default trades exactness against Metal for compile time (pipeline making) or GPU
time. The fast form is the default. The accurate form stays behind its opt-in switch, and both
are kept always. An accurate form becomes the default only where it costs nothing measurable in
compile or GPU time. An accuracy difference is logged (`metal2vulkan-degrade-<code>`, first line,
then counted) and the process runs on. The process ends only where Metal itself ends it, or where
an operation cannot run at all. `REIMS_VGPU_STRICT=1` ends it at each difference instead, for
tests.

Switches: `REIMS_VGPU_EXACT=<field>[,<field>...]` sets the runtime's `DevicePolicy` field to its
exact form at device creation. The runtime's own ledger, with every bound in full, is
metal2vulkan's `docs/FAST-PATHS.md` at the pinned commit. Costs are the RTX 5080 Laptop GPU's,
cold, from that file: the making of the big glass pipelines (A2X, Xgc14, Xgc15), then GPU time
per draw.

Two runtime defaults are not taken, and have no switch: completion and heap compression. Every
device takes `CompletionPolicy::Exact`, each command buffer in a kernel submission of its own and
seen complete at its own end, as Metal's, where the runtime's default (`Grouped`) on Mesa's drivers
sends up to four ready together and sees each but the last complete only once the last completed.
When a command buffer is seen complete is what the guest observes of its work (the stamps and
interrupts core writes as it completes), and this platform does not diverge from Metal there, in
timing as in order: the rows below trade values alone. The grouped form also bought no frame rate in
a guest's drag on the Iris Xe (15.7-17.3 frames a second grouped, 16.4-17.6 exact), for 0.4-1.7 ms
of the runtime's scheduler CPU a frame.

Every device also takes `HeapCompressionPolicy::Uncompressed`: a heap's colour texture that Metal
keeps losslessly compressed (`allowGPUOptimizedContents`, no `ShaderWrite`, `PixelFormatView` or
`ShaderAtomic` usage, 16 texels or more a side) takes an image the driver keeps uncompressed in the
heap's bytes, as one Metal keeps uncompressed does, where the runtime's default (`Compressed`) takes
the driver's compressed image on a driver whose compressed images may keep their texels apart from
their memory (ANV). That image changes what `heapTextureSizeAndAlignWithDescriptor:` answers the
guest: on the Iris Xe 0x4100 bytes at 0x10000 for a 64x64 `BGRA8Unorm` texture, and 0x8100 at
0x10000 for a 64x64 `R32Uint` one of seven levels, where the uncompressed images answer 0x4080 and
0x8000 at 0x1000, as before this default came (Metal: 0x4080 and 0x5800 at 0x80). An answer stays as
it was until the guests boot with the new one. A write through a buffer or texture over its bytes
may also leave the compressed image's texels as they were (`HeapTextureCompressed`). The compressed
image cleared a 2048x2048 `RGBA8Unorm` render target in a `Private` heap in 26.9 us against 676 us
on the Iris Xe, and copied into another in 648-649 us against 674-693 us.

| Fast default | Accurate switch | Bound of the fast form | Cost of the accurate form (5080) |
|---|---|---|---|
| The device's sampler (filter, footprint, level, anisotropy, sRGB, comparisons) | `sampling` | linear filter 1 code, sRGB 1.83; the rest unmeasured | Xgc15 making about 2x |
| The device's filter of a two-plane YCbCr texture, under the device's sampler (on MoltenVK, Metal's own sampler over the texture's converted texels) | `ycbcr_filter` (Metal's filter of the format, in the slot's YCbCr variant of the pipeline) | 1.18 codes of Metal's sum on the RTX 3090, 0.88 on RADV, 1.06 on ANV; on MoltenVK one step of the format's grid (0.1245 codes on 1/2048, 0.016 on 1/8192), a `half` result 0.1245 codes, and a `half` sample of one texel of an `_sRGB` twin through a runtime sampler rounds a tie to even (one half ulp) | a pipeline of its own, made at the first draw that binds such a texture: ANV, A2Xghfc's 37.8 s (ms01) |
| The device's conversion of a texel's codes | `texel_conversion` (a read alone; a sample or gather with `sampling`) | 1 f32 ulp (8/16-bit), 0.18 code (5/6-bit packed) | reads: about +0.03 s making, GPU +4-18% |
| Device maths library, division, fma | `float_maths` | Vulkan's precision table (e.g. Atan 4096 ulp) against Metal's 4-6 | A2X 1.0 → 4.6 s, GPU 2x |
| Device NaN bits; device min/max/clamp | `nan_and_min_max` | NaN bits, ±0 and denormal picks; on ANV and RADV a `half` read from its bits widens by the device's conversion, every value but a NaN exact, a NaN's bits the device's (ANV quiets a signalling one) | A2X 1.0 → 1.9 s, GPU +18%; its full texel rule (gathers included, but not a cube gather) adds A2X 1.85 → 2.0 s, and a first draw with a linear runtime sampler bound remakes its pipeline (about 2 s), GPU +8% |
| Device `dot`, `mix`, `fast::sqrt` | `rounding_order` | dot (3n−1)·2^-24·Σ, mix 2^-21, sqrt 4.5 ulp | about +0.05 s making |
| Device half rounding, widening and folding; a half texel's NaN | `half_rounding`; the texel NaN also under `nan_and_min_max` (no cost over that switch) | AMD and Intel (RADV, ANV) round a computed half through an f32→f16→f32 conversion pair under Vulkan's 16-bit float controls (round to nearest even, denormals and signed zeros kept), which the spec makes exact, under either form and announcing nothing (0 mismatches outside NaN inputs over every half and every half +, −, × pair on Iris Xe, the 780M and the 7900 XTX). The RTX 5080 rounds exactly (8192 of 8192 narrowed sums are Metal's). The RTX 3090, on the same 615.71.09 driver, and lavapipe drop the rounding: a half result keeps float precision, its differences compounding across operations. A value past half's range stays finite until a half store saturates it to infinity. Elsewhere 1 half ulp. NaN bits; half denormals may flush | A2X 1.0 → 7.2 s, GPU +40% |
| The device's blend of a safe-math half output into a blended 16-bit float attachment | `half_blend` (a `half4` output into R16/RG16/RGBA16Float, safe or relaxed math; elsewhere the device blend stays) | at the largest operand: 16-bit float 2.5 half ulps on NVIDIA (blend-colour factors), 2 on AMD, Intel and lavapipe; 32-bit float 1 ulp on NVIDIA, 2 elsewhere | making 11.1 → 24.0 ms; GPU 25 → 82 µs for one 1080p draw, 0.51 → 2.29 ms for 32 overlapping draws |
| A decoded twin of compact codes for a texture of a format the device lacks, which the runtime holds in a carrier: an ASTC LDR texture's (not an sRGB one's) of 8-bit codes, an EAC 11-bit one's of 16-bit normalized codes (Apple's GPUs have the formats: no twin on MoltenVK) | `decoded_texels` (Metal's values in a twin of floats) | an ASTC channel within 1/510 of Metal's 12-bit value, an EAC 11-bit one within half a 16-bit step (`DecodedTexelPrecision`) | the twin's memory: 16 bytes a texel against 4 (a 2048x2048 chain 85 MiB against 21 MiB), an EAC twin 4 bytes a channel against 2; Iris Xe, the CPU decode of that chain 79 ms against 68 ms; sampling not measured |
| The device's load and store of a single-sampled signed normalized (`R8Snorm` to `RGBA16Snorm`) or `RGB9E5Float` colour attachment that a drawing pass loads: the texels its draws do not write keep their bytes (on MoltenVK the pass is Metal's: no difference) | `tile_store` (the runtime's load pass rewrites them as Metal's tile store leaves them, in the tiles the draws reach) | one code, at a minimum code alone (-128 or -32768 kept where Metal stores -127 or -32767, both -1.0), and a shared-exponent word kept unnormalized (the same value); only uploaded or copied bytes hold either (`SnormMinimumKept`, `SharedExponentKept`) | the load pass before each such pass, the render area copied into a scratch buffer and back with a dispatch: about 1.3 ms a 4-byte attachment at 1920x1080 on the Radeon 780M (RADV), as timed for the extended-range attachments, which take it under either form |
| The device's `AVERAGE` resolve of a multisampled colour attachment whose format has Metal's resolve rule (on MoltenVK Metal's own resolve: no difference; an attachment a fetching pass holds in sample planes takes Metal's rule under either form) | `resolve` (Metal's rule, in a render pass instance after the pass for each resolved attachment) | at 2 and 4 samples a unorm, snorm or packed 16-bit channel within one code of Metal's and an sRGB one within two (NVIDIA at 4 samples; one elsewhere) on RADV, ANV, NVIDIA and lavapipe, each rounding an 8-bit tie down; a float channel in its last bits, unbounded; wholly at a NaN or infinite sample (`DeviceRoundedResolve`) | a 1080p frame of 97 command buffers, every 4th pass resolving an `RGBA8Unorm` x4 attachment, GPU a frame: Iris Xe (ANV) 2.41-2.52 ms against 1.92 ms, Radeon 780M 2.33-2.78 against 2.00-2.13 ms, RX 7900 XTX 1.38 against 1.36 ms |
| Batches of one queue ordered by their ends on a driver measured to start no kernel submission of a queue before the one before it completed (ANV, RADV, NVIDIA): each ends with a barrier before every later command of its queue, so a dependent batch begins with no barrier of its own | `batch_order` (a barrier for each access a batch's work depends on in an earlier batch's) | none: on those drivers each batch starts once the one before it completed under either form, so order, results and GPU times are the same. Not announced | UHD 770 (ANV), 67 command buffers a frame over 520 host imports: 67 command buffers submitted a frame against 129, a frame through completion 2.13-2.22 ms against 2.27-2.38 ms |
| An extended-range blend source rounded as a loaded colour | none yet: parked (the 5080 effort first; no 5080 boot reaches it) | 1 code | not measured |
| Bicubic near-tie rounding | none yet: parked (the 5080 effort first; no 5080 boot reaches it) | 1 grid step (1/4080) within 2^-21 of a tie | not measured |

All exact at once (`all`: every switch but `sampling`): A2X 1.0 → 17.5 s, Xgc14 0.9 → 5.0 s,
Xgc15 0.65 → 3.1 s, GPU about 2-2.7x, and the costs of the forms that change no pipeline, in their
rows.

Differences no switch can fix (device capabilities, undefined values) are not speed choices. They
are listed in the runtime's `docs/FAST-PATHS.md`, "Not speed choices", and are logged the same way.

## Logged gaps on our side (no accurate form yet; not speed choices)

Each is logged once per kind and counted, and the work runs. `REIMS_VGPU_STRICT=1` ends the
process there instead.

| What runs instead of Metal's answer | Where |
|---|---|
| A line width other than 1 draws 1 pixel wide | `reims-vgpu-vulkan` draw encoding |
| A `half` colour output stored past the bound its rounding was measured within, as blended or converted at its attachment, stores the runtime's `half` rounding | draw encoding |
| A draw in a pass that loads and stores a bfloat colour attachment leaves the runtime's tiles of it: codes moved through `half` where no draw reaches, kept where one does | draw encoding |
| A multisampled 32-bit float attachment cleared to a NaN other than the plain quiet one, in a pass whose only draws record nothing, resolves keeping the NaN's bits, where Metal's resolve stores `0x7fc00000` | render passes |
| Under `textureWriteRoundingMode` 2, a half-float attachment of a sample count no one measured at the mode resolves by the runtime's rules | render states |
| A heap texture of unmeasured size or layout takes the runtime's size and layout | heaps |
| A Metal-level answer for an unmeasured host release, or a format's description off macOS 26.7, comes from that release's description | device answers |
| A draw or tile dispatch in a varying-rate pass runs on the runtime's own regions | rate maps |
| A retyped YCbCr view reads as the texture itself | YCbCr views |
| A blit fill packed in another format repeats the packed texel | blits |
| A late resolve texture beside a dropped resolve leaves the colour resolve dropped | render passes |
| A texture of protection options from `newTextureWithDescriptor:` or `newSharedTextureWithDescriptor:` is made unprotected, as Metal's other creators make it, where Metal answers nil without the protected-content entitlement; its contents are not kept from the host | device textures |
| The `uniqueIdentifier` of a pipeline state made with indirect command buffer support, which the guest reads, is the runtime's offset in its own allocation of such states, not Metal's address (`unmeasured-unique-identifier`); one value for the state's life, unique among live states, the last released taken again first by kind, as Metal's | pipeline states |
| A function that reads a sampler state through its ID in an argument buffer reads the neutral sampler where the device's sampler tables do not map the state for its command: more sampler states live than the command's set-1 sampler arrays hold, an ID the tables do not reach, a sampler the device cannot make, or a cubic state beside a fragment stage (`unlisted-argument-sampler`) | argument samplers |
| A heap's two-plane XR10 texture of more than one level, or a 1D array, keeps its first level's texels in the heap's bytes as Metal's CPU copies and GPU writes lay them and its later levels apart from them, a 1D array's GPU reads reading each texel, where Metal's paths address those texels at bytes that differ by path (`heap-two-plane-layout`) | heaps |
| A draw under the last provoking vertex that the runtime draws otherwise than Metal: a line from its last vertex to its first (`provoking-vertex-lines`); flat inputs, a computed primitive ID or a selected layer or viewport from another vertex, barycentric weights beside flat inputs Metal's but for rounding, and a mode from 2 up, its top bit set or not, drawn as mode 1 (`provoking-vertex-unmatched`) | the runtime's provoking vertex |
| A command predication record of the runtime's whose condition word lies past its buffer, the 256 bytes Metal rounds a lone buffer up to, or its heap reads zero, where Metal reads what its address space holds there (`predication-word-past-buffer`). No record reaches the runtime's predication yet: this platform's own carries it | the runtime's predication |
| A no-copy buffer, heap or rebind over pages the process maps read-only, imported as dma-bufs of its pages, lets the GPU's writes reach the pages, where after Metal's the CPU reads them as they were (`read-only-pages-written`). This platform imports a writable twin of such pages on Linux, whose writes die where they are encoded, so none reaches it | host imports (Linux) |
| Under `textureWriteRoundingMode` 2, a pipeline that covers by alpha, whose function writes a `float` alpha to a half-float colour 0 stored as written, covers by the alpha rounded to the `half` Metal stores (and so where the write mask leaves the alpha out), where Metal's coverage reads it before rounding: a fragment within half a `half` step of a threshold may cover one sample more or fewer (`coverage-reads-rounded-alpha`) | render states |
| A blend the fragment function computes, of an attachment it reads through the pixel interlock (MoltenVK, RADV, ANV), in a format whose Metal blend arithmetic the runtime has not built (32-bit float, 16-bit normalized, 8-bit snorm, 10-bit, packed float, `A8Unorm`, or an 8-bit unorm one written by fewer than four lanes), is the shader's generic blend in `float`, where Metal's for these formats is not measured (`unmeasured-fetch-blend`) | draw encoding |

