//! Prepare: the guest's command records into the ICB's indirect commands, one command per
//! slot of the ICB's size.
//!
//! Each slot reads its record at the layout's offsets and sends, in order: a reset for an empty
//! slot, the pipeline state (unless inherited), the vertex and fragment (or kernel) buffers
//! (unless inherited), the threadgroup memory lengths (compute), the barrier, then the command's
//! own arguments. A record that cannot be read ends the walk with its error, after whatever the
//! slot already sent.

use crate::memory::cursor::Cursor;
use crate::platform::Platform;
use crate::platform::icb::{
    IcbComputeCmd, IcbRenderCmd, IndirectCommandBuffer, IndirectComputeCommand,
    IndirectRenderCommand,
};
use crate::platform::types::MtlRegion;
use crate::raise::RaiseResult;
use crate::resources::icb::{IndirectCommandBufferFlags, IndirectCommandLayout};
use crate::resources::manager::records::IndirectCommandBufferBind;
use crate::rt::Hold;

use super::lookup::resource_and_offset_for_bind;
use super::records::{
    BIND_SIZE, ComputeCommand, ConcurrentDispatch, Dispatch, DrawIndexedPatches,
    DrawIndexedPrimitives, DrawPatches, DrawPrimitives, RENDER_COMMAND_TYPES, RenderCommand,
    TessellationFactor, WORD_SIZE,
};
use super::{DeathKind, IcbErrorText, IcbWalkOps};

/// The value, or `Ok(Err(..))` from the enclosing walk with `$what: <error>`.
macro_rules! read_or_fail {
    ($e:expr, $what:literal) => {
        match $e {
            Ok(v) => v,
            Err(e) => return Ok(Err(format!(concat!($what, ": {}"), e).into())),
        }
    };
}

/// Sends `cmd` to the render command; a nil command sends nothing.
pub(super) fn send_render<P: Platform>(
    rc: Option<&P::IndirectRenderCommand>,
    cmd: IcbRenderCmd<'_, P>,
) -> RaiseResult<()> {
    rc.map_or(Ok(()), |c| c.encode(cmd))
}

/// Sends `cmd` to the compute command; a nil command sends nothing.
pub(super) fn send_compute<P: Platform>(
    cc: Option<&P::IndirectComputeCommand>,
    cmd: IcbComputeCmd<'_, P>,
) -> RaiseResult<()> {
    cc.map_or(Ok(()), |c| c.encode(cmd))
}

/// The stages a render command binds buffers for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum RenderStage {
    /// The vertex buffers, with their attribute strides when the ICB has them.
    Vertex,
    /// The fragment buffers.
    Fragment,
}

/// The walk: a render walk when the ICB's command types include a render type, a compute walk
/// otherwise.
pub(crate) fn prepare<P: Platform, M: IcbWalkOps<P>>(
    hold: Hold<'_>,
    m: &M,
    buffer: &P::Icb,
    layout: &IndirectCommandLayout,
    flags: &IndirectCommandBufferFlags,
    cursor: Cursor,
) -> RaiseResult<Result<(), IcbErrorText>> {
    if flags.command_types & RENDER_COMMAND_TYPES != 0 {
        return prepare_render(hold, m, buffer, layout, flags, cursor);
    }
    Ok(prepare_compute(hold, m, buffer, layout, flags, cursor)?.map_err(IcbErrorText::from))
}

/// Binds `count` buffers of `stage` from the bind records at `binds`.
///
/// A bind that does not resolve is skipped silently, and a skipped vertex bind does not read its
/// stride, so each later bind takes the stride before its own.
pub(super) fn set_render_buffers<P: Platform, M: IcbWalkOps<P>>(
    hold: Hold<'_>,
    m: &M,
    rc: Option<&P::IndirectRenderCommand>,
    binds: &mut Cursor,
    count: u64,
    strides: Option<&mut Cursor>,
    stage: RenderStage,
) -> RaiseResult<Result<(), String>> {
    let mut strides = strides;
    let mut i: u32 = 0;
    while u64::from(i) < count {
        let index = u64::from(i);
        let bind = read_or_fail!(
            binds.read::<IndirectCommandBufferBind>(),
            "buffer bind not readable"
        );
        if let Ok(r) = resource_and_offset_for_bind(hold, m, &bind) {
            let buffer = r.buffer.as_ref();
            let offset = r.offset;
            let cmd: IcbRenderCmd<'_, P> = match (stage, strides.as_deref_mut()) {
                (RenderStage::Vertex, Some(s)) => {
                    let stride = read_or_fail!(s.read::<u64>(), "attribute stride not readable");
                    IcbRenderCmd::SetVertexBufferAttributeStride {
                        buffer,
                        offset,
                        stride,
                        index,
                    }
                }
                (RenderStage::Vertex, None) => IcbRenderCmd::SetVertexBuffer {
                    buffer,
                    offset,
                    index,
                },
                (RenderStage::Fragment, _) => IcbRenderCmd::SetFragmentBuffer {
                    buffer,
                    offset,
                    index,
                },
            };
            send_render::<P>(rc, cmd)?;
        }
        i = i.wrapping_add(1);
    }
    Ok(Ok(()))
}

/// Binds `count` kernel buffers from the bind records at `binds`, skipping those that do not
/// resolve as [`set_render_buffers`] does.
pub(super) fn set_kernel_buffers<P: Platform, M: IcbWalkOps<P>>(
    hold: Hold<'_>,
    m: &M,
    cc: Option<&P::IndirectComputeCommand>,
    binds: &mut Cursor,
    count: u64,
    strides: Option<&mut Cursor>,
) -> RaiseResult<Result<(), String>> {
    let mut strides = strides;
    let mut i: u32 = 0;
    while u64::from(i) < count {
        let index = u64::from(i);
        let bind = read_or_fail!(
            binds.read::<IndirectCommandBufferBind>(),
            "buffer bind not readable"
        );
        // As for the vertex binds.
        if let Ok(r) = resource_and_offset_for_bind(hold, m, &bind) {
            let buffer = r.buffer.as_ref();
            let offset = r.offset;
            let cmd: IcbComputeCmd<'_, P> = match strides.as_deref_mut() {
                Some(s) => {
                    let stride = read_or_fail!(s.read::<u64>(), "attribute stride not readable");
                    IcbComputeCmd::SetKernelBufferAttributeStride {
                        buffer,
                        offset,
                        stride,
                        index,
                    }
                }
                None => IcbComputeCmd::SetKernelBuffer {
                    buffer,
                    offset,
                    index,
                },
            };
            send_compute::<P>(cc, cmd)?;
        }
        i = i.wrapping_add(1);
    }
    Ok(Ok(()))
}

/// The text of a patch draw whose tessellation factor cannot be read.
///
/// The error of a result that holds the draw record is formatted from the record's first 24 bytes
/// as a string in libc++'s layout: byte 23's bit 7 is the long flag.
/// Short: the text is the record's first `byte23 & 0x7f` bytes, raw (past the record the bytes are
/// unlockable, and only the log text shows them). Long: bytes 8-15 are the size and bytes 0-7 a
/// guest-chosen host pointer; a size of 0 is empty, and any other size reads through that pointer,
/// which faults. Byte 23 is byte 1 of the patch index buffer's GPU address in both patch records.
fn draw_error_bytes(record: &[u8]) -> Vec<u8> {
    let byte23 = record.get(23).copied().unwrap_or(0);
    if byte23 & 0x80 == 0 {
        let len = usize::from(byte23);
        return record.get(..len).unwrap_or(record).to_vec();
    }
    let size = record
        .get(8..16)
        .and_then(|b| <[u8; 8]>::try_from(b).ok())
        .map_or(0, u64::from_le_bytes);
    if size == 0 {
        return Vec::new();
    }
    crate::fatal!(DeathKind::TessellationText)
}

/// The failure of a patch draw's tessellation factor read.
fn tessellation_failure(record: &[u8]) -> IcbErrorText {
    IcbErrorText::with_bytes(
        "tessellation factor not readable: ",
        &draw_error_bytes(record),
    )
}

pub(super) fn prepare_render<P: Platform, M: IcbWalkOps<P>>(
    hold: Hold<'_>,
    m: &M,
    buffer: &P::Icb,
    layout: &IndirectCommandLayout,
    flags: &IndirectCommandBufferFlags,
    mut cursor: Cursor,
) -> RaiseResult<Result<(), IcbErrorText>> {
    let mut i: u32 = 0;
    // The ICB's size is asked for at every slot
    while u64::from(i) < buffer.size() {
        let index = u64::from(i);
        i = i.wrapping_add(1);
        let command = read_or_fail!(
            cursor.sub_cursor(u64::from(layout.command_size)),
            "command not readable"
        );
        let rc = buffer.render_command(index)?;
        let rc = rc.as_ref();
        let ty = read_or_fail!(
            command.read_at::<u32>(u64::from(layout.command_type_offset)),
            "command type not readable"
        );
        let kind = RenderCommand::try_from(ty);
        if kind == Ok(RenderCommand::Empty) {
            send_render::<P>(rc, IcbRenderCmd::Reset)?;
        }
        if !flags.inherit_pipeline_state() {
            let state = read_or_fail!(
                command.read_at::<u32>(u64::from(layout.pipeline_state_offset)),
                "command's pipeline state not readable"
            );
            // A failed pipeline lookup is ignored.
            if state != 0
                && let Ok(pipeline) = m.get_render_pipeline_object(hold, state)?
            {
                crate::fatal_unless!(pipeline.is_some(), DeathKind::NilRenderPipeline);
                send_render::<P>(
                    rc,
                    IcbRenderCmd::SetRenderPipelineState {
                        pipeline_state: pipeline.as_ref(),
                    },
                )?;
            }
        }
        if !flags.inherit_buffers() {
            let mut strides = None;
            if flags.support_dynamic_attribute_stride() {
                strides = Some(read_or_fail!(
                    command.sub_cursor_at(
                        u64::from(layout.attribute_stride_offset),
                        flags.max_vertex_buffer_bind_count.wrapping_mul(WORD_SIZE),
                    ),
                    "attribute strides not readable"
                ));
            }
            // Both are cut before either is checked
            let vertex = command.sub_cursor_at(
                u64::from(layout.vertex_buffer_bind_offset),
                flags.max_vertex_buffer_bind_count.wrapping_mul(BIND_SIZE),
            );
            let fragment = command.sub_cursor_at(
                u64::from(layout.fragment_buffer_bind_offset),
                flags.max_fragment_buffer_bind_count.wrapping_mul(BIND_SIZE),
            );
            let mut vertex = read_or_fail!(vertex, "vertex buffer binds not readable");
            let mut fragment = read_or_fail!(fragment, "fragment buffer binds not readable");
            if let Err(e) = set_render_buffers(
                hold,
                m,
                rc,
                &mut vertex,
                flags.max_vertex_buffer_bind_count,
                strides.as_mut(),
                RenderStage::Vertex,
            )? {
                return Ok(Err(format!("vertex buffers not set: {e}").into()));
            }
            if let Err(e) = set_render_buffers(
                hold,
                m,
                rc,
                &mut fragment,
                flags.max_fragment_buffer_bind_count,
                strides.as_mut(),
                RenderStage::Fragment,
            )? {
                return Ok(Err(format!("fragment buffers not set: {e}").into()));
            }
        }
        let barrier = read_or_fail!(
            command.read_at::<u32>(u64::from(layout.barrier_offset)),
            "barrier not readable"
        );
        send_render::<P>(
            rc,
            if barrier != 0 {
                IcbRenderCmd::SetBarrier
            } else {
                IcbRenderCmd::ClearBarrier
            },
        )?;
        let arguments = u64::from(layout.command_arguments_offset);
        let factor_offset = u64::from(layout.tessellation_factor_offset);
        match kind {
            Ok(RenderCommand::Empty) => {}
            Ok(RenderCommand::Draw) => {
                let draw = read_or_fail!(
                    command.read_at::<DrawPrimitives>(arguments),
                    "draw not readable"
                );
                send_render::<P>(
                    rc,
                    IcbRenderCmd::DrawPrimitives {
                        primitive_type: u64::from(draw.primitive_type),
                        vertex_start: draw.vertex_start,
                        vertex_count: draw.vertex_count,
                        instance_count: draw.instance_count,
                        base_instance: draw.base_instance,
                    },
                )?;
            }
            Ok(RenderCommand::DrawIndexed) => {
                let draw = read_or_fail!(
                    command.read_at::<DrawIndexedPrimitives>(arguments),
                    "indexed draw not readable"
                );
                let index_bind = IndirectCommandBufferBind {
                    buffer_reference: draw.index_buffer_reference,
                    gpu_address: draw.index_buffer_gpu_address,
                    offset: draw.index_buffer_offset,
                };
                // An index buffer that is not found skips the draw.
                let Ok(ib) = resource_and_offset_for_bind(hold, m, &index_bind) else {
                    continue;
                };
                send_render::<P>(
                    rc,
                    IcbRenderCmd::DrawIndexedPrimitives {
                        primitive_type: u64::from(draw.primitive_type),
                        index_count: draw.index_count,
                        index_type: u64::from(draw.index_type),
                        index_buffer: ib.buffer.as_ref(),
                        index_buffer_offset: ib.offset,
                        instance_count: draw.instance_count,
                        base_vertex: draw.base_vertex,
                        base_instance: draw.base_instance,
                    },
                )?;
            }
            Ok(RenderCommand::DrawPatches) => {
                let draw = read_or_fail!(
                    command.read_at::<DrawPatches>(arguments),
                    "patch draw not readable"
                );
                let patch_bind = IndirectCommandBufferBind {
                    buffer_reference: draw.patch_index_buffer_reference,
                    gpu_address: draw.patch_index_buffer_gpu_address,
                    offset: draw.patch_index_buffer_offset,
                };
                // A buffer that is not found skips the draw.
                let Ok(patch_index) = resource_and_offset_for_bind(hold, m, &patch_bind) else {
                    continue;
                };
                let Ok(factor) = command.read_at::<TessellationFactor>(factor_offset) else {
                    return Ok(Err(tessellation_failure(zerocopy::IntoBytes::as_bytes(
                        &draw,
                    ))));
                };
                let factor_bind = IndirectCommandBufferBind {
                    buffer_reference: factor.buffer_reference,
                    gpu_address: factor.buffer_gpu_address,
                    offset: factor.buffer_offset,
                };
                // As above.
                let Ok(tessellation) = resource_and_offset_for_bind(hold, m, &factor_bind) else {
                    continue;
                };
                // The factor's scale is not passed
                send_render::<P>(
                    rc,
                    IcbRenderCmd::DrawPatches {
                        number_of_patch_control_points: u64::from(
                            draw.number_of_patch_control_points,
                        ),
                        patch_start: draw.patch_start,
                        patch_count: draw.patch_count,
                        patch_index_buffer: patch_index.buffer.as_ref(),
                        patch_index_buffer_offset: patch_index.offset,
                        instance_count: draw.instance_count,
                        base_instance: draw.base_instance,
                        buffer: tessellation.buffer.as_ref(),
                        offset: tessellation.offset,
                        instance_stride: factor.instance_stride,
                    },
                )?;
            }
            Ok(RenderCommand::DrawIndexedPatches) => {
                let draw = read_or_fail!(
                    command.read_at::<DrawIndexedPatches>(arguments),
                    "indexed patch draw not readable"
                );
                let patch_bind = IndirectCommandBufferBind {
                    buffer_reference: draw.patch_index_buffer_reference,
                    gpu_address: draw.patch_index_buffer_gpu_address,
                    offset: draw.patch_index_buffer_offset,
                };
                // A buffer that is not found skips the draw.
                let Ok(patch_index) = resource_and_offset_for_bind(hold, m, &patch_bind) else {
                    continue;
                };
                let control_bind = IndirectCommandBufferBind {
                    buffer_reference: draw.control_point_index_buffer_reference,
                    gpu_address: draw.control_point_index_buffer_gpu_address,
                    offset: draw.control_point_index_buffer_offset,
                };
                // As above.
                let Ok(control_point) = resource_and_offset_for_bind(hold, m, &control_bind) else {
                    continue;
                };
                let Ok(factor) = command.read_at::<TessellationFactor>(factor_offset) else {
                    return Ok(Err(tessellation_failure(zerocopy::IntoBytes::as_bytes(
                        &draw,
                    ))));
                };
                let factor_bind = IndirectCommandBufferBind {
                    buffer_reference: factor.buffer_reference,
                    gpu_address: factor.buffer_gpu_address,
                    offset: factor.buffer_offset,
                };
                // As above.
                let Ok(tessellation) = resource_and_offset_for_bind(hold, m, &factor_bind) else {
                    continue;
                };
                send_render::<P>(
                    rc,
                    IcbRenderCmd::DrawIndexedPatches {
                        number_of_patch_control_points: u64::from(
                            draw.number_of_patch_control_points,
                        ),
                        patch_start: draw.patch_start,
                        patch_count: draw.patch_count,
                        patch_index_buffer: patch_index.buffer.as_ref(),
                        patch_index_buffer_offset: patch_index.offset,
                        control_point_index_buffer: control_point.buffer.as_ref(),
                        control_point_index_buffer_offset: control_point.offset,
                        instance_count: draw.instance_count,
                        base_instance: draw.base_instance,
                        buffer: tessellation.buffer.as_ref(),
                        offset: tessellation.offset,
                        instance_stride: factor.instance_stride,
                    },
                )?;
            }
            Err(other) => {
                return Ok(Err(
                    format!("unknown indirect render command type {other}").into()
                ));
            }
        }
    }
    Ok(Ok(()))
}

pub(super) fn prepare_compute<P: Platform, M: IcbWalkOps<P>>(
    hold: Hold<'_>,
    m: &M,
    buffer: &P::Icb,
    layout: &IndirectCommandLayout,
    flags: &IndirectCommandBufferFlags,
    mut cursor: Cursor,
) -> RaiseResult<Result<(), String>> {
    let mut i: u32 = 0;
    while u64::from(i) < buffer.size() {
        let index = u64::from(i);
        i = i.wrapping_add(1);
        let command = read_or_fail!(
            cursor.sub_cursor(u64::from(layout.command_size)),
            "command not readable"
        );
        let cc = buffer.compute_command(index)?;
        let cc = cc.as_ref();
        let ty = read_or_fail!(
            command.read_at::<u32>(u64::from(layout.command_type_offset)),
            "command type not readable"
        );
        let kind = ComputeCommand::try_from(ty);
        if kind == Ok(ComputeCommand::Empty) {
            send_compute::<P>(cc, IcbComputeCmd::Reset)?;
        }
        if !flags.inherit_pipeline_state() {
            let state = read_or_fail!(
                command.read_at::<u32>(u64::from(layout.pipeline_state_offset)),
                "command's pipeline state not readable"
            );
            // A failed pipeline lookup is ignored.
            if state != 0
                && let Ok(pipeline) = m.get_compute_pipeline_object(hold, state)?
            {
                crate::fatal_unless!(pipeline.is_some(), DeathKind::NilComputePipeline);
                send_compute::<P>(
                    cc,
                    IcbComputeCmd::SetComputePipelineState {
                        pipeline_state: pipeline.as_ref(),
                    },
                )?;
            }
        }
        if !flags.inherit_buffers() {
            let mut strides = None;
            // One word a kernel bind, as the synchronize writes them.
            if flags.support_dynamic_attribute_stride() {
                strides = Some(read_or_fail!(
                    command.sub_cursor_at(
                        u64::from(layout.attribute_stride_offset),
                        flags.max_kernel_buffer_bind_count.wrapping_mul(WORD_SIZE),
                    ),
                    "attribute strides not readable"
                ));
            }
            let mut kernel = read_or_fail!(
                command.sub_cursor_at(
                    u64::from(layout.kernel_buffer_bind_offset),
                    flags.max_kernel_buffer_bind_count.wrapping_mul(BIND_SIZE),
                ),
                "kernel buffer binds not readable"
            );
            if let Err(e) = set_kernel_buffers(
                hold,
                m,
                cc,
                &mut kernel,
                flags.max_kernel_buffer_bind_count,
                strides.as_mut(),
            )? {
                return Ok(Err(format!("kernel buffers not set: {e}")));
            }
        }
        let lengths_count = flags.max_kernel_threadgroup_memory_bind_count;
        if lengths_count != 0 {
            let mut lengths = read_or_fail!(
                command.sub_cursor_at(
                    u64::from(layout.threadgroup_memory_length_offset),
                    lengths_count.wrapping_mul(WORD_SIZE),
                ),
                "threadgroup memory lengths not readable"
            );
            let mut j: u32 = 0;
            while u64::from(j) < lengths_count {
                let length = read_or_fail!(
                    lengths.read::<u64>(),
                    "threadgroup memory length not readable"
                );
                send_compute::<P>(
                    cc,
                    IcbComputeCmd::SetThreadgroupMemoryLength {
                        length,
                        index: u64::from(j),
                    },
                )?;
                j = j.wrapping_add(1);
            }
        }
        let barrier = read_or_fail!(
            command.read_at::<u32>(u64::from(layout.barrier_offset)),
            "barrier not readable"
        );
        send_compute::<P>(
            cc,
            if barrier != 0 {
                IcbComputeCmd::SetBarrier
            } else {
                IcbComputeCmd::ClearBarrier
            },
        )?;
        let arguments = u64::from(layout.command_arguments_offset);
        match kind {
            Ok(ComputeCommand::Empty) => {}
            Ok(ComputeCommand::DispatchThreadgroups) => {
                let d = read_or_fail!(
                    command.read_at::<ConcurrentDispatch>(arguments),
                    "dispatch not readable"
                );
                send_compute::<P>(
                    cc,
                    IcbComputeCmd::ConcurrentDispatchThreadgroups {
                        threadgroups_per_grid: d.grid.size(),
                        threads_per_threadgroup: d.threads_per_threadgroup.size(),
                    },
                )?;
            }
            Ok(ComputeCommand::DispatchThreads) => {
                let d = read_or_fail!(
                    command.read_at::<ConcurrentDispatch>(arguments),
                    "dispatch not readable"
                );
                send_compute::<P>(
                    cc,
                    IcbComputeCmd::ConcurrentDispatchThreads {
                        threads_per_grid: d.grid.size(),
                        threads_per_threadgroup: d.threads_per_threadgroup.size(),
                    },
                )?;
            }
            Err(other) => {
                return Ok(Err(format!(
                    "unknown indirect compute command type {other}"
                )));
            }
        }
        // Every command, an empty one included
        let dispatch = read_or_fail!(
            command.read_at::<Dispatch>(u64::from(layout.kernel_dispatch_arguments_offset)),
            "dispatch arguments not readable"
        );
        send_compute::<P>(
            cc,
            IcbComputeCmd::SetStageInRegion {
                region: MtlRegion {
                    origin: dispatch.stage_in_region_origin.origin(),
                    size: dispatch.stage_in_region_size.size(),
                },
            },
        )?;
        send_compute::<P>(
            cc,
            IcbComputeCmd::SetImageblockWidth {
                width: u64::from(dispatch.imageblock_width),
                height: u64::from(dispatch.imageblock_height),
            },
        )?;
    }
    Ok(Ok(()))
}

#[cfg(test)]
mod tests {
    use zerocopy::{Immutable, IntoBytes};

    use super::super::testing::*;
    use super::*;
    use crate::platform::Handle;
    use crate::platform::types::{MtlOrigin, MtlSize};
    use crate::testkit::calllog::Arg;
    use crate::testkit::capture::catch_death;
    use crate::testkit::{FakePlatform, Fk, Step, kind};

    mod render {
        use super::*;

        fn setup() -> (
            FakeOps,
            Fk<kind::Buffer>,
            Fk<kind::Buffer>,
            Fk<kind::RenderPipeline>,
        ) {
            let mut o = FakeOps::new();
            let b5 = o.resident_buffer(5, 0x100);
            let b6 = o.resident_buffer(6, 0x100);
            let rp = o.render_pipeline();
            o.renders.push((9, Lookup::Found(Some(rp.clone_ref()))));
            (o, b5, b6, rp)
        }

        fn prepare(o: &FakeOps, icb: &Fk<kind::Icb>, flags: u32, c: &Cmds) -> Result<(), String> {
            prepare_render(
                Hold::NONE,
                o,
                icb,
                &layout(),
                &render_flags(flags),
                c.cursor(),
            )
            .unwrap()
            .map_err(|e| e.to_string())
        }

        #[test]
        fn a_draw_sets_pipeline_binds_strides_barrier_and_draws() {
            let (o, b5, b6, rp) = setup();
            let icb = o.icb(1);
            let mut c = Cmds::new(1);
            draw_command(&mut c, 0);
            prepare(&o, &icb, 0b100, &c).unwrap();
            assert_eq!(
                render_cmds(&o),
                vec![
                    "setRenderPipelineState:",
                    "setVertexBuffer:offset:attributeStride:atIndex:",
                    "setVertexBuffer:offset:attributeStride:atIndex:",
                    "setFragmentBuffer:offset:atIndex:",
                    "setBarrier",
                    "drawPrimitives:vertexStart:vertexCount:instanceCount:baseInstance:",
                ]
            );
            let vb = o
                .log()
                .calls_to("setVertexBuffer:offset:attributeStride:atIndex:");
            assert_eq!(
                vb[0].args,
                vec![Arg::obj(&b5), Arg::U64(0x10), Arg::U64(0x30), Arg::U64(0)]
            );
            assert_eq!(
                vb[1].args,
                vec![Arg::obj(&b6), Arg::U64(0x20), Arg::U64(0x40), Arg::U64(1)]
            );
            assert_eq!(
                last_args(&o, "setFragmentBuffer:offset:atIndex:"),
                vec![Arg::obj(&b5), Arg::U64(8), Arg::U64(0)]
            );
            assert_eq!(
                last_args(&o, "setRenderPipelineState:"),
                vec![Arg::obj(&rp)]
            );
            assert_eq!(
                last_args(
                    &o,
                    "drawPrimitives:vertexStart:vertexCount:instanceCount:baseInstance:"
                ),
                vec![
                    Arg::U64(3),
                    Arg::U64(1),
                    Arg::U64(2),
                    Arg::U64(4),
                    Arg::U64(5)
                ]
            );
            assert_eq!(
                o.calls(),
                vec![
                    "get_render_pipeline_object(9)",
                    "get_buffer_object(5)",
                    "get_buffer_object(6)",
                    "get_buffer_object(5)"
                ]
            );
        }

        #[test]
        fn without_dynamic_strides_the_plain_setter_is_used() {
            let (o, _b5, _b6, _rp) = setup();
            let icb = o.icb(1);
            let mut c = Cmds::new(1);
            draw_command(&mut c, 0);
            prepare(&o, &icb, 0, &c).unwrap();
            assert_eq!(o.log().count("setVertexBuffer:offset:atIndex:"), 2);
            assert_eq!(
                o.log()
                    .count("setVertexBuffer:offset:attributeStride:atIndex:"),
                0
            );
        }

        #[test]
        fn inherited_pipeline_and_buffers_are_not_read() {
            let (o, _b5, _b6, _rp) = setup();
            let icb = o.icb(1);
            let mut c = Cmds::new(1);
            draw_command(&mut c, 0);
            prepare(&o, &icb, 0b11, &c).unwrap();
            assert!(o.calls().is_empty());
            assert_eq!(
                render_cmds(&o),
                vec![
                    "setBarrier",
                    "drawPrimitives:vertexStart:vertexCount:instanceCount:baseInstance:"
                ]
            );
        }

        #[test]
        fn an_empty_slot_resets_and_draws_nothing() {
            let (o, _b5, _b6, _rp) = setup();
            let icb = o.icb(1);
            let c = Cmds::new(1);
            prepare(&o, &icb, 0b11, &c).unwrap();
            assert_eq!(render_cmds(&o), vec!["reset", "clearBarrier"]);
        }

        #[test]
        fn a_zero_pipeline_is_not_looked_up_and_a_failed_lookup_is_ignored() {
            // A failed pipeline lookup is ignored
            let (mut o, _b5, _b6, _rp) = setup();
            let icb = o.icb(2);
            let mut c = Cmds::new(2);
            draw_command(&mut c, 0);
            c.put(0, 8, &0u32);
            draw_command(&mut c, 1);
            c.put(1, 8, &11u32);
            o.renders.push((11, Lookup::Missing("gone")));
            prepare(&o, &icb, 0b10, &c).unwrap();
            assert_eq!(o.calls(), vec!["get_render_pipeline_object(11)"]);
            assert_eq!(o.log().count("setRenderPipelineState:"), 0);
            assert_eq!(
                o.log()
                    .count("drawPrimitives:vertexStart:vertexCount:instanceCount:baseInstance:"),
                2
            );
        }

        #[test]
        fn a_nil_pipeline_dies() {
            let (mut o, _b5, _b6, _rp) = setup();
            o.renders.insert(0, (9, Lookup::Found(None)));
            let icb = o.icb(1);
            let mut c = Cmds::new(1);
            draw_command(&mut c, 0);
            let d = catch_death(|| prepare(&o, &icb, 0b10, &c).is_ok()).unwrap_err();
            assert_eq!(d, death(DeathKind::NilRenderPipeline));
        }

        #[test]
        fn an_unresolved_bind_is_skipped_without_consuming_a_stride() {
            // An unresolved bind is skipped silently, and does not read its stride, shifting the
            // later strides by one
            let (o, _b5, b6, _rp) = setup();
            let icb = o.icb(1);
            let mut c = Cmds::new(1);
            draw_command(&mut c, 0);
            c.put(0, 0x10, &bind(77, 0, 0));
            prepare(&o, &icb, 0b101, &c).unwrap();
            let vb = o
                .log()
                .calls_to("setVertexBuffer:offset:attributeStride:atIndex:");
            assert_eq!(vb.len(), 1);
            assert_eq!(
                vb[0].args,
                vec![Arg::obj(&b6), Arg::U64(0x20), Arg::U64(0x30), Arg::U64(1)]
            );
        }

        #[test]
        fn an_indexed_draw_with_an_unresolved_index_buffer_is_skipped() {
            // The draw is skipped after the barrier was set.
            let (o, b5, _b6, _rp) = setup();
            let icb = o.icb(2);
            let mut c = Cmds::new(2);
            let draw = |r: u32| DrawIndexedPrimitives {
                primitive_type: 1,
                index_type: 1,
                index_buffer_reference: r,
                index_count: 6,
                index_buffer_gpu_address: 0,
                index_buffer_offset: 0x10,
                instance_count: 2,
                base_vertex: -3,
                base_instance: 4,
            };
            c.ty(0, DRAW_INDEXED)
                .put(0, 4, &1u32)
                .put(0, 0x90, &draw(88));
            c.ty(1, DRAW_INDEXED).put(1, 0x90, &draw(5));
            prepare(&o, &icb, 0b11, &c).unwrap();
            let sel = "drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:instanceCount:baseVertex:baseInstance:";
            assert_eq!(o.log().count(sel), 1);
            assert_eq!(o.log().count("setBarrier"), 1);
            assert_eq!(
                last_args(&o, sel),
                vec![
                    Arg::U64(1),
                    Arg::U64(6),
                    Arg::U64(1),
                    Arg::obj(&b5),
                    Arg::U64(0x10),
                    Arg::U64(2),
                    Arg::I64(-3),
                    Arg::U64(4)
                ]
            );
        }

        #[test]
        fn patch_draws_pass_the_factor_buffer_without_its_scale() {
            let (o, b5, b6, _rp) = setup();
            let icb = o.icb(2);
            let mut c = Cmds::new(2);
            let factor = TessellationFactor {
                buffer_reference: 6,
                buffer_gpu_address: 0,
                buffer_offset: 0x18,
                instance_stride: 0x40,
                scale: 2.0,
            };
            c.ty(0, DRAW_PATCHES).put(0, 0x70, &factor).put(
                0,
                0x90,
                &DrawPatches {
                    number_of_patch_control_points: 3,
                    patch_start: 1,
                    patch_count: 2,
                    patch_index_buffer_reference: 5,
                    patch_index_buffer_gpu_address: 0,
                    patch_index_buffer_offset: 8,
                    instance_count: 4,
                    base_instance: 5,
                    field_36: [0; 2],
                },
            );
            c.ty(1, DRAW_INDEXED_PATCHES).put(1, 0x70, &factor).put(
                1,
                0x90,
                &DrawIndexedPatches {
                    number_of_patch_control_points: 4,
                    patch_start: 1,
                    patch_count: 2,
                    patch_index_buffer_reference: 5,
                    patch_index_buffer_gpu_address: 0,
                    patch_index_buffer_offset: 8,
                    control_point_index_buffer_reference: 6,
                    control_point_index_buffer_gpu_address: 0,
                    control_point_index_buffer_offset: 0xc,
                    instance_count: 4,
                    base_instance: 5,
                    field_4a: [0; 2],
                },
            );
            prepare(&o, &icb, 0b11, &c).unwrap();
            let dp = "drawPatches:patchStart:patchCount:patchIndexBuffer:patchIndexBufferOffset:instanceCount:baseInstance:tessellationFactorBuffer:tessellationFactorBufferOffset:tessellationFactorBufferInstanceStride:";
            assert_eq!(
                last_args(&o, dp),
                vec![
                    Arg::U64(3),
                    Arg::U64(1),
                    Arg::U64(2),
                    Arg::obj(&b5),
                    Arg::U64(8),
                    Arg::U64(4),
                    Arg::U64(5),
                    Arg::obj(&b6),
                    Arg::U64(0x18),
                    Arg::U64(0x40)
                ]
            );
            let dip = "drawIndexedPatches:patchStart:patchCount:patchIndexBuffer:patchIndexBufferOffset:controlPointIndexBuffer:controlPointIndexBufferOffset:instanceCount:baseInstance:tessellationFactorBuffer:tessellationFactorBufferOffset:tessellationFactorBufferInstanceStride:";
            assert_eq!(
                last_args(&o, dip),
                vec![
                    Arg::U64(4),
                    Arg::U64(1),
                    Arg::U64(2),
                    Arg::obj(&b5),
                    Arg::U64(8),
                    Arg::obj(&b6),
                    Arg::U64(0xc),
                    Arg::U64(4),
                    Arg::U64(5),
                    Arg::obj(&b6),
                    Arg::U64(0x18),
                    Arg::U64(0x40)
                ]
            );
        }

        #[test]
        fn an_unreadable_factor_fails_with_the_records_text() {
            // A GPU address of 0 makes the string short and empty.
            let e = short_factor(DRAW_PATCHES, &patches(0, 0, 0, 0)).unwrap_err();
            assert_eq!(e, "tessellation factor not readable: ");
        }

        fn patches(control_points: u16, start: u64, count: u64, address: u64) -> DrawPatches {
            DrawPatches {
                number_of_patch_control_points: control_points,
                patch_start: start,
                patch_count: count,
                patch_index_buffer_reference: 5,
                patch_index_buffer_gpu_address: address,
                ..Default::default()
            }
        }

        /// Prepares one command of type `ty` whose record is `draw` and whose tessellation
        /// factor read runs past the command.
        fn short_factor<T: IntoBytes + Immutable>(ty: u32, draw: &T) -> Result<(), IcbErrorText> {
            let (o, _b5, _b6, _rp) = setup();
            let icb = o.icb(1);
            let mut c = Cmds::new(1);
            c.ty(0, ty).put(0, 0x90, draw);
            let mut l = layout();
            l.tessellation_factor_offset = (CMD - 4) as u16;
            prepare_render(Hold::NONE, &o, &icb, &l, &render_flags(0b11), c.cursor()).unwrap()
        }

        #[test]
        fn a_short_string_reads_the_records_leading_bytes() {
            // Byte 23 (bits 8-14 of the GPU address) is the length; the text is the record's first
            // bytes: control points "AB", then patchStart "CD".
            let draw = patches(0x4241, 0x4443, 0, 0x0400);
            let e = short_factor(DRAW_PATCHES, &draw).unwrap_err();
            assert_eq!(e, "tessellation factor not readable: ABCD");
        }

        #[test]
        fn a_long_string_of_size_zero_is_empty() {
            // Bit 15 of the GPU address sets the long flag; bytes 8-15 (patchStart's top two bytes,
            // patchCount's low six) are the size.
            let draw = patches(1, 0x0000_ffff_ffff_ffff, 0xffff_0000_0000_0000, 0x8000);
            let e = short_factor(DRAW_PATCHES, &draw).unwrap_err();
            assert_eq!(e, "tessellation factor not readable: ");
        }

        #[test]
        fn a_long_string_with_a_size_dies() {
            // A read through the guest-chosen pointer.
            let draw = patches(0, 0, 1, 0x8000);
            let d = catch_death(|| short_factor(DRAW_PATCHES, &draw).is_ok()).unwrap_err();
            assert_eq!(d, death(DeathKind::TessellationText));
        }

        #[test]
        fn indexed_patches_format_the_same_bytes() {
            // As for a patch draw.
            let draw = DrawIndexedPatches {
                number_of_patch_control_points: 0x4241,
                patch_index_buffer_reference: 5,
                patch_index_buffer_gpu_address: 0x0200,
                control_point_index_buffer_reference: 6,
                ..Default::default()
            };
            let e = short_factor(DRAW_INDEXED_PATCHES, &draw).unwrap_err();
            assert_eq!(e, "tessellation factor not readable: AB");
            let long = DrawIndexedPatches {
                patch_count: 1,
                patch_index_buffer_gpu_address: 0x8000,
                ..draw
            };
            let d = catch_death(|| short_factor(DRAW_INDEXED_PATCHES, &long).is_ok()).unwrap_err();
            assert_eq!(d, death(DeathKind::TessellationText));
        }

        #[test]
        fn the_short_string_keeps_non_utf8_bytes_raw() {
            // The record's bytes are copied unchanged; the log's %s prints them as bytes. 0x80
            // stays 0x80.
            let draw = patches(0x0180, 0, 0, 0x0200);
            let e = short_factor(DRAW_PATCHES, &draw).unwrap_err();
            assert_eq!(e.as_bytes(), b"tessellation factor not readable: \x80\x01");
            let draw = patches(0x0080, 0, 0, 0x0200);
            let e = short_factor(DRAW_PATCHES, &draw).unwrap_err();
            assert_eq!(e.as_bytes(), b"tessellation factor not readable: \x80\x00");
            // The same for the indexed form.
            let draw = DrawIndexedPatches {
                number_of_patch_control_points: 0xfeff,
                patch_start: 0xc3,
                patch_index_buffer_reference: 5,
                patch_index_buffer_gpu_address: 0x0300,
                control_point_index_buffer_reference: 6,
                ..Default::default()
            };
            let e = short_factor(DRAW_INDEXED_PATCHES, &draw).unwrap_err();
            assert_eq!(
                e.as_bytes(),
                b"tessellation factor not readable: \xff\xfe\xc3"
            );
        }

        #[test]
        fn an_unknown_type_fails() {
            let (o, _b5, _b6, _rp) = setup();
            let icb = o.icb(1);
            let mut c = Cmds::new(1);
            c.ty(0, 0x10);
            let e = prepare(&o, &icb, 0b11, &c).unwrap_err();
            assert_eq!(e, "unknown indirect render command type 16");
        }

        #[test]
        fn a_short_cursor_fails_to_read_the_command() {
            let (o, _b5, _b6, _rp) = setup();
            let icb = o.icb(2);
            let c = Cmds::new(1);
            let e = prepare(&o, &icb, 0b11, &c).unwrap_err();
            assert!(e.starts_with("command not readable: "), "{e}");
            assert_eq!(o.log().count("reset"), 1);
        }

        #[test]
        fn a_bind_block_outside_the_command_fails_with_the_cursors_error() {
            // The cut fails before any bind is set.
            let (o, _b5, _b6, _rp) = setup();
            let icb = o.icb(1);
            let c = Cmds::new(1);
            let mut l = layout();
            l.vertex_buffer_bind_offset = (CMD - 8) as u32;
            let e = prepare_render(Hold::NONE, &o, &icb, &l, &render_flags(0b01), c.cursor())
                .unwrap()
                .unwrap_err();
            assert!(
                e.to_string()
                    .starts_with("vertex buffer binds not readable: "),
                "{e}"
            );
        }

        #[test]
        fn a_short_bind_list_fails() {
            let (o, _b5, _b6, _rp) = setup();
            let mut binds = cursor(&bytes(&bind(5, 0, 0)));
            let e = set_render_buffers::<FakePlatform, _>(
                Hold::NONE,
                &o,
                None,
                &mut binds,
                2,
                None,
                RenderStage::Fragment,
            )
            .unwrap()
            .unwrap_err();
            assert!(e.starts_with("buffer bind not readable: "), "{e}");
        }

        #[test]
        fn a_nil_command_is_sent_nothing() {
            let (o, _b5, _b6, _rp) = setup();
            let icb = o.icb(1);
            o.p.script()
                .push("indirectRenderCommandAtIndex:", Step::Nil);
            let mut c = Cmds::new(1);
            draw_command(&mut c, 0);
            prepare(&o, &icb, 0, &c).unwrap();
            assert!(render_cmds(&o).is_empty());
            assert_eq!(o.calls().len(), 4);
        }
    }

    mod compute {
        use super::*;

        fn setup() -> (
            FakeOps,
            Fk<kind::Buffer>,
            Fk<kind::Buffer>,
            Fk<kind::ComputePipeline>,
        ) {
            let mut o = FakeOps::new();
            let b5 = o.resident_buffer(5, 0x100);
            let b6 = o.resident_buffer(6, 0x100);
            let cp = o.compute_pipeline();
            o.computes.push((9, Lookup::Found(Some(cp.clone_ref()))));
            (o, b5, b6, cp)
        }

        fn prepare(o: &FakeOps, icb: &Fk<kind::Icb>, flags: u32, c: &Cmds) -> Result<(), String> {
            prepare_compute(
                Hold::NONE,
                o,
                icb,
                &layout(),
                &compute_flags(flags),
                c.cursor(),
            )
            .unwrap()
        }

        #[test]
        fn a_dispatch_sets_everything_in_order() {
            let (o, b5, _b6, cp) = setup();
            let icb = o.icb(1);
            let mut c = Cmds::new(1);
            compute_command(&mut c, 0, CONCURRENT_DISPATCH);
            prepare(&o, &icb, 0b100, &c).unwrap();
            assert_eq!(
                compute_cmds(&o),
                vec![
                    "setComputePipelineState:",
                    "setKernelBuffer:offset:attributeStride:atIndex:",
                    "setKernelBuffer:offset:attributeStride:atIndex:",
                    "setThreadgroupMemoryLength:atIndex:",
                    "setThreadgroupMemoryLength:atIndex:",
                    "clearBarrier",
                    "concurrentDispatchThreadgroups:threadsPerThreadgroup:",
                    "setStageInRegion:",
                    "setImageblockWidth:height:",
                ]
            );
            assert_eq!(
                last_args(&o, "setComputePipelineState:"),
                vec![Arg::obj(&cp)]
            );
            let kb = o
                .log()
                .calls_to("setKernelBuffer:offset:attributeStride:atIndex:");
            assert_eq!(
                kb[0].args,
                vec![Arg::obj(&b5), Arg::U64(0x10), Arg::U64(0x30), Arg::U64(0)]
            );
            let tg = o.log().calls_to("setThreadgroupMemoryLength:atIndex:");
            assert_eq!(tg[1].args, vec![Arg::U64(0x200), Arg::U64(1)]);
            assert_eq!(
                last_args(&o, "concurrentDispatchThreadgroups:threadsPerThreadgroup:"),
                vec![
                    Arg::val(&MtlSize {
                        width: 1,
                        height: 2,
                        depth: 3
                    }),
                    Arg::val(&MtlSize {
                        width: 4,
                        height: 5,
                        depth: 6
                    })
                ]
            );
            assert_eq!(
                last_args(&o, "setStageInRegion:"),
                vec![Arg::val(&MtlRegion {
                    origin: MtlOrigin { x: 1, y: 1, z: 0 },
                    size: MtlSize {
                        width: 7,
                        height: 8,
                        depth: 9
                    }
                })]
            );
            assert_eq!(
                last_args(&o, "setImageblockWidth:height:"),
                vec![Arg::U64(16), Arg::U64(32)]
            );
        }

        #[test]
        fn threads_dispatch() {
            let (o, _b5, _b6, _cp) = setup();
            let icb = o.icb(1);
            let mut c = Cmds::new(1);
            compute_command(&mut c, 0, CONCURRENT_DISPATCH_THREADS);
            prepare(&o, &icb, 0b11, &c).unwrap();
            assert_eq!(
                o.log()
                    .count("concurrentDispatchThreads:threadsPerThreadgroup:"),
                1
            );
        }

        #[test]
        fn an_empty_slot_resets_and_still_sets_the_dispatch_arguments() {
            let (o, _b5, _b6, _cp) = setup();
            let icb = o.icb(1);
            let mut c = Cmds::new(1);
            compute_command(&mut c, 0, 0);
            prepare(&o, &icb, 0b11, &c).unwrap();
            let sels = compute_cmds(&o);
            assert_eq!(sels[0], "reset");
            assert!(sels.ends_with(&[
                "clearBarrier",
                "setStageInRegion:",
                "setImageblockWidth:height:"
            ]));
        }

        #[test]
        fn strides_are_sized_by_the_kernel_count() {
            let (o, _b5, _b6, _cp) = setup();
            let icb = o.icb(1);
            let mut c = Cmds::new(1);
            compute_command(&mut c, 0, CONCURRENT_DISPATCH);
            let mut f = compute_flags(0b101);
            f.max_vertex_buffer_bind_count = 0;
            prepare_compute(Hold::NONE, &o, &icb, &layout(), &f, c.cursor())
                .unwrap()
                .unwrap();
            assert_eq!(
                o.log()
                    .count("setKernelBuffer:offset:attributeStride:atIndex:"),
                2
            );
        }

        #[test]
        fn a_nil_pipeline_dies() {
            let (mut o, _b5, _b6, _cp) = setup();
            o.computes.insert(0, (9, Lookup::Found(None)));
            let icb = o.icb(1);
            let mut c = Cmds::new(1);
            compute_command(&mut c, 0, CONCURRENT_DISPATCH);
            let d = catch_death(|| prepare(&o, &icb, 0b10, &c).is_ok()).unwrap_err();
            assert_eq!(d, death(DeathKind::NilComputePipeline));
        }

        #[test]
        fn a_failed_pipeline_lookup_is_ignored() {
            let (mut o, _b5, _b6, _cp) = setup();
            o.computes.insert(0, (9, Lookup::Missing("gone")));
            let icb = o.icb(1);
            let mut c = Cmds::new(1);
            compute_command(&mut c, 0, CONCURRENT_DISPATCH);
            prepare(&o, &icb, 0b10, &c).unwrap();
            assert_eq!(o.log().count("setComputePipelineState:"), 0);
        }

        #[test]
        fn an_unknown_type_fails_after_the_barrier() {
            let (o, _b5, _b6, _cp) = setup();
            let icb = o.icb(1);
            let mut c = Cmds::new(1);
            compute_command(&mut c, 0, 1);
            let e = prepare(&o, &icb, 0b11, &c).unwrap_err();
            assert_eq!(e, "unknown indirect compute command type 1");
            assert_eq!(o.log().count("clearBarrier"), 1);
        }

        #[test]
        fn threadgroup_lengths_outside_the_command_fail() {
            let (o, _b5, _b6, _cp) = setup();
            let icb = o.icb(1);
            let c = Cmds::new(1);
            let mut l = layout();
            l.threadgroup_memory_length_offset = (CMD - 8) as u32;
            let e = prepare_compute(Hold::NONE, &o, &icb, &l, &compute_flags(0b11), c.cursor())
                .unwrap()
                .unwrap_err();
            assert!(
                e.starts_with("threadgroup memory lengths not readable: "),
                "{e}"
            );
        }

        #[test]
        fn a_short_bind_list_fails() {
            let (o, _b5, _b6, _cp) = setup();
            let mut binds = cursor(&bytes(&bind(5, 0, 0)));
            let e =
                set_kernel_buffers::<FakePlatform, _>(Hold::NONE, &o, None, &mut binds, 2, None)
                    .unwrap()
                    .unwrap_err();
            assert!(e.starts_with("buffer bind not readable: "), "{e}");
        }
    }

    #[test]
    fn the_command_types_choose_the_render_or_compute_walk() {
        for (types, render) in [
            (DRAW, true),
            (0x80, true),
            (0x100, true),
            (0x20, false),
            (0x40, false),
            (0x10, false),
        ] {
            let o = FakeOps::new();
            let icb = o.icb(1);
            let mut f = render_flags(0b11);
            f.command_types = types;
            prepare(Hold::NONE, &o, &icb, &layout(), &f, Cmds::new(1).cursor())
                .unwrap()
                .unwrap();
            let (_mem, w) = writable(CMD as u64);
            super::super::synchronize::synchronize(Hold::NONE, &o, &icb, &layout(), &f, w)
                .unwrap()
                .unwrap();
            let sels = sels_of(o.log(), "MTLIndirectCommandBuffer");
            let want = if render {
                "indirectRenderCommandAtIndex:"
            } else {
                "indirectComputeCommandAtIndex:"
            };
            assert_eq!(sels.iter().filter(|s| **s == want).count(), 2, "{types:#x}");
        }
    }

    #[test]
    fn with_no_manager_a_short_cursor_is_a_refusal() {
        let o = FakeOps::new();
        let icb = o.icb(1);
        let e = super::super::prepare_indirect_command_buffer_on_null::<FakePlatform>(
            Hold::NONE,
            &icb,
            &layout(),
            &IndirectCommandBufferFlags {
                command_types: 1,
                ..IndirectCommandBufferFlags::default()
            },
            cursor(&[]),
        )
        .unwrap()
        .unwrap_err();
        assert!(e.to_string().starts_with("command not readable: "), "{e}");
    }
}
