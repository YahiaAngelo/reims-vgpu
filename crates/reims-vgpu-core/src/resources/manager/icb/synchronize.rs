//! Synchronize: the ICB's indirect commands back into the guest's command records, one record
//! per slot of the ICB's size.
//!
//! Each slot writes its command type, the pipeline state's reference (unless inherited), its
//! buffers as binds (unless inherited), the threadgroup memory lengths (compute: one word each,
//! where the prepare reads them), the barrier and the command's arguments. Only a failed cursor,
//! command type or pipeline write ends the walk; the other writes' errors are dropped.
//!
//! A command's arguments are read whole ([`DrawArgs`] and its kin), then each address they
//! hold is looked up for its bind. The `get*AtIndex:` reads, whose index is the guest's, go out
//! one at a time between the guest writes, in order.

use crate::memory::cursor::WritableCursor;
use crate::num::Truncate;
use crate::platform::Platform;
use crate::platform::icb::{
    DrawArgs, IndirectCommandBuffer, IndirectComputeCommand, IndirectRenderCommand,
    TessellationFactorArgs,
};
use crate::raise::RaiseResult;
use crate::resources::icb::{IndirectCommandBufferFlags, IndirectCommandLayout};
use crate::resources::manager::records::IndirectCommandBufferBind;
use crate::rt::Hold;

use super::IcbWalkOps;
use super::lookup::fill_buffer_bind;
use super::prepare::RenderStage;
use super::records::{
    BIND_SIZE, ComputeCommand, ConcurrentDispatch, Dispatch, DrawIndexedPatches,
    DrawIndexedPrimitives, DrawPatches, DrawPrimitives, RENDER_COMMAND_TYPES, RenderCommand,
    TessellationFactor, Triple, WORD_SIZE,
};

/// The value, or `Ok(Err(..))` from the enclosing walk with `$what: <error>`.
macro_rules! write_or_fail {
    ($e:expr, $what:literal) => {
        match $e {
            Ok(v) => v,
            Err(e) => return Ok(Err(format!(concat!($what, ": {}"), e))),
        }
    };
}

/// The walk: a render walk when the ICB's command types include a render type, a compute walk
/// otherwise.
pub(crate) fn synchronize<P: Platform, M: IcbWalkOps<P>>(
    hold: Hold<'_>,
    m: &M,
    buffer: &P::Icb,
    layout: &IndirectCommandLayout,
    flags: &IndirectCommandBufferFlags,
    cursor: WritableCursor,
) -> RaiseResult<Result<(), String>> {
    if flags.command_types & RENDER_COMMAND_TYPES != 0 {
        return synchronize_render(hold, m, buffer, layout, flags, cursor);
    }
    synchronize_compute(hold, m, buffer, layout, flags, cursor)
}

/// Writes `count` binds of `stage` from the command's buffer addresses, and each vertex bind's
/// attribute stride when `strides` is given.
///
/// A failed stride write is dropped and the loop goes on; only a failed bind write ends it.
pub(super) fn get_render_buffers<P: Platform, M: IcbWalkOps<P>>(
    hold: Hold<'_>,
    m: &M,
    rc: Option<&P::IndirectRenderCommand>,
    binds: &mut WritableCursor,
    count: u64,
    strides: Option<&mut WritableCursor>,
    stage: RenderStage,
) -> RaiseResult<Result<(), String>> {
    let mut strides = strides;
    let mut i: u32 = 0;
    while u64::from(i) < count {
        let index = u64::from(i);
        let address = match (stage, rc) {
            (_, None) => 0,
            (RenderStage::Vertex, Some(c)) => c.vertex_buffer_address(index)?,
            (RenderStage::Fragment, Some(c)) => c.fragment_buffer_address(index)?,
        };
        let bind = fill_buffer_bind(hold, m, address);
        write_or_fail!(binds.write(&bind), "buffer bind not writable");
        if stage == RenderStage::Vertex
            && let Some(s) = strides.as_deref_mut()
        {
            let stride = match rc {
                Some(c) => c.vertex_attribute_stride(index)?,
                None => 0,
            };
            let _ = s.write(&stride);
        }
        i = i.wrapping_add(1);
    }
    Ok(Ok(()))
}

/// Writes `count` kernel binds from the command's buffer addresses, and each one's attribute
/// stride when `strides` is given.
///
/// Here a failed stride write ends the loop.
pub(super) fn get_kernel_buffers<P: Platform, M: IcbWalkOps<P>>(
    hold: Hold<'_>,
    m: &M,
    cc: Option<&P::IndirectComputeCommand>,
    binds: &mut WritableCursor,
    count: u64,
    strides: Option<&mut WritableCursor>,
) -> RaiseResult<Result<(), String>> {
    let mut strides = strides;
    let mut i: u32 = 0;
    while u64::from(i) < count {
        let index = u64::from(i);
        let address = match cc {
            Some(c) => c.kernel_buffer_address(index)?,
            None => 0,
        };
        let bind = fill_buffer_bind(hold, m, address);
        write_or_fail!(binds.write(&bind), "buffer bind not writable");
        if let Some(s) = strides.as_deref_mut() {
            let stride = match cc {
                Some(c) => c.kernel_attribute_stride(index)?,
                None => 0,
            };
            write_or_fail!(s.write(&stride), "attribute stride not writable");
        }
        i = i.wrapping_add(1);
    }
    Ok(Ok(()))
}

/// The reference registered for a command's pipeline, or 0 when none is.
fn pipeline_reference<P: Platform, M: IcbWalkOps<P>>(m: &M, identifier: u64) -> u32 {
    m.get_pipeline_ref(identifier).unwrap_or(0)
}

/// The tessellation factor record: its buffer's bind, then the arguments' stride and scale.
fn tessellation_factor(
    t: &TessellationFactorArgs,
    fb: IndirectCommandBufferBind,
) -> TessellationFactor {
    TessellationFactor {
        buffer_reference: fb.buffer_reference,
        buffer_gpu_address: fb.gpu_address,
        buffer_offset: fb.offset,
        instance_stride: t.instance_stride,
        scale: t.scale,
    }
}

pub(super) fn synchronize_render<P: Platform, M: IcbWalkOps<P>>(
    hold: Hold<'_>,
    m: &M,
    buffer: &P::Icb,
    layout: &IndirectCommandLayout,
    flags: &IndirectCommandBufferFlags,
    mut cursor: WritableCursor,
) -> RaiseResult<Result<(), String>> {
    let mut i: u32 = 0;
    while u64::from(i) < buffer.size() {
        let index = u64::from(i);
        i = i.wrapping_add(1);
        let mut command = write_or_fail!(
            cursor.writable_sub_cursor(u64::from(layout.command_size)),
            "command not writable"
        );
        let rc = buffer.render_command(index)?;
        let rc = rc.as_ref();
        let ty = rc.map_or(0, |c| c.command_type()).low_u32();
        // The walk ends at this point with the write's own error.
        write_or_fail!(
            command.write_at(u64::from(layout.command_type_offset), &ty),
            "command type not writable"
        );
        if !flags.inherit_pipeline_state() {
            let identifier = rc.map_or(0, |c| c.pipeline_state_unique_identifier());
            let reference = pipeline_reference(m, identifier);
            // As for the command type.
            write_or_fail!(
                command.write_at(u64::from(layout.pipeline_state_offset), &reference),
                "command's pipeline state not writable"
            );
        }
        if !flags.inherit_buffers() {
            let mut strides = None;
            if flags.support_dynamic_attribute_stride() {
                strides = Some(write_or_fail!(
                    command.writable_sub_cursor_at(
                        u64::from(layout.attribute_stride_offset),
                        flags.max_vertex_buffer_bind_count.wrapping_mul(WORD_SIZE),
                    ),
                    "attribute strides not writable"
                ));
            }
            // Both are cut before either is checked
            let vertex = command.writable_sub_cursor_at(
                u64::from(layout.vertex_buffer_bind_offset),
                flags.max_vertex_buffer_bind_count.wrapping_mul(BIND_SIZE),
            );
            let fragment = command.writable_sub_cursor_at(
                u64::from(layout.fragment_buffer_bind_offset),
                flags.max_fragment_buffer_bind_count.wrapping_mul(BIND_SIZE),
            );
            let mut vertex = write_or_fail!(vertex, "vertex buffer binds not writable");
            let mut fragment = write_or_fail!(fragment, "fragment buffer binds not writable");
            // Both results are dropped; the fragment binds get no strides.
            let _ = get_render_buffers(
                hold,
                m,
                rc,
                &mut vertex,
                flags.max_vertex_buffer_bind_count,
                strides.as_mut(),
                RenderStage::Vertex,
            )?;
            let _ = get_render_buffers(
                hold,
                m,
                rc,
                &mut fragment,
                flags.max_fragment_buffer_bind_count,
                None,
                RenderStage::Fragment,
            )?;
        }
        let barrier = u32::from(rc.is_some_and(|c| c.has_barrier()));
        let _ = command.write_at(u64::from(layout.barrier_offset), &barrier);
        let arguments = u64::from(layout.command_arguments_offset);
        let factor_offset = u64::from(layout.tessellation_factor_offset);
        match RenderCommand::try_from(ty) {
            Ok(RenderCommand::Empty) => {}
            Ok(RenderCommand::Draw) => {
                let args: DrawArgs = rc.and_then(|c| c.draw_args()).unwrap_or_default();
                let draw = DrawPrimitives {
                    primitive_type: args.primitive_type.low_u16(),
                    vertex_start: args.vertex_start,
                    vertex_count: args.vertex_count,
                    instance_count: args.instance_count,
                    base_instance: args.base_instance,
                    field_22: [0; 2],
                };
                let _ = command.write_at(arguments, &draw);
            }
            Ok(RenderCommand::DrawIndexed) => {
                let a = rc.and_then(|c| c.draw_indexed_args()).unwrap_or_default();
                let ib = fill_buffer_bind(hold, m, a.index_buffer_gpu_virtual_address);
                let draw = DrawIndexedPrimitives {
                    primitive_type: a.primitive_type.low_u16(),
                    index_type: a.index_type.low_u16(),
                    index_buffer_reference: ib.buffer_reference,
                    index_count: a.index_count,
                    index_buffer_gpu_address: ib.gpu_address,
                    index_buffer_offset: ib.offset,
                    instance_count: a.instance_count,
                    base_vertex: a.base_vertex.cast_signed(),
                    base_instance: a.base_instance,
                };
                let _ = command.write_at(arguments, &draw);
            }
            Ok(RenderCommand::DrawPatches) => {
                let a = rc.and_then(|c| c.draw_patches_args()).unwrap_or_default();
                let t = rc
                    .and_then(|c| c.tessellation_factor_args())
                    .unwrap_or_default();
                let pib = fill_buffer_bind(hold, m, a.patch_index_buffer_gpu_virtual_address);
                let fb = fill_buffer_bind(hold, m, t.gpu_virtual_address);
                let draw = DrawPatches {
                    number_of_patch_control_points: a.number_of_patch_control_points.low_u16(),
                    patch_start: a.patch_start,
                    patch_count: a.patch_count,
                    patch_index_buffer_reference: pib.buffer_reference,
                    patch_index_buffer_gpu_address: pib.gpu_address,
                    patch_index_buffer_offset: pib.offset,
                    instance_count: a.instance_count,
                    base_instance: a.base_instance,
                    field_36: [0; 2],
                };
                let factor = tessellation_factor(&t, fb);
                let _ = command.write_at(arguments, &draw);
                let _ = command.write_at(factor_offset, &factor);
            }
            Ok(RenderCommand::DrawIndexedPatches) => {
                let a = rc
                    .and_then(|c| c.draw_indexed_patches_args())
                    .unwrap_or_default();
                let t = rc
                    .and_then(|c| c.tessellation_factor_args())
                    .unwrap_or_default();
                let pib = fill_buffer_bind(hold, m, a.patch_index_buffer_gpu_virtual_address);
                let cpib =
                    fill_buffer_bind(hold, m, a.control_point_index_buffer_gpu_virtual_address);
                let fb = fill_buffer_bind(hold, m, t.gpu_virtual_address);
                let draw = DrawIndexedPatches {
                    number_of_patch_control_points: a.number_of_patch_control_points.low_u16(),
                    patch_start: a.patch_start,
                    patch_count: a.patch_count,
                    patch_index_buffer_reference: pib.buffer_reference,
                    patch_index_buffer_gpu_address: pib.gpu_address,
                    patch_index_buffer_offset: pib.offset,
                    control_point_index_buffer_reference: cpib.buffer_reference,
                    control_point_index_buffer_gpu_address: cpib.gpu_address,
                    control_point_index_buffer_offset: cpib.offset,
                    instance_count: a.instance_count,
                    base_instance: a.base_instance,
                    field_4a: [0; 2],
                };
                let factor = tessellation_factor(&t, fb);
                let _ = command.write_at(arguments, &draw);
                let _ = command.write_at(factor_offset, &factor);
            }
            Err(other) => {
                return Ok(Err(format!("unknown indirect render command type {other}")));
            }
        }
    }
    Ok(Ok(()))
}

pub(super) fn synchronize_compute<P: Platform, M: IcbWalkOps<P>>(
    hold: Hold<'_>,
    m: &M,
    buffer: &P::Icb,
    layout: &IndirectCommandLayout,
    flags: &IndirectCommandBufferFlags,
    mut cursor: WritableCursor,
) -> RaiseResult<Result<(), String>> {
    let mut i: u32 = 0;
    while u64::from(i) < buffer.size() {
        let index = u64::from(i);
        i = i.wrapping_add(1);
        let mut command = write_or_fail!(
            cursor.writable_sub_cursor(u64::from(layout.command_size)),
            "command not writable"
        );
        let cc = buffer.compute_command(index)?;
        let cc = cc.as_ref();
        let ty = cc.map_or(0, |c| c.command_type()).low_u32();
        // As in the render walk.
        write_or_fail!(
            command.write_at(u64::from(layout.command_type_offset), &ty),
            "command type not writable"
        );
        let kind = ComputeCommand::try_from(ty);
        // An empty compute slot writes only its type.
        if kind == Ok(ComputeCommand::Empty) {
            continue;
        }
        if !flags.inherit_pipeline_state() {
            let identifier = cc.map_or(0, |c| c.pipeline_state_unique_identifier());
            let reference = pipeline_reference(m, identifier);
            // As for the command type.
            write_or_fail!(
                command.write_at(u64::from(layout.pipeline_state_offset), &reference),
                "command's pipeline state not writable"
            );
        }
        if !flags.inherit_buffers() {
            let mut strides = None;
            // Sized by the vertex bind count.
            if flags.support_dynamic_attribute_stride() {
                strides = Some(write_or_fail!(
                    command.writable_sub_cursor_at(
                        u64::from(layout.attribute_stride_offset),
                        flags.max_vertex_buffer_bind_count.wrapping_mul(WORD_SIZE),
                    ),
                    "attribute strides not writable"
                ));
            }
            let mut kernel = write_or_fail!(
                command.writable_sub_cursor_at(
                    u64::from(layout.kernel_buffer_bind_offset),
                    flags.max_kernel_buffer_bind_count.wrapping_mul(BIND_SIZE),
                ),
                "kernel buffer binds not writable"
            );
            let _ = get_kernel_buffers(
                hold,
                m,
                cc,
                &mut kernel,
                flags.max_kernel_buffer_bind_count,
                strides.as_mut(),
            )?;
        }
        // One word a length, as the prepare reads them: a cursor over all of them first, whose
        // failure ends the walk, then each length in turn, a failed write dropped.
        let lengths_count = flags.max_kernel_threadgroup_memory_bind_count;
        if lengths_count != 0 {
            let mut lengths = write_or_fail!(
                command.writable_sub_cursor_at(
                    u64::from(layout.threadgroup_memory_length_offset),
                    lengths_count.wrapping_mul(WORD_SIZE),
                ),
                "threadgroup memory lengths not writable"
            );
            let mut j: u32 = 0;
            while u64::from(j) < lengths_count {
                let length = match cc {
                    Some(c) => c.threadgroup_memory_length(u64::from(j))?,
                    None => 0,
                };
                let _ = lengths.write(&length);
                j = j.wrapping_add(1);
            }
        }
        let barrier = u32::from(cc.is_some_and(|c| c.has_barrier()));
        let _ = command.write_at(u64::from(layout.barrier_offset), &barrier);
        let arguments = u64::from(layout.command_arguments_offset);
        match kind {
            Ok(ComputeCommand::DispatchThreadgroups) => {
                let a = cc
                    .and_then(|c| c.dispatch_threadgroups_args())
                    .unwrap_or_default();
                let d = ConcurrentDispatch {
                    grid: Triple::of_size(a.threadgroups_per_grid),
                    threads_per_threadgroup: Triple::of_size(a.threads_per_threadgroup),
                };
                let _ = command.write_at(arguments, &d);
            }
            Ok(ComputeCommand::DispatchThreads) => {
                let a = cc
                    .and_then(|c| c.dispatch_threads_args())
                    .unwrap_or_default();
                let d = ConcurrentDispatch {
                    grid: Triple::of_size(a.threads_per_grid),
                    threads_per_threadgroup: Triple::of_size(a.threads_per_threadgroup),
                };
                let _ = command.write_at(arguments, &d);
            }
            // An empty slot went on above
            Ok(ComputeCommand::Empty) | Err(_) => {
                return Ok(Err(format!("unknown indirect compute command type {ty}")));
            }
        }
        let region = cc.map_or_else(Default::default, |c| c.stage_in_region());
        let block = cc.and_then(|c| c.image_block_args()).unwrap_or_default();
        let dispatch = Dispatch {
            stage_in_region_size: Triple::of_size(region.size),
            stage_in_region_origin: Triple::of_origin(region.origin),
            imageblock_width: block.width.low_u32(),
            imageblock_height: block.height.low_u32(),
        };
        let _ = command.write_at(
            u64::from(layout.kernel_dispatch_arguments_offset),
            &dispatch,
        );
    }
    Ok(Ok(()))
}

#[cfg(test)]
mod tests {
    use crate::platform::icb::IcbRenderCmd;
    use zerocopy::IntoBytes;

    use super::super::prepare::{prepare_compute, prepare_render};
    use super::super::testing::*;
    use super::*;
    use std::sync::Arc;

    use crate::platform::Handle;
    use crate::platform::resource::Buffer;
    use crate::resources::manager::records::IndirectCommandBufferBind;
    use crate::testkit::FakePlatform;
    use crate::testkit::guest_ram::RawRegion;

    mod render {
        use super::*;

        #[test]
        fn a_prepared_draw_round_trips() {
            let mut o = FakeOps::new();
            let b5 = o.resident_buffer(5, 0x100);
            let b6 = o.resident_buffer(6, 0x100);
            let rp = o.render_pipeline();
            o.renders.push((9, Lookup::Found(Some(rp.clone_ref()))));
            o.pipeline_refs.push((rp.unique_id_value(), 9));
            let icb = o.icb(1);
            let mut c = Cmds::new(1);
            draw_command(&mut c, 0);
            let f = render_flags(0b100);
            prepare_render(Hold::NONE, &o, &icb, &layout(), &f, c.cursor())
                .unwrap()
                .unwrap();
            let (mem, w) = writable(CMD as u64);
            synchronize_render(Hold::NONE, &o, &icb, &layout(), &f, w)
                .unwrap()
                .unwrap();
            assert_eq!(read_back::<u32>(&mem, 0), DRAW);
            assert_eq!(read_back::<u32>(&mem, 4), 1);
            assert_eq!(read_back::<u32>(&mem, 8), 9);
            let v0: IndirectCommandBufferBind = read_back(&mem, 0x10);
            assert_eq!(
                (v0.buffer_reference, v0.gpu_address, v0.offset),
                (5, b5.gpu_address() + 0x10, 0x10)
            );
            let v1: IndirectCommandBufferBind = read_back(&mem, 0x24);
            assert_eq!(
                (v1.buffer_reference, v1.gpu_address, v1.offset),
                (6, b6.gpu_address() + 0x20, 0x20)
            );
            let fr: IndirectCommandBufferBind = read_back(&mem, 0x40);
            assert_eq!(
                (fr.buffer_reference, fr.gpu_address, fr.offset),
                (5, b5.gpu_address() + 8, 8)
            );
            assert_eq!(read_back::<u64>(&mem, 0x58), 0x30);
            assert_eq!(read_back::<u64>(&mem, 0x60), 0x40);
            assert_eq!(
                read_back::<DrawPrimitives>(&mem, 0x90).as_bytes(),
                c.0[0x90..0x90 + 0x24].as_ref()
            );
        }

        #[test]
        fn an_unknown_pipeline_and_unresolved_addresses_write_zero_and_all_ones() {
            let o = FakeOps::new();
            let icb = o.icb(1);
            let rc = icb.render_command(0).unwrap().unwrap();
            rc.encode(IcbRenderCmd::SetVertexBuffer {
                buffer: None,
                offset: 0x5000,
                index: 0,
            })
            .unwrap();
            let f = render_flags(0);
            let (mem, w) = writable(CMD as u64);
            synchronize_render(Hold::NONE, &o, &icb, &layout(), &f, w)
                .unwrap()
                .unwrap();
            assert_eq!(read_back::<u32>(&mem, 8), 0);
            let v0: IndirectCommandBufferBind = read_back(&mem, 0x10);
            assert_eq!({ v0.buffer_reference }, u32::MAX);
            assert_eq!({ v0.offset }, 0);
            assert!(o.calls().contains(&"get_pipeline_ref(0)".to_owned()));
        }

        #[test]
        fn an_unknown_command_type_fails_after_the_barrier() {
            let o = FakeOps::new();
            let icb = o.icb(1);
            let rc = icb.render_command(0).unwrap().unwrap();
            rc.encode(IcbRenderCmd::SetBarrier).unwrap();
            rc.set_command_type(0x10);
            let (mem, w) = writable(CMD as u64);
            let e = synchronize_render(Hold::NONE, &o, &icb, &layout(), &render_flags(0b11), w)
                .unwrap()
                .unwrap_err();
            assert_eq!(e, "unknown indirect render command type 16");
            assert_eq!(read_back::<u32>(&mem, 0), 0x10);
            assert_eq!(read_back::<u32>(&mem, 4), 1);
        }

        #[test]
        fn a_type_outside_the_command_fails() {
            // The walk ends with the write's own error.
            let o = FakeOps::new();
            let icb = o.icb(1);
            let mut l = layout();
            l.command_type_offset = (CMD - 2) as u16;
            let (_mem, w) = writable(CMD as u64);
            let e = synchronize_render(Hold::NONE, &o, &icb, &l, &render_flags(0b11), w)
                .unwrap()
                .unwrap_err();
            assert!(e.starts_with("command type not writable: "), "{e}");
        }

        #[test]
        fn a_short_cursor_fails() {
            let o = FakeOps::new();
            let icb = o.icb(2);
            let (_mem, w) = writable(CMD as u64);
            let e = synchronize_render(Hold::NONE, &o, &icb, &layout(), &render_flags(0b11), w)
                .unwrap()
                .unwrap_err();
            assert!(e.starts_with("command not writable: "), "{e}");
        }

        #[test]
        fn a_failed_stride_write_does_not_stop_the_loop() {
            // A stride write's error is dropped
            let o = FakeOps::new();
            let (mem, mut w) = writable(0x28);
            let (_s, mut strides) = writable(8);
            get_render_buffers::<FakePlatform, _>(
                Hold::NONE,
                &o,
                None,
                &mut w,
                2,
                Some(&mut strides),
                RenderStage::Vertex,
            )
            .unwrap()
            .unwrap();
            let b1: IndirectCommandBufferBind = read_back(&mem, 0x14);
            assert_eq!({ b1.buffer_reference }, u32::MAX);
        }

        #[test]
        fn a_patch_draw_and_its_tessellation_factor_round_trip() {
            let mut o = FakeOps::new();
            let b5 = o.resident_buffer(5, 0x100);
            let _b6 = o.resident_buffer(6, 0x100);
            let icb = o.icb(1);
            let rc = icb.render_command(0).unwrap().unwrap();
            rc.encode(IcbRenderCmd::DrawPatches {
                number_of_patch_control_points: 3,
                patch_start: 1,
                patch_count: 2,
                patch_index_buffer: Some(&b5),
                patch_index_buffer_offset: 8,
                instance_count: 4,
                base_instance: 5,
                buffer: Some(&b5),
                offset: 0x18,
                instance_stride: 0x40,
            })
            .unwrap();
            let (mem, w) = writable(CMD as u64);
            synchronize_render(Hold::NONE, &o, &icb, &layout(), &render_flags(0b11), w)
                .unwrap()
                .unwrap();
            let d: DrawPatches = read_back(&mem, 0x90);
            assert_eq!(
                (
                    { d.number_of_patch_control_points },
                    { d.patch_start },
                    { d.patch_count },
                    { d.patch_index_buffer_reference },
                    { d.patch_index_buffer_offset },
                    { d.instance_count },
                    { d.base_instance }
                ),
                (3, 1, 2, 5, 8, 4, 5)
            );
            let t: TessellationFactor = read_back(&mem, 0x70);
            assert_eq!(
                ({ t.buffer_reference }, { t.buffer_offset }, {
                    t.instance_stride
                }),
                (5, 0x18, 0x40)
            );
        }
    }

    mod compute {
        use super::*;

        /// One prepared compute command of type `ty`, synchronized: what the walk wrote.
        fn sync_one(ty: u32) -> Arc<RawRegion> {
            let mut o = FakeOps::new();
            let _b5 = o.resident_buffer(5, 0x100);
            let _b6 = o.resident_buffer(6, 0x100);
            let icb = o.icb(1);
            let mut c = Cmds::new(1);
            compute_command(&mut c, 0, ty);
            let f = compute_flags(0b11);
            prepare_compute(Hold::NONE, &o, &icb, &layout(), &f, c.cursor())
                .unwrap()
                .unwrap();
            let (mem, w) = writable(CMD as u64);
            synchronize_compute(Hold::NONE, &o, &icb, &layout(), &f, w)
                .unwrap()
                .unwrap();
            mem
        }

        #[test]
        fn both_dispatch_kinds_round_trip_their_sizes() {
            for ty in [CONCURRENT_DISPATCH, CONCURRENT_DISPATCH_THREADS] {
                let mem = sync_one(ty);
                assert_eq!(read_back::<u32>(&mem, 0), ty);
                let d: ConcurrentDispatch = read_back(&mem, 0x90);
                assert_eq!(d.grid, Triple { a: 1, b: 2, c: 3 });
                assert_eq!(d.threads_per_threadgroup, Triple { a: 4, b: 5, c: 6 });
            }
        }

        #[test]
        fn a_prepared_dispatch_round_trips_every_length() {
            let mut o = FakeOps::new();
            let b5 = o.resident_buffer(5, 0x100);
            let _b6 = o.resident_buffer(6, 0x100);
            let cp = o.compute_pipeline();
            o.computes.push((9, Lookup::Found(Some(cp.clone_ref()))));
            o.pipeline_refs.push((cp.unique_id_value(), 9));
            let icb = o.icb(1);
            let mut c = Cmds::new(1);
            compute_command(&mut c, 0, CONCURRENT_DISPATCH);
            let mut f = compute_flags(0b100);
            f.max_vertex_buffer_bind_count = 2;
            prepare_compute(Hold::NONE, &o, &icb, &layout(), &f, c.cursor())
                .unwrap()
                .unwrap();
            let (mem, w) = writable(CMD as u64);
            synchronize_compute(Hold::NONE, &o, &icb, &layout(), &f, w)
                .unwrap()
                .unwrap();
            assert_eq!(read_back::<u32>(&mem, 0), CONCURRENT_DISPATCH);
            assert_eq!(read_back::<u32>(&mem, 8), 9);
            let k0: IndirectCommandBufferBind = read_back(&mem, 0x100);
            assert_eq!(
                (k0.buffer_reference, k0.gpu_address, k0.offset),
                (5, b5.gpu_address() + 0x10, 0x10)
            );
            assert_eq!(read_back::<u64>(&mem, 0x58), 0x30);
            assert_eq!(read_back::<u64>(&mem, 0x60), 0x40);
            // Each length at its own word, as the prepare read them.
            assert_eq!(read_back::<u64>(&mem, 0x140), 0x100);
            assert_eq!(read_back::<u64>(&mem, 0x148), 0x200);
            let d: ConcurrentDispatch = read_back(&mem, 0x90);
            assert_eq!(d.grid, Triple { a: 1, b: 2, c: 3 });
            assert_eq!(d.threads_per_threadgroup, Triple { a: 4, b: 5, c: 6 });
            let x: Dispatch = read_back(&mem, 0x160);
            assert_eq!(x.stage_in_region_size, Triple { a: 7, b: 8, c: 9 });
            assert_eq!(x.stage_in_region_origin, Triple { a: 1, b: 1, c: 0 });
            assert_eq!((x.imageblock_width, x.imageblock_height), (16, 32));
        }

        #[test]
        fn strides_are_sized_by_the_vertex_count() {
            // One vertex slot for two kernel binds; the second stride write fails and stops the
            // loop, its bind already written.
            let mut o = FakeOps::new();
            let _b5 = o.resident_buffer(5, 0x100);
            let _b6 = o.resident_buffer(6, 0x100);
            let icb = o.icb(1);
            let mut c = Cmds::new(1);
            compute_command(&mut c, 0, CONCURRENT_DISPATCH);
            let mut f = compute_flags(0b111);
            f.bits = 0b100;
            prepare_compute(Hold::NONE, &o, &icb, &layout(), &f, c.cursor())
                .unwrap()
                .unwrap();
            f.max_vertex_buffer_bind_count = 1;
            let (mem, w) = writable(CMD as u64);
            synchronize_compute(Hold::NONE, &o, &icb, &layout(), &f, w)
                .unwrap()
                .unwrap();
            assert_eq!(read_back::<u64>(&mem, 0x58), 0x30);
            assert_eq!(read_back::<u64>(&mem, 0x60), 0);
            assert_eq!(
                { read_back::<IndirectCommandBufferBind>(&mem, 0x114).buffer_reference },
                6
            );
            assert_eq!(o.log().count("getKernelAttributeStrideAtIndex:"), 2);
        }

        /// Lengths that run past the command end the walk before any length is asked for or
        /// written; what came before them is written.
        #[test]
        fn lengths_past_the_command_end_the_walk() {
            let mut o = FakeOps::new();
            let _b5 = o.resident_buffer(5, 0x100);
            let _b6 = o.resident_buffer(6, 0x100);
            let icb = o.icb(1);
            let mut c = Cmds::new(1);
            compute_command(&mut c, 0, CONCURRENT_DISPATCH);
            let f = compute_flags(0b11);
            prepare_compute(Hold::NONE, &o, &icb, &layout(), &f, c.cursor())
                .unwrap()
                .unwrap();
            // 0x140 + 13 words runs past the 0x1a0-byte command.
            let mut far = f;
            far.max_kernel_threadgroup_memory_bind_count = 13;
            let mark = o.log().mark();
            let (mem, w) = writable(CMD as u64);
            let e = synchronize_compute(Hold::NONE, &o, &icb, &layout(), &far, w)
                .unwrap()
                .unwrap_err();
            assert!(
                e.starts_with("threadgroup memory lengths not writable: "),
                "{e}"
            );
            assert!(
                !o.log()
                    .since(mark)
                    .iter()
                    .any(|c| c.sel == "getThreadgroupMemoryLengthAtIndex:")
            );
            assert_eq!(read_back::<u32>(&mem, 0), CONCURRENT_DISPATCH);
            assert_eq!(read_back::<u64>(&mem, 0x140), 0);
            // The arguments come after the lengths: not written.
            let d: ConcurrentDispatch = read_back(&mem, 0x90);
            assert_eq!(d.grid, Triple { a: 0, b: 0, c: 0 });
            // Twelve fit exactly.
            let mut fits = f;
            fits.max_kernel_threadgroup_memory_bind_count = 12;
            let (mem, w) = writable(CMD as u64);
            synchronize_compute(Hold::NONE, &o, &icb, &layout(), &fits, w)
                .unwrap()
                .unwrap();
            assert_eq!(read_back::<u64>(&mem, 0x140), 0x100);
            assert_eq!(read_back::<u64>(&mem, 0x148), 0x200);
        }

        #[test]
        fn an_empty_slot_writes_only_its_type() {
            let o = FakeOps::new();
            let icb = o.icb(1);
            let (mem, w) = writable(CMD as u64);
            synchronize_compute(Hold::NONE, &o, &icb, &layout(), &compute_flags(0), w)
                .unwrap()
                .unwrap();
            assert_eq!(mem.read_vec(0, CMD as u64).unwrap(), vec![0; CMD]);
            assert!(o.calls().is_empty());
        }

        #[test]
        fn an_unknown_type_fails() {
            let o = FakeOps::new();
            let icb = o.icb(1);
            icb.compute_command(0).unwrap().unwrap().set_command_type(1);
            let (_mem, w) = writable(CMD as u64);
            let e = synchronize_compute(Hold::NONE, &o, &icb, &layout(), &compute_flags(0b11), w)
                .unwrap()
                .unwrap_err();
            assert_eq!(e, "unknown indirect compute command type 1");
        }

        #[test]
        fn a_failed_stride_write_stops_the_loop() {
            // Unlike a vertex stride's, a failed kernel stride write ends the loop.
            let o = FakeOps::new();
            let (_mem, mut w) = writable(0x28);
            let (_s, mut strides) = writable(8);
            let e = get_kernel_buffers::<FakePlatform, _>(
                Hold::NONE,
                &o,
                None,
                &mut w,
                2,
                Some(&mut strides),
            )
            .unwrap()
            .unwrap_err();
            assert!(e.starts_with("attribute stride not writable: "), "{e}");
        }
    }

    #[test]
    fn with_no_manager_nothing_is_written_for_an_empty_icb() {
        let o = FakeOps::new();
        let icb = o.icb(0);
        let (mem, w) = writable(CMD as u64);
        super::super::synchronize_indirect_command_buffer_on_null::<FakePlatform>(
            Hold::NONE,
            &icb,
            &layout(),
            &render_flags(0),
            w,
        )
        .unwrap()
        .unwrap();
        assert_eq!(mem.read_vec(0, CMD as u64).unwrap(), vec![0; CMD]);
    }
}
