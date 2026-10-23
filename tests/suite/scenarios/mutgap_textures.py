"""Textures, resources and the resource manager at their edges: each scenario
locks one behaviour, as its docstring says (docs/testing.md, sections 4.10,
6.11, 6.19, 6.20)."""
import struct

from pgtest import abi
from pgtest.abi import PAGE
from pgtest.registry import scenario
from pgtest.resources_build import (BC1_RGBA, BGRA8Unorm, Depth32Float_Stencil8, OPT_SHARED, RGBA8Unorm, RTask,
                                    TT_2DArray, TT_3D, TT_Cube, TT_CubeArray, b_copy_buf_tex, b_copy_tex_buf,
                                    b_fill_buf, b_fill_tex_color, backing_rec, cmd_synchronize, desc1, dimension,
                                    iosurface_op, mip, ref_texture_rec, shared_backing_rec, shared_texture_rec,
                                    simple_dim, texture_rec)
from pgtest.scenario import Scenario
from pgtest.task import seg

PATTERN = bytes((i * 7 + 3) & 0xFF for i in range(0x100))
FULL = (4, 4, 1)
BGRA = 0x42475241  # 'BGRA', an IOSurface pixel format


def _pattern(n, seed):
    return bytes((i * 13 + seed * 29 + (i >> 8)) & 0xFF for i in range(n))


@scenario
def mutgap_tex_compressed_dims():
    """The texture dimension check's compressed path on a BC1_RGBA 8x8
    texture (4x4 blocks of 8 bytes), each record listed in its own exec
    beside a probe buffer: a block height of 8 is refused at the block
    comparison ("Compressed texture block size mismatch"), though its
    computed size would fit; a mip whose height is 4 (its depth right) and
    one whose depth is 2 (its height right) are refused at the size
    comparison ("Size mismatch at mip(0)"). A well-formed record closes."""
    from scenarios.resources import creation_case
    bcd = desc1(fmt=BC1_RGBA, w=8, h=8)

    def bc(mips, block=(4, 4, 1, 8)):
        return (abi.T_TEXTURE, texture_rec(PAGE, 8, [dimension(mips, block=block, compressed=1)], bcd), None, True)
    cases = [bc([mip(0, 32, 16, 8, 8)], block=(4, 8, 1, 8)),   # block height 8
             bc([mip(0, 32, 16, 8, 4)]),                        # mip height 4
             bc([mip(0, 32, 16, 8, 8, 2)])]                     # mip depth 2
    ok = bc([mip(0, 32, 16, 8, 8)])
    cases.append(ok[:3] + (False,))
    return creation_case("mutgap_tex_compressed_dims", mutgap_tex_compressed_dims.__doc__, cases)


@scenario(host=True)
def mutgap_tex_bc3d_depth_slices():
    """An 8x8x4 BC1 3D texture (type 2, one level, rows of 16 bytes, 2 block
    rows, so 32 bytes a depth slice) over distinct guest bytes. Paging it in
    copies with bytesPerImage = bytesPerRow x ceil(height / blockHeight):
    depth slice z comes from guest offset 32z. Copies out of depth slice 1
    alone and of all four show the guest's bytes in order. Host-keyed:
    whether Metal makes a BC1 3D texture is the host's
    (textures_gap_dimension_branches)."""
    s = Scenario("mutgap_tex_bc3d_depth_slices", host=True, doc=mutgap_tex_bc3d_depth_slices.__doc__)
    s.bring_up()
    t = RTask(s)
    va = t.area(PAGE, contents=_pattern(PAGE, 1))
    tex = t.add(abi.T_TEXTURE, texture_rec(PAGE, va // PAGE, [dimension([mip(0, 128, 16, 8, 8, 4)], block=(4, 4, 1, 8),
                                                                       compressed=1)],
                                           desc1(fmt=BC1_RGBA, w=8, h=8, d=4, tt=TT_3D, usage=1)))
    b, b_va = t.shared(PAGE)
    t.define()
    t.install()
    t.blit(b_copy_tex_buf(tex, b, 0, 16, 32, (8, 8, 1), origin=(0, 0, 1)),
           b_copy_tex_buf(tex, b, 0x40, 16, 32, (8, 8, 4)),
           resources=[(tex, 0), (b, 0)], label="copy depth slice 1, then all four")
    t.dump(b_va, 0x20, "B: depth slice 1")
    t.dump(b_va + 0x40, 0x80, "B+0x40: the four depth slices")
    t.delete()
    s.events("end")
    return s


@scenario
def mutgap_tex_leaf_rewrite():
    """A normal texture (type 2, 4x4 RGBA8Unorm) keeps the transfer buffer
    its first prepare made (it makes one only when it has none). The guest
    then moves the texture's VA page to another RAM page holding other bytes,
    and an exec lists it with invalidateHost (flag 1), so it is paged in
    again from that same buffer. In the blocks setup the host task's mapping
    of the VA page is the old RAM page anyway (memory_gap_table_rewrite); in
    map and isolate a new buffer would walk the table again and read the new
    page, but the kept one still reads the old: every setup copies out the
    old page's bytes."""
    s = Scenario("mutgap_tex_leaf_rewrite", doc=mutgap_tex_leaf_rewrite.__doc__)
    s.bring_up()
    t = RTask(s)
    va = t.area(PAGE, contents=_pattern(0x40, 2))
    tex = t.add(abi.T_TEXTURE, texture_rec(PAGE, va // PAGE, [simple_dim(4, 4)], desc1(usage=3)))
    b, b_va = t.shared(PAGE)
    t.define()
    t.install()
    t.blit(b_copy_tex_buf(tex, b, 0, 16, 64, FULL), resources=[(tex, 0), (b, 0)], label="first copy out")
    other = s.alloc_page()
    s.mem(s.pa(other), _pattern(0x40, 3))
    s.u32(s.pa(t.pt_root, 4 * (va // PAGE)), s.pfn(other) | abi.PTE_WRITABLE)
    t.blit(b_copy_tex_buf(tex, b, 0x40, 16, 64, FULL), resources=[(tex, 1), (b, 0)],
           label="after the leaf moved, listed with invalidateHost")
    t.dump(b_va, 0x80, "B: before and after the leaf moved")
    t.delete()
    s.events("end")
    return s


@scenario(setups=("blocks", "map"))
def mutgap_tex_backing_exact_fit():
    """An IOSurface backing (type 4) in task 0 whose one BGRA plane of 4x4
    (rows of 16 bytes) ends exactly at its allocation length, 0x1000: the
    check is offset + height x bytesPerRow > length, so it is made, and a
    texture over it in task 1 (type 5, commandType 12) copies out the plane's
    guest bytes. A plane one byte further is refused ("Plane overruns
    backingAllocationLength", "Failed to validate plane descriptor") and its
    texture's exec faults. In-process only: a type 5 texture aborts the
    service (textures_backing_ref_service)."""
    from scenarios.textures_shared import define_both, kernel_and_user
    s = Scenario("mutgap_tex_backing_exact_fit", setups=("blocks", "map"), doc=mutgap_tex_backing_exact_fit.__doc__)
    s.bring_up()
    k, t = kernel_and_user(s)
    va = k.area(PAGE, contents=_pattern(PAGE, 4))
    bk = k.add(abi.T_IOSURFACE_BACKING, backing_rec(PAGE, va // PAGE, BGRA, [(PAGE - 64, 4, 4, 16, 4)]))
    over = k.add(abi.T_IOSURFACE_BACKING, backing_rec(PAGE, va // PAGE, BGRA, [(PAGE - 63, 4, 4, 16, 4)]))
    d = desc1(fmt=BGRA8Unorm, options=OPT_SHARED, usage=3)
    r = t.add(abi.T_BACKING_TEXTURE, ref_texture_rec(bk, op=iosurface_op(12, d)))
    r2 = t.add(abi.T_BACKING_TEXTURE, ref_texture_rec(over, op=iosurface_op(12, d)))
    b, b_va = t.shared(PAGE)
    define_both(k, t)
    t.blit(b_copy_tex_buf(r, b, 0, 16, 64, FULL), resources=[(r, 0), (b, 0)], label="the plane ending at 0x1000")
    t.blit(b_fill_buf(b, 0x40, 0x10, 1), resources=[(r2, 0), (b, 0)], fault=True, label="one byte further")
    t.dump(b_va, 0x50, "B")
    t.delete()
    k.delete()
    s.events("end")
    return s


@scenario(setups=("blocks", "map"))
def mutgap_tex_backing_guest_valid():
    """An IOSurface backing (type 4) starts with its guest copy valid. A copy
    out through a type 5 texture prepares it (host-valid now too); the guest
    then writes 0x99 over the plane's bytes, and 53 on task 0 naming the
    backing finds the guest copy valid and writes nothing: the pages keep
    0x99. A second copy out still reads the host copy. In-process only (type
    5)."""
    from scenarios.textures_shared import define_both, kernel_and_user
    s = Scenario("mutgap_tex_backing_guest_valid", setups=("blocks", "map"),
                 doc=mutgap_tex_backing_guest_valid.__doc__)
    s.bring_up()
    k, t = kernel_and_user(s)
    va = k.area(PAGE, contents=bytes(0x100) + PATTERN[:64])
    bk = k.add(abi.T_IOSURFACE_BACKING, backing_rec(PAGE, va // PAGE, BGRA, [(0x100, 4, 4, 16, 4)]))
    r = t.add(abi.T_BACKING_TEXTURE, ref_texture_rec(bk, op=iosurface_op(12, desc1(fmt=BGRA8Unorm, options=OPT_SHARED,
                                                                                   usage=3))))
    b, b_va = t.shared(PAGE)
    define_both(k, t)
    t.blit(b_copy_tex_buf(r, b, 0, 16, 64, FULL), resources=[(r, 0), (b, 0)], label="copy out: prepares the backing")
    k.write(va + 0x100, b"\x99" * 64)
    cmd_synchronize(s, 0, [bk], label="53 on task 0, on the backing")
    k.dump(va + 0x100, 0x40, "the backing's pages after 53")
    t.blit(b_copy_tex_buf(r, b, 0x40, 16, 64, FULL), resources=[(r, 0), (b, 0)], label="copy out again")
    t.dump(b_va, 0x80, "B")
    t.delete()
    k.delete()
    s.events("end")
    return s


@scenario
def mutgap_tex_shared_guest_valid():
    """A shared texture backing (type 17, 4x4 RGBA8Unorm, private) in task 0
    starts with its guest copy valid. A type 16 texture in task 1 reads it
    (the backing is paged in); the guest then writes 0x99 over the backing's
    pages, and 53 on task 0 naming it finds the guest copy valid and writes
    nothing: the pages keep 0x99."""
    from scenarios.textures_shared import define_both, kernel_and_user
    s = Scenario("mutgap_tex_shared_guest_valid", doc=mutgap_tex_shared_guest_valid.__doc__)
    s.bring_up()
    k, t = kernel_and_user(s)
    k_va = k.area(PAGE, contents=PATTERN[:64])
    kb = k.add(abi.T_SHARED_BACKING1, shared_backing_rec(PAGE, k_va // PAGE, [simple_dim(4, 4)],
                                                         desc1(options=0x20, usage=3)))
    st = t.add(abi.T_SHARED_TEXTURE, shared_texture_rec(kb))
    b, b_va = t.shared(PAGE)
    define_both(k, t)
    t.blit(b_copy_tex_buf(st, b, 0, 16, 64, FULL), resources=[(st, 0), (b, 0)], label="read the shared texture")
    k.write(k_va, b"\x99" * 64)
    cmd_synchronize(s, 0, [kb], label="53 on task 0, on the backing")
    k.dump(k_va, 0x40, "the backing's pages after 53")
    t.dump(b_va, 0x40, "B")
    t.delete()
    k.delete()
    s.events("end")
    return s


def _shared_slices(name, doc, dims, desc, nslices, bps, fill_slice, host=False):
    """A type 17 shared backing in task 0 over nslices x bps distinct guest
    bytes (one pattern a slice), read from task 1 through a type 16
    texture: every slice copied out, then one slice filled blue with
    invalidateGuest only and 53 on task 0 pages the backing out."""
    from scenarios.textures_shared import define_both, kernel_and_user
    s = Scenario(name, host=host, doc=doc)
    s.bring_up()
    k, t = kernel_and_user(s)
    n = nslices * bps
    k_va = k.area(max(PAGE, n), contents=b"".join(_pattern(bps, 10 + i) for i in range(nslices)))
    kb = k.add(abi.T_SHARED_BACKING1, shared_backing_rec(max(PAGE, n), k_va // PAGE, dims, desc))
    st = t.add(abi.T_SHARED_TEXTURE, shared_texture_rec(kb))
    b, b_va = t.shared(PAGE)
    define_both(k, t)
    t.blit(*[b_copy_tex_buf(st, b, 0x40 * i, 16, 64, FULL, slice_=i) for i in range(nslices)],
           resources=[(st, 0), (b, 0)], label="copy out every slice")
    t.dump(b_va, 0x40 * nslices, "B: the slices in order")
    t.blit(b_fill_tex_color(st, FULL, (0, 0, 1.0, 1.0), RGBA8Unorm, slice_=fill_slice), resources=[(st, 2), (b, 0)],
           label="fill slice %d, invalidateGuest only" % fill_slice)
    cmd_synchronize(s, 0, [kb], label="53 on task 0, on the backing")
    k.dump(k_va, n, "the backing's pages after 53")
    t.delete()
    k.delete()
    s.events("end")
    return s


@scenario
def mutgap_tex_shared_2d_array():
    """A shared texture backing (type 17) that is a 2D array of two 4x4
    RGBA8Unorm slices, 64 bytes apart: a non-cube slice pages to texture
    slice s from guest offset bytesPerSlice x s. Both slices copy out through
    a type 16 texture as the guest wrote them; slice 1 filled blue pages out
    to guest offset 0x40."""
    return _shared_slices("mutgap_tex_shared_2d_array", mutgap_tex_shared_2d_array.__doc__,
                          [simple_dim(4, 4, array=2, slice_bytes=64)],
                          desc1(tt=TT_2DArray, array=2, options=0x20, usage=3), 2, 64, 1)


@scenario
def mutgap_tex_shared_cube_array():
    """A shared texture backing (type 17) that is a cube array of two cubes
    of 4x4 RGBA8Unorm faces, 64 bytes a face: face f of array slice s pages
    to texture slice 6s + f at guest offset 64 x (6s + f). All twelve copy
    out as the guest wrote them; texture slice 11 filled blue pages out to
    guest offset 0x2c0."""
    return _shared_slices("mutgap_tex_shared_cube_array", mutgap_tex_shared_cube_array.__doc__,
                          [simple_dim(4, 4, array=2, cube=1, slice_bytes=64)],
                          desc1(tt=TT_CubeArray, array=2, options=0x20, usage=3), 12, 64, 11)


def _ds_cube_dims(arrays, depth_base=0, stencil_base=0x400):
    """Depth32Float_Stencil8 cube faces of 4x4: plane 0 (4 bytes a texel,
    64 a face) at depth_base, plane 1 (1 byte a texel, 16 a face) at
    stencil_base, each face bytesPerSlice after the last."""
    return [dimension([mip(0, 64, 16, 4, 4)], array=arrays, cube=1, base=depth_base, slice_bytes=64,
                      block=(1, 1, 1, 4)),
            dimension([mip(0, 16, 4, 4, 4)], array=arrays, cube=1, base=stencil_base, slice_bytes=16,
                      block=(1, 1, 1, 1))]


def _depth_faces(n):
    """n faces of 16 distinct depth floats in [0, 1)."""
    return struct.pack("<%df" % (16 * n), *[(i + 1) / 256.0 for i in range(16 * n)])


@scenario
def mutgap_tex_dual_plane_cube_array():
    """A two-plane Depth32Float_Stencil8 texture (type 12) that is a cube
    array of two cubes of 4x4 faces: face f of array slice s pages to and
    from texture slice 6s + f at guest offset bytesPerSlice x (6s + f), a
    plane at a time. Copying texture slice 7 (array slice 1, face 1) out
    shows that face's guest depth and stencil bytes; new stencil bytes copied
    into it with flags 6 page out to its stencil face at +0x470."""
    s = Scenario("mutgap_tex_dual_plane_cube_array", doc=mutgap_tex_dual_plane_cube_array.__doc__)
    s.bring_up()
    t = RTask(s)
    va = t.area(2 * PAGE, contents=_depth_faces(12))
    t.write(va + 0x400, bytes(range(0x40, 0x40 + 12 * 16)))
    d = t.add(abi.T_DUAL_PLANE_TEXTURE, texture_rec(2 * PAGE, va // PAGE, _ds_cube_dims(2),
                                                    desc1(fmt=Depth32Float_Stencil8, tt=TT_CubeArray, array=2,
                                                          usage=3)))
    b, b_va = t.shared(PAGE)
    t.write(b_va + 0x200, bytes(range(0xA0, 0xB0)))
    t.define()
    t.install()
    t.blit(b_copy_tex_buf(d, b, 0, 16, 64, FULL, slice_=7, options=1),
           b_copy_tex_buf(d, b, 0x40, 4, 16, FULL, slice_=7, options=2),
           resources=[(d, 0), (b, 0)], label="copy texture slice 7's two planes out")
    t.dump(b_va, 0x50, "B: slice 7's depth, then its stencil")
    t.blit(b_copy_buf_tex(b, d, 0x200, 4, 16, FULL, slice_=7, options=2), resources=[(d, 6), (b, 0)],
           label="new stencil bytes into slice 7, synchronize")
    t.dump(va + 0x400, 12 * 16, "the stencil plane's pages")
    t.delete()
    s.events("end")
    return s


@scenario(host=True)
def mutgap_tex_shared_dual_plane_cube():
    """A two-plane Depth32Float_Stencil8 shared texture backing (type 18) in
    task 0 that is one cube of 4x4 faces, used from task 1 through a type 16
    texture. Paging goes over faces 0-5 only: the guest bytes after the sixth
    face of each plane (a 0xEE sentinel) are never read or written. Face 5's
    two planes copy out as the guest wrote them; new stencil bytes copied
    into face 5 with invalidateGuest only, then 53 on task 0, page out, and
    the sentinels stay. Host-keyed: whether Metal makes a shareable
    depth-stencil cube is the host's."""
    from scenarios.textures_shared import define_both, kernel_and_user
    s = Scenario("mutgap_tex_shared_dual_plane_cube", host=True, doc=mutgap_tex_shared_dual_plane_cube.__doc__)
    s.bring_up()
    k, t = kernel_and_user(s)
    k_va = k.area(PAGE, fill=0xEE)
    k.write(k_va, _depth_faces(6))
    k.write(k_va + 0x200, bytes(range(0x40, 0x40 + 6 * 16)))
    kb = k.add(abi.T_SHARED_BACKING2, shared_backing_rec(PAGE, k_va // PAGE, _ds_cube_dims(1, stencil_base=0x200),
                                                         desc1(fmt=Depth32Float_Stencil8, tt=TT_Cube, usage=3)))
    st = t.add(abi.T_SHARED_TEXTURE, shared_texture_rec(kb))
    b, b_va = t.shared(PAGE)
    t.write(b_va + 0x200, bytes(range(0xA0, 0xB0)))
    define_both(k, t)
    t.blit(b_copy_tex_buf(st, b, 0, 16, 64, FULL, slice_=5, options=1),
           b_copy_tex_buf(st, b, 0x40, 4, 16, FULL, slice_=5, options=2),
           resources=[(st, 0), (b, 0)], label="copy face 5's two planes out")
    t.dump(b_va, 0x50, "B: face 5's depth, then its stencil")
    t.blit(b_copy_buf_tex(b, st, 0x200, 4, 16, FULL, slice_=5, options=2), resources=[(st, 2), (b, 0)],
           label="new stencil bytes into face 5, invalidateGuest only")
    cmd_synchronize(s, 0, [kb], label="53 on task 0, on the backing")
    k.dump(k_va + 0x140, 0x80, "the depth plane's pages: face 5 and the sentinel after it")
    k.dump(k_va + 0x250, 0x20, "the stencil plane's pages: face 5 and the sentinel after it")
    t.delete()
    k.delete()
    s.events("end")
    return s


@scenario(setups=("isolate",))
def mutgap_tex_detach_buffer_hole():
    """In the service a buffer is a detach buffer whose backing is replaced
    with the host ranges of its pages when it is prepared.
    memory_gap_ram_holes' layout: an exec lists a buffer whose page is in the
    RAM hole beside a buffer in R4, and only fills the one in R4. The hole
    buffer's backing cannot be replaced ("Failed to replace buffer backing"),
    so the exec faults in its prepare and the fill of the other never lands;
    an exec of the R4 buffer alone then fills it. Isolate only: detach
    buffers are the service's."""
    from scenarios.memory_gaps import FIVE, HOLE, R1, R4, pg
    from pgtest.gaps_memory import Pool, fill_exec, page_of, placed_task
    from pgtest.tasks_lib import XGPUTask, fault_and_resume
    s = Scenario("mutgap_tex_detach_buffer_hole", setups=("isolate",), ram=FIVE,
                 doc=mutgap_tex_detach_buffer_hole.__doc__)
    s.bring_up()
    pool = Pool([pg(R1, 0), pg(R1, 1), pg(R4, 0)])
    leaves = {v: s.alloc_page() for v in range(8)}
    leaves.update({12: page_of(HOLE) + 3, 13: pg(R4, 1)})
    t = placed_task(s, pool, cls=XGPUTask, tid=1, pages=14, depth=2, leaves=leaves)
    hole, _ = t.buffer(length=PAGE, va=12 * PAGE)
    far, _ = t.buffer(length=PAGE, va=13 * PAGE)
    s.fill(R4[0] + PAGE, 0x20, 0x11)
    t.define()
    t.install()
    fill_exec(s, t, far, 0, 0x10, 0x5A, resources=[(hole, 0), (far, 0)], fault=True)
    fault_and_resume(s, "the hole buffer listed beside the R4 buffer")
    s.dump(R4[0] + PAGE, 0x20, "the buffer in R4: not filled")
    fill_exec(s, t, far, 0x10, 0x10, 0x6B)
    s.dump(R4[0] + PAGE, 0x20, "the buffer in R4 after an exec of its own")
    t.delete()
    s.events("end")
    return s


@scenario(setups=("blocks", "map"))
def mutgap_tex_legacy_heap_discard():
    """In-process a heap is a legacy heap (private, paged through a transfer
    buffer). A fill of a buffer placed in it stays on the host. 63 discards
    the heap, which makes it volatile without synchronizing; the next exec
    that lists it prepares it again (needsPrepare), devolatilizes it, finds
    it host-valid and returns without copying the guest pages over it: a copy
    out of the heap buffer still shows the fill, and the guest pages still
    hold their pattern. (Were the OS to purge the volatile heap in between,
    the prepare would page the guest bytes in: under no memory pressure it
    does not.)"""
    from pgtest.resources_build import b_copy_buf_buf, cmd_discard, heap_buffer_rec, heap_rec
    s = Scenario("mutgap_tex_legacy_heap_discard", setups=("blocks", "map"),
                 doc=mutgap_tex_legacy_heap_discard.__doc__)
    s.bring_up()
    t = RTask(s, pages=80)
    h_va = t.area(4 * PAGE, contents=PATTERN)
    h = t.add(abi.T_HEAP, heap_rec(h_va // PAGE, 4 * PAGE, flags=1))
    hb = t.add(abi.T_HEAP_BUFFER, heap_buffer_rec(h, 0, PAGE))
    b, b_va = t.shared(PAGE)
    t.define()
    t.install()
    t.blit(b_fill_buf(hb, 0, 0x10, 0xA1), resources=[(h, 0), (hb, 0), (b, 0)], label="fill the heap buffer")
    cmd_discard(s, 1, [h], label="63 on the heap")
    t.blit(b_copy_buf_buf(hb, b, 0, 0, 0x20), resources=[(h, 0), (hb, 0), (b, 0)], label="copy the heap buffer out")
    t.dump(b_va, 0x20, "B: the fill kept on the host")
    t.dump(h_va, 0x20, "the heap's guest pages")
    t.delete()
    s.events("end")
    return s


@scenario(setups=("isolate",))
def mutgap_tex_readonly_heap_write():
    """In the service a read-only heap (flag bit 2; argument buffers are on)
    is a detach heap made with the private resource option 0x1000000, and so
    is its cover buffer, which is what the heap is as a blit buffer. A fill
    of the heap as a buffer, and a fill of a buffer placed in it, both
    complete and leave its guest pages as they were (the GPU drops the
    writes, as for a read-only buffer, resources_readonly_buffer); a fill of
    a shared heap lands. Isolate only: in-process a read-only heap is refused
    (resources_heap_setups_inproc)."""
    from pgtest.resources_build import heap_buffer_rec, heap_rec
    s = Scenario("mutgap_tex_readonly_heap_write", setups=("isolate",), doc=mutgap_tex_readonly_heap_write.__doc__)
    s.bring_up()
    t = RTask(s, pages=80)
    ro_va = t.area(4 * PAGE, fill=0x42)
    ro = t.add(abi.T_HEAP, heap_rec(ro_va // PAGE, 4 * PAGE, 4))
    rob = t.add(abi.T_HEAP_BUFFER, heap_buffer_rec(ro, PAGE, PAGE))
    sh_va = t.area(4 * PAGE, fill=0x31)
    sh = t.add(abi.T_HEAP, heap_rec(sh_va // PAGE, 4 * PAGE, 0))
    b, b_va = t.shared(PAGE)
    t.define()
    t.install()
    t.blit(b_fill_buf(ro, 0, 0x10, 0x3C), b_fill_buf(b, 0, 0x10, 1), resources=[(ro, 0), (b, 0)],
           label="fill the read-only heap as a buffer")
    t.blit(b_fill_buf(rob, 0, 0x10, 0x3D), b_fill_buf(b, 0x10, 0x10, 2), resources=[(ro, 0), (rob, 0), (b, 0)],
           label="fill a buffer in the read-only heap")
    t.blit(b_fill_buf(sh, 0, 0x10, 0x3E), b_fill_buf(b, 0x20, 0x10, 3), resources=[(sh, 0), (b, 0)],
           label="fill the shared heap as a buffer")
    t.dump(ro_va, 0x20, "the read-only heap's pages at 0")
    t.dump(ro_va + PAGE, 0x20, "the read-only heap's pages at 0x1000")
    t.dump(sh_va, 0x20, "the shared heap's pages at 0")
    t.dump(b_va, 0x30, "B: every exec ran")
    t.delete()
    s.events("end")
    return s


@scenario(setups=("map", "isolate"))
def mutgap_tex_mapper_buffer_twice():
    """A mapper-ref buffer (type 15) over mapped surface 1, listed twice in
    one exec: the second entry's prepare finds the buffer made and returns,
    so the surface is retained once. After 37 deletes the buffer, the
    IOSurface host device unmaps the surface: the ring entry's status is
    success (the device refuses to unmap a surface still retained elsewhere).
    The IOSurface host device needs a memory map: map and isolate."""
    from pgtest.iosfc_guest import CMD_UNMAP
    from pgtest.resources_build import b_copy_buf_buf, cmd_delete_resource, mapper_ref_rec
    from scenarios.resources_mapper import _surfaces
    s = Scenario("mutgap_tex_mapper_buffer_twice", setups=("map", "isolate"), iosfc=True,
                 doc=mutgap_tex_mapper_buffer_twice.__doc__)
    s.bring_up()
    d, pages = _surfaces(s, {1: (64, 16, 4, b"BGRA", None)})
    t = RTask(s)
    m = t.add(abi.T_MAPPER_BUFFER, mapper_ref_rec(1))
    b, b_va = t.shared(PAGE)
    t.define()
    t.install()
    t.blit(b_copy_buf_buf(m, b, 0, 0, 0x20), resources=[(m, 0), (m, 0), (b, 0)], label="listed twice")
    t.dump(b_va, 0x20, "B: the surface's page")
    cmd_delete_resource(s, 1, m, label="37 on the mapper buffer")
    idx = d.submit((CMD_UNMAP, 1))
    d.status(idx, "unmap after 37")
    t.delete()
    s.events("end")
    return s


def _reservations(name, doc, places):
    """A service task with argument buffers and a placement list: sampler
    S2 (argument-buffer, zero entry) and sampler S1, texture B (zero entry)
    and texture A; places gives S1's and A's entries. One compute exec
    binds S2 then S1 (0xcc), creating them in that order; one blit exec
    copies B then A out, creating them in that order. The placement list
    shows what each was given."""
    from pgtest.compute_objects import Compute as C, sampler_record
    from pgtest.task import seg
    from scenarios.textures_placement import install_with_placements
    s = Scenario(name, setups=("isolate",), host=True, doc=doc)
    s.bring_up()
    t = RTask(s, pages=96)
    s2 = t.add(abi.T_SERIALIZED, sampler_record(arg_buffers=1))
    s1 = t.add(abi.T_SERIALIZED, sampler_record(arg_buffers=1, s_mode=2))
    tb = t.add(abi.T_TEXTURE, texture_rec(PAGE, t.area(PAGE, contents=PATTERN[:64]) // PAGE, [simple_dim(4, 4)],
                                          desc1(usage=0x13)))
    ta = t.add(abi.T_TEXTURE, texture_rec(PAGE, t.area(PAGE, contents=PATTERN[64:128]) // PAGE, [simple_dim(4, 4)],
                                          desc1(usage=0x13)))
    b, b_va = t.shared(PAGE)
    t.define()
    pl = install_with_placements(t, {s1: places["S1"], ta: places["A"]})
    t.dump(pl, 16 * len(t.ol.entries), "placement list before")
    t.reset_cmdbufs()
    cb = t.cmdbuf(seg(abi.ENC_COMPUTE, C.set_samplers(0, [s2, s1])))
    t.exec3([cb], resources=[(b, 0)], label="bind S2, then S1")
    t.blit(b_copy_tex_buf(tb, b, 0, 16, 64, FULL), b_copy_tex_buf(ta, b, 0x40, 16, 64, FULL),
           resources=[(tb, 0), (ta, 0), (b, 0)], label="copy B, then A, out")
    t.dump(b_va, 0x80, "B: B's texels, then A's")
    t.dump(pl, 16 * len(t.ol.entries), "placement list after")
    t.delete()
    s.events("end")
    return s


@scenario(setups=("isolate",), host=True)
def mutgap_tex_placement_reservations():
    """Guest-held indices are reserved when the lists are first mapped (the
    task's rebuild reserves them): a type 7 record whose header is
    commandType 3 (a sampler) with a nonzero resourceID reserves its entry's
    index as a sampler index, and a texture's as a texture index. Sampler S1
    holds {2, 2} and texture A {1, 1}, the indices this host's Metal gives
    the first argument-buffer sampler and the first texture of a fresh
    service (measured with zero entries). B, made first with a zero entry, is
    therefore given index 2 and writes it back, and A is then made at 1. The
    sampler reservation changes nothing on this host: S2 is still given index
    2, as without it, and S1 is made at 2 too; both write {2, 2}. Every
    creation passes the resourceID checks. Host-keyed: the indices are the
    driver's."""
    return _reservations("mutgap_tex_placement_reservations", mutgap_tex_placement_reservations.__doc__,
                         {"S1": (2, 2), "A": (1, 1)})


@scenario(setups=("isolate",), host=True)
def mutgap_tex_placement_delete_function():
    """A deletion (40) of a function (operation 1005) and of a sampler (1003)
    forgets the object and zeroes its 16-byte placement entry in guest
    memory. Both entries hold a guest pattern, installed with opcode 68; an
    exec maps the lists; after the two deletions both entries are zero.
    Isolate only (placements)."""
    from pgtest.compute_objects import function_record, sampler_record
    from pgtest.resources_build import cmd_delete_object
    from scenarios.textures_placement import install_with_placements
    s = Scenario("mutgap_tex_placement_delete_function", setups=("isolate",), host=True,
                 doc=mutgap_tex_placement_delete_function.__doc__)
    s.bring_up()
    t = RTask(s)
    fva = t.reserve(0x2000)
    n = s.mem_file(t.gpa(fva), "shaders/kernel.metallib")
    f = t.add(abi.T_FUNCTION, function_record(fva, n))
    smp = t.add(abi.T_SERIALIZED, sampler_record())
    b, b_va = t.shared(PAGE)
    t.define()
    pl = install_with_placements(t, {f: (0x1111, 0x2222), smp: (0x3333, 0x4444)})
    t.blit(b_fill_buf(b, 0, 0x10, 1), resources=[(b, 0)], label="an exec: the lists are mapped")
    t.dump(pl, 16 * len(t.ol.entries), "placement list before the deletions")
    cmd_delete_object(s, 1, 1005, f, label="40, 1005 on the function")
    cmd_delete_object(s, 1, 1003, smp, label="40, 1003 on the sampler")
    t.dump(pl, 16 * len(t.ol.entries), "placement list after")
    t.delete()
    s.events("end")
    return s


@scenario(setups=("isolate",), host=True)
def mutgap_tex_placement_views_used():
    """Texture views with forced placements {0x40 + i, 0x40 + i}
    (textures_placement_forced's entries) used by a copy out, not only named
    by 314: a view by pixel format (commandType 7), by type (8) and by type
    with a swizzle (27) of a normal texture. Each is made at its index and
    its gpuResourceID matches, so each copy completes with the base texture's
    texels (a blit copies the swizzled view's bytes as they are). Host-keyed:
    the IDs are the driver's."""
    from pgtest.resources_build import ser_view_format, ser_view_type
    from scenarios.textures_placement import install_with_placements
    s = Scenario("mutgap_tex_placement_views_used", setups=("isolate",), host=True,
                 doc=mutgap_tex_placement_views_used.__doc__)
    s.bring_up()
    t = RTask(s, pages=96)
    base = t.add(abi.T_TEXTURE, texture_rec(PAGE, t.area(PAGE, contents=PATTERN[:64]) // PAGE, [simple_dim(4, 4)],
                                            desc1(usage=0x13)))
    views = [t.add(abi.T_SERIALIZER_TEXTURE, ser_view_format(base, RGBA8Unorm)),
             t.add(abi.T_SERIALIZER_TEXTURE, ser_view_type(base, RGBA8Unorm, 2)),
             t.add(abi.T_SERIALIZER_TEXTURE, ser_view_type(base, RGBA8Unorm, 2, swizzle=(5, 4, 3, 2)))]
    b, b_va = t.shared(PAGE)
    t.define()
    pl = install_with_placements(t, {v: (0x40 + i, 0x40 + i) for i, v in enumerate(views)})
    for i, v in enumerate(views):
        t.blit(b_copy_tex_buf(v, b, 0x40 * i, 16, 64, FULL), resources=[(base, 0), (b, 0)],
               label="copy out of view %d" % i)
    t.dump(b_va, 0xC0, "B: the three views' texels")
    t.dump(pl, 16 * len(t.ol.entries), "placement list")
    t.delete()
    s.events("end")
    return s


@scenario
def mutgap_tex_unplaced_textures_sampled():
    """Two normal textures (type 2, 4x1 RGBA8Unorm, shader-read) with no
    placement entry (every in-process task, and a service task without a
    placement list) are made with no resource index, each where Metal puts
    it. Both are listed (made A then B), then a compute pass samples A into
    one buffer and B into another (sample_k, Repeat): each shows its own
    texels."""
    from pgtest.compute_objects import Compute as C, Rig
    import struct as _st
    r = Rig("mutgap_tex_unplaced_textures_sampled", mutgap_tex_unplaced_textures_sampled.__doc__)
    t = r.t
    texs = []
    for words in ((0xff0000ff, 0xff00ff00, 0xffff0000, 0x80808080), (0x11223344, 0x55667788, 0x99aabbcc, 0xddeeff00)):
        va = t.reserve(PAGE)
        t.write(va, _st.pack("<4I", *words))
        ref = t.ol.add(abi.T_TEXTURE, texture_rec(PAGE, va // PAGE, [simple_dim(4, 1)], desc1(w=4, h=1, usage=1)))
        r.buffers.append(ref)
        texs.append(ref)
    out, out_va = r.buffer()
    smp = r.sampler(s_mode=2, t_mode=2)
    pso = r.kernel("shaders/compute_sample.metallib")
    r.ready()
    r.compute(C.set_pipeline(pso), C.set_samplers(0, [smp]),
              C.set_textures(0, [texs[0]]), C.set_buffers(0, [(out, 0)]), C.dispatch_threads((4, 1, 1), (4, 1, 1)),
              C.set_textures(0, [texs[1]]), C.set_buffers(0, [(out, 0x10)]), C.dispatch_threads((4, 1, 1), (4, 1, 1)))
    r.dump(out_va, 0x20, "samples: A, then B")
    return r.end()


@scenario(setups=("isolate",), host=True)
def mutgap_tex_icb_sync_more():
    """The ICB write-back (53 after an exec that lists each ICB with
    invalidateGuest; icb_sync's rig) for what icb_sync's ICBs leave out: an
    indexed-patches command whose baseInstance is 5 and whose tessellation
    factor offset is 8, both written back; a render ICB with neither dynamic
    strides nor inherited buffers, whose vertex bind prepare set without a
    stride and which is written back with no strides cursor; a dynamic-stride
    compute ICB of two kernel binds with strides 4 and 8: the stride cursor
    is sized by the kernel bind count, so whether the descriptor's vertex bind
    count is 2 or none, both binds and both strides, as prepare set them with
    their binds, are written back. GPU addresses are the host's: keyed by the
    host. Isolate
    only: in-process the write-back aborts (icb_sync_in_process)."""
    from scenarios.icb import ICBTask, sync_refs
    s = Scenario("mutgap_tex_icb_sync_more", setups=("isolate",), host=True, doc=mutgap_tex_icb_sync_more.__doc__)
    s.bring_up()
    w = ICBTask(s)
    O = w.out
    idx, _ = w.t.buffer(PAGE, contents=struct.pack("<3H", 0, 1, 2))
    from pgtest.exec_objects import DYNAMIC_STRIDE, ICB_DRAW_INDEXED_PATCHES
    patches = w.icb([render_cmd_patch(w, idx)], kind="render", command_types=ICB_DRAW_INDEXED_PATCHES,
                    flags=DYNAMIC_STRIDE)
    from pgtest.icb_cmds import compute_cmd, render_cmd
    plain = w.icb([render_cmd(pso=w.rpso, vbinds=[(O, 0, 0x40)], fbinds=[(w.tb, 0, 0x50)])], kind="render")
    from pgtest.exec_objects import ICB_DISPATCH_THREADS, icb_descriptor, icb_record
    from pgtest.icb_cmds import CMD, layout
    two_va = w.t.reserve(PAGE)
    # A compute ICB whose descriptor also says 2 vertex binds, which the
    # write-back's stride cursor does not use.
    two = w.icb([compute_cmd(pso=w.cpso, binds=[(O, 0, 0x30), (w.tb, 0, 0x40)], strides=[4, 8], grid=(4, 1, 1),
                             tg=(4, 1, 1))], va=two_va,
                record=icb_record(CMD, two_va // PAGE, icb_descriptor(ICB_DISPATCH_THREADS, vbufs=2, kbufs=2,
                                                                      flags=DYNAMIC_STRIDE),
                                  layout("compute"), max_count=1))
    one = w.icb([compute_cmd(pso=w.cpso, binds=[(O, 0, 0x30), (w.tb, 0, 0x40)], strides=[4, 8], grid=(4, 1, 1),
                             tg=(4, 1, 1))], flags=DYNAMIC_STRIDE, kbufs=2)
    icbs = [patches, plain, two, one]
    w.start()
    w.t.exec_segments([seg(abi.ENC_BLIT)], resources=w.res() + [(idx, 0)] +
                      [(i, abi.RES_INVALIDATE_GUEST) for i in icbs], label="prepared, guest invalidated")
    s.fault_state("prepared")
    for i in icbs:
        va, size = w.icbs[i]
        w.t.fill(va, size, 0xEE)
    sync_refs(s, w.t.tid, icbs, label="synchronize")
    for i, name in zip(icbs, ("indexed patches", "render, no strides", "compute, two strided binds, 2 vertex binds",
                                 "compute, two strided binds, no vertex binds")):
        va, size = w.icbs[i]
        w.t.dump(va, size, "written back: " + name)
    s.events("end")
    return s


def render_cmd_patch(w, idx):
    from pgtest.icb_cmds import draw_indexed_patches_args, render_cmd, tess_args
    return render_cmd(ctype=8, pso=w.rpso, args=draw_indexed_patches_args(points=3, count=1, index=(idx, 0, 0),
                                                                          cp_index=(idx, 0, 2), base_instance=5),
                      tess=tess_args(idx, 0, 8, 0, 1.0))


@scenario
def mutgap_tex_icb_reset_again():
    """Prepare resets a render ICB command whose guest type is 0. An ICB's
    command 0 draws the triangle; executed, the target shows it. The guest
    then writes type 0 into command 0 and lists the ICB with invalidateHost,
    so prepare runs again: executed into a freshly cleared target, the
    command is reset and only the clear shows."""
    from pgtest.icb_cmds import render_cmd
    from scenarios.icb import ICBTask
    s = Scenario("mutgap_tex_icb_reset_again", doc=mutgap_tex_icb_reset_again.__doc__)
    s.bring_up()
    w = ICBTask(s)
    O = w.out
    a = w.icb([render_cmd(pso=w.rpso, vbinds=[(O, 0, 0)], fbinds=[(O, 0, 0x10)])], kind="render")
    w.start()
    w.t.write(w.tb_va, bytes(0x40))
    w.run_render(a, 0, 1, label="the draw")
    w.pixels("the draw")
    va, _ = w.icbs[a]
    w.t.write(va, render_cmd(ctype=0, pso=w.rpso, vbinds=[(O, 0, 0)], fbinds=[(O, 0, 0x10)]))
    w.t.write(w.tb_va, bytes(0x40))
    w.run_render(a, 0, 1, flags=abi.RES_INVALIDATE_HOST, label="type 0, prepared again")
    w.pixels("type 0, prepared again")
    s.events("end")
    return s

