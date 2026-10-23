"""Indirect command buffers (object type 20): creation (operation 54), the
guest's command layouts prepared into Metal by the CPU, execution from
compute and render passes, the blit commands that edit them, and the CPU
write-back of their commands into guest memory (docs/testing.md, sections
4.3 W13, 4.10, 6.19).

What an ICB holds is read back by executing it: a compute ICB dispatches
double_gid (fixtures/shaders/kernel.metal: out[gid] = gid * 2) into a shared
buffer, a render ICB draws the fixture triangle (colour {1, 0.5, 0.25, 1})
into a texture over a shared buffer.
"""
import struct

from pgtest import abi
from pgtest.exec_objects import (ICB_DISPATCH, ICB_DISPATCH_THREADS, ICB_DRAW, ICB_DRAW_INDEXED,
                                 ICB_DRAW_INDEXED_PATCHES, ICB_DRAW_PATCHES, INH_BUFFERS, INH_PIPELINE,
                                 DYNAMIC_STRIDE, ObjTask, buffer_texture_record, c_buffers, c_pipeline,
                                 icb_descriptor, icb_record, info_ref, r_pipeline, render_pass, submit_last)
from pgtest.icb_cmds import CMD, compute_cmd, layout, render_cmd
from pgtest.registry import scenario
from pgtest.scenario import Scenario
from pgtest.task import rec, seg

PAGE = abi.PAGE
BLIT, COMPUTE, RENDER, INFO = abi.ENC_BLIT, abi.ENC_COMPUTE, abi.ENC_RENDER, abi.ENC_INFO


def exec_ranged(icb, loc, n):
    """Compute 0xe4 ExecuteCommandsInBufferRanged, packed {u32 icb, u64
    location, u64 length}."""
    return rec(0xE4, struct.pack("<IQQ", icb, loc, n))


def exec_indirect(icb, range_buf, off):
    """Compute 0xe5 ExecuteCommandsInBuffer {u32 icb, u32 range buffer, u64
    offset}; the range buffer holds {u32 location, u32 length}."""
    return rec(0xE5, struct.pack("<IIQ", icb, range_buf, off))


def use_resources(*refs, usage=3):
    """Compute 0x87 UseResources {u32 count, u32 usage}, then the refs."""
    return rec(0x87, struct.pack("<II", len(refs), usage) + b"".join(struct.pack("<I", r) for r in refs))


class ICBTask:
    """A task with an output buffer, a texture over a second buffer, ICB
    pipelines, and room for ICBs: each ICB's commands live at their own
    task pages (icb()), written before define."""

    def __init__(self, s, pages=96, **kw):
        t = self.t = ObjTask(s, pages=pages, **kw)
        self.out, self.out_va = t.buffer(PAGE)
        self.tb, self.tb_va = t.buffer(PAGE)
        self.tex = t.ol.add(abi.T_SERIALIZER_TEXTURE, buffer_texture_record(self.tb))
        self.rng, self.rng_va = t.buffer(PAGE)
        self.cpso = t.compute_pipeline(icb=True)
        self.rpso = t.render_pipeline(icb=True)
        self.cpso_plain = t.compute_pipeline()
        self.icbs = {}

    def icb(self, commands, kind="compute", command_types=None, max_count=None, flags=0, kbufs=1, vbufs=1,
            fbufs=1, ktg=0, lay=None, length=None, va=None, record=None):
        """An ICB whose guest memory holds commands (each CMD bytes); returns
        its reference."""
        t = self.t
        n = max_count if max_count is not None else len(commands)
        size = max(n * CMD, CMD)
        if va is None:
            va = t.reserve(max(size, PAGE))
        data = b"".join(commands)
        if data:
            t.write(va, data)
        if command_types is None:
            command_types = ICB_DISPATCH_THREADS if kind == "compute" else ICB_DRAW
        if kind == "compute":
            desc = icb_descriptor(command_types, kbufs=kbufs, ktg=ktg, flags=flags)
        else:
            desc = icb_descriptor(command_types, vbufs=vbufs, fbufs=fbufs, flags=flags)
        rec_ = record if record is not None else icb_record(size if length is None else length, va // PAGE, desc,
                                                            lay if lay is not None else layout(kind), max_count=n)
        ref = t.ol.add(abi.T_ICB, rec_)
        self.icbs[ref] = (va, size)
        return ref

    def start(self):
        self.t.define()
        self.t.install()

    def res(self, *icbs, flags=0):
        return [(self.out, 0), (self.tb, 0), (self.rng, 0)] + [(i, flags) for i in icbs]

    def run_compute(self, icb, loc=0, n=None, label=None, fault=False, flags=0, extra=b"", res=None):
        t = self.t
        n = n if n is not None else 1
        body = use_resources(self.out) + extra + exec_ranged(icb, loc, n)
        t.exec_segments([seg(COMPUTE, body)], resources=res if res is not None else self.res(icb, flags=flags),
                        fault=fault, label=label)
        t.s.fault_state(label)
        if fault:
            t.s.resume_root()

    def run_render(self, icb, loc=0, n=1, label=None, fault=False, flags=0, clear=(0.0, 0.0, 1.0, 1.0), extra=b"",
                   res=None):
        t = self.t
        body = render_pass(self.tex, clear=clear) + extra + rec(0x15, struct.pack("<IQQ", icb, loc, n))
        t.exec_segments([seg(RENDER, body)], resources=res if res is not None else self.res(icb, flags=flags),
                        fault=fault, label=label)
        t.s.fault_state(label)
        if fault:
            t.s.resume_root()

    def out_dump(self, label, n=0x80, off=0):
        self.t.dump(self.out_va + off, n, label)

    def clear_out(self):
        self.t.write(self.out_va, bytes(0x400))

    def pixels(self, label):
        self.t.dump(self.tb_va, 0x40, label)


@scenario
def icb_compute_execute():
    """A compute ICB prepared by the CPU from the guest's commands
    (prepareIndirectComputeCommandBuffer) and executed from a compute pass
    (0xe4 ranged, 0xe5 through a range buffer): each command's pipeline,
    kernel buffer bind (reference and offset) and dispatch size reach the
    GPU; a command of type 0 is reset and dispatches nothing; a range
    executes only its commands."""
    s = Scenario("icb_compute_execute", doc=icb_compute_execute.__doc__)
    s.bring_up()
    w = ICBTask(s)
    O = w.out
    icb = w.icb([compute_cmd(pso=w.cpso, binds=[(O, 0, 0)], grid=(8, 1, 1), tg=(8, 1, 1)),
                 compute_cmd(pso=w.cpso, binds=[(O, 0, 0x40)], grid=(4, 1, 1), tg=(4, 1, 1), barrier=1),
                 compute_cmd(ctype=0, pso=w.cpso, binds=[(O, 0, 0x80)]),
                 compute_cmd(ctype=0x20, pso=w.cpso, binds=[(O, 0, 0xC0)], grid=(2, 1, 1), tg=(4, 1, 1))],
                command_types=ICB_DISPATCH_THREADS | ICB_DISPATCH)
    w.t.write(w.rng_va, struct.pack("<II", 1, 1))
    w.start()
    w.run_compute(icb, 0, 4, label="execute all four")
    w.out_dump("all four", 0x100)
    w.clear_out()
    w.run_compute(icb, 1, 1, label="execute command 1")
    w.out_dump("command 1", 0x100)
    w.clear_out()
    w.t.exec_segments([seg(COMPUTE, use_resources(O) + exec_indirect(icb, w.rng, 0))], resources=w.res(icb),
                      label="execute through a range buffer {1, 1}")
    s.fault_state("0xe5")
    w.out_dump("0xe5", 0x100)
    s.events("end")
    return s


@scenario
def icb_compute_variants():
    """What else a compute ICB's layout carries (prepare): an ICB that
    inherits its pipeline or its buffers from the encoder (the command's
    own are not read), threadgroup memory lengths, dynamic attribute
    strides, a pipeline reference of 0 or naming something that is not a
    pipeline (skipped silently), and a bind that names no buffer
    (reference ~0 with GPU address 0 or 1: skipped). Each result read back
    after executing its ICB."""
    s = Scenario("icb_compute_variants", doc=icb_compute_variants.__doc__)
    s.bring_up()
    w = ICBTask(s)
    O = w.out
    inh_p = w.icb([compute_cmd(pso=0xDEAD, binds=[(O, 0, 0)], grid=(4, 1, 1), tg=(4, 1, 1))], flags=INH_PIPELINE)
    inh_b = w.icb([compute_cmd(pso=w.cpso, binds=[(0xDEAD, 0, 0)], grid=(4, 1, 1), tg=(4, 1, 1))], flags=INH_BUFFERS)
    tgm = w.icb([compute_cmd(pso=w.cpso, binds=[(O, 0, 0x10)], tg_lengths=[64, 128], grid=(4, 1, 1), tg=(4, 1, 1))],
                ktg=2)
    stride = w.icb([compute_cmd(pso=w.cpso, binds=[(O, 0, 0x20)], strides=[4], grid=(4, 1, 1), tg=(4, 1, 1))],
                   flags=DYNAMIC_STRIDE)
    nopso = w.icb([compute_cmd(pso=0, binds=[(O, 0, 0x30)], grid=(4, 1, 1), tg=(4, 1, 1)),
                   compute_cmd(pso=O, binds=[(O, 0, 0x40)], grid=(4, 1, 1), tg=(4, 1, 1)),
                   compute_cmd(pso=w.cpso, binds=[(0xFFFFFFFF, 0, 0x50)], grid=(4, 1, 1), tg=(4, 1, 1)),
                   compute_cmd(pso=w.cpso, binds=[(0xFFFFFFFF, 1, 0x60)], grid=(4, 1, 1), tg=(4, 1, 1)),
                   compute_cmd(pso=w.cpso, binds=[(0xFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0)], grid=(4, 1, 1), tg=(4, 1, 1)),
                   compute_cmd(pso=w.cpso, binds=[(O, 0, 0x70)], grid=(4, 1, 1), tg=(4, 1, 1))])
    w.start()
    w.run_compute(inh_p, extra=c_pipeline(w.cpso), label="inherits the pipeline set by 0xd0")
    w.out_dump("inherit pipeline", 0x10)
    w.clear_out()
    w.run_compute(inh_b, extra=c_buffers(0, [(O, 0x80)]), label="inherits the buffer set by 0xcb")
    w.out_dump("inherit buffers", 0x10, off=0x80)
    w.run_compute(tgm, label="threadgroup memory lengths")
    w.run_compute(stride, label="dynamic attribute stride")
    w.run_compute(nopso, 0, 6, extra=c_pipeline(w.cpso_plain), label="pipeline 0, a buffer as pipeline, unfound binds")
    w.out_dump("results", 0x80)
    s.events("end")
    return s


@scenario
def icb_render_execute():
    """A render ICB prepared by the CPU (prepareIndirectRenderCommandBuffer)
    and executed from a render pass (render 0x15): a draw; a command of type
    0, reset; an indexed draw through an index buffer named by reference; an
    indexed draw whose index buffer is not found (skipped, and the next
    command still prepared); patch draws whose patch index or tessellation
    factor buffer is not found (skipped before Metal is called); vertex and
    fragment buffer binds; an ICB that inherits pipeline and buffers from
    the encoder. The target's pixels say whether a triangle was drawn
    (fixture colour) or only cleared."""
    from pgtest.icb_cmds import draw_indexed_args, draw_indexed_patches_args, draw_patches_args, tess_args
    s = Scenario("icb_render_execute", doc=icb_render_execute.__doc__)
    s.bring_up()
    w = ICBTask(s)
    O = w.out
    idx, idx_va = w.t.buffer(PAGE, contents=struct.pack("<3H", 0, 1, 2))
    NF = (0xFFFFFFFF, 0, 0)
    draw = w.icb([render_cmd(pso=w.rpso, vbinds=[(O, 0, 0)], fbinds=[(O, 0, 0x10)], barrier=1)], kind="render")
    reset = w.icb([render_cmd(ctype=0, pso=w.rpso)], kind="render")
    indexed = w.icb([render_cmd(ctype=2, pso=w.rpso, args=draw_indexed_args(index_ref=idx))], kind="render",
                    command_types=ICB_DRAW_INDEXED)
    unfound = w.icb([render_cmd(ctype=2, pso=w.rpso, args=draw_indexed_args(index_ref=NF[0])),
                     render_cmd(ctype=1, pso=w.rpso)], kind="render", command_types=ICB_DRAW | ICB_DRAW_INDEXED)
    patches = w.icb([render_cmd(ctype=4, pso=w.rpso, args=draw_patches_args(index=NF)),
                     render_cmd(ctype=4, pso=w.rpso, args=draw_patches_args(index=(idx, 0, 0)), tess=tess_args(*NF)),
                     render_cmd(ctype=8, pso=w.rpso, args=draw_indexed_patches_args(index=NF)),
                     render_cmd(ctype=8, pso=w.rpso, args=draw_indexed_patches_args(index=(idx, 0, 0), cp_index=NF)),
                     render_cmd(ctype=8, pso=w.rpso, args=draw_indexed_patches_args(index=(idx, 0, 0),
                                                                                    cp_index=(idx, 0, 0)),
                                tess=tess_args(*NF))],
                    kind="render", command_types=ICB_DRAW_PATCHES | ICB_DRAW_INDEXED_PATCHES)
    inherit = w.icb([render_cmd(pso=0xBAD, vbinds=[(0xBAD, 0, 0)])], kind="render", flags=INH_PIPELINE | INH_BUFFERS)
    stride = w.icb([render_cmd(pso=w.rpso, vbinds=[(O, 0, 0)], strides=[16])], kind="render", flags=DYNAMIC_STRIDE)
    w.start()
    for ref, n, label in ((draw, 1, "a draw"), (reset, 1, "type 0"), (indexed, 1, "an indexed draw"),
                          (unfound, 2, "index buffer not found, then a draw"),
                          (patches, 5, "patch draws whose buffers are not found"),
                          (stride, 1, "a dynamic attribute stride")):
        w.t.write(w.tb_va, bytes(0x40))
        w.run_render(ref, 0, n, label=label, extra=use_resources(idx, usage=1), res=w.res(ref) + [(idx, 0)])
        w.pixels(label)
    w.t.write(w.tb_va, bytes(0x40))
    body = (render_pass(w.tex) + r_pipeline(w.rpso) + rec(0x15, struct.pack("<IQQ", inherit, 0, 1)))
    w.t.exec_segments([seg(RENDER, body)], resources=w.res(inherit), label="inherits from the encoder")
    s.fault_state("inherit")
    w.pixels("inherit")
    s.events("end")
    return s


@scenario
def icb_prepare_errors():
    """Where prepare stops (prepareIndirect{Render,Compute}CommandBuffer): a
    field the layout places past commandSize fails its read ("Failed to read
    command type / pipeline state / attribute strides / vertex, fragment or
    kernel buffers / threadgroup memories / barrier / draw / indexed draw /
    patch draw / tessellation args / dispatch / command dispatch args"), a
    command past the ICB's guest range fails ("Failed to read command"), and
    an unknown command type is refused. Each failure returns an error that
    prepareWithCPU only logs: the exec runs, and commands encoded before the
    failure stay encoded, so executing each ICB shows how far its prepare
    got. The ICB's guest range outside the task: "Failed to make cursor for
    prepare"."""
    from pgtest.icb_cmds import draw_indexed_args, draw_indexed_patches_args, draw_patches_args
    s = Scenario("icb_prepare_errors", doc=icb_prepare_errors.__doc__)
    s.bring_up()
    w = ICBTask(s, pages=128)
    O = w.out
    FAR = 0x1000  # past commandSize (0x100)
    ok_c = lambda off: compute_cmd(pso=w.cpso, binds=[(O, 0, off)], grid=(2, 1, 1), tg=(2, 1, 1))
    ok_r = render_cmd(pso=w.rpso)
    cases = []

    def c(label, cmds, **kw):
        cases.append(("compute", label, w.icb(cmds, **kw), len(cmds)))

    def r(label, cmds, **kw):
        cases.append(("render", label, w.icb(cmds, kind="render", **kw), len(cmds)))

    c("compute type past the command", [ok_c(0)], lay=layout("compute", commandTypeOffset=0xFFFF))
    c("compute pipeline past the command", [ok_c(0)], lay=layout("compute", pipelineStateOffset=FAR))
    c("compute strides past the command", [ok_c(0)], flags=DYNAMIC_STRIDE, lay=layout("compute", attributeStrideOffset=FAR))
    c("kernel buffers past the command", [ok_c(0)], lay=layout("compute", kernelBufferBindOffset=FAR))
    c("threadgroup memories past the command", [ok_c(0)], ktg=1, lay=layout("compute", threadgroupMemoryLengthOffset=FAR))
    c("compute barrier past the command", [ok_c(0)], lay=layout("compute", barrierOffset=0xFFFF))
    c("dispatch past the command", [ok_c(0)], lay=layout("compute", commandArgumentsOffset=FAR))
    c("threadgroups dispatch past the command", [compute_cmd(ctype=0x20, pso=w.cpso, binds=[(O, 0, 0)])],
      command_types=ICB_DISPATCH, lay=layout("compute", commandArgumentsOffset=FAR))
    c("dispatch args past the command", [ok_c(0)], lay=layout("compute", kernelDispatchArgumentsOffset=0xFFFF))
    c("compute type 1, after a good command", [ok_c(0x40), compute_cmd(ctype=1, pso=w.cpso)])
    c("the second command past the range", [ok_c(0x80), ok_c(0xC0)], length=CMD + CMD // 2)
    r("render type past the command", [ok_r], lay=layout("render", commandTypeOffset=0xFFFF))
    r("render pipeline past the command", [ok_r], lay=layout("render", pipelineStateOffset=FAR))
    r("render strides past the command", [ok_r], flags=DYNAMIC_STRIDE, lay=layout("render", attributeStrideOffset=FAR))
    r("vertex buffers past the command", [ok_r], lay=layout("render", vertexBufferBindOffset=FAR))
    r("fragment buffers past the command", [ok_r], lay=layout("render", fragmentBufferBindOffset=FAR))
    r("render barrier past the command", [ok_r], lay=layout("render", barrierOffset=0xFFFF))
    r("draw past the command", [ok_r], lay=layout("render", commandArgumentsOffset=FAR))
    r("indexed draw past the command", [render_cmd(ctype=2, pso=w.rpso, args=draw_indexed_args())],
      command_types=ICB_DRAW_INDEXED, lay=layout("render", commandArgumentsOffset=FAR))
    r("patch draw past the command", [render_cmd(ctype=4, pso=w.rpso, args=draw_patches_args())],
      command_types=ICB_DRAW_PATCHES, lay=layout("render", commandArgumentsOffset=FAR))
    r("indexed patches past the command", [render_cmd(ctype=8, pso=w.rpso, args=draw_indexed_patches_args())],
      command_types=ICB_DRAW_INDEXED_PATCHES, lay=layout("render", commandArgumentsOffset=FAR))
    r("tessellation past the command", [render_cmd(ctype=4, pso=w.rpso, args=draw_patches_args(index=(O, 0, 0)))],
      command_types=ICB_DRAW_PATCHES, lay=layout("render", tessellationFactorOffset=0xFFFF))
    r("indexed patches tessellation past the command",
      [render_cmd(ctype=8, pso=w.rpso, args=draw_indexed_patches_args(index=(O, 0, 0), cp_index=(O, 0, 0)))],
      command_types=ICB_DRAW_INDEXED_PATCHES, lay=layout("render", tessellationFactorOffset=0xFFFF))
    r("render type 0x10", [render_cmd(ctype=0x10, pso=w.rpso)])
    r("the second render command past the range", [ok_r, ok_r], length=CMD + CMD // 2)
    c("an ICB outside the task", [ok_c(0)], va=127 * PAGE, length=4 * PAGE)
    w.start()
    for kind, label, ref, n in cases:
        if kind == "compute":
            w.clear_out()
            w.run_compute(ref, 0, n, label=label)
            w.out_dump(label, 0x100)
        else:
            w.t.write(w.tb_va, bytes(0x40))
            w.run_render(ref, 0, n, label=label)
            w.pixels(label)
    s.events("end")
    return s


@scenario
def icb_create_errors():
    """Records that fail to become an ICB (createIndirectCommandBuffer):
    a record under 12 bytes, a missing operation header, an operation
    other than 54 ("Invalid object type"), an operation under 0x50 bytes
    ("Failed to read descriptor"). Listing one in an exec, or naming it
    in 0xe4 or 465, faults."""
    s = Scenario("icb_create_errors", doc=icb_create_errors.__doc__)
    s.bring_up()
    w = ICBTask(s)
    good = icb_record(CMD, 80, icb_descriptor(ICB_DISPATCH_THREADS, kbufs=1), layout("compute"))
    bad = [("an 11-byte record", good[:11]), ("no operation header", good[:16]),
           ("operation 55", good[:12] + struct.pack("<II", 55, 0) + good[20:]),
           ("an operation of 0x4f bytes", good[:20 + 0x4F])]
    refs = [(label, w.icb([], record=r, max_count=1)) for label, r in bad]
    w.start()
    for label, ref in refs:
        w.t.exec_segments([seg(BLIT)], resources=[(ref, 0)], fault=True, label=label + ": listed")
        s.fault_state(label)
        s.resume_root()
        w.t.exec_segments([seg(COMPUTE, exec_ranged(ref, 0, 1))], resources=w.res(), fault=True,
                          label=label + ": executed")
        s.fault_state(label + ": executed")
        s.resume_root()
        w.t.exec_segments([seg(INFO, info_ref(465, ref, w.out, 0))], fault=True, label=label + ": 465")
        s.fault_state(label + ": 465")
        s.resume_root()
    s.events("end")
    return s


def copy_icb(src, dst, loc, n, index):
    """Blit 305 CopyIndirectCommandBuffer {u32 source, u32 destination, u64
    source range location, length, u64 destination index}."""
    return rec(305, struct.pack("<IIQQQ", src, dst, loc, n, index))


def optimize_icb(icb, loc, n):
    """Blit 312 OptimizeIndirectCommandBuffer, packed {u32 icb, u64
    location, u64 length}."""
    return rec(312, struct.pack("<IQQ", icb, loc, n))


def reset_icb(icb, loc, n):
    """Blit 313 ResetCommandsInBuffer, as 312."""
    return rec(313, struct.pack("<IQQ", icb, loc, n))


@scenario
def icb_blit_commands():
    """The blit commands on ICBs: 305 copies commands from one ICB into
    another at an index, 312 optimizes a range, 313 resets a range; each
    checks its range against the ICB's size, 305 its destination index,
    and each its references ("Got invalid indirect command buffer").
    Executing the destination shows what it holds."""
    s = Scenario("icb_blit_commands", doc=icb_blit_commands.__doc__)
    s.bring_up()
    w = ICBTask(s)
    O = w.out
    a = w.icb([compute_cmd(pso=w.cpso, binds=[(O, 0, 0)], grid=(2, 1, 1), tg=(2, 1, 1)),
               compute_cmd(pso=w.cpso, binds=[(O, 0, 0x10)], grid=(2, 1, 1), tg=(2, 1, 1))])
    b = w.icb([compute_cmd(ctype=0), compute_cmd(ctype=0), compute_cmd(ctype=0)])
    w.start()
    both = [(a, 0), (b, 0)]

    def blit(records, label, fault=False):
        w.t.exec_segments([seg(BLIT, records)], resources=w.res() + both, fault=fault, label=label)
        s.fault_state(label)
        if fault:
            s.resume_root()

    blit(copy_icb(a, b, 0, 2, 1), "305: a[0,2) into b at 1")
    w.clear_out()
    w.run_compute(b, 0, 3, res=w.res() + both, label="execute b")
    w.out_dump("b after 305", 0x20)
    blit(optimize_icb(b, 0, 3), "312 over b")
    w.clear_out()
    w.run_compute(b, 0, 3, res=w.res() + both, label="execute b after 312")
    w.out_dump("b after 312", 0x20)
    blit(reset_icb(b, 1, 1), "313: reset b[1]")
    w.clear_out()
    w.run_compute(b, 0, 3, res=w.res() + both, label="execute b after 313")
    w.out_dump("b after 313", 0x20)
    for records, label in ((copy_icb(a, b, 1, 2, 0), "305 source range past a"),
                           (copy_icb(a, b, 0, 1, 3), "305 destination index 3"),
                           (copy_icb(a, b, 0xFFFFFFFFFFFFFFFF, 2, 0), "305 range overflows"),
                           (copy_icb(0, b, 0, 1, 0), "305 source 0"),
                           (copy_icb(a, 0, 0, 1, 0), "305 destination 0"),
                           (optimize_icb(b, 2, 2), "312 range past b"),
                           (optimize_icb(0, 0, 1), "312 reference 0"),
                           (reset_icb(b, 3, 1), "313 range past b"),
                           (reset_icb(0, 0, 1), "313 reference 0"),
                           (reset_icb(w.out, 0, 1), "313 on a buffer")):
        blit(records, label, fault=True)
    s.events("end")
    return s


@scenario
def icb_execute_errors():
    """The compute and render execute commands' checks: an ICB reference 0
    ("Failed to get indirect command buffer"), a range past the ICB's size
    or overflowing, an indirect range buffer reference 0 ("Failed to get
    indirect range buffer"). Each faults."""
    s = Scenario("icb_execute_errors", doc=icb_execute_errors.__doc__)
    s.bring_up()
    w = ICBTask(s)
    icb = w.icb([compute_cmd(pso=w.cpso, binds=[(w.out, 0, 0)])])
    ricb = w.icb([render_cmd(pso=w.rpso)], kind="render")
    w.start()
    for body, label in ((exec_ranged(0, 0, 1), "0xe4 reference 0"), (exec_ranged(icb, 0, 2), "0xe4 range past"),
                        (exec_ranged(icb, 0xFFFFFFFFFFFFFFFF, 2), "0xe4 range overflows"),
                        (exec_indirect(0, w.rng, 0), "0xe5 reference 0"),
                        (exec_indirect(icb, 0, 0), "0xe5 range buffer 0")):
        w.t.exec_segments([seg(COMPUTE, body)], resources=w.res(icb), fault=True, label=label)
        s.fault_state(label)
        s.resume_root()
    for body, label in ((rec(0x15, struct.pack("<IQQ", 0, 0, 1)), "0x15 reference 0"),
                        (rec(0x15, struct.pack("<IQQ", ricb, 1, 1)), "0x15 range past"),
                        (rec(0x14, struct.pack("<IIQ", 0, w.rng, 0)), "0x14 reference 0"),
                        (rec(0x14, struct.pack("<IIQ", ricb, 0, 0)), "0x14 range buffer 0")):
        w.t.exec_segments([seg(RENDER, render_pass(w.tex) + body)], resources=w.res(ricb), fault=True, label=label)
        s.fault_state(label)
        s.resume_root()
    w.t.write(w.rng_va, struct.pack("<II", 0, 1))
    w.t.exec_segments([seg(RENDER, render_pass(w.tex) + rec(0x14, struct.pack("<IIQ", ricb, w.rng, 0)))],
                      resources=w.res(ricb), label="0x14 through a range buffer {0, 1}")
    s.fault_state("0x14")
    w.pixels("0x14")
    s.events("end")
    return s


def sync_refs(s, tid, refs, **kw):
    """SynchronizeResources (53): {u32 taskID, u32 count} + count x u32
    reference."""
    return s.submit(abi.OP_SYNCHRONIZE_RESOURCES,
                    struct.pack("<II", tid, len(refs)) + b"".join(struct.pack("<I", r) for r in refs), **kw)


RO_PAGE = 95


class SyncWorld:
    """ICBs of every kind the write-back (synchronizeIndirect{Render,
    Compute}CommandBuffer) reads out of Metal, each over its own guest
    pages, and a compute ICB over the task's one read-only page
    (RO_PAGE). prepare() lists them all with invalidateGuest in an exec
    that only opens a blit pass: each is created, prepared from its guest
    commands, and marked guest-invalid; the guest's copy is then
    overwritten with 0xee, so what a write-back puts there is all that
    reads back."""

    def __init__(self, s):
        from pgtest.icb_cmds import draw_indexed_args, draw_indexed_patches_args, draw_patches_args, tess_args
        w = self.w = ICBTask(s, pages=96, writable=set(range(96)) - {RO_PAGE})
        O = w.out
        idx, _ = w.t.buffer(PAGE, contents=struct.pack("<3H", 0, 1, 2))
        self.idx = idx
        NF = (0xFFFFFFFF, 0, 0)
        c = self.compute = w.icb(
            [compute_cmd(pso=w.cpso, binds=[(O, 0, 0x10), NF], tg_lengths=[64, 128], barrier=1, grid=(8, 2, 1),
                         tg=(4, 2, 1), stage_size=(1, 2, 3), stage_origin=(4, 5, 6), imageblock=(8, 8)),
             compute_cmd(ctype=0, pso=w.cpso, binds=[(O, 0, 0)]),
             compute_cmd(ctype=0x20, pso=w.cpso, binds=[(w.tb, 0, 0x20)], grid=(3, 1, 1), tg=(2, 1, 1))],
            command_types=ICB_DISPATCH | ICB_DISPATCH_THREADS, kbufs=2, ktg=2)
        st = self.cstride = w.icb([compute_cmd(pso=w.cpso, binds=[(O, 0, 0x30)], strides=[4], grid=(4, 1, 1),
                                               tg=(4, 1, 1))], flags=DYNAMIC_STRIDE)
        ci = self.cinherit = w.icb([compute_cmd(pso=w.cpso, binds=[(O, 0, 0)], grid=(4, 1, 1), tg=(4, 1, 1))],
                                   flags=INH_PIPELINE | INH_BUFFERS)
        r = self.render = w.icb(
            [render_cmd(pso=w.rpso, vbinds=[(O, 0, 0x40)], fbinds=[(w.tb, 0, 0x50)], barrier=1, prim=4, start=1,
                        count=6, instances=2, base_instance=1, strides=[16]),
             render_cmd(ctype=2, pso=w.rpso, args=draw_indexed_args(index_type=0, index_ref=idx, index_off=2,
                                                                     count=2, base_vertex=1, instances=3)),
             render_cmd(ctype=4, pso=w.rpso, args=draw_patches_args(points=4, start=1, count=2, index=(idx, 0, 4)),
                        tess=tess_args(idx, 0, 8, 16, 2.0)),
             render_cmd(ctype=8, pso=w.rpso, args=draw_indexed_patches_args(points=3, count=1, index=(idx, 0, 0),
                                                                             cp_index=(idx, 0, 2)),
                        tess=tess_args(idx, 0, 0, 0, 1.0)),
             render_cmd(ctype=0, pso=w.rpso)],
            kind="render", command_types=ICB_DRAW | ICB_DRAW_INDEXED | ICB_DRAW_PATCHES | ICB_DRAW_INDEXED_PATCHES,
            flags=DYNAMIC_STRIDE)
        ri = self.rinherit = w.icb([render_cmd(pso=w.rpso, vbinds=[(O, 0, 0)])], kind="render",
                                   flags=INH_PIPELINE | INH_BUFFERS)
        ro = self.ro = w.icb([compute_cmd(pso=w.cpso, binds=[(O, 0, 0)])], va=RO_PAGE * PAGE, length=CMD)
        self.all = [c, st, ci, r, ri, ro]
        self.names = {c: "compute", st: "compute, dynamic stride", ci: "compute, inherits",
                      r: "render", ri: "render, inherits", ro: "compute, read-only page"}

    def prepare(self, label, refs=None, extra=()):
        w = self.w
        refs = self.all if refs is None else refs
        w.t.exec_segments([seg(BLIT)], resources=w.res() + [(self.idx, 0)] + [(i, abi.RES_INVALIDATE_GUEST)
                                                                            for i in refs] + list(extra),
                          label=label)
        w.t.s.fault_state(label)
        for i in refs:
            if i != self.ro:
                va, size = w.icbs[i]
                w.t.fill(va, size, 0xEE)

    def dump(self, label, refs=None):
        w = self.w
        for i in (self.all if refs is None else refs):
            va, size = w.icbs[i]
            w.t.dump(va, min(size, 0x500) if i != self.ro else CMD, "%s: %s" % (label, self.names[i]))


@scenario(setups=("isolate",), host=True)
def icb_sync():
    """The write-back (W13): an ICB listed with invalidateGuest in an exec
    is prepared and marked guest-invalid; SynchronizeResources (53) then
    writes each one's commands back into its guest pages through the
    service's writable cursor (synchronizeWithCPU and
    synchronizeIndirect{Render,Compute}CommandBuffer): each command's type,
    pipeline (by reference, getPipelineRef), buffer binds (reference, GPU
    address and offset, fillBufferBind; an unbound slot reads ~0), strides,
    barrier and arguments; a compute slot of type 0 stops after its type, a
    render one does not; an inheriting ICB writes neither pipeline nor
    binds; each threadgroup memory length lands in its own word, where
    prepare reads it. An ICB over a read-only page fails "Failed to make
    cursor for synchronize" and stays as it was. A second 53 finds every
    guest copy valid and writes nothing.
    GPU addresses are the host's: keyed by the host."""
    s = Scenario("icb_sync", setups=("isolate",), host=True, doc=icb_sync.__doc__)
    s.bring_up()
    x = SyncWorld(s)
    w = x.w
    w.start()
    x.prepare("prepared, guest invalidated")
    sync_refs(s, w.t.tid, x.all, label="synchronize")
    x.dump("written back")
    for i in x.all[:-1]:
        va, size = w.icbs[i]
        w.t.fill(va, size, 0xEE)
    sync_refs(s, w.t.tid, x.all, label="synchronize again")
    w.t.dump(w.icbs[x.compute][0], 0x40, "the second synchronize: compute")
    s.events("end")
    return s


@scenario(setups=("isolate",), host=True)
def icb_sync_after_blit():
    """What the write-back reads is Metal's copy, not the guest's: after
    a blit 313 resets command 1 and 305 copies command 0 over command 2,
    a SynchronizeResources (53) writes the reset command's type 0 and
    the copy into the guest. Keyed by the host (GPU addresses)."""
    s = Scenario("icb_sync_after_blit", setups=("isolate",), host=True, doc=icb_sync_after_blit.__doc__)
    s.bring_up()
    w = ICBTask(s)
    O = w.out
    a = w.icb([compute_cmd(pso=w.cpso, binds=[(O, 0, 0x10 * i)], grid=(i + 1, 1, 1), tg=(1, 1, 1)) for i in range(3)])
    w.start()
    w.t.exec_segments([seg(BLIT, reset_icb(a, 1, 1) + copy_icb(a, a, 0, 1, 2))],
                      resources=w.res(a, flags=abi.RES_INVALIDATE_GUEST), label="313, 305")
    s.fault_state("313, 305")
    va, size = w.icbs[a]
    w.t.fill(va, size, 0xEE)
    sync_refs(s, w.t.tid, [a], label="synchronize")
    w.t.dump(va, size, "written back")
    s.events("end")
    return s


@scenario(setups=("isolate",), host=True)
def icb_evict():
    """Suspending flushes every task: the service evicts each resource that
    needs a sync, and a guest-invalid ICB's evictWithCPU writes it back as a
    53 would. The read-only ICB's failure is logged and suspend goes on.
    Keyed by the host."""
    s = Scenario("icb_evict", setups=("isolate",), host=True, doc=icb_evict.__doc__)
    s.bring_up()
    x = SyncWorld(s)
    x.w.start()
    x.prepare("prepared, guest invalidated")
    s.api("willSuspend")
    s.api("finishSuspend")
    x.dump("evicted")
    s.events("end")
    return s


@scenario(setups=("blocks", "map"))
def icb_sync_in_process():
    """In-process, the write-back's cursor comes from the in-process task's
    writable cursor, which aborts: a SynchronizeResources (53) on a
    guest-invalid ICB kills the host. The exec before it prepared the ICB
    and ran it."""
    s = Scenario("icb_sync_in_process", setups=("blocks", "map"), doc=icb_sync_in_process.__doc__)
    s.bring_up()
    w = ICBTask(s)
    a = w.icb([compute_cmd(pso=w.cpso, binds=[(w.out, 0, 0)], grid=(4, 1, 1), tg=(4, 1, 1))])
    w.start()
    w.run_compute(a, flags=abi.RES_INVALIDATE_GUEST, label="execute, guest invalidated")
    w.out_dump("executed", 0x10)
    submit_last(s, abi.OP_SYNCHRONIZE_RESOURCES, struct.pack("<III", w.t.tid, 1, a), label="53: the host dies")
    return s


@scenario(setups=("blocks", "map"))
def icb_evict_in_process():
    """In-process, suspend's flush evicts a guest-invalid ICB through the
    same abort()ing writable cursor: rvg_device_will_suspend and
    rvg_device_finish_suspend kill the host."""
    s = Scenario("icb_evict_in_process", setups=("blocks", "map"), doc=icb_evict_in_process.__doc__)
    s.bring_up()
    w = ICBTask(s)
    a = w.icb([compute_cmd(pso=w.cpso, binds=[(w.out, 0, 0)], grid=(4, 1, 1), tg=(4, 1, 1))])
    w.start()
    w.run_compute(a, flags=abi.RES_INVALIDATE_GUEST, label="execute, guest invalidated")
    s.api("willSuspend")
    s.api("finishSuspend")
    return s


@scenario
def icb_prepare_invalid_guest():
    """An ICB whose host and guest copies are both marked invalid
    (InvalidateResources, 52: clearHost and clearGuest) is prepared from
    its guest pages anyway at its next exec, with only a log ("Trying to
    prepare from invalid guest resource"): the guest changed its command
    in between, and the new one runs. An ICB listed with invalidateHost
    (flag 1) is prepared again the same way, without the log."""
    s = Scenario("icb_prepare_invalid_guest", doc=icb_prepare_invalid_guest.__doc__)
    s.bring_up()
    w = ICBTask(s)
    a = w.icb([compute_cmd(pso=w.cpso, binds=[(w.out, 0, 0)], grid=(2, 1, 1), tg=(2, 1, 1))])
    w.start()
    va, _ = w.icbs[a]
    w.run_compute(a, label="first")
    w.out_dump("first", 0x40)
    w.clear_out()
    w.t.write(va, compute_cmd(pso=w.cpso, binds=[(w.out, 0, 0x20)], grid=(2, 1, 1), tg=(2, 1, 1)))
    w.run_compute(a, label="changed, not invalidated")
    w.out_dump("the old command", 0x40)
    w.clear_out()
    s.submit(abi.OP_INVALIDATE_RESOURCES, struct.pack("<IIIBBBB", w.t.tid, 1, a, 1, 0, 1, 0), label="52: clear host and guest")
    w.run_compute(a, label="after 52")
    w.out_dump("after 52", 0x40)
    w.clear_out()
    w.t.write(va, compute_cmd(pso=w.cpso, binds=[(w.out, 0, 0x30)], grid=(2, 1, 1), tg=(2, 1, 1)))
    w.run_compute(a, flags=abi.RES_INVALIDATE_HOST, label="invalidateHost")
    w.out_dump("after invalidateHost", 0x40)
    s.events("end")
    return s


@scenario(setups=("isolate",), host=True)
def icb_placement():
    """Placements (W8, section 6.11) for ICBs: with an object and placement
    list (opcode 68), the service creates each ICB with its entry
    (newIndirectCommandBufferWithDescriptor). An entry of zeros, or with a
    resourceID but index 0 (not checked), gets the ICB's resourceIndex and
    gpuResourceID written into it; an entry with an index and a resourceID
    the new ICB does not get fails ("Failed to create indirect command
    buffer with correct resourceID"), and the exec listing it faults. Each
    good ICB still executes. Host values: keyed by the host. In-process
    tasks pass no placement."""
    s = Scenario("icb_placement", setups=("isolate",), host=True, doc=icb_placement.__doc__)
    s.bring_up()
    w = ICBTask(s)
    O = w.out
    mk = lambda off: w.icb([compute_cmd(pso=w.cpso, binds=[(O, 0, off)], grid=(2, 1, 1), tg=(2, 1, 1))])
    zero, unchecked, bad = mk(0), mk(0x10), mk(0x20)
    t = w.t
    t.define()
    n = len(t.ol.entries)
    place_va = t.reserve(PAGE)
    entries = bytearray(16 * n)
    struct.pack_into("<QQ", entries, 16 * unchecked, 0, 0x1234)
    struct.pack_into("<QQ", entries, 16 * bad, 5, 0x1234)
    t.write(place_va, bytes(entries))
    t.ol.write()
    t.set_object_and_placement_list(0, place_va, t.ol.length(), 16 * n)
    for ref, label in ((zero, "entry of zeros"), (unchecked, "resourceID without an index")):
        w.run_compute(ref, label=label)
        t.dump(place_va + 16 * ref, 16, label + ": the entry after")
    w.out_dump("executed", 0x20)
    w.run_compute(bad, label="index 5, resourceID 0x1234", fault=True)
    t.dump(place_va + 16 * bad, 16, "index 5, resourceID 0x1234: the entry after")
    s.events("end")
    return s


@scenario(setups=("isolate",), host=True)
def icb_sync_errors():
    """Where the write-back stops
    (synchronizeIndirect{Render,Compute}CommandBuffer). Each target ICB's
    layout places one field past its command (or its guest range is shorter
    than a command), so its own prepare fails and leaves it host-invalid; a
    blit 305 then copies a well-formed ICB's commands into it, and
    InvalidateResources (52: set host valid, clear guest valid) makes it need a
    write-back. The 53 then writes each command until the failing field
    ("Failed to read command", "Failed to write commandType / pipeline state",
    "Failed to get attribute strides / kernel, vertex or fragment buffers
    cursor"): the error is dropped (synchronizeWithCPU ignores it) and the
    guest copy is marked valid. A compute ICB with dynamic strides sizes its
    write-back stride cursor by the kernel bind count, as prepare does: one
    with 31 vertex binds prepares, executes and is written back whole. Keyed
    by the host (GPU addresses)."""
    s = Scenario("icb_sync_errors", setups=("isolate",), host=True, doc=icb_sync_errors.__doc__)
    s.bring_up()
    w = ICBTask(s)
    O = w.out
    FAR = 0x1000
    ca = w.icb([compute_cmd(pso=w.cpso, binds=[(O, 0, 0)], grid=(2, 1, 1), tg=(2, 1, 1), barrier=1)])
    cs = w.icb([compute_cmd(pso=w.cpso, binds=[(O, 0, 0)], strides=[4], grid=(2, 1, 1), tg=(2, 1, 1))],
               flags=DYNAMIC_STRIDE)
    ra = w.icb([render_cmd(pso=w.rpso, vbinds=[(O, 0, 0)], fbinds=[(O, 0, 0x10)])], kind="render")
    rs = w.icb([render_cmd(pso=w.rpso, vbinds=[(O, 0, 0)], strides=[16])], kind="render", flags=DYNAMIC_STRIDE)
    targets = []

    def b(label, src, kind="compute", flags=0, length=None, **lay):
        ref = w.icb([], kind=kind, max_count=1, flags=flags, lay=layout(kind, **lay), length=length)
        targets.append((label, src, ref))

    b("compute, type past the command", ca, commandTypeOffset=0xFFFF)
    b("compute, pipeline past the command", ca, pipelineStateOffset=FAR)
    b("compute, strides past the command", cs, flags=DYNAMIC_STRIDE, attributeStrideOffset=FAR)
    b("compute, kernel buffers past the command", ca, kernelBufferBindOffset=FAR)
    b("compute, half a command", ca, length=CMD // 2)
    b("render, type past the command", ra, "render", commandTypeOffset=0xFFFF)
    b("render, pipeline past the command", ra, "render", pipelineStateOffset=FAR)
    b("render, strides past the command", rs, "render", flags=DYNAMIC_STRIDE, attributeStrideOffset=FAR)
    b("render, vertex buffers past the command", ra, "render", vertexBufferBindOffset=FAR)
    b("render, fragment buffers past the command", ra, "render", fragmentBufferBindOffset=FAR)
    b("render, half a command", ra, "render", length=CMD // 2)
    va31 = w.t.reserve(PAGE)
    v31 = w.icb([compute_cmd(pso=w.cpso, binds=[(O, 0, 0x40)], strides=[4], grid=(2, 1, 1), tg=(2, 1, 1))], va=va31,
                record=icb_record(CMD, va31 // PAGE, icb_descriptor(ICB_DISPATCH_THREADS, vbufs=31, kbufs=1,
                                                                     flags=DYNAMIC_STRIDE), layout("compute")))
    w.start()
    srcs = [ca, cs, ra, rs]
    copies = b"".join(copy_icb(src, ref, 0, 1, 0) for _, src, ref in targets)
    w.t.exec_segments([seg(BLIT, copies)], resources=w.res(*srcs, *(r for _, _, r in targets)),
                      label="prepare (each target fails), 305 into each")
    s.fault_state("305")
    w.run_compute(v31, flags=abi.RES_INVALIDATE_GUEST, label="31 vertex binds, guest invalidated")
    w.out_dump("31 vertex binds: executed", 0x10, off=0x40)
    refs = [r for _, _, r in targets]
    s.submit(abi.OP_INVALIDATE_RESOURCES, struct.pack("<II", w.t.tid, len(refs))
             + b"".join(struct.pack("<IBBBB", r, 0, 1, 1, 0) for r in refs), label="52: set host valid, clear guest")
    for ref in refs + [v31]:
        va, size = w.icbs[ref]
        w.t.fill(va, size, 0xEE)
    sync_refs(s, w.t.tid, refs + [v31], label="synchronize")
    for label, _, ref in targets + [("compute, 31 vertex binds", 0, v31)]:
        va, size = w.icbs[ref]
        w.t.dump(va, CMD, label)
    s.events("end")
    return s


@scenario
def icb_bind_lookups():
    """How prepare resolves a buffer bind (getResourceAndOffset): a reference
    past the object list, reference 0, one naming an object that fails to
    create, a texture, the ICB itself, a buffer whose record lies outside the
    task and one whose pages do are each not found and skip their bind (the
    command still dispatches, writing nowhere). The ICB also appears in a 0x87
    useResources (getObject<id<MTLResource>>, its getResource) and is discarded
    by DiscardResources (63; its discard does nothing), after which it executes
    as prepared: the guest's change to its last bind is not read again."""
    s = Scenario("icb_bind_lookups", doc=icb_bind_lookups.__doc__)
    s.bring_up()
    w = ICBTask(s)
    O = w.out
    bad = w.t.ol.add(10, bytes(16))
    cmd = lambda ref, off: compute_cmd(pso=w.cpso, binds=[(ref, 0, off)], grid=(2, 1, 1), tg=(2, 1, 1))
    rec_out = w.t.ol.add_raw(abi.T_BUFFER, 16, 200 * PAGE)
    pages_out = w.t.ol.add(abi.T_BUFFER, struct.pack("<QIB3x", PAGE, 200, 0))
    icb = w.icb([cmd(0x7FFF, 0), cmd(bad, 0x10), cmd(w.tex, 0x20), cmd(0, 0x30), cmd(rec_out, 0),
                 cmd(pages_out, 0), cmd(O, 0x40)])
    w.t.write(w.icbs[icb][0] + 3 * CMD + 0x10, struct.pack("<I", icb))
    w.start()
    w.run_compute(icb, 0, 7, extra=use_resources(icb), label="unfound binds, then a found one")
    w.out_dump("unfound binds, then a found one", 0x60)
    s.submit(abi.OP_DISCARD_RESOURCES, struct.pack("<III", w.t.tid, 1, icb), label="63: discard the ICB")
    w.clear_out()
    w.t.write(w.icbs[icb][0] + 6 * CMD, cmd(O, 0x50))
    w.run_compute(icb, 0, 7, label="after the discard")
    w.out_dump("after the discard", 0x60)
    s.events("end")
    return s
