"""FIFOs, barriers and ordering in interplay: several FIFOs (the root and
child FIFOs) carrying GPU work that depends on each other, with the result
in guest memory proving the order (docs/testing.md, sections 6.5-6.7,
6.12; decision 8).

What the implementation orders (pgtest/interplay_util.py says more): a FIFO
publishes its stamps in command order; a barrier holds its FIFO until
another FIFO's published slot reaches a value, and an exec's stamp is
published when its command buffer completed, so a gated exec reads what the
exec it waits for wrote; DeviceCore::signal_stamp raises the publisher's
interrupt under the mutex the released wait re-checks under, so the
publisher's interrupt is recorded first. Every scenario here makes the
FIFOs' order total with barriers, or observes only what does not depend on
the interleaving (final stamps, interrupt counts).
"""
from pgtest import abi
from pgtest.abi import PAGE
from pgtest.compute_objects import Blit as B, Compute as C
from pgtest.fifo_util import HEALTHY, NODE_FAULT, NODE_READ, child, child_fault, resume_child, track_stamps
from pgtest.interplay_util import ADD, INC, World, blit, compute, u32s
from pgtest.registry import scenario
from pgtest.task import ObjectList
from pgtest.tasks_lib import XGPUTask

PATTERN = 0x04030201


def ramp_u32(n, start=0):
    return u32s(*[(start + i) & 0xFFFFFFFF for i in range(n)])


def add_one(pso, out, inp, n=64):
    """out[i] = in[i] + 1 for i < n (compute_add.metal)."""
    return compute(C.set_pipeline(pso), C.set_buffers(0, [(out, 0), (inp, 0)]),
                   C.dispatch_threads((n, 1, 1), (n, 1, 1)))


@scenario
def interplay_fifo_blit_chain():
    """A chain of blits across four FIFOs, written in reverse: child 3
    copies C to D gated on child 2's stamp 1, child 2 copies B to C gated
    on child 1's, child 1 copies A to B gated on the root's next stamp. All
    three block (their read counts stay 0, and B, C and D stay zero); the
    root's exec then fills A with a pattern, and the release runs down the
    chain: the interrupts come root, child 1, child 2, child 3, each
    snapshot showing only the FIFOs before it published, and D ends up
    holding the root's pattern."""
    w = World("interplay_fifo_blit_chain", interplay_fifo_blit_chain.__doc__, children=(1, 2, 3))
    s = w.s
    a, a_va = w.buffer()
    b, b_va = w.buffer()
    c, c_va = w.buffer()
    d, d_va = w.buffer()
    w.ready()
    rs = w.f.stamp + 1
    s3 = w.exec(3, blit(B.copy_buffer_to_buffer(c, d, 0, 0, 0x100)), barriers=[(2, 1)], wait=False)
    s2 = w.exec(2, blit(B.copy_buffer_to_buffer(b, c, 0, 0, 0x100)), barriers=[(1, 1)], wait=False)
    s1 = w.exec(1, blit(B.copy_buffer_to_buffer(a, b, 0, 0, 0x100)), barriers=[(0, rs)], wait=False)
    w.blocked(3, s3, "child 3 blocked on child 2")
    w.blocked(2, s2, "child 2 blocked on child 1", timeout_ms=100)
    w.blocked(1, s1, "child 1 blocked on the root", timeout_ms=100)
    for fid in (1, 2, 3):
        s.dump(w.c[fid].node_gpa(NODE_READ), 4, "child %d read count" % fid)
    w.t.dump(b_va, 0x10, "B before")
    w.t.dump(d_va, 0x10, "D before")
    s.events("defined; three children blocked")
    w.exec(0, blit(B.fill_pattern(a, 0, 0x100, PATTERN)), wait=False)
    w.c[3].wait_stamp(s3)
    s.events("root, child 1, child 2, child 3", order=True, irq_stamps=True)
    for label, va in (("A", a_va), ("B", b_va), ("C", c_va), ("D", d_va)):
        w.t.dump(va, 0x110, label)
    w.stamps()
    return s


@scenario
def interplay_fifo_compute_join():
    """A diamond of add_one kernels over four FIFOs: the root computes
    A = S + 1; child 2 C = A + 1 (gated on the root); child 1 B = C + 1
    (gated on child 2); child 3 runs D = B + 1 and E = C + 1 behind two
    barriers, on child 2 and then child 1, so it passes the first as child
    2 publishes and blocks again on the second until child 1 does. Written
    child 3, child 1, child 2, root; released root, child 2, child 1,
    child 3. D = S + 4 and E = S + 3 prove every read came after its
    writer."""
    w = World("interplay_fifo_compute_join", interplay_fifo_compute_join.__doc__, children=(1, 2, 3))
    s = w.s
    src, _ = w.buffer(contents=ramp_u32(64, 0x100))
    bufs = {n: w.buffer() for n in "ABCDE"}
    add = w.kernel(ADD)
    w.ready()
    A, B_, C_, D, E = (bufs[n][0] for n in "ABCDE")
    rs = w.f.stamp + 1
    s3 = w.exec(3, add_one(add, D, B_), add_one(add, E, C_), barriers=[(2, 1), (1, 1)], wait=False)
    s1 = w.exec(1, add_one(add, B_, C_), barriers=[(2, 1)], wait=False)
    s2 = w.exec(2, add_one(add, C_, A), barriers=[(0, rs)], wait=False)
    w.blocked(3, s3, "child 3 blocked")
    w.blocked(1, s1, "child 1 blocked", timeout_ms=100)
    w.blocked(2, s2, "child 2 blocked", timeout_ms=100)
    s.events("defined; three children blocked")
    w.exec(0, add_one(add, A, src), wait=False)
    w.c[3].wait_stamp(s3)
    s.events("root, child 2, child 1, child 3", order=True, irq_stamps=True)
    for n in "ABCDE":
        w.t.dump(bufs[n][1], 0x104, n)
    w.stamps()
    return s


@scenario
def interplay_fifo_serial_stamps():
    """Barriers compare by serial arithmetic, (int32_t)(slot - value) >= 0,
    against the slot as it is now, whatever order the guest's stamp values
    come in. Child 2 blocks on child 1 reaching 0x7ffffff8: 0x7ffffff4 does
    not release it, 0x80000002 does. Child 1 then publishes 5, lower: a
    barrier for 3 and one for 0xfffffffe (behind 5 across the wrap) pass at
    once, while one for 0x80000002, a value the slot held before, now
    blocks, until 0x80000010. The root blocks on child 1 reaching
    0xfffffff0 with the slot at 0x80000010 (ahead by more than 2^31);
    0xfffffff8 releases it, and after child 1 wraps to 3 a root barrier for
    0xfffffffc passes."""
    w = World("interplay_fifo_serial_stamps", interplay_fifo_serial_stamps.__doc__, children=(1, 2))
    s = w.s
    w.define_children()
    c2 = w.c[2]
    w.nop(1, stamp=0x7FFFFFF0)
    x = w.nop(2, barriers=[(1, 0x7FFFFFF8)], wait=False)
    w.blocked(2, x, "child 2 blocked on 1:0x7ffffff8")
    w.nop(1, stamp=0x7FFFFFF4)
    w.blocked(2, x, "0x7ffffff4 does not release it")
    w.nop(1, stamp=0x80000002)
    c2.wait_stamp(x)
    s.events("0x7ffffff0, 0x7ffffff4, 0x80000002, then child 2", order=True, irq_stamps=True)
    w.nop(1, stamp=5)
    w.nop(2, barriers=[(1, 3)], label="1:3 with the slot at 5")
    w.nop(2, barriers=[(1, 0xFFFFFFFE)], label="1:0xfffffffe with the slot at 5")
    y = w.nop(2, barriers=[(1, 0x80000002)], wait=False)
    w.blocked(2, y, "1:0x80000002 blocks again with the slot at 5")
    w.nop(1, stamp=0x80000010)
    c2.wait_stamp(y)
    r = w.nop(0, barriers=[(1, 0xFFFFFFF0)], wait=False)
    w.blocked(0, r, "root blocked on 1:0xfffffff0 with the slot at 0x80000010")
    w.nop(1, stamp=0xFFFFFFF8)
    w.f.wait_stamp(r)
    w.nop(1, stamp=3)
    w.nop(0, barriers=[(1, 0xFFFFFFFC)], label="root: 1:0xfffffffc with the slot at 3")
    s.events("the rest", order=True, irq_stamps=True)
    w.stamps()
    s.fault_state("no fault")
    return s


@scenario
def interplay_fifo_held_stamps_release():
    """Stamps signalled out of publication order release a barrier only when
    published. Child 1's first command faults (opcode 99), and two NOPs
    after it complete but are held behind it. Child 2's exec, copying A to
    B, is gated on child 1's stamp 3, the last held one, and stays blocked
    while the root fills A. 0x1028 = 1 publishes 3 in one interrupt, and
    child 2's copy then runs: B holds the root's pattern."""
    w = World("interplay_fifo_held_stamps_release", interplay_fifo_held_stamps_release.__doc__, children=(1, 2))
    s = w.s
    a, a_va = w.buffer()
    b, b_va = w.buffer()
    w.ready()
    c1 = w.c[1]
    c1.cmd(99)
    c1.cmd(abi.OP_NOP)
    h = c1.cmd(abi.OP_NOP)
    c1.doorbell()
    child_fault(s, c1)
    x = w.exec(2, blit(B.copy_buffer_to_buffer(a, b, 0, 0, 0x40)), barriers=[(1, h)], wait=False)
    w.blocked(2, x, "child 2 blocked on child 1's held stamp")
    w.exec(0, blit(B.fill_pattern(a, 0, 0x40, PATTERN)), label="the root fills A while child 1 is faulted")
    w.blocked(2, x, "child 2 still blocked", timeout_ms=100)
    w.t.dump(b_va, 0x10, "B before")
    w.node(1, "child 1 faulted")
    s.events("child 1's fault, the root's exec", irq_stamps=True)
    resume_child(s, 1)
    w.c[2].wait_stamp(x)
    s.events("child 1 publishes 3, then child 2", order=True, irq_stamps=True)
    w.t.dump(b_va, 0x44, "B")
    w.node(1, "child 1 resumed")
    s.fault_state("resumed")
    w.stamps()
    return s


@scenario
def interplay_fifo_fault_others_carry_on():
    """An exec that faults on child 1 (a fill naming ref 0) leaves child 1
    faulted (node +8 is the exec's ring offset, 0x102c reads 1) while the
    root and child 2 go on running execs: the root fills A, child 2 copies A
    to B, and both publish. Child 1's next exec, a copy of B to C gated on
    child 2's stamp 2, waits behind its barrier; 0x1028 = 1 publishes the
    faulted stamp, a NOP on child 2 then releases child 1's copy, which
    runs and publishes: C holds the root's pattern."""
    w = World("interplay_fifo_fault_others_carry_on", interplay_fifo_fault_others_carry_on.__doc__,
              children=(1, 2))
    s = w.s
    a, a_va = w.buffer()
    b, b_va = w.buffer()
    c, c_va = w.buffer()
    w.ready()
    c1 = w.c[1]
    w.exec(1, blit(B.fill(0, 0, 0x10, 0x11)), wait=False)
    child_fault(s, c1)
    s.fault_state("child 1 faulted")
    w.node(1, "child 1 faulted")
    s.events("child 1's fault")
    w.exec(0, blit(B.fill_pattern(a, 0, 0x40, PATTERN)), label="the root runs")
    w.exec(2, blit(B.copy_buffer_to_buffer(a, b, 0, 0, 0x40)), label="child 2 runs")
    s.fault_state("child 1 still faulted")
    s.events("the root and child 2 carried on", order=True, irq_stamps=True)
    x = w.exec(1, blit(B.copy_buffer_to_buffer(b, c, 0, 0, 0x40)), barriers=[(2, 2)], wait=False)
    resume_child(s, 1)
    c1.wait_stamp(x - 1, label="the faulted stamp, published by the resume")
    w.blocked(1, x, "child 1's copy waits on child 2", timeout_ms=200)
    w.t.dump(c_va, 0x10, "C before")
    w.nop(2)
    c1.wait_stamp(x)
    s.events("resume, child 2's NOP, child 1's copy", order=True, irq_stamps=True)
    w.t.dump(c_va, 0x44, "C")
    s.fault_state("resumed")
    w.stamps()
    return s


@scenario
def interplay_fifo_round_robin():
    """Sixty NOPs interleaved over three FIFOs in lockstep, each FIFO's
    whole queue written and rung at once: the root's NOP k waits for child
    2's k-1, child 1's k for the root's k, child 2's k for child 1's k.
    The barriers make the order total, so the sixty interrupts, and the
    stamp slots each one saw, are locked in order; 0x1018 then has bits 0,
    1 and 2."""
    w = World("interplay_fifo_round_robin", interplay_fifo_round_robin.__doc__, children=(1, 2))
    s = w.s
    w.define_children()
    base = w.f.stamp
    n = 20
    for k in range(1, n + 1):
        w.c[2].cmd(abi.OP_NOP, barriers=[(1, k)])
    for k in range(1, n + 1):
        w.c[1].cmd(abi.OP_NOP, barriers=[(0, base + k)])
    w.c[2].doorbell()
    w.c[1].doorbell()
    for k in range(1, n + 1):
        w.f.cmd(abi.OP_NOP, barriers=[(2, k - 1)] if k > 1 else [])
    w.f.doorbell()
    w.c[2].wait_stamp(n)
    s.events("root, child 1, child 2, twenty times", order=True, irq_stamps=True)
    s.r(abi.R_IRQ_STAMP, "0x1018")
    w.stamps()
    return s


@scenario
def interplay_fifo_free_running():
    """Three FIFOs with twenty NOPs each and no barriers between them run
    on their own threads, so how their interrupts interleave is timing and
    is not locked (decision 8; measured: the same scenario with an ordered
    events checkpoint gave different orders across bless's three runs).
    What is: every FIFO publishes its last stamp,
    each NOP raises exactly one interrupt (sixty), and each ring's read
    count reaches its end."""
    w = World("interplay_fifo_free_running", interplay_fifo_free_running.__doc__, children=(1, 2))
    s = w.s
    w.define_children()
    s.events("defined")
    for fid in (0, 1, 2):
        f = w.fifo(fid)
        for k in range(20):
            f.cmd(abi.OP_NOP, bytes(4 * (k % 3)))
    for fid in (1, 2, 0):
        w.fifo(fid).doorbell()
    for fid in (0, 1, 2):
        f = w.fifo(fid)
        f.wait_stamp(f.stamp)
    s.events("sixty interrupts")
    s.r(abi.R_FIFO_READ)
    w.node(1)
    w.node(2)
    w.stamps()
    return s


@scenario
def interplay_fifo_circular_three():
    """A circular wait across three children: child 1 waits for child 2's
    stamp 5, child 2 for child 3's 5, then child 3 for child 1's 5; none is
    pushed, so DeviceCore::add_wait's walk 3 -> 1 -> 2 -> 3 comes back to
    child 3, which faults with "Circular wait". (Each wait is registered
    before the next is written: the scenario waits on each blocked stamp
    first.) 0x1028 = 3 publishes child 3's stamp 1; its NOPs 2-5 then
    release child 2, whose five stamps release child 1."""
    w = World("interplay_fifo_circular_three", interplay_fifo_circular_three.__doc__, children=(1, 2, 3))
    s = w.s
    w.define_children()
    c1, c2, c3 = w.c[1], w.c[2], w.c[3]
    c1.cmd(abi.OP_NOP, barriers=[(2, 5)])
    c1.cmd(abi.OP_NOP)
    c1.doorbell()
    w.blocked(1, 1, "child 1 blocked on 2:5")
    c2.cmd(abi.OP_NOP, barriers=[(3, 5)])
    for _ in range(4):
        c2.cmd(abi.OP_NOP)
    c2.doorbell()
    w.blocked(2, 1, "child 2 blocked on 3:5")
    c3.cmd(abi.OP_NOP, barriers=[(1, 5)])
    c3.doorbell()
    child_fault(s, c3, label="child 3: circular")
    w.node(3, "child 3 faulted")
    s.fault_state("child 3 faulted")
    s.events("defined; child 3 faulted")
    resume_child(s, 3)
    c3.wait_stamp(1)
    for _ in range(4):
        c3.cmd(abi.OP_NOP)
    c3.doorbell()
    c1.wait_stamp(2)
    s.events("child 3, then child 2, then child 1", order=True, irq_stamps=True)
    s.fault_state("resumed")
    w.stamps()
    return s


@scenario
def interplay_fifo_chain_pushed_not_circular():
    """DeviceCore::add_wait's walk ends at a FIFO that has already pushed
    the value waited for, even unpublished (its waiting-info entry's
    last_pending_stamp). Child 1's opcode 99 faults and its NOP (stamp 2) is
    held behind it; its next command then blocks on child 2's 5, child 2's
    on child 3's 5, and child 3's on child 1's 2: the cycle 3 -> 1 -> 2 -> 3
    is not circular, because child 1 pushed 2. 0x1028 = 1 publishes 2, and
    the chain unwinds: child 3's five stamps, child 2's five, then child 1's
    last two."""
    w = World("interplay_fifo_chain_pushed_not_circular", interplay_fifo_chain_pushed_not_circular.__doc__,
              children=(1, 2, 3))
    s = w.s
    w.define_children()
    c1, c2, c3 = w.c[1], w.c[2], w.c[3]
    c1.cmd(99)
    held = c1.cmd(abi.OP_NOP)
    c1.cmd(abi.OP_NOP, barriers=[(2, 5)])
    last = c1.cmd(abi.OP_NOP)
    c1.doorbell()
    child_fault(s, c1)
    w.blocked(1, held + 1, "child 1 blocked on 2:5, behind its fault")
    c2.cmd(abi.OP_NOP, barriers=[(3, 5)])
    for _ in range(4):
        c2.cmd(abi.OP_NOP)
    c2.doorbell()
    w.blocked(2, 1, "child 2 blocked on 3:5")
    c3.cmd(abi.OP_NOP, barriers=[(1, held)])
    for _ in range(4):
        c3.cmd(abi.OP_NOP)
    c3.doorbell()
    w.blocked(3, 1, "child 3 blocked on 1:2, not circular")
    w.node(3, "child 3 not faulted")
    s.events("defined; child 1 faulted")
    resume_child(s, 1)
    c1.wait_stamp(last)
    s.events("child 1's 2, child 3, child 2, child 1", order=True, irq_stamps=True)
    s.fault_state("no fault")
    w.stamps()
    return s


@scenario
def interplay_fifo_stamp_index_false_circular():
    """DeviceCore::add_wait's walk reads a waited-on FIFO's pushed stamp
    from the waiting-info entry of the barrier's index, but a child stores
    it in the entry of its id (DeviceCore::create_fifo_internal gives the
    new child waiting_info[fifo_id]'s last pending stamp), so for a child
    whose stampIndex differs from its id the walk sees 0.
    Control: child 1 (stampIndex 1) has a held stamp 2 behind a fault and
    blocks on child 2's 1; child 2's barrier on slot 1 for 2 is not circular
    and blocks, and 0x1028 = 1 unwinds both. The same with child 3 on
    stampIndex 5 and child 4: child 4's barrier on slot 5 for 2, which child
    3 has pushed, faults with "Circular wait". 0x1028 = 4 resumes child 4
    (raising an interrupt, as bit 5 is still set), whose stamp 1 releases
    child 3 behind its fault; 0x1028 = 3 resumes child 3 (bit 3 is not set:
    a stamp-wait report, then the resume), publishing slot 5, and 0x1028 = 5
    clears bit 5."""
    w = World("interplay_fifo_stamp_index_false_circular", interplay_fifo_stamp_index_false_circular.__doc__,
              children=(1, 2, 3, 4), stamp_indices={3: 5})
    s = w.s
    w.define_children()
    c1, c2, c3, c4 = (w.c[i] for i in (1, 2, 3, 4))
    # Control: ids and stamp indices agree.
    c1.cmd(99)
    held = c1.cmd(abi.OP_NOP)
    x1 = c1.cmd(abi.OP_NOP, barriers=[(2, 1)])
    c1.doorbell()
    child_fault(s, c1)
    w.blocked(1, x1, "child 1 blocked on 2:1, behind its fault")
    x2 = c2.cmd(abi.OP_NOP, barriers=[(1, held)])
    c2.doorbell()
    w.blocked(2, x2, "child 2 blocked on 1:2: not circular")
    w.node(2, "child 2 not faulted")
    resume_child(s, 1)
    c1.wait_stamp(x1)
    s.events("control: child 1's 2, child 2's 1, child 1's 3", order=True, irq_stamps=True)
    s.fault_state("control resumed")
    # Child 3 on stamp index 5.
    c3.cmd(99)
    held = c3.cmd(abi.OP_NOP)
    x3 = c3.cmd(abi.OP_NOP, barriers=[(4, 1)])
    c3.doorbell()
    child_fault(s, c3)
    w.blocked(3, x3, "child 3 blocked on 4:1, behind its fault")
    x4 = c4.cmd(abi.OP_NOP, barriers=[(5, held)])
    c4.doorbell()
    s.wait(gpa=c4.node_gpa(NODE_FAULT), ne=HEALTHY, label="child 4: circular")
    w.node(4, "child 4 faulted")
    w.node(3, "child 3 faulted")
    s.events("child 3's fault, child 4's fault")
    resume_child(s, 4)
    c4.wait_stamp(x4)
    s.wait(gpa=c3.node_gpa(NODE_READ), eq=c3.written, label="child 3 released")
    s.events("0x1028 = 4", order=True, irq_stamps=True)
    s.fault_state("bit 5 still set")
    resume_child(s, 3)
    c3.wait_stamp(x3)
    s.fault_state("after 0x1028 = 3")
    resume_child(s, 5)
    s.wait(mmio=abi.R_FAULT_MASK, mmio_eq=0)
    s.events("0x1028 = 3, then 5", order=True, irq_stamps=True)
    s.fault_state("cleared")
    w.stamps()
    return s


@scenario
def interplay_fifo_define_over_waiting_stamp_index_dies():
    """DeviceCore::create_fifo asserts that the new id's waiting-info entry
    is idle, but a blocked child registers its wait under its stampIndex:
    child 1 on stampIndex 2 blocks on slot 5 (the scenario waits 300 ms for
    the wait to register), and defining child 2 then kills the host."""
    w = World("interplay_fifo_define_over_waiting_stamp_index_dies",
              interplay_fifo_define_over_waiting_stamp_index_dies.__doc__, children=(1, 2),
              stamp_indices={1: 2}, slots=6)
    s = w.s
    c1 = w.c[1]
    c1.define()
    x = c1.cmd(abi.OP_NOP, barriers=[(5, 1)])
    c1.doorbell()
    w.blocked(1, x, "child 1 (stampIndex 2) blocked on slot 5")
    s.events("child 1 blocked")
    w.c[2].define()
    s.fault_state("not reached")
    return s


@scenario
def interplay_fifo_delete_blocked():
    """Deleting a child FIFO that is blocked on a barrier:
    DeviceCore::delete_fifo stops it and wakes every wait, the quiesced wait
    gives up, and the blocked command is never consumed or pushed: node 1's
    read count stays 0. The delete signals, and zeroes node 2's
    pageListPage. Child 2, blocked on child 1's slot, stays blocked, until
    child 1 is defined again over a fresh ring and its NOP publishes slot
    1."""
    w = World("interplay_fifo_delete_blocked", interplay_fifo_delete_blocked.__doc__, children=(1, 2), slots=4)
    s = w.s
    w.define_children()
    c1, c2 = w.c[1], w.c[2]
    x1 = c1.cmd(abi.OP_NOP, barriers=[(3, 1)])
    c1.doorbell()
    x2 = c2.cmd(abi.OP_NOP, barriers=[(1, 1)])
    c2.doorbell()
    w.blocked(1, x1, "child 1 blocked on slot 3")
    w.blocked(2, x2, "child 2 blocked on child 1", timeout_ms=100)
    c1.delete()
    w.node(1, "child 1 deleted while blocked")
    w.node(2, "child 2, its pageListPage zeroed")
    w.blocked(2, x2, "child 2 still blocked")
    s.events("deleted", order=True, irq_stamps=True)
    c1.written = 0
    c1.setup_node()
    c1.define()
    c1.cmd(abi.OP_NOP, stamp=1)
    c1.doorbell()
    c2.wait_stamp(x2)
    s.events("child 1 redefined: its NOP, then child 2", order=True, irq_stamps=True)
    w.node(1, "child 1 redefined")
    w.stamps()
    return s


@scenario
def interplay_fifo_delete_exec_pending():
    """Deleting a child FIFO whose exec was read: the exec was pushed before
    the delete's quiesce, so DeviceCore::delete_fifo drains it: the fill
    lands, child 1 publishes its stamp, and only then does the delete's
    stamp publish. Child 1 is then defined again over a fresh ring and
    carries an exec on the same task (in the service, the task was told the
    FIFO was deleted and is sent the new FIFO's event): it copies what the
    first one wrote."""
    w = World("interplay_fifo_delete_exec_pending", interplay_fifo_delete_exec_pending.__doc__, children=(1,))
    s = w.s
    b, b_va = w.buffer(length=4 * PAGE)
    c, c_va = w.buffer()
    w.ready()
    c1 = w.c[1]
    x = w.exec(1, blit(B.fill(b, 0, 4 * PAGE, 0x3C)), wait=False)
    s.wait(gpa=c1.node_gpa(NODE_READ), eq=c1.written, label="child 1 read its exec")
    c1.delete()
    s.events("child 1's exec, then the delete", order=True, irq_stamps=True)
    s.dump(c1.stamp_gpa(), 4, "slot 1")
    w.t.dump(b_va + 4 * PAGE - 0x10, 0x10, "B's end")
    w.node(1, "deleted")
    c1.written = 0
    c1.setup_node()
    c1.define()
    y = w.exec(1, blit(B.copy_buffer_to_buffer(b, c, 4 * PAGE - 0x20, 0, 0x20)))
    s.events("redefined: its exec", order=True, irq_stamps=True)
    w.t.dump(c_va, 0x30, "C")
    c1.delete()
    w.exec(0, blit(B.fill(c, 0x20, 0x10, 0x77)), label="the task runs on the root after")
    w.t.dump(c_va, 0x30, "C after the root's fill")
    s.events("the delete, the root's exec")
    return s


@scenario
def interplay_fifo_token_ring_wrap():
    """A token passed around four FIFOs for eight rounds while every ring
    wraps: each exec increments a counter (inc_k) and copies it to log[k],
    and is gated on the previous FIFO's exec. Each exec command is 0x1554
    bytes (its payload padded), so the root's 0x3c00-byte ring and the
    children's one-page rings wrap more than twice, with commands, headers
    and barriers straddling their ends. The guest writes two rounds at a
    time, rings all four doorbells, and waits for the last stamp. The 32
    interrupts come in token order, the counter is 32, and log[k] = k + 1:
    each exec saw every earlier one's write."""
    w = World("interplay_fifo_token_ring_wrap", interplay_fifo_token_ring_wrap.__doc__, children=(1, 2, 3),
              root_pages=1)
    s = w.s
    k_ref, k_va = w.buffer()
    log, log_va = w.buffer()
    inc = w.kernel(INC)
    w.ready()
    size = 0x1554
    root_base = w.f.stamp
    order = [0, 1, 2, 3]

    def stamp_of(fid, r):
        return (root_base if fid == 0 else 0) + r + 1

    for batch in range(4):
        for r in (2 * batch, 2 * batch + 1):
            for i, fid in enumerate(order):
                k = 4 * r + i
                if k == 0:
                    bar = []
                else:
                    pf = order[(i - 1) % 4]
                    pr = r if i > 0 else r - 1
                    bar = [(pf, stamp_of(pf, pr))]
                w.exec_cmd(fid, compute(C.set_pipeline(inc), C.set_buffers(0, [(k_ref, 0)]),
                                        C.dispatch_threads((1, 1, 1), (1, 1, 1))),
                           blit(B.copy_buffer_to_buffer(k_ref, log, 0, 4 * k, 4)),
                           barriers=bar, size=size)
        for fid in (3, 2, 1, 0):
            w.fifo(fid).doorbell()
        w.c[3].wait_stamp(stamp_of(3, 2 * batch + 1))
    s.events("32 execs in token order", order=True, irq_stamps=True)
    w.t.dump(k_va, 0x40, "counter")
    w.t.dump(log_va, 0x84, "log")
    s.r(abi.R_FIFO_READ)
    for fid in (1, 2, 3):
        w.node(fid)
    w.stamps()
    return s


@scenario
def interplay_fifo_fences():
    """A fence carried across FIFOs on one task: child 1 fills A and updates
    fence F; child 2, gated on child 1, waits for F and copies A to B; the
    root, gated on child 2, waits for F in a compute pass and computes
    C = B + 1 (Deserializer blit records 316, 317; compute 0xd4, 0xd5).
    Written root, child 2, child 1. Then InfoIndirect (69) with no command
    buffers on child 1 waits for child 1's event (its exec's value) and on
    child 2 for child 2's; in-process it advances the FIFO's event value, in
    the service the event alone, and a later exec on each still runs."""
    w = World("interplay_fifo_fences", interplay_fifo_fences.__doc__, children=(1, 2))
    s = w.s
    a, a_va = w.buffer()
    b, b_va = w.buffer()
    c, c_va = w.buffer()
    f = w.fence()
    add = w.kernel(ADD)
    w.ready()
    rs = w.exec(0, compute(C.fence(0xD5, f), C.set_pipeline(add), C.set_buffers(0, [(c, 0), (b, 0)]),
                           C.dispatch_threads((16, 1, 1), (16, 1, 1))), barriers=[(2, 1)], wait=False)
    x2 = w.exec(2, blit(B.fence(317, f), B.copy_buffer_to_buffer(a, b, 0, 0, 0x40)), barriers=[(1, 1)],
                wait=False)
    w.blocked(0, rs, "root blocked")
    w.blocked(2, x2, "child 2 blocked", timeout_ms=100)
    w.exec(1, blit(B.fill_pattern(a, 0, 0x40, PATTERN), B.fence(316, f)), wait=False)
    w.f.wait_stamp(rs)
    s.events("child 1, child 2, root", order=True, irq_stamps=True)
    for label, va in (("A", a_va), ("B", b_va), ("C", c_va)):
        w.t.dump(va, 0x44, label)
    for fid in (1, 2):
        st = w.fifo(fid).cmd(abi.OP_INFO_INDIRECT, w.payload([]))
        w.fifo(fid).doorbell()
        w.fifo(fid).wait_stamp(st, label="child %d InfoIndirect" % fid)
    w.exec(1, blit(B.fill(a, 0x40, 0x10, 0x5A)))
    w.exec(2, blit(B.copy_buffer_to_buffer(a, b, 0x40, 0x40, 0x10)))
    s.events("InfoIndirect on each child, then an exec on each", order=True, irq_stamps=True)
    w.t.dump(b_va + 0x40, 0x10, "B + 0x40")
    s.fault_state("no fault")
    return s


@scenario
def interplay_fifo_private_across():
    """A private buffer P (type 1, flags bit 0) used from three FIFOs. Child
    2's copy of P to S1 is gated on child 1's fill of P: S1 holds the fill
    in every setup; P's guest pages hold it only in the service, whose
    buffers are over the guest pages (docs/testing.md, the
    resources_private_buffer divergence). Child 1's second fill lists P with
    invalidateGuest and synchronize (6), which copies the host copy back.
    The guest then writes P's pages, and the root's copy lists P with
    invalidateHost (1): its paging command buffer waits for the root's own
    FIFO event and S2 holds the guest's bytes."""
    w = World("interplay_fifo_private_across", interplay_fifo_private_across.__doc__, children=(1, 2))
    s = w.s
    p, p_va = w.buffer(contents=bytes(range(0x80)), flags=1, listed=False)
    s1, s1_va = w.buffer()
    s2, s2_va = w.buffer()
    w.ready()
    res = lambda pf: [(p, pf), (s1, 0), (s2, 0)]
    x2 = w.exec(2, blit(B.copy_buffer_to_buffer(p, s1, 0, 0, 0x40)), barriers=[(1, 1)], resources=res(0),
                wait=False)
    w.blocked(2, x2, "child 2 blocked on child 1")
    w.exec(1, blit(B.fill(p, 0, 0x20, 0xCD)), resources=res(0), wait=False)
    w.c[2].wait_stamp(x2)
    s.events("child 1's fill, child 2's copy", order=True, irq_stamps=True)
    w.t.dump(s1_va, 0x40, "S1")
    w.t.dump(p_va, 0x40, "P's pages after a fill with no flags")
    w.exec(1, blit(B.fill(p, 0x20, 0x10, 0xEF)), resources=res(6))
    w.t.dump(p_va, 0x40, "P's pages after invalidateGuest + synchronize")
    w.t.write(p_va + 0x30, b"\x55" * 8)
    w.exec(0, blit(B.copy_buffer_to_buffer(p, s2, 0, 0, 0x40)), resources=res(1))
    w.t.dump(s2_va, 0x40, "S2: after invalidateHost on the root")
    s.events("the rest", order=True, irq_stamps=True)
    return s


@scenario
def interplay_fifo_delete_faulted_releases():
    """Deleting a faulted child publishes what it held:
    DeviceCore::delete_fifo's stop quiesces its stamp queue, whose pump then
    passes the faulted head, so child 1's opcode 99 (stamp 1) and the NOP
    held behind it (stamp 2) publish together, in one interrupt, and the
    delete clears child 1's fault bit. That releases child 2's first barrier
    (child 1's stamp 2); its second, on the delete's own root stamp, holds
    it until the delete signals. Child 2's copy then runs. Interrupts: child
    1, the root, child 2."""
    w = World("interplay_fifo_delete_faulted_releases", interplay_fifo_delete_faulted_releases.__doc__,
              children=(1, 2))
    s = w.s
    a, a_va = w.buffer()
    b, b_va = w.buffer()
    w.ready()
    w.exec(0, blit(B.fill_pattern(a, 0, 0x40, PATTERN)))
    c1 = w.c[1]
    c1.cmd(99)
    held = c1.cmd(abi.OP_NOP)
    c1.doorbell()
    child_fault(s, c1)
    d = w.f.stamp + 1
    x = w.exec(2, blit(B.copy_buffer_to_buffer(a, b, 0, 0, 0x40)), barriers=[(1, held), (0, d)], wait=False)
    w.blocked(2, x, "child 2 blocked on child 1's held stamp")
    s.fault_state("child 1 faulted")
    s.events("child 1 faulted, child 2 blocked")
    c1.delete()
    w.c[2].wait_stamp(x)
    s.events("child 1's two stamps at once, the delete, child 2", order=True, irq_stamps=True)
    s.fault_state("child 1 deleted")
    w.node(1, "child 1 deleted")
    w.t.dump(b_va, 0x44, "B")
    w.stamps()
    return s


@scenario
def interplay_fifo_timeout_report_mismatch_dies():
    """A barrier blocked for 2 s posts a stamp-wait report (from
    DeviceCore::wait_stamp), which looks every waiting stampIndex up as a
    child id and asserts that the child there has that stampIndex
    (DeviceCore::fifo_for_stamp_index). Child 1 on stampIndex 3 blocks on
    slot 6, and child 3 exists on stampIndex 5: the host dies when the
    report runs, with no MMIO write to cause it (docs/testing.md, section
    7.4)."""
    w = World("interplay_fifo_timeout_report_mismatch_dies", interplay_fifo_timeout_report_mismatch_dies.__doc__,
              children=(1, 3), stamp_indices={1: 3, 3: 5}, slots=7)
    s = w.s
    w.define_children()
    c1 = w.c[1]
    x = c1.cmd(abi.OP_NOP, barriers=[(6, 1)])
    c1.doorbell()
    s.events("defined")
    c1.wait_stamp(x, timeout_ms=4000, label="blocked past the 2 s timeout: the report kills the host")
    s.fault_state("not reached")
    return s


@scenario
def interplay_fifo_two_tasks_aliased():
    """Two tasks over the same guest pages, each carried by its own FIFO:
    task 2's page table maps task 1's data pages, and both object lists name
    a buffer X and a buffer Y at the same task addresses. Child 1 fills X
    through task 1; child 2's copy of X to Y through task 2 is gated on
    child 1; the root's copy of Y back into X + 0x80 through task 1 is
    gated on child 2. Written root, child 2, child 1; the guest pages show
    each step's result."""
    w = World("interplay_fifo_two_tasks_aliased", interplay_fifo_two_tasks_aliased.__doc__, children=(1, 2))
    s = w.s
    t1 = w.t
    t2 = XGPUTask(s, tid=2, pages=64, leaves={v: t1.data + v for v in range(64)})
    t2.ol = ObjectList(t2, 3 * PAGE, 3 * PAGE + 0x800)  # clear of task 1's list and records
    w.tasks.append(t2)
    w.listed[2] = []
    x1, x_va = w.buffer()
    y1, y_va = w.buffer()
    x2, _ = t2.buffer(va=x_va)
    y2, _ = t2.buffer(va=y_va)
    w.listed[2] = [x2, y2]
    t2._cmd = 6 * PAGE  # its command buffers after task 1's, over the same pages
    w.ready()
    r = w.exec(0, blit(B.copy_buffer_to_buffer(y1, x1, 0, 0x80, 0x40)), barriers=[(2, 1)], wait=False)
    x = w.exec(2, blit(B.copy_buffer_to_buffer(x2, y2, 0, 0, 0x40)), barriers=[(1, 1)], task=t2, wait=False)
    w.blocked(0, r, "root blocked")
    w.blocked(2, x, "child 2 blocked", timeout_ms=100)
    w.exec(1, blit(B.fill_pattern(x1, 0, 0x40, PATTERN)), wait=False)
    w.f.wait_stamp(r)
    s.events("child 1 (task 1), child 2 (task 2), root (task 1)", order=True, irq_stamps=True)
    t1.dump(x_va, 0xC4, "X")
    t1.dump(y_va, 0x44, "Y")
    return s


@scenario
def interplay_fifo_stamp_index_alias_exec():
    """Child 1 on stampIndex 0 shares the root's stamp slot and interrupt
    bit, while it keeps a FIFO event of its own. The root runs two execs on
    a task, child 1 one, and the root one more, each waited for: the stamps
    all land in slot 0. In the service the task registers one FIFO event per
    stampIndex, the first it saw, the root's, and keeps each exec's event
    values per stampIndex, so child 1's exec signals the root's event with
    its own count, lower than the root's; the root's next exec still
    completes. Child 1's stamps are 0x101 and up, the root's third exec's
    0x200, so every wait needs its own command."""
    w = World("interplay_fifo_stamp_index_alias_exec", interplay_fifo_stamp_index_alias_exec.__doc__,
              children=(1,), stamp_indices={1: 0}, slots=2)
    s = w.s
    a, a_va = w.buffer()
    w.ready()
    c1 = w.c[1]
    c1.stamp = 0x100
    w.exec(0, blit(B.fill(a, 0, 0x10, 0x11)))
    w.exec(0, blit(B.fill(a, 0x10, 0x10, 0x22)))
    w.exec(1, blit(B.fill(a, 0x20, 0x10, 0x33)), label="child 1's exec, in slot 0")
    # Stamp 0x200: above child 1's 0x101, so the wait needs this exec.
    w.exec(0, blit(B.fill(a, 0x30, 0x10, 0x44)), stamp=0x200, label="the root's third exec")
    s.events("four execs", order=True, irq_stamps=True)
    w.t.dump(a_va, 0x40, "A")
    return s


@scenario
def interplay_fifo_root_fault_children_carry_on():
    """The root faults (opcode 99: 0x103c holds its ring offset) and the
    children carry on: child 1 fills A and child 2 copies A to B, each
    publishing while 0x102c reads 1. The root's next command, an exec
    copying B to C gated on child 2's stamp 2, waits behind its barrier;
    0x1024 publishes the faulted stamp, child 2's NOP releases the root's
    copy, and C holds child 1's pattern."""
    w = World("interplay_fifo_root_fault_children_carry_on", interplay_fifo_root_fault_children_carry_on.__doc__,
              children=(1, 2))
    s = w.s
    a, a_va = w.buffer()
    b, b_va = w.buffer()
    c, c_va = w.buffer()
    w.ready()
    f = w.f
    bad = f.cmd(99)
    f.doorbell()
    s.wait_fault()
    s.fault_state("the root faulted")
    s.events("the root's fault")
    w.exec(1, blit(B.fill_pattern(a, 0, 0x40, PATTERN)), label="child 1 runs")
    w.exec(2, blit(B.copy_buffer_to_buffer(a, b, 0, 0, 0x40)), label="child 2 runs")
    s.fault_state("the root still faulted")
    s.events("the children carried on", order=True, irq_stamps=True)
    x = w.exec(0, blit(B.copy_buffer_to_buffer(b, c, 0, 0, 0x40)), barriers=[(2, 2)], wait=False)
    s.resume_root()
    f.wait_stamp(bad, label="the faulted stamp, published by 0x1024")
    w.blocked(0, x, "the root's copy waits on child 2", timeout_ms=200)
    w.t.dump(c_va, 0x10, "C before")
    w.nop(2)
    f.wait_stamp(x)
    s.events("0x1024, child 2's NOP, the root's copy", order=True, irq_stamps=True)
    w.t.dump(c_va, 0x44, "C")
    s.fault_state("resumed")
    w.stamps()
    return s


@scenario
def interplay_fifo_fan_out_fan_in():
    """All 63 children at once (maxFIFOCount 64 at version 245). Fan-out:
    each child's exec copies its own 4 bytes of A into B, gated on the
    root's exec that fills A; the root's release lets all 63 run on their
    own threads, so their order is timing and is not locked: each raises one
    interrupt, and B holds all of A. Fan-in: a root NOP with 63 barriers,
    one on each child's stamp, written before the children are rung,
    publishes after all of them. Not in the blocks setup: there each exec
    reads its command buffer and page-table entries through the host's
    read_memory callback from its own FIFO's thread, so 63 concurrent execs
    interleave their block calls, which the events record (measured: bless
    found them in different orders)."""
    kids = tuple(range(1, 64))
    w = World("interplay_fifo_fan_out_fan_in", interplay_fifo_fan_out_fan_in.__doc__, children=kids, slots=64,
              setups=("map", "isolate"))
    s = w.s
    a, a_va = w.buffer()
    b, b_va = w.buffer()
    w.ready()
    rs = w.f.stamp + 1
    for fid in kids:
        w.exec_cmd(fid, blit(B.copy_buffer_to_buffer(a, b, 4 * fid, 4 * fid, 4)), barriers=[(0, rs)])
    s.events("63 children defined")
    w.f.cmd(abi.OP_EXEC_INDIRECT3, w.payload([blit(B.fill_pattern(a, 0, 0x100, PATTERN))]))
    fan_in = w.f.cmd(abi.OP_NOP, barriers=[(fid, 1) for fid in kids])
    w.f.doorbell()
    w.f.wait_stamp(rs)
    w.f.wait_stamp(fan_in, timeout_ms=500, label="the fan-in waits for the children")
    for fid in kids:
        w.c[fid].doorbell()
    w.f.wait_stamp(fan_in)
    s.events("the root's exec, 63 children, the fan-in")
    w.t.dump(b_va, 0x100, "B")
    w.stamps()
    return s


@scenario
def interplay_fifo_task_deleted_under_gated_exec():
    """The root deletes a task (32) while child 1 holds an exec on it behind
    a barrier on child 2: the delete completes and clears the task's word in
    the root page. Child 2's NOP releases child 1, whose exec then finds no
    task (DeviceCore::get_task_id raises) and faults at its ring offset.
    After 0x1028 = 1, the root defines the task again with the same id and
    object list, and child 1's next exec on it runs."""
    w = World("interplay_fifo_task_deleted_under_gated_exec", interplay_fifo_task_deleted_under_gated_exec.__doc__,
              children=(1, 2))
    s = w.s
    a, a_va = w.buffer()
    w.ready()
    c1 = w.c[1]
    w.exec(1, blit(B.fill(a, 0, 0x10, 0x11)), label="child 1's first exec")
    x = w.exec(1, blit(B.fill(a, 0x10, 0x10, 0x22)), barriers=[(2, 1)], wait=False)
    w.blocked(1, x, "child 1's exec blocked on child 2")
    w.t.delete()
    s.dump(s.pa(s.root_page, 4 * w.t.tid), 4, "the task's root-page word after the delete")
    w.nop(2)
    s.wait(gpa=c1.node_gpa(NODE_FAULT), ne=HEALTHY, label="child 1's exec faults")
    w.node(1, "child 1 faulted")
    s.events("child 1, the delete, child 2, child 1's fault", order=True, irq_stamps=True)
    resume_child(s, 1)
    c1.wait_stamp(x)
    w.t.define()
    w.t.install()
    w.exec(1, blit(B.fill(a, 0x20, 0x10, 0x33)), label="child 1's exec on the task defined again")
    s.events("resumed, redefined, run", order=True, irq_stamps=True)
    w.t.dump(a_va, 0x40, "A")
    s.fault_state("no fault")
    return s


@scenario
def interplay_fifo_pending_full_across():
    """A child's stamp queue holds 128 pending stamps; child 1's opcode 99
    faults and 129 NOPs follow, so after 127 of them its thread blocks
    pushing the 128th: the node's read count already covers that command
    (Fifo::advance precedes Fifo::push_stamp). Child 2 waits on child 1's
    last stamp. 0x1028 = 1 publishes stamps up to 128 in one interrupt, the
    blocked push goes on, stamps 129 and 130 publish one each, and child 2
    is released."""
    w = World("interplay_fifo_pending_full_across", interplay_fifo_pending_full_across.__doc__, children=(1, 2))
    s = w.s
    w.define_children()
    c1 = w.c[1]
    c1.cmd(99)
    for _ in range(129):
        last = c1.cmd(abi.OP_NOP)
    c1.doorbell()
    child_fault(s, c1)
    s.wait(gpa=c1.node_gpa(NODE_READ), eq=12 * 129, label="child 1's thread blocked pushing stamp 129")
    w.node(1, "child 1: 128 pending, one more read")
    x = w.nop(2, barriers=[(1, last)], wait=False)
    w.blocked(2, x, "child 2 blocked on child 1's stamp 130")
    s.events("child 1 faulted")
    resume_child(s, 1)
    w.c[2].wait_stamp(x)
    s.events("child 1's 128, 129, 130, then child 2", order=True, irq_stamps=True)
    w.node(1, "child 1 drained")
    w.stamps()
    return s


@scenario
def interplay_fifo_scatter_rings():
    """Children over three scattered pages each (every ring page followed
    in guest memory by a page of 0xff), and the root, pass a token for
    nine rounds: each exec increments a counter and logs it, gated on the
    previous FIFO's exec. Every exec command is 0x1998 bytes, so on the
    children's 0xc000-byte rings commands span page boundaries, the eighth
    round's header sits at 0x7ff8 across the second boundary, and the ring
    wraps with a command straddling its end. Three rounds per doorbell.
    The 27 interrupts come in token order and log[k] = k + 1."""
    w = World("interplay_fifo_scatter_rings", interplay_fifo_scatter_rings.__doc__)
    s = w.s
    for fid in (1, 2):
        w.c[fid] = child(s, fid, define=False, pages=3, scatter=True)
        for p in w.c[fid].pages:
            s.fill(s.pa(p + 1), PAGE, 0xFF)
    track_stamps(s, 3)
    w.nslots = 3
    k_ref, k_va = w.buffer()
    log, log_va = w.buffer()
    inc = w.kernel(INC)
    w.ready()
    size = 0x1998
    root_base = w.f.stamp
    order = [0, 1, 2]

    def stamp_of(fid, r):
        return (root_base if fid == 0 else 0) + r + 1

    for batch in range(3):
        for r in range(3 * batch, 3 * batch + 3):
            for i, fid in enumerate(order):
                k = 3 * r + i
                if k == 0:
                    bar = []
                else:
                    pf = order[(i - 1) % 3]
                    bar = [(pf, stamp_of(pf, r if i > 0 else r - 1))]
                w.exec_cmd(fid, compute(C.set_pipeline(inc), C.set_buffers(0, [(k_ref, 0)]),
                                        C.dispatch_threads((1, 1, 1), (1, 1, 1))),
                           blit(B.copy_buffer_to_buffer(k_ref, log, 0, 4 * k, 4)),
                           barriers=bar, size=size)
        for fid in (2, 1, 0):
            w.fifo(fid).doorbell()
        w.c[2].wait_stamp(stamp_of(2, 3 * batch + 2))
    s.events("27 execs in token order", order=True, irq_stamps=True)
    w.t.dump(k_va, 0x10, "counter")
    w.t.dump(log_va, 0x70, "log")
    w.node(1)
    w.node(2)
    w.stamps()
    return s
