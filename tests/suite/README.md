# The behaviour-lock suite

This suite writes down what this codebase does, as a hypervisor sees it, so
that no change can alter it unnoticed. [docs/testing.md](../../docs/testing.md)
is its specification; its section 1 holds the decisions everything here
follows:

- **Drive it as a hypervisor does.** The library's C API, called the way
  QEMU calls it. Nothing of the implementation is included or imported.
- **Assert what the machine does**: registers, guest memory, interrupts, the
  host's callbacks, display callbacks and pixels, the suspend blob, and
  whether a run completes, faults, dies or hangs.
- **Lock it exactly.** Expected files are recorded from the checkout's own
  build, a person reviews them, and a run may differ from them in nothing.
  Only diagnostics (`t_*` timing and `d_*` fields: signals, messages) are
  ignored.
- **Lock order** wherever the implementation establishes one.
- **No VM boots.**

**Every change is checked against them.** The expected files record what
the build did when they were recorded, and a tree must reproduce every one
its predecessor passed (AGENTS.md, "Checking a change"). A change meant to
alter behaviour records the affected files again from its own build (`make
bless ARGS=--update`), and the diff of those files is reviewed in the same
commit.

**Citations.** The suite cites where a layout or a behaviour comes from:
docs/testing.md's guest ABI reference (section 6), the code that implements
it, or, for QEMU's side, `vendor/qemu/hw/display/apple-gfx*.c`. A scenario's
docstring is written into its generated file's `#` line, which the input
hash leaves out ("The input hash", below), so a docstring can change without
any expected file being recorded again.

## Layout

```
runner/pgrun.m        the host: plays the hypervisor and guest RAM through the C API
                      (reims_vgpu.h), writes observations
runner/build.sh       builds it against a dist tree (`cargo xtask dist`), linked with its
                      libreims_vgpu.dylib by LC_RPATH (bin/pgrun-*)
pgtest/               the library and the orchestrator (python3 -m pgtest)
  abi.py              the guest ABI's constants, each citing its source
  scenario.py         Scenario: guest memory, registers, commands, waits, checkpoints
  fifo.py             the root FIFO and child FIFOs
  task.py             tasks, page tables, object lists, segments and records
  <area>.py           builders for one area (objects, decoders, displays...)
  cli.py              gen, list, run, bless, cov, diffsum
  diffsum.py          a change to expected files by record kind and by scenario
  coverage.py         cov: the library instrumented, and the report
  normalize.py        what of a log is compared
  test_*.py           unit tests of the tooling's pure parts (no GPU, no runner)
scenarios/<area>.py   scenarios, registered with @scenario
fixtures/             committed inputs: shader sources, their metallibs and
                      the build.sh that reproduces them
generated/            the scenarios as JSON Lines (committed, generated)
expected/<area>/<scenario>/<setup>[@host].jsonl   the locked observations
                      (<setup> is a chain such as map-isolate for a scenario in phases)
```

## Using it

```sh
make test                      # every scenario, against the checkout's library (target/dist)
make test TEST='fifo*'         # scenarios or modules matching fnmatch patterns
make bless TEST=fifo           # record expected files for new or changed scenarios
make bless TEST=fifo ARGS=--update   # record them again after a deliberate change of behaviour
make coverage                  # the whole suite on the library built instrumented, in build/cov/dist
```

From `tests/suite`, `python3 -m pgtest` takes the same commands with more
options: `run -k PATTERNS --setup map -j N -v --dist DIR`,
`bless -k PATTERNS --runs N --update`,
`cov -k PATTERNS --tag NAME --files fifo/,device/` (a run's report, limited
to those files), `cov --tag NAME --show fifo/fifo.rs` (the lines that run
never reached) and `cov --per-case` (which cases reach each line). `run` and
`bless` default to the checkout's dist tree (`$PGTEST_DIST` overrides it).
`python3 -m unittest pgtest.test_reach pgtest.test_coverage
pgtest.test_diffsum` runs the tooling's unit tests, with no GPU.

**Reviewing a re-record.** `python3 -m pgtest diffsum [A [B]]` summarizes
how the expected files changed between revision A (default `HEAD`) and B
(default the working tree), and `diffsum --dirs OLD NEW` between two result
sets laid out as `expected/` is: per record kind, the records added, removed
and changed, each field added, removed or changed with its commonest value
changes; per scenario, the files and records changed. `-k` limits it to
some scenarios or modules.

**Selective runs.** With `-k`, `run`, `bless` and `cov` import and build only
the scenarios the patterns need (`registry.load`), so one case starts in a
few hundredths of a second instead of the 6 s that building all of them
takes. A name is known only once its function has run, so an index,
`~/.cache/pvd-pgtest/index-<checkout>.json`, records the names each
function builds, keyed by the sources that function's module is built from:
its own and every pgtest or scenario module it imports. A module whose
sources changed since it was indexed is built whole and indexed again. An
edit to a helper every module imports (`scenario.py`), or a missing index,
costs one full build. The index decides only which functions run: what a
case runs is built from the sources, and a function whose names differ from
its entry makes the run build everything. `gen`, `list` and a run without
`-k` build everything.

**Speed.** A case is latency-bound: the whole suite spends about 4% of its
wall time on a CPU, and the rest in deadlines, settle windows and watchdogs
that are meant to run out. So `-j` defaults to twice the CPUs (`PGTEST_JOBS`
overrides it), and the full suite runs in a few minutes on a 16-CPU M3 Max.
Those waits are what make "nothing more happened" an observation the
library does not vouch for: no hook in the library (an idle or quiescence
report) ends them early, and the suite's wall time is not a target.
The orchestrator (`cli.run_all`) starts the longest cases first, by each
case's last wall time (`out/durations.json`, written after every run; a case
without one is estimated from its expected file). A case starts only while
the machine has fewer than 600 IOSurface clients open
(`PGTEST_IOSURFACE_HIGH`), and the scenarios that declare many isolated
tasks (`Scenario.services`) share a budget of 300 (`PGTEST_SERVICE_BUDGET`),
so no two of the four cases that define 255 isolated tasks ramp up together
(the services run on threads of the runner's process, so the budget spaces
out load, not IOSurface clients). `bless` schedules each of a case's three
runs as a task of its own: every case's first runs, then the second, then
the third, so the runs of one case meet different moments of the run.

**Blessing.** `bless` records from the build under test: the checkout's
library, which `make bless` builds first. It runs each case three times
(`--runs`), one process each, and records it only if all three complete and
agree; a case whose runs differ is reported nondeterministic, and nothing is
written for it (rule 9). By default it records only the expected files that
are missing or were recorded from other input, and never rewrites one whose
input is unchanged: an expected file changes when its scenario does.
`--update` records the selected cases again even when their input is
unchanged. It is for a deliberate change of behaviour, and the diff of the
expected files it rewrites is reviewed in the same commit as the change.
Review every new or changed expected file before committing it: it is now
the definition of correct.

**The input hash.** An expected file's header records the hash of its
input: the generated scenario's ops (its file without the `#` doc line) and
the sources of `runner/pgrun.m` and `pgtest/normalize.py`. `run` fails a
case whose expected file was recorded from other input, and `bless` records
it again. So a change to the runner or to `normalize.py` fails every case
until `make bless` records the whole suite again, and that diff is reviewed
like any other.

**Host keys.** The expected files of a `host=True` scenario (rule 5) are
keyed by the host, `<setup>@<key>.jsonl`, the key being the GPU model, its
core count and the macOS build (`AppleM3Max-40c-26A434`). After an OS update
the new build's files are missing, and `run` reports those cases missing;
`make bless` records them, and the old build's files stay while a host
still runs it, then go. Review the new files against the old key's before
committing them.

**Load-sensitive cases.** These can fail under concurrent GPU load. Each
failed at least once in a run beside other GPU work and passed when run
again alone. A name with no setup means every setup the scenario runs in.
- `mutgap4_res_view_synchronize_isolate[isolate]`,
  `covgap_svc_*_hole_synchronize_isolate[isolate]` (among them
  `covgap_svc_texture_hole_synchronize_isolate[isolate]`) and
  `mutgap3_res_texture_record_clamps`: the synchronize paths, each seen in
  a whole-suite run;
- `fresh_res_delete_while_kernel_held` and
  `fresh_res_info_while_kernel_held`, which hold a kernel for 8 s: their
  cases saw the FIFO fault while the kernel ran, or the kernel end early;
- `render_objects_depth_stencil[blocks]` and `render_objects_pipeline[map]`,
  which run alone within their own orchestrator (`exclusive`) but not
  beside another;
- `interplay_boot_7[isolate]`, `interplay_dag_3[map]`,
  `interplay_dag_45[map]`, `interplay_decodewalk_7[map]`,
  `interplay_displaywalk_43[blocks]`, `interplay_displaywalk_44[map]`,
  `interplay_fifo_compute_join[map]`,
  `interplay_fifo_delete_exec_pending[blocks]`,
  `interplay_fifo_fan_out_fan_in[map]`, `interplay_iosfcwalk_9[map]`,
  `interplay_limits_all_tasks[isolate]`, `interplay_mix_43[isolate]`,
  `interplay_rand_4`, `interplay_rand_20`, `interplay_rand_33[isolate]`,
  `interplay_rand_39[isolate]`, `interplay_rand_42[blocks]`,
  `interplay_scanout_service_death_beside_frames[isolate]`,
  `interplay_session_efi_live_handover[isolate]`,
  `interplay_versions_efi_245[map]`,
  `interplay_versions_efi_after_2[blocks]`,
  `interplay_versions_efi_after_2[isolate]`,
  `mutgap2_rdec_sampler_bindings[map]`,
  `newops_held_present_timeout[blocks]` and `suspend_resume_efi[isolate]`:
  a present read back zero pixels, a wait saw its FIFO fault or timed
  out, or a dump lost a write;
- `interplay_boot_8`, `interplay_boot_15`, `interplay_boot_17`,
  `interplay_boot_18` and `interplay_marathon_17`, beside other whole
  suites: a frame's wait returned `RVG_ERR_GPU`;
- `textures_formats[blocks]`: a format's probe slot stayed unwritten;
- `icb_blit_commands[blocks]`, `interplay_iosfcwalk_16[map]`,
  `interplay_iosfcwalk_18[map]`, `interplay_iosfcwalk_18[isolate]`,
  `interplay_marathon_24[isolate]` and
  `mutgap2_dev_iosfc_host_only_texture[isolate]`, each seen once in a
  whole-suite run;
- `interplay_life_resume_rich[blocks]`, `mutgap_tex_shared_guest_valid[map]`
  and `resources_private_buffer_sync_commands[isolate]`, together in one
  whole-suite run of the `checked` build: a fault's flag read clear after
  its offset was set, a dump lost a write, and a wait saw its FIFO fault.

Run such a case again alone before calling it a failure.

## Writing scenarios

A scenario module is `scenarios/<area>.py`. Each function decorated with
`@scenario` returns a `Scenario` or a list of them; a scenario's name is
unique across the suite and starts with its area. `smoke.py` and
`smoke_host.py` are short examples of every mechanism. A function builds the
same scenarios whatever the process built before it: a selective run builds
only the functions it needs ("Selective runs", above), so no function may
depend on state another left behind.

```python
@scenario
def fifo_nop_twice():
    """What it locks, in a sentence or two."""
    s = Scenario("fifo_nop_twice", doc=fifo_nop_twice.__doc__)
    f = s.bring_up()                 # version 245, root page, root FIFO
    s.submit(abi.OP_NOP)             # write, doorbell, wait for its stamp
    s.submit(abi.OP_NOP)
    s.events("two NOPs")             # interrupts and callbacks so far
    s.r(abi.R_FIFO_READ)
    s.diff("after")                  # every byte of guest RAM the device changed
    return s
```

The rules:

1. **Checkpoint after a wait.** Take an observation (`r`, `diff`, `dump`,
   `events`) only after waiting for something the guest sees: `submit()`
   waits for the command's stamp, `submit(..., fault=True)` for the fault
   (0x102c reads 1); `wait()` takes a guest u32 (`gpa`, `eq`/`ge`/`ne`), a
   register that reading does not clear (`mmio`, `mmio_eq`/`mmio_ne`), event
   counts (`irqs`, `callbacks`, `blocks`, `iosfc_irqs`) or a callback kind
   (`cb="newFrame"`). Never read 0x1014 or 0x1018 (read-to-clear) inside a
   wait. `events()` also settles first, so its counts are exact. But one
   interrupt publishes every stamp ready at a FIFO's head, so after a
   batch with GPU work in it (an exec with commands behind it) how many
   interrupts came is timing: count them with `events(collapse=["irq"])`
   (runs of interrupts count once; without `order`, the count says only
   whether any came). The runner settles before its `willSuspend` op
   (`rvg_device_will_suspend`) too, which drops any interrupt not yet raised
   (`no_settle=True` skips it).
2. **After a fault, resume** (`s.resume_root()`, or 0x1028 for a child FIFO)
   before submitting more; a faulted FIFO publishes nothing else.
3. **One death per scenario, at its end.** A host death ends the process: put
   the command that kills it last. With isolation, a death in the service's
   own code is a fault the guest sees, and the scenario can go on; a death
   inside Metal ends the process from the task's thread, asynchronously to
   the ops after its doorbell, so its command goes behind every other, its
   doorbell is rung last and its fault is the first wait
   (`interplay_servicewalk` does; `exec_objects.submit_last` puts 256 NOPs
   ahead of it). A death on another thread (a FIFO's, the display thread's)
   waits for the record of the MMIO access, api call or encode in progress,
   if that call returns (`on_signal` in `runner/pgrun.m`): so the doorbell
   that kills the host is always in the log, however the threads were
   scheduled. Nothing is recorded after the
   death. A death still races with later ops that can complete without it:
   a wait that something else satisfies, or the end's settle window.
4. **Hangs**: give a scenario that hangs a short `watchdog_s` (8), and a wait
   that is expected to time out a short `timeout_ms` (500). The expected
   outcome is then `watchdog` at the op that hung.
5. **Host values.** Set `host=True` when an observation reports the host
   rather than computes something: device info, compute info, heap sizes
   and alignments, GPU addresses and resource IDs, and any pixel that
   the library's shaders draw (`encode`). Its expected files are then keyed
   by the host (GPU model, core count, macOS build). Where a value can be
   computed from Metal in the test instead, do so (`s.metal(...)`).
6. **Setups.** A scenario runs in all three host setups (`blocks`, `map`,
   `isolate`) unless it cannot mean anything in one; where the setups
   differ, that difference is behaviour to lock (docs/testing.md,
   section 6.10). Restrict with `setups=(...)` only with a reason in the
   docstring.
7. **Never observe memory nothing wrote** (docs/testing.md, sections 5.3
   and 7.5).
   Prefer `task.dump(va, n)` or `s.diff(gpa=..., length=...)` over whole-RAM
   diffs when only a region matters, and keep expected files readable.
8. **Cite the source** of every layout a builder writes ("Citations"
   above). Read the code to learn the ABI; never import or include it.
9. **Deterministic by construction.** If `bless` reports a case
   nondeterministic, the scenario is observing timing: wait for the right
   thing. Never loosen a comparison. One trap for walkers: in the blocks
   setup the runner numbers host tasks in `create_task` call order, and
   block records name those numbers, so two task defines in flight at once
   on different FIFOs make the records vary. Make each define wait on the
   one before (`interplay_servicewalk` does).
10. **GPU faults and compiler crashes run alone.** A command buffer that
    faults or hangs the GPU makes Metal discard other processes' work too
    ("victim of GPU error" in the system log), and a pipeline that crashes
    Metal's compiler service ("Compilation failed with XPC_ERROR" in the
    log) costs retries and crash reports that under load outlast the
    watchdog. A scenario that causes either passes `exclusive=True`:
    the orchestrator runs it by itself, after the others. Nothing keeps
    two orchestrators apart: a case run beside another orchestrator can
    fail as the victim of its fault. Such a failure shows "Discarded
    (victim of GPU error/recovery)" for the case's process in the log; run
    the case again. `log show
    --predicate 'eventMessage CONTAINS "Caused GPU"'` names the culprit's
    process; the runner's first record carries its `d_pid`.

## Phases: a migration to a new process

A suspend and resume on one host API is a migration: the VM is restored
by a new process, often on another host. `Scenario.phase()` splits a
scenario there, and each phase runs in a runner process of its own:

```python
s = Scenario("fresh_resume_x", setups=("map", ("map", "isolate")))
...                              # phase 0: bring up, drive, then
s.api("willSuspend")
s.api("finishSuspend")           # the blob, in slot "s"
s.phase()                        # a new process: RAM and slots handed on
s.api("willResume", slot="s")
s.api("didResume")
...                              # phase 1 drives the guest on
```

- **What is handed on.** `phase()` ends the phase with `handoff_save`: every
  guest RAM range (the host's bytes and the scenario's shadow of them, so
  a diff in the next process reports what the device changed, as in one
  process) and every blob slot, as an image in a directory the
  orchestrator owns (a temporary one a case run, removed after). The next
  phase's `config` loads it (`handoff_load`) before creating its device,
  by guest physical address: whatever of the image overlaps the target's
  RAM is copied, and the rest starts zeroed. That is what a VM restore
  carries; the IOSurface host device's state is not (no public or QEMU
  API saves it), so the target's is new.
- **Each phase may differ.** `phase(ram_ranges=[(gpa, len), ...])` gives the
  target other RAM (smaller, larger, split, elsewhere); `displays=[...]`,
  `descriptor={...}`, `iosfc=...` change its config (a value of `None`
  drops the key: `displays=None` for a device with no display, then
  `api new_display` where the host attaches one); `ram=False` loads the
  blobs only (a lost memory image); `save=False` ends a phase without an
  image, so the next loads the earlier one again (one image resumed twice).
  Guest addresses (`pa`, `pfn`) stay the first phase's.
- **Setups are chains.** Each entry of `setups` is a setup for every phase
  (`"map"`) or one a phase (`("map", "isolate")`, written `map-isolate`):
  the case `name[map-isolate]` and its expected file `map-isolate.jsonl`.
  List the chains a scenario means; never a cross product. `--setup map`
  selects the chains made only of map, or `--setup map-isolate` the one.
- **Counters start again.** Interrupts, callbacks, block calls and the
  blocks setup's host task numbers are counted per process, so every
  `wait(irqs=..., callbacks=..., cb_count=...)` in a phase counts from 0.
  A builder that keeps such a count registers `s.on_phase(fn)` to reset it
  (`IOSFC`, `Session`, the compute objects do).
- **The expected file** holds the phases' observations in order, each record
  tagged `"phase": k`, and each phase ends in its own outcome record. A
  phase that does not complete (it dies, hangs or errs) ends the case: the
  phases after it do not run. A one-phase scenario's records carry no
  `phase` tag.
- **Mutating a blob.** The `willResume` op takes `patch=[[offset, u32],
  ...]` and `truncate=n`, applied to the blob it passes. A slot's blob is
  the library's own (version 1, docs/testing.md section 6.24), with every
  byte written, so a mutation of it reads nothing unwritten (rule 7).
  Before mutating a slot's blob whose second word is 2, the layout of
  Apple's suspend state, which the library never writes, the runner zeroes
  its unwritten tail; `canonical=True` asks for that step with no
  mutation.

`run`, `bless` and `cov` all run a case through `cli.run_case`, so they
handle phases alike; bless runs the whole chain three times.

## Coverage

`make coverage` (or `python3 -m pgtest cov`) builds `build/cov`: the library,
with the workspace's crates instrumented (`cargo llvm-cov show-env`) in its
own target directory. It runs the selected scenarios on it and reports, per
crate file, the lines the library reached (`cov/<tag>/report.txt`;
`summary.json` adds its functions, regions and branches). It is never a
build under test: `make test` and `make bless` use the plain library.
Isolated tasks run in the case's own process, so each case's profile holds
every line it reached. `--per-case` gives each case its own profiles and
writes `map.json`, each line and the cases that reach it.
`pgtest/coverage.py`'s docstring has the method.
