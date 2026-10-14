/*
 * api_pins.c: black-box pins of the C API (include/reims_vgpu.h), as a host sees it.
 *
 * A C program a host could be. It pins what the suite's records do not show of the API:
 * - the order, arguments and thread of the device's callbacks;
 * - each table's one destroy: when, and on which thread, for a device, a display and an
 *   IOSurface device, at a release and at a refused create;
 * - the display thread: display callbacks, the EFI display's among them, never run on the
 *   host's thread, run one at a time, and on one thread;
 * - a display kept alive through an uncommitted and a committed frame;
 * - a display with a window, created and released in a process whose main thread never runs a
 *   run loop: no hang, and each table's one destroy (a window frame's hold on its display is
 *   the unit test window::tests::a_window_frame_keeps_its_display_until_it_completes of
 *   reims-vgpu, since no host reaches a window's frames);
 * - add_trace_range without remove_trace_range, a refusal;
 * - the registers a fresh device reads, a resume's refusals, and a display's first cursor
 *   position;
 * - the EFI display's traced range: the host calls, in order, from its start to its end, and
 *   the display calls it starts with;
 * - resume-child writes (0x1028) of ids past the 64 child FIFOs, to a plain device and to one
 *   with use_iosurface_mapper: they resume no FIFO, and the process runs on;
 * - the configurable inputs the suite's runner never sets, as the guest reads them: the
 *   feature level (the binary version it may choose, the display port count, and the death
 *   of a level above 4) and the display port connection types (the register, and a resume
 *   of a state taken with other types);
 * - the log hook: the lines a handler gets at each `max_level`, and none once removed;
 * - the files behind a memory map's ranges: each refusal, in order, with the line saying which,
 *   no descriptor left open by a refusal, and none kept by an accepted device, which on macOS
 *   checks the files and does not use them;
 * - the deaths that remain: no raise_interrupt at the first interrupt, a host function that
 *   raises, and a raise under a scope. A display released with a callback job queued is
 *   pinned by the unit test display::tests::a_display_released_with_a_job_queued_dies: only
 *   a guest's cursor doorbell queues a job that does not hold its display, and a host cannot
 *   ring it without a guest's shared display state.
 *
 * tests/pins/main.rs builds it against this crate's dylib, runs it, and compares its whole
 * output with tests/pins/expected.tsv, recorded from this build (main.rs says how to re-record).
 * Each check prints `<id>\t<observation>`: what happened, not a verdict. Deaths run in a child
 * (this executable with the death's name). A callback's thread prints as `@m` for the main
 * thread and `@x` for any other.
 *
 * The checks need a Metal device.
 *
 *   api_pins               every check
 *   api_pins --die <name>  one death's body
 */
#include "reims_vgpu.h"

#include <fcntl.h>
#include <objc/message.h>
#include <objc/runtime.h>
#include <pthread.h>
#include <spawn.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

extern void objc_exception_throw(id exception);

static void pin(const char *name, const char *fmt, ...) __attribute__((format(printf, 2, 3)));
static void pin(const char *name, const char *fmt, ...) {
    va_list ap;
    printf("%s\t", name);
    va_start(ap, fmt);
    vprintf(fmt, ap);
    va_end(ap);
    printf("\n");
    fflush(stdout);
}

static const char *status_name(RvgStatus s) {
    static const char *names[] = {
        "RVG_OK",
        "RVG_ERR_INVALID_ARGUMENT",
        "RVG_ERR_UNSUPPORTED",
        "RVG_ERR_DISPLAY_PORT_COUNT",
        "RVG_ERR_NO_METAL_DEVICE",
        "RVG_ERR_MEMORY_CONFLICT",
        "RVG_ERR_MEMORY_MAP",
        "RVG_ERR_ISOLATION_WITHOUT_MAP",
        "RVG_ERR_METAL",
        "RVG_ERR_PORT_OUT_OF_RANGE",
        "RVG_ERR_PORT_IN_USE",
        "RVG_ERR_SERIAL_IN_USE",
        "RVG_ERR_NO_MEMORY_MAP",
        "RVG_ERR_SUSPEND",
        "RVG_ERR_RESUME",
        "RVG_ERR_GPU",
        "RVG_ERR_FRAME_STATE",
    };
    return s < sizeof names / sizeof *names ? names[s] : "?";
}

/* ---- a log of calls, each marked with its thread ---- */

static pthread_mutex_t g_log_lock = PTHREAD_MUTEX_INITIALIZER;
static char g_log[8192];
static void tlog(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
static void tlog(const char *fmt, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    pthread_mutex_lock(&g_log_lock);
    strlcat(g_log, buf, sizeof g_log);
    strlcat(g_log, pthread_main_np() ? "@m," : "@x,", sizeof g_log);
    pthread_mutex_unlock(&g_log_lock);
}
static void log_reset(void) {
    pthread_mutex_lock(&g_log_lock);
    g_log[0] = 0;
    pthread_mutex_unlock(&g_log_lock);
}
static void named_destroy(void *ctx) { tlog("%s", (const char *)ctx); }

/* Waits up to 10 s for `flag`. */
static int wait_flag(atomic_int *flag) {
    for (int i = 0; i < 1000 && !atomic_load(flag); i++) {
        usleep(10000);
    }
    return atomic_load(flag);
}

/* ---- the tracing host, over the table ----
 *
 * createTask reserves the root task's range and hands back g_task; mapMemory copies the guest
 * pages it maps (g_guest_root) into the task's range at the offset it is given, as a
 * hypervisor's mapping would show them, so the library reads the guest's root node there. */

static int g_task;
static void *g_task_base;
static uint64_t g_task_size;
#define GUEST_ROOT_PAGE 0x10u
#define GUEST_TASK_ROOT 0x20u
static uint8_t g_guest_root[0x4000];
static int g_create_raises;
static int g_read_reenters;
static int g_read_raises;
static RvgDevice *g_trace_dev;

static void throw_host_raise(const char *reason) {
    id name = ((id(*)(id, SEL, const char *))objc_msgSend)(
        (id)objc_getClass("NSString"), sel_registerName("stringWithUTF8String:"), "HostRaise");
    id why = ((id(*)(id, SEL, const char *))objc_msgSend)(
        (id)objc_getClass("NSString"), sel_registerName("stringWithUTF8String:"), reason);
    id e = ((id(*)(id, SEL, id, id, id))objc_msgSend)(
        (id)objc_getClass("NSException"), sel_registerName("exceptionWithName:reason:userInfo:"),
        name, why, (id)NULL);
    objc_exception_throw(e);
}

static const char *task_name(const void *task) {
    return task == &g_task ? "T" : task == NULL ? "NULL" : "?";
}

static void *t_create(void *ctx, uint64_t size, void **base) {
    (void)ctx;
    tlog("createTask(%#llx,base_in=%s)", (unsigned long long)size,
         *base == NULL ? "0" : *base == g_task_base ? "prev" : "?");
    if (g_create_raises) {
        throw_host_raise("createTask");
    }
    if (!g_task_base) {
        g_task_size = size;
        g_task_base = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
    }
    *base = g_task_base;
    return &g_task;
}
static void t_destroy_task(void *ctx, void *task) {
    (void)ctx;
    tlog("destroyTask(%s)", task_name(task));
}
static bool t_map(void *ctx, void *task, uint32_t count, uint64_t offset, bool read_only,
                  const RvgPhysicalMemoryRange *ranges) {
    (void)ctx;
    char r[256] = "";
    uint64_t at = offset;
    for (uint32_t i = 0; i < count; i++) {
        char one[64];
        snprintf(one, sizeof one, "%s%#llx+%#llx", i ? ";" : "",
                 (unsigned long long)ranges[i].physical_address,
                 (unsigned long long)ranges[i].physical_length);
        strlcat(r, one, sizeof r);
        if (task == &g_task && g_task_base &&
            ranges[i].physical_address == (uint64_t)GUEST_ROOT_PAGE << 14 &&
            at + sizeof g_guest_root <= g_task_size) {
            memcpy((uint8_t *)g_task_base + at, g_guest_root, sizeof g_guest_root);
        }
        at += ranges[i].physical_length;
    }
    tlog("map(%s,%u,%#llx,ro=%d,%s)", task_name(task), count, (unsigned long long)offset,
         read_only, r);
    return true;
}
static bool t_unmap(void *ctx, void *task, uint64_t offset, uint64_t length) {
    (void)ctx;
    tlog("unmap(%s,%#llx,%#llx)", task_name(task), (unsigned long long)offset,
         (unsigned long long)length);
    return true;
}
static bool t_read(void *ctx, uint64_t address, uint64_t length, void *dst) {
    (void)ctx;
    tlog("read(%#llx,%#llx)", (unsigned long long)address, (unsigned long long)length);
    if (g_read_raises) {
        throw_host_raise("readMemory");
    }
    if (g_read_reenters) {
        /* A host call back into the device, from a callback the library called: the root
         * page written again, whose createTask raises. */
        g_read_reenters = 0;
        g_create_raises = 1;
        rvg_device_mmio_write(g_trace_dev, 0x101c, GUEST_ROOT_PAGE);
        g_create_raises = 0;
        tlog("reentry returned");
    }
    memset(dst, 0, length);
    return false;
}
static void t_raise(void *ctx, uint32_t vector) {
    (void)ctx;
    tlog("raise(%u)", vector);
}

static RvgHostCallbacks trace_callbacks(void) {
    RvgHostCallbacks cb;
    memset(&cb, 0, sizeof cb);
    cb.struct_size = sizeof cb;
    cb.ctx = "destroy";
    cb.destroy = named_destroy;
    cb.create_task = t_create;
    cb.destroy_task = t_destroy_task;
    cb.map_memory = t_map;
    cb.unmap_memory = t_unmap;
    cb.read_memory = t_read;
    cb.raise_interrupt = t_raise;
    return cb;
}

static RvgDeviceConfig device_config(const RvgHostCallbacks *cb, uint32_t ports) {
    RvgDeviceConfig c;
    memset(&c, 0, sizeof c);
    c.struct_size = sizeof c;
    c.display_port_count = ports;
    c.callbacks = cb;
    return c;
}

static RvgDevice *make_device(const RvgHostCallbacks *cb, uint32_t ports) {
    RvgDeviceConfig c = device_config(cb, ports);
    RvgDevice *dev = NULL;
    RvgStatus s = rvg_device_create(&c, &dev);
    if (s != RVG_OK) {
        fprintf(stderr, "rvg_device_create: %s\n", status_name(s));
        exit(3);
    }
    return dev;
}

/* A device over the tracing host whose guest wrote a root node with one task root. */
static RvgDevice *trace_device(const RvgHostCallbacks *cb) {
    memset(g_guest_root, 0, sizeof g_guest_root);
    uint32_t task_root = GUEST_TASK_ROOT;
    memcpy(g_guest_root, &task_root, sizeof task_root);
    g_trace_dev = make_device(cb, 1);
    return g_trace_dev;
}

static void suspend_resume(RvgDevice *dev, int log_status) {
    rvg_device_will_suspend(dev);
    RvgBytes state = {NULL, 0};
    RvgStatus fs = rvg_device_finish_suspend(dev, &state);
    RvgResumeError e = 99;
    RvgStatus rs = rvg_device_will_resume(dev, state.data, state.length, &e);
    rvg_bytes_free(state);
    if (log_status) {
        tlog("finish=%s resume=%s error=%d", status_name(fs), status_name(rs),
             e == RVG_RESUME_ERROR_NONE ? -1 : (int)e);
    }
    rvg_device_did_resume(dev);
}

/* The root page's write (createTask, then the map of the root page), a suspend and resume, a
 * root page of 0, and the last release, which calls no task or memory function: only the
 * table's destroy. */
static void calls(void) {
    RvgHostCallbacks cb = trace_callbacks();
    RvgDevice *dev = trace_device(&cb);
    log_reset();
    rvg_device_mmio_write(dev, 0x101c, GUEST_ROOT_PAGE);
    pin("api.calls.root_page", "%s", g_log);

    log_reset();
    suspend_resume(dev, 1);
    pin("api.calls.suspend_resume", "%s", g_log);

    log_reset();
    rvg_device_mmio_write(dev, 0x101c, 0);
    pin("api.calls.root_page_zero", "%s", g_log);

    log_reset();
    rvg_device_release(dev);
    pin("api.calls.release", "%s", g_log);
    g_trace_dev = NULL;
}

/* ---- each table's one destroy ---- */

static void on_raise(void *ctx, uint32_t vector) {
    (void)ctx;
    (void)vector;
}
static void *on_add_trace(void *ctx, RvgPhysicalMemoryRange range, RvgTraceRangeHandler *h) {
    (void)ctx;
    (void)range;
    (void)h;
    return NULL;
}
static void on_remove_trace(void *ctx, void *trace_range) {
    (void)ctx;
    (void)trace_range;
}
static void on_mode_change(void *ctx, uint16_t w, uint16_t h, uint32_t format) {
    (void)ctx;
    (void)w;
    (void)h;
    (void)format;
}
static void on_void(void *ctx) { (void)ctx; }

static RvgDisplayCallbacks display_callbacks(const char *destroy_name, int new_frame) {
    RvgDisplayCallbacks cb;
    memset(&cb, 0, sizeof cb);
    cb.struct_size = sizeof cb;
    cb.ctx = (void *)destroy_name;
    cb.destroy = named_destroy;
    cb.mode_change = on_mode_change;
    cb.cursor_move = on_void;
    if (new_frame) {
        cb.new_frame = on_void;
    }
    return cb;
}

static RvgStatus make_display(RvgDevice *dev, const RvgDisplayCallbacks *cb, uint64_t port,
                              uint32_t serial, RvgDisplay **out) {
    RvgDisplayConfig c;
    memset(&c, 0, sizeof c);
    c.struct_size = sizeof c;
    c.serial_number = serial;
    c.port = port;
    c.callbacks = cb;
    return rvg_display_create(dev, &c, out);
}

static void *release_display_thread(void *disp) {
    rvg_display_release((RvgDisplay *)disp);
    return NULL;
}

static atomic_int g_named_seen;
/* A destroy whose thread is not pinned: it counts, and logs nothing. */
static void seen_destroy(void *ctx) {
    (void)ctx;
    atomic_fetch_add(&g_named_seen, 1);
}

static void destroys(void) {
    /* A device's table at the device's teardown, and none before. */
    RvgHostCallbacks cb = trace_callbacks();
    cb.add_trace_range = on_add_trace;
    cb.remove_trace_range = on_remove_trace;
    RvgDevice *dev = make_device(&cb, 1);
    log_reset();
    tlog("made");
    rvg_device_release(dev);
    pin("api.destroy.device_teardown", "%s", g_log);

    /* A table with no function: at the host's release. */
    RvgHostCallbacks none;
    memset(&none, 0, sizeof none);
    none.struct_size = sizeof none;
    none.ctx = "destroy";
    none.destroy = named_destroy;
    log_reset();
    dev = make_device(&none, 1);
    tlog("made");
    rvg_device_release(dev);
    pin("api.destroy.device_no_function", "%s", g_log);

    /* Refused creates: the device at its port count, a display at a taken port, and one whose
     * new-frame source was made before the refusal: each destroy before the create returns. */
    log_reset();
    RvgDeviceConfig c = device_config(&cb, 0);
    RvgStatus s = rvg_device_create(&c, &dev);
    tlog("returned=%s", status_name(s));
    dev = make_device(&cb, 1);
    RvgDisplayCallbacks d0 = display_callbacks("ddestroy0", 0);
    RvgDisplay *disp = NULL;
    make_display(dev, &d0, 0, 0x5eedd000, &disp);
    RvgDisplayCallbacks d1 = display_callbacks("ddestroy1", 0);
    RvgDisplay *refused = NULL;
    s = make_display(dev, &d1, 0, 0x5eedd001, &refused);
    tlog("returned=%s", status_name(s));
    RvgDisplayCallbacks d2 = display_callbacks("ddestroy2", 1);
    s = make_display(dev, &d2, 0, 0x5eedd002, &refused);
    tlog("returned=%s", status_name(s));
    pin("api.destroy.refused", "%s", g_log);

    /* A display holding the device's last reference: its table, then the device's, on the
     * thread of the display's last release (the main thread, then another). */
    log_reset();
    rvg_device_release(dev);
    tlog("device_released");
    rvg_display_release(disp);
    pin("api.destroy.display_last_release_main", "%s", g_log);

    dev = make_device(&cb, 1);
    make_display(dev, &d0, 0, 0x5eedd003, &disp);
    log_reset();
    rvg_device_release(dev);
    tlog("device_released");
    pthread_t t;
    pthread_create(&t, NULL, release_display_thread, disp);
    pthread_join(t, NULL);
    pin("api.destroy.display_last_release_thread", "%s", g_log);

    /* With new_frame, the display's end drops its closures on the releasing thread and its
     * new-frame source's cancel drops that one on the display thread: the display's destroy
     * runs once, on whichever of them is last, maybe after the release returned; the
     * device's stays at its teardown. */
    dev = make_device(&cb, 1);
    RvgDisplayCallbacks d3 = display_callbacks("ddestroy3", 1);
    d3.destroy = seen_destroy;
    atomic_store(&g_named_seen, 0);
    make_display(dev, &d3, 0, 0x5eedd004, &disp);
    log_reset();
    rvg_display_release(disp);
    wait_flag(&g_named_seen);
    tlog("seen=%d", atomic_load(&g_named_seen));
    rvg_device_release(dev);
    tlog("seen=%d", atomic_load(&g_named_seen));
    pin("api.destroy.display_with_new_frame", "%s", g_log);

    /* The IOSurface device's table at its release. */
    static uint8_t page[0x4000] __attribute__((aligned(0x4000)));
    RvgMemoryRange r = {0, sizeof page, page};
    RvgMemoryMap m = {&r, 1};
    RvgIOSurfaceCallbacks icb = {sizeof icb, "idestroy", named_destroy, on_raise};
    RvgIOSurfaceDeviceConfig ic = {sizeof ic, NULL, &icb, 0};
    RvgIOSurfaceDevice *io = NULL;
    log_reset();
    s = rvg_iosurface_device_create(&ic, &io);
    tlog("returned=%s", status_name(s));
    ic.memory_map = &m;
    s = rvg_iosurface_device_create(&ic, &io);
    tlog("returned=%s", status_name(s));
    rvg_iosurface_device_release(io);
    pin("api.destroy.iosurface", "%s", g_log);
}

/* add_trace_range without remove_trace_range: a refusal, with the table's destroy. */
static void add_trace_alone(void) {
    RvgHostCallbacks cb = trace_callbacks();
    cb.add_trace_range = on_add_trace;
    RvgDeviceConfig c = device_config(&cb, 1);
    RvgDevice *dev = (RvgDevice *)0x1;
    log_reset();
    RvgStatus s = rvg_device_create(&c, &dev);
    tlog("returned=%s written=%d", status_name(s), dev != (RvgDevice *)0x1);
    pin("api.refusal.add_trace_without_remove", "%s", g_log);
}

/* ---- the EFI display: a host whose task functions give it memory ---- */

static void *efi_create(void *ctx, uint64_t size, void **base) {
    (void)ctx;
    static char task;
    if (!*base) {
        *base = calloc(1, size);
    }
    return &task;
}
static void efi_destroy_task(void *ctx, void *task) {
    (void)ctx;
    (void)task;
}
static bool efi_map(void *ctx, void *task, uint32_t count, uint64_t offset, bool read_only,
                    const RvgPhysicalMemoryRange *ranges) {
    (void)ctx;
    (void)task;
    (void)count;
    (void)offset;
    (void)read_only;
    (void)ranges;
    return true;
}
static bool efi_unmap(void *ctx, void *task, uint64_t offset, uint64_t length) {
    (void)ctx;
    (void)task;
    (void)offset;
    (void)length;
    return true;
}
static bool efi_read(void *ctx, uint64_t address, uint64_t length, void *dst) {
    (void)ctx;
    (void)address;
    memset(dst, 0, length);
    return false;
}
/* The trace functions: the handler is kept, and the range is its own token. */
static RvgTraceRangeHandler *g_kept[4];
static void *efi_add_trace(void *ctx, RvgPhysicalMemoryRange range, RvgTraceRangeHandler *h) {
    (void)ctx;
    (void)range;
    for (int i = 0; i < 4; i++) {
        if (!g_kept[i]) {
            g_kept[i] = rvg_trace_range_handler_retain(h);
            return &g_kept[i];
        }
    }
    return NULL;
}
static void efi_remove_trace(void *ctx, void *token) {
    (void)ctx;
    RvgTraceRangeHandler **slot = token;
    if (slot) {
        rvg_trace_range_handler_release(*slot);
        *slot = NULL;
    }
}

static RvgHostCallbacks efi_callbacks(int trace) {
    RvgHostCallbacks cb;
    memset(&cb, 0, sizeof cb);
    cb.struct_size = sizeof cb;
    cb.create_task = efi_create;
    cb.destroy_task = efi_destroy_task;
    cb.map_memory = efi_map;
    cb.unmap_memory = efi_unmap;
    cb.read_memory = efi_read;
    cb.raise_interrupt = on_raise;
    if (trace) {
        cb.add_trace_range = efi_add_trace;
        cb.remove_trace_range = efi_remove_trace;
    }
    return cb;
}

/* The suite's EFI bring-up (efi_on in tests/suite/scenarios/fresh_host.py): binary version, then
 * EFI on at 64x32, 32 bits, stride 0x100, over guest page 0x10002. */
static void efi_on(RvgDevice *dev) {
    rvg_device_mmio_write(dev, 0x1034, 0xf5);
    rvg_device_mmio_write(dev, 0x1200, 1);
    rvg_device_mmio_write(dev, 0x121c, 0x200040);
    rvg_device_mmio_write(dev, 0x1218, 0x20);
    rvg_device_mmio_write(dev, 0x1228, 0x100);
    rvg_device_mmio_write(dev, 0x1210, 0x10002);
    rvg_device_mmio_write(dev, 0x1214, 0x4000);
}

static void set_one_mode(RvgDisplay *disp) {
    RvgDisplayMode mode = {64, 32, 0, 60.0};
    rvg_display_set_modes(disp, &mode, 1);
}

/* ---- the display thread ----
 *
 * Two devices, each with the EFI display on (100 ms presents), each presenting to its port-0
 * display: every display callback of the process runs on the display thread. Each callback
 * notes its thread and whether another display callback was running; it takes a millisecond,
 * so overlapping calls would be seen. */
static pthread_t g_main;
static pthread_t g_display_thread;
static atomic_int g_display_threads;
static atomic_int g_inside, g_overlaps, g_on_main;
static atomic_int g_mode_changes[2], g_new_frames[2];
static pthread_mutex_t g_thread_lock = PTHREAD_MUTEX_INITIALIZER;

static void display_call(void) {
    if (atomic_fetch_add(&g_inside, 1) != 0) {
        atomic_fetch_add(&g_overlaps, 1);
    }
    if (pthread_equal(pthread_self(), g_main)) {
        atomic_fetch_add(&g_on_main, 1);
    }
    pthread_mutex_lock(&g_thread_lock);
    if (!atomic_load(&g_display_threads)) {
        g_display_thread = pthread_self();
        atomic_store(&g_display_threads, 1);
    } else if (!pthread_equal(g_display_thread, pthread_self())) {
        g_display_thread = pthread_self();
        atomic_fetch_add(&g_display_threads, 1);
    }
    pthread_mutex_unlock(&g_thread_lock);
    usleep(1000);
    atomic_fetch_sub(&g_inside, 1);
}
static void thread_mode_change(void *ctx, uint16_t w, uint16_t h, uint32_t format) {
    (void)w;
    (void)h;
    (void)format;
    display_call();
    atomic_fetch_add(&g_mode_changes[(intptr_t)ctx], 1);
}
static void thread_new_frame(void *ctx) {
    display_call();
    atomic_fetch_add(&g_new_frames[(intptr_t)ctx], 1);
}
static void thread_destroy(void *ctx) {
    (void)ctx;
    atomic_fetch_add(&g_named_seen, 1);
}

static void display_thread(void) {
    g_main = pthread_self();
    RvgHostCallbacks cb = efi_callbacks(0);
    RvgDevice *dev[2];
    RvgDisplay *disp[2];
    atomic_store(&g_named_seen, 0);
    for (intptr_t i = 0; i < 2; i++) {
        dev[i] = make_device(&cb, 1);
        RvgDisplayCallbacks dcb;
        memset(&dcb, 0, sizeof dcb);
        dcb.struct_size = sizeof dcb;
        dcb.ctx = (void *)i;
        dcb.destroy = thread_destroy;
        dcb.mode_change = thread_mode_change;
        dcb.new_frame = thread_new_frame;
        RvgStatus s = make_display(dev[i], &dcb, 0, 0x5eedb000u + (uint32_t)i, &disp[i]);
        if (s != RVG_OK) {
            fprintf(stderr, "display: %s\n", status_name(s));
            exit(3);
        }
        set_one_mode(disp[i]);
    }
    for (int i = 0; i < 2; i++) {
        efi_on(dev[i]);
    }
    for (int i = 0; i < 1000; i++) {
        if (atomic_load(&g_new_frames[0]) >= 3 && atomic_load(&g_new_frames[1]) >= 3) {
            break;
        }
        usleep(10000);
    }
    for (int i = 0; i < 2; i++) {
        rvg_device_mmio_write(dev[i], 0x1200, 0);
    }
    for (int i = 0; i < 2; i++) {
        rvg_display_release(disp[i]);
        rvg_device_release(dev[i]);
    }
    for (int i = 0; i < 1000 && atomic_load(&g_named_seen) < 2; i++) {
        usleep(10000);
    }
    pin("api.threads.display",
        "mode_change=%d,%d new_frame=%d,%d on_main=%d overlaps=%d threads=%d destroys=%d",
        atomic_load(&g_mode_changes[0]) > 0, atomic_load(&g_mode_changes[1]) > 0,
        atomic_load(&g_new_frames[0]) >= 3, atomic_load(&g_new_frames[1]) >= 3,
        atomic_load(&g_on_main), atomic_load(&g_overlaps), atomic_load(&g_display_threads),
        atomic_load(&g_named_seen));
}

/* ---- a display's life through a frame ----
 *
 * A display outlives the host's release while a frame holds it: its port stays taken and its
 * table stays until the frame completes, or until it is released uncommitted. */
static atomic_int g_life_destroyed;
static void life_destroy(void *ctx) {
    (void)ctx;
    atomic_fetch_add(&g_life_destroyed, 1);
}

static void display_lifetime(void) {
    RvgHostCallbacks cb = efi_callbacks(0);
    RvgDevice *dev = make_device(&cb, 1);
    RvgFrameTargetConfig tc = {sizeof tc, 64, 64, RVG_PIXEL_FORMAT_BGRA8_UNORM};
    RvgFrameTarget *target = NULL;
    rvg_frame_target_create(dev, &tc, &target);
    for (uint32_t commit = 0; commit < 2; commit++) {
        RvgDisplayCallbacks dcb;
        memset(&dcb, 0, sizeof dcb);
        dcb.struct_size = sizeof dcb;
        dcb.destroy = life_destroy;
        dcb.cursor_move = on_void;
        atomic_store(&g_life_destroyed, 0);
        RvgDisplay *disp = NULL;
        make_display(dev, &dcb, 0, 0x5eed0f00u + commit, &disp);
        RvgFrame *frame = NULL;
        RvgStatus enc = rvg_display_encode_frame(disp, target, NULL, &frame);
        if (commit) {
            rvg_frame_commit(frame);
        }
        rvg_display_release(disp);
        int after_release = atomic_load(&g_life_destroyed);
        RvgDisplay *probe = NULL;
        int port_free_before =
            make_display(dev, NULL, 0, 0x5eed0f10u + commit, &probe) == RVG_OK;
        rvg_display_release(probe);
        RvgStatus waited = commit ? rvg_frame_wait(frame) : RVG_OK;
        rvg_frame_release(frame);
        /* Metal lets a completed buffer's handler go on its own thread, maybe after the wait
         * returned: wait up to 5 s for the display's end. */
        int port_free_after = 0;
        for (uint32_t tries = 0; tries < 500 && !port_free_after; tries++) {
            if (tries) {
                usleep(10000);
            }
            if (atomic_load(&g_life_destroyed)) {
                probe = NULL;
                port_free_after = make_display(dev, NULL, 0,
                                               0x5eed1000u + 0x1000u * commit + tries,
                                               &probe) == RVG_OK;
                rvg_display_release(probe);
            }
        }
        pin(commit ? "api.lifetime.display_through_a_committed_frame"
                   : "api.lifetime.display_through_an_uncommitted_frame",
            "encoded=%s waited=%s destroyed=%d,%d port_free=%d,%d", status_name(enc),
            status_name(waited), after_release, atomic_load(&g_life_destroyed),
            port_free_before, port_free_after);
    }
    rvg_frame_target_release(target);
    rvg_device_release(dev);
}

/* ---- a window display with no run loop ----
 *
 * This process's main thread never runs a run loop, so no window opens. A display created with
 * a window (flags 0: no host screen is read) and released does not hang: the window's table
 * ends once, at the release, on the releasing thread; the display's once, maybe on the display
 * thread after the release returned, since a window display has new_frame; the device's at its
 * teardown. */
static void on_window_pointer(void *ctx, uint16_t x, uint16_t y, uint16_t w, uint16_t h,
                              uint32_t buttons) {
    (void)ctx;
    (void)x;
    (void)y;
    (void)w;
    (void)h;
    (void)buttons;
}
static void on_window_wheel(void *ctx, int32_t dy, int32_t dx) {
    (void)ctx;
    (void)dy;
    (void)dx;
}
static void on_window_key(void *ctx, uint16_t usage, bool down) {
    (void)ctx;
    (void)usage;
    (void)down;
}

static void window_no_run_loop(void) {
    RvgHostCallbacks cb = efi_callbacks(0);
    cb.destroy = named_destroy;
    cb.ctx = "destroy";
    RvgDevice *dev = make_device(&cb, 1);
    RvgWindowCallbacks wcb;
    memset(&wcb, 0, sizeof wcb);
    wcb.struct_size = sizeof wcb;
    wcb.ctx = "wdestroy";
    wcb.destroy = named_destroy;
    wcb.pointer = on_window_pointer;
    wcb.wheel = on_window_wheel;
    wcb.key = on_window_key;
    wcb.close = on_void;
    RvgWindowConfig wc = {sizeof wc, 0, &wcb, 0, 0, 0};
    RvgDisplayCallbacks dcb = display_callbacks("ddestroy", 0);
    dcb.destroy = seen_destroy;
    RvgDisplayConfig c;
    memset(&c, 0, sizeof c);
    c.struct_size = sizeof c;
    c.serial_number = 0x5eedf000;
    c.callbacks = &dcb;
    c.window = &wc;
    atomic_store(&g_named_seen, 0);
    log_reset();
    RvgDisplay *disp = NULL;
    RvgStatus s = rvg_display_create(dev, &c, &disp);
    tlog("created=%s", status_name(s));
    rvg_display_release(disp);
    tlog("released");
    wait_flag(&g_named_seen);
    tlog("ddestroys=%d", atomic_load(&g_named_seen));
    rvg_device_release(dev);
    tlog("ddestroys=%d", atomic_load(&g_named_seen));
    pin("api.window.no_run_loop", "%s", g_log);
}

/* ---- the modes as the guest reads them ----
 *
 * The EFI mode registers (0x1204 the count, 0x1208 the select, 0x120c the value) answer the
 * guest from the mode list of the display on port 0. */
static void efi_modes(void) {
    RvgHostCallbacks cb = efi_callbacks(0);
    RvgDevice *dev = make_device(&cb, 1);
    RvgDisplay *disp = NULL;
    make_display(dev, NULL, 0, 0x5eed0e02, &disp);
    RvgDisplayMode m[3] = {{64, 32, 0, 60.0},
                           {1920, 1080, RVG_DISPLAY_MODE_EDR, 60.0},
                           {800, 600, RVG_DISPLAY_MODE_10BPC, 30.0}};
    rvg_display_set_modes(disp, m, 3);
    rvg_device_mmio_write(dev, 0x1034, 0xf5);
    rvg_device_mmio_write(dev, 0x1200, 1);
    char buf[256] = "";
    uint32_t count = rvg_device_mmio_read(dev, 0x1204);
    snprintf(buf, sizeof buf, "count=%u", count);
    for (uint32_t i = 0; i <= count && i < 8; i++) {
        char v[32];
        rvg_device_mmio_write(dev, 0x1208, i);
        snprintf(v, sizeof v, " %u=%#x", i, rvg_device_mmio_read(dev, 0x120c));
        strlcat(buf, v, sizeof buf);
    }
    rvg_device_mmio_write(dev, 0x1200, 0);
    pin("api.display.efi_modes", "%s", buf);
    rvg_display_release(disp);
    rvg_device_release(dev);
}

/* ---- the device's registers and a resume's refusals ---- */

static void registers(RvgDevice *dev, const char *id_) {
    char buf[1024] = "";
    const uint32_t spans[2][2] = {{0x1000, 0x1040}, {0x1200, 0x1238}};
    for (int s = 0; s < 2; s++) {
        for (uint32_t o = spans[s][0]; o < spans[s][1]; o += 4) {
            char v[32];
            snprintf(v, sizeof v, "%x=%x,", o, rvg_device_mmio_read(dev, o));
            strlcat(buf, v, sizeof buf);
        }
    }
    pin(id_, "%s", buf);
}

/* A fresh device's registers, with three ports, and again after the guest wrote its binary
 * version; a resume of no state, of no state with no error to write, and of an empty state; a
 * display's cursor position before the guest placed it. */
static void device_basics(void) {
    RvgHostCallbacks cb = efi_callbacks(0);
    RvgDevice *dev = make_device(&cb, 3);
    registers(dev, "api.device.registers");
    rvg_device_mmio_write(dev, 0x1034, 0xf5);
    registers(dev, "api.device.registers_after_version");

    RvgResumeError e = 99, e_empty = 99;
    RvgStatus none = rvg_device_will_resume(dev, NULL, 0, &e);
    RvgStatus no_error = rvg_device_will_resume(dev, NULL, 0, NULL);
    static const uint8_t empty_state[1];
    RvgStatus empty = rvg_device_will_resume(dev, empty_state, 0, &e_empty);
    pin("api.device.resume_errors", "none=%s,%d no_error=%s empty=%s,%d", status_name(none),
        e == RVG_RESUME_ERROR_NONE ? -1 : (int)e, status_name(no_error), status_name(empty),
        e_empty == RVG_RESUME_ERROR_NONE ? -1 : (int)e_empty);

    RvgDisplay *disp = NULL;
    make_display(dev, NULL, 0, 0x5eed0c01, &disp);
    RvgDisplayCoord at = rvg_display_cursor_position(disp);
    pin("api.display.cursor_position", "%u,%u", at.x, at.y);
    rvg_display_release(disp);
    rvg_device_release(dev);
}

/* ---- the EFI display's traced range ----
 *
 * The EFI display over guest page 0x10002 asks the host to trace its range; the host reports
 * part of it dirty through the handler it was lent, and the display presents again. Its end
 * removes the range and unmaps it. The host's calls, in order, with the marks between them; and
 * the first two display calls, both on the display thread. */
static RvgTraceRangeHandler *g_traced;
static void *tr_create(void *ctx, uint64_t size, void **base) {
    (void)ctx;
    static char task;
    tlog("createTask(%#llx,base_in=%d)", (unsigned long long)size, *base != NULL);
    if (!*base) {
        *base = calloc(1, size);
    }
    return &task;
}
static void tr_destroy_task(void *ctx, void *task) {
    (void)ctx;
    (void)task;
    tlog("destroyTask(T)");
}
static bool tr_map(void *ctx, void *task, uint32_t count, uint64_t offset, bool read_only,
                   const RvgPhysicalMemoryRange *ranges) {
    (void)ctx;
    (void)task;
    tlog("map(%u,%#llx,ro=%d,%#llx+%#llx)", count, (unsigned long long)offset, read_only,
         (unsigned long long)(count ? ranges[0].physical_address : 0),
         (unsigned long long)(count ? ranges[0].physical_length : 0));
    return true;
}
static bool tr_unmap(void *ctx, void *task, uint64_t offset, uint64_t length) {
    (void)ctx;
    (void)task;
    tlog("unmap(%#llx,%#llx)", (unsigned long long)offset, (unsigned long long)length);
    return true;
}
static void *tr_add_trace(void *ctx, RvgPhysicalMemoryRange range, RvgTraceRangeHandler *h) {
    (void)ctx;
    tlog("addTraceRange(%#llx+%#llx,handler=%d)", (unsigned long long)range.physical_address,
         (unsigned long long)range.physical_length, h != NULL);
    g_traced = rvg_trace_range_handler_retain(h);
    return &g_traced;
}
static void tr_remove_trace(void *ctx, void *token) {
    (void)ctx;
    tlog("removeTraceRange(%s)", token == &g_traced ? "the token" : token ? "?" : "NULL");
}

static pthread_mutex_t g_dcalls_lock = PTHREAD_MUTEX_INITIALIZER;
static char g_dcalls[256];
static int g_dcall_count;
static atomic_int g_tr_frames;
static void dcall(const char *what) {
    pthread_mutex_lock(&g_dcalls_lock);
    if (g_dcall_count++ < 2) {
        strlcat(g_dcalls, what, sizeof g_dcalls);
        strlcat(g_dcalls, pthread_main_np() ? "@m," : "@x,", sizeof g_dcalls);
    }
    pthread_mutex_unlock(&g_dcalls_lock);
}
static void tr_mode_change(void *ctx, uint16_t w, uint16_t h, uint32_t format) {
    (void)ctx;
    (void)format;
    char v[48];
    snprintf(v, sizeof v, "modeChange(%ux%u)", w, h);
    dcall(v);
}
static void tr_new_frame(void *ctx) {
    (void)ctx;
    dcall("newFrame");
    atomic_fetch_add(&g_tr_frames, 1);
}
static int wait_frames(int n) {
    for (int i = 0; i < 500; i++) {
        if (atomic_load(&g_tr_frames) >= n) {
            return 1;
        }
        usleep(10000);
    }
    return 0;
}

static void trace_ranges(void) {
    RvgHostCallbacks cb = efi_callbacks(0);
    cb.create_task = tr_create;
    cb.destroy_task = tr_destroy_task;
    cb.map_memory = tr_map;
    cb.unmap_memory = tr_unmap;
    cb.add_trace_range = tr_add_trace;
    cb.remove_trace_range = tr_remove_trace;
    RvgDevice *dev = make_device(&cb, 1);
    RvgDisplayCallbacks dcb;
    memset(&dcb, 0, sizeof dcb);
    dcb.struct_size = sizeof dcb;
    dcb.mode_change = tr_mode_change;
    dcb.new_frame = tr_new_frame;
    RvgDisplay *disp = NULL;
    make_display(dev, &dcb, 0, 0x5eed0e01, &disp);
    set_one_mode(disp);
    log_reset();
    efi_on(dev);
    int first = wait_frames(1);
    tlog("presented");
    int fired = 0;
    if (g_traced) {
        RvgPhysicalMemoryRange dirty = {0x40008000, 0x100};
        rvg_trace_range_handler_invoke(g_traced, dirty);
        fired = 1;
    }
    int second = wait_frames(2);
    tlog("dirty-presented");
    rvg_device_mmio_write(dev, 0x1200, 0);
    tlog("efi-off");
    rvg_display_release(disp);
    rvg_device_release(dev);
    tlog("released");
    pin("api.trace.efi_ranges", "%s first=%d fired=%d second=%d", g_log, first, fired, second);
    pthread_mutex_lock(&g_dcalls_lock);
    pin("api.trace.efi_display_calls", "%s", g_dcalls);
    pthread_mutex_unlock(&g_dcalls_lock);
    rvg_trace_range_handler_release(g_traced);
    g_traced = NULL;
}

/* ---- the feature level and the connection types, as the guest reads them ----
 *
 * The suite's runner leaves both at 0. A device with three ports at each level: its create,
 * the binary version the guest reads back (0x1034) after asking for 0xf5, and the display port
 * count it reads (0x122c). Levels above 4 are accepted at the create; the guest's version write
 * then ends the process (api.death.feature_level_above_4_*). */
static void feature_levels(void) {
    RvgHostCallbacks cb = efi_callbacks(0);
    const uint64_t levels[] = {0, 1, 2, 3, 4, 5, UINT64_MAX};
    char buf[512] = "";
    for (size_t i = 0; i < sizeof levels / sizeof *levels; i++) {
        RvgDeviceConfig c = device_config(&cb, 3);
        c.feature_level = levels[i];
        RvgDevice *dev = NULL;
        RvgStatus s = rvg_device_create(&c, &dev);
        char v[96];
        if (s == RVG_OK && levels[i] <= 4) {
            rvg_device_mmio_write(dev, 0x1034, 0xf5);
            snprintf(v, sizeof v, "%s%llu=%s,%#x,%u", i ? " " : "",
                     (unsigned long long)levels[i], status_name(s),
                     rvg_device_mmio_read(dev, 0x1034), rvg_device_mmio_read(dev, 0x122c));
        } else {
            snprintf(v, sizeof v, " %#llx=%s", (unsigned long long)levels[i], status_name(s));
        }
        strlcat(buf, v, sizeof buf);
        rvg_device_release(dev);
    }
    pin("api.device.feature_levels", "%s", buf);
}

/* The connection types 0xa5 on a device with three ports: the register (0x1234) before the
 * guest's version write, after it asks for 0xf5, and at feature level 1, where 0xf5 is stored
 * as 0x3e; then the state of the first device, at version 0xf5, offered to a device with the
 * types 0x5a and to one with 0xa5: the resume's status and reason. */
static RvgDevice *typed_device(const RvgHostCallbacks *cb, uint32_t types, uint64_t level) {
    RvgDeviceConfig c = device_config(cb, 3);
    c.display_port_connection_types = types;
    c.feature_level = level;
    RvgDevice *dev = NULL;
    RvgStatus s = rvg_device_create(&c, &dev);
    if (s != RVG_OK) {
        fprintf(stderr, "rvg_device_create: %s\n", status_name(s));
        exit(3);
    }
    return dev;
}

static void connection_types(void) {
    RvgHostCallbacks cb = efi_callbacks(0);
    RvgDevice *dev = typed_device(&cb, 0xa5, 0);
    uint32_t before = rvg_device_mmio_read(dev, 0x1234);
    rvg_device_mmio_write(dev, 0x1034, 0xf5);
    uint32_t after = rvg_device_mmio_read(dev, 0x1234);
    RvgDevice *level1 = typed_device(&cb, 0xa5, 1);
    rvg_device_mmio_write(level1, 0x1034, 0xf5);
    uint32_t at_level1 = rvg_device_mmio_read(level1, 0x1234);
    rvg_device_release(level1);

    rvg_device_will_suspend(dev);
    RvgBytes state = {NULL, 0};
    RvgStatus fs = rvg_device_finish_suspend(dev, &state);
    RvgDevice *other = typed_device(&cb, 0x5a, 0);
    RvgResumeError e_other = 99, e_same = 99;
    RvgStatus r_other = rvg_device_will_resume(other, state.data, state.length, &e_other);
    RvgDevice *same = typed_device(&cb, 0xa5, 0);
    RvgStatus r_same = rvg_device_will_resume(same, state.data, state.length, &e_same);
    rvg_device_did_resume(same);
    rvg_bytes_free(state);
    pin("api.device.connection_types",
        "before_version=%#x after=%#x level1=%#x finish=%s resume_other=%s,%d resume_same=%s,%d",
        before, after, at_level1, status_name(fs), status_name(r_other),
        e_other == RVG_RESUME_ERROR_NONE ? -1 : (int)e_other, status_name(r_same),
        e_same == RVG_RESUME_ERROR_NONE ? -1 : (int)e_same);
    rvg_device_release(same);
    rvg_device_release(other);
    rvg_device_release(dev);
}

/* ---- the log hook ----
 *
 * In a child (the hook is the process's), with REIMS_VGPU_LOG unset, since its filter would
 * pick the lines in place of `max_level`. Each phase sets the hook, then makes and releases a
 * device at feature level 1, which logs an info line as it is made, and one at level 5, which
 * logs an error line: the levels of the lines the handler gets on this thread, lowest and
 * highest, or none. Lines of other threads are not counted, since when they come is not
 * fixed; every line is checked for the ctx and for its level and strings. */
static atomic_int g_log_lines, g_log_lowest, g_log_highest, g_log_bad;

static void on_log_line(void *ctx, RvgLogLevel level, const char *target, size_t target_length,
                        const char *message, size_t message_length) {
    if (ctx != (void *)&g_log_lines || level < RVG_LOG_ERROR || level > RVG_LOG_TRACE ||
        target == NULL || target_length == 0 || (message == NULL && message_length != 0)) {
        atomic_fetch_add(&g_log_bad, 1);
    }
    if (!pthread_main_np()) {
        return;
    }
    atomic_fetch_add(&g_log_lines, 1);
    if ((int)level < atomic_load(&g_log_lowest)) {
        atomic_store(&g_log_lowest, (int)level);
    }
    if ((int)level > atomic_load(&g_log_highest)) {
        atomic_store(&g_log_highest, (int)level);
    }
}

static void log_phase(const char *name, RvgLogFn handler, RvgLogLevel max_level) {
    rvg_log_set_handler(handler, (void *)&g_log_lines, max_level);
    atomic_store(&g_log_lines, 0);
    atomic_store(&g_log_lowest, 99);
    atomic_store(&g_log_highest, 0);
    RvgHostCallbacks cb = efi_callbacks(0);
    const uint64_t levels[] = {1, 5};
    for (int i = 0; i < 2; i++) {
        RvgDeviceConfig c = device_config(&cb, 1);
        c.feature_level = levels[i];
        RvgDevice *dev = NULL;
        if (rvg_device_create(&c, &dev) == RVG_OK) {
            rvg_device_release(dev);
        }
    }
    if (atomic_load(&g_log_lines)) {
        printf("%s: %d-%d\n", name, atomic_load(&g_log_lowest), atomic_load(&g_log_highest));
    } else {
        printf("%s: none\n", name);
    }
}

static int log_handler(void) {
    unsetenv("REIMS_VGPU_LOG");
    log_phase("error", on_log_line, RVG_LOG_ERROR);
    log_phase("info", on_log_line, RVG_LOG_INFO);
    log_phase("off", on_log_line, RVG_LOG_OFF);
    log_phase("error_again", on_log_line, RVG_LOG_ERROR);
    log_phase("removed", NULL, RVG_LOG_TRACE);
    printf("bad=%d\n", atomic_load(&g_log_bad));
    return 0;
}

/* ---- the files behind the memory map's ranges ----
 *
 * In a child, since the log handler is the process's. Each create prints its status and the
 * destroys its table has run, and each refusal the C layer's line saying which check refused:
 * files without a map, an fd below -1, flags, an fd fstat cannot read, a file shorter than the
 * range, an unaligned offset, in that order, the first range's first. `fds` counts the
 * process's open descriptors against the count before the first case, taken after a device
 * made and released without files has opened what the process keeps for Metal: a refusal
 * leaves none open, and on macOS, where the files are checked and not used, an accepted device
 * keeps no duplicate past its create. */

static atomic_int g_files_destroys;

static void files_destroy(void *ctx) {
    (void)ctx;
    atomic_fetch_add(&g_files_destroys, 1);
}

static void on_files_line(void *ctx, RvgLogLevel level, const char *target, size_t target_length,
                          const char *message, size_t message_length) {
    (void)ctx;
    static const char want[] = "reims_vgpu_c::device";
    if (level == RVG_LOG_WARN && target_length == sizeof want - 1 &&
        memcmp(target, want, target_length) == 0) {
        printf("line: %.*s\n", (int)message_length, message);
    }
}

static int open_fds(void) {
    int n = 0;
    for (int fd = 0; fd < 1024; fd++) {
        if (fcntl(fd, F_GETFD) != -1) {
            n++;
        }
    }
    return n;
}

static int memory_files(void) {
    unsetenv("REIMS_VGPU_LOG");
    rvg_log_set_handler(on_files_line, NULL, RVG_LOG_WARN);
    char path[] = "/tmp/rvg-api-pins-ram.XXXXXX";
    int fd = mkstemp(path);
    unlink(path);
    if (fd < 0 || ftruncate(fd, 0x10000) != 0) {
        return 3;
    }
    static uint8_t ram[0x8000] __attribute__((aligned(0x4000)));
    RvgMemoryRange ranges[2] = {{0, 0x4000, ram}, {0x4000, 0x4000, ram + 0x4000}};
    RvgMemoryMap map = {ranges, 2};
    RvgMemoryRangeFile files[2] = {{fd, 0, 0}, {fd, 0, 0x4000}};
    RvgHostCallbacks cb;
    memset(&cb, 0, sizeof cb);
    cb.struct_size = sizeof cb;
    cb.destroy = files_destroy;
    RvgDeviceConfig c = device_config(&cb, 1);
    c.memory_map = &map;
    RvgDevice *warm = NULL;
    if (rvg_device_create(&c, &warm) != RVG_OK) {
        return 4;
    }
    rvg_device_release(warm);
    atomic_store(&g_files_destroys, 0);
    c.memory_files = files;
    const int before = open_fds();
    struct {
        const char *name;
        int map, fd0, fd1;
        uint32_t flags1;
        uint64_t offset0, offset1;
        uint32_t struct_size;
    } cases[] = {
        {"no_map", 0, fd, fd, 0, 0, 0x4000, sizeof c},
        {"fd_below", 1, fd, -2, 0, 0, 0x4000, sizeof c},
        {"flags", 1, fd, 1 << 20, 1, 0, 0x4000, sizeof c},
        {"unreadable", 1, fd, 1 << 20, 0, 0, 0x4000, sizeof c},
        {"short", 1, fd, fd, 0, 0, 0xc010, sizeof c},
        {"unaligned", 1, fd, fd, 0, 0, 0x10, sizeof c},
        {"first_range_first", 1, fd, -2, 0, 0x10, 0x4000, sizeof c},
        {"first_version", 1, fd, -2, 1, 0x10, 0x4000, 48},
        {"no_file", 1, -1, -1, 0, 0x10, 0x10, sizeof c},
        {"ok", 1, fd, fd, 0, 0, 0x4000, sizeof c},
    };
    for (size_t i = 0; i < sizeof cases / sizeof cases[0]; i++) {
        c.memory_map = cases[i].map ? &map : NULL;
        files[0].fd = cases[i].fd0;
        files[0].offset = cases[i].offset0;
        files[1].fd = cases[i].fd1;
        files[1].flags = cases[i].flags1;
        files[1].offset = cases[i].offset1;
        c.struct_size = cases[i].struct_size;
        RvgDevice *dev = NULL;
        RvgStatus s = rvg_device_create(&c, &dev);
        printf("%s=%s destroys=%d fds=%+d\n", cases[i].name, status_name(s),
               atomic_load(&g_files_destroys), open_fds() - before);
        if (s == RVG_OK) {
            rvg_device_release(dev);
            printf("released destroys=%d fds=%+d\n", atomic_load(&g_files_destroys),
                   open_fds() - before);
        }
    }
    close(fd);
    return 0;
}

/* ---- deaths ---- */

extern char **environ;
static const char *g_self;

/* The child's fate: "signal=N" or "exit=N", with what the child printed (its lines, joined by
 * `|`) when it printed anything. */
static void child(const char *id_, const char *name) {
    pid_t pid;
    char *argv[] = {(char *)g_self, "--die", (char *)name, NULL};
    char out_path[] = "/tmp/rvg-api-pins-child.XXXXXX";
    int out_fd = mkstemp(out_path);
    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_adddup2(&fa, out_fd, 1);
    posix_spawn_file_actions_addopen(&fa, 2, "/dev/null", O_WRONLY, 0);
    int status = -1;
    if (posix_spawn(&pid, g_self, &fa, NULL, argv, environ) == 0) {
        waitpid(pid, &status, 0);
    }
    posix_spawn_file_actions_destroy(&fa);
    char printed[4096] = "";
    ssize_t n = pread(out_fd, printed, sizeof printed - 1, 0);
    printed[n > 0 ? n : 0] = 0;
    for (char *p = printed; *p; p++) {
        if (*p == '\n') {
            *p = p[1] ? '|' : 0;
        }
    }
    close(out_fd);
    unlink(out_path);
    if (WIFSIGNALED(status) && printed[0]) {
        pin(id_, "signal=%d printed=%s", WTERMSIG(status), printed);
    } else if (WIFSIGNALED(status)) {
        pin(id_, "signal=%d", WTERMSIG(status));
    } else if (printed[0]) {
        pin(id_, "exit=%d printed=%s", WEXITSTATUS(status), printed);
    } else {
        pin(id_, "exit=%d", WEXITSTATUS(status));
    }
}

static int die(const char *name) {
    if (strcmp(name, "survives") == 0) {
        puts("survived");
        return 0;
    }
    if (strcmp(name, "no_raise_interrupt") == 0) {
        /* rvg_device_did_resume raises vector 0 (api.calls.suspend_resume): the first
         * interrupt. */
        RvgHostCallbacks cb = trace_callbacks();
        cb.raise_interrupt = NULL;
        RvgDevice *dev = trace_device(&cb);
        rvg_device_mmio_write(dev, 0x101c, GUEST_ROOT_PAGE);
        suspend_resume(dev, 0);
        puts("survived");
        return 0;
    }
    if (strcmp(name, "host_raise_uncaught") == 0) {
        RvgHostCallbacks cb = trace_callbacks();
        RvgDevice *dev = trace_device(&cb);
        g_create_raises = 1;
        rvg_device_mmio_write(dev, 0x101c, GUEST_ROOT_PAGE);
        puts("survived");
        return 0;
    }
    if (strcmp(name, "raise_under_a_scope") == 0 ||
        strcmp(name, "reentrant_raise_under_a_scope") == 0) {
        /* rvg_device_did_resume reads the guest's task root through read_memory. A
         * read_memory that raises ends the process at the call; one that re-enters the device,
         * whose create_task then raises, ends it too. */
        RvgHostCallbacks cb = trace_callbacks();
        RvgDevice *dev = trace_device(&cb);
        rvg_device_mmio_write(dev, 0x101c, GUEST_ROOT_PAGE);
        rvg_device_will_suspend(dev);
        RvgBytes state = {NULL, 0};
        rvg_device_finish_suspend(dev, &state);
        rvg_device_will_resume(dev, state.data, state.length, NULL);
        rvg_bytes_free(state);
        if (strcmp(name, "raise_under_a_scope") == 0) {
            g_read_raises = 1;
        } else {
            g_read_reenters = 1;
        }
        rvg_device_did_resume(dev);
        puts("survived");
        return 0;
    }
    if (strcmp(name, "resume_child_mapper") == 0) {
        /* A resume-child write of 92, past the 64 child FIFOs, to a device made with
         * use_iosurface_mapper. */
        RvgHostCallbacks cb = efi_callbacks(0);
        RvgDeviceConfig c = device_config(&cb, 1);
        c.use_iosurface_mapper = true;
        RvgDevice *dev = NULL;
        if (rvg_device_create(&c, &dev) != RVG_OK) {
            puts("no device");
            return 3;
        }
        puts("resume 92");
        fflush(stdout);
        rvg_device_mmio_write(dev, 0x1028, 92);
        _exit(0);
    }
    if (strcmp(name, "resume_child_past_the_fifos") == 0) {
        /* Resume-child writes of ids past the 64 child FIFOs, the first past them among them:
         * none resumes a FIFO. */
        RvgHostCallbacks cb = efi_callbacks(0);
        RvgDevice *dev = make_device(&cb, 1);
        const uint32_t fifos[] = {64, 92, 198};
        for (size_t i = 0; i < sizeof fifos / sizeof fifos[0]; i++) {
            printf("resume %u\n", fifos[i]);
            fflush(stdout);
            rvg_device_mmio_write(dev, 0x1028, fifos[i]);
        }
        _exit(0); /* the device's teardown is not what this pins */
    }
    if (strcmp(name, "feature_level_above_4_version_write") == 0 ||
        strcmp(name, "feature_level_above_4_reset") == 0) {
        /* A device at feature level 5, accepted at its create; the guest's first binary
         * version write, or the host's reset, ends the process. */
        RvgHostCallbacks cb = efi_callbacks(0);
        RvgDeviceConfig c = device_config(&cb, 1);
        c.feature_level = 5;
        RvgDevice *dev = NULL;
        printf("created=%s\n", status_name(rvg_device_create(&c, &dev)));
        fflush(stdout);
        if (strcmp(name, "feature_level_above_4_reset") == 0) {
            rvg_device_reset(dev);
        } else {
            rvg_device_mmio_write(dev, 0x1034, 0xf5);
        }
        puts("survived");
        return 0;
    }
    if (strcmp(name, "log_handler") == 0) {
        return log_handler();
    }
    if (strcmp(name, "memory_files") == 0) {
        return memory_files();
    }
    return 2;
}

static void deaths(void) {
    child("api.death.control", "survives");
    child("api.death.no_raise_interrupt", "no_raise_interrupt");
    child("api.death.host_raise_uncaught", "host_raise_uncaught");
    child("api.death.raise_under_a_scope", "raise_under_a_scope");
    child("api.death.reentrant_raise_under_a_scope", "reentrant_raise_under_a_scope");
    child("api.device.resume_child_mapper_alias", "resume_child_mapper");
    child("api.device.resume_child_past_the_fifos", "resume_child_past_the_fifos");
}

int main(int argc, char **argv) {
    g_self = argv[0];
    if (argc == 3 && strcmp(argv[1], "--die") == 0) {
        return die(argv[2]);
    }
    calls();
    destroys();
    add_trace_alone();
    display_thread();
    display_lifetime();
    window_no_run_loop();
    efi_modes();
    device_basics();
    trace_ranges();
    deaths();
    feature_levels();
    connection_types();
    child("api.death.feature_level_above_4_version_write", "feature_level_above_4_version_write");
    child("api.death.feature_level_above_4_reset", "feature_level_above_4_reset");
    child("api.log.handler", "log_handler");
    child("api.device.memory_files", "memory_files");
    return 0;
}
