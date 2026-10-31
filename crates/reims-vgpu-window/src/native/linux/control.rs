//! The windows' control socket on Linux: with `REIMS_VGPU_WINDOW_CONTROL=<path>` set, a Unix
//! stream socket at that path, which a client (vphone-linux's) drives the windows through, in
//! `crate::control`'s protocol. Unset, nothing here runs: no socket, no thread, and the windows
//! are as they are without it.
//!
//! **Threads.** The socket is bound at the first window's opening ([`start`]), a stale file at
//! the path removed first, and served on threads of its own, so neither the UI thread nor a
//! render thread ever waits on it: `reims-vgpu.ctl` accepts, one reader per client reads its
//! lines, and `reims-vgpu.ctl-out` writes what the windows tell the client. One client at a time:
//! a new connection replaces the one before, which is shut down. Each request read is posted to
//! the UI thread ([`super::ui::post`]), which applies it to every window
//! (`super::sdl::control`), in the order read; a toolbar's atlas is read from its file on the
//! reader's thread first, and one that is not the size its request says is refused there. What a
//! window tells the client ([`send`]) is queued for the writer, which drops it where no client
//! is connected or the client is gone.

use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Sender};
use std::sync::{Arc, OnceLock};

use reims_vgpu_core::rt::spawn_named;
use reims_vgpu_core::sync::Mutex;
use reims_vgpu_vulkan::os::OsThreads;

use crate::control::{self, Request};
use crate::decor::Atlas;

/// The control socket's variable.
pub(crate) const CONTROL_VAR: &str = "REIMS_VGPU_WINDOW_CONTROL";

/// The control threads' stacks.
const STACK: usize = 256 * 1024;

/// What the UI thread is asked to do, the atlas read.
pub(crate) enum Control {
    /// Show the bar with these buttons.
    Toolbar {
        /// Their images.
        atlas: Arc<Atlas>,
        /// Their ids, in the atlas's order.
        ids: Vec<String>,
    },
    /// Hide the bar.
    ToolbarOff,
    /// Turn the image.
    Rotate(crate::decor::Rotation),
}

/// The socket, once served.
struct Served {
    /// The lines for the client, to the writer.
    out: Mutex<Sender<String>>,
}

/// The socket, once [`start`] has bound it.
static SERVED: OnceLock<Option<Served>> = OnceLock::new();

/// Binds the control socket and starts serving it, once, where `REIMS_VGPU_WINDOW_CONTROL` names a
/// path (the module doc). A socket that cannot be bound is logged, and the windows go on without
/// one.
pub(crate) fn start() {
    SERVED.get_or_init(|| {
        let path = std::env::var_os(CONTROL_VAR).filter(|p| !p.is_empty())?;
        serve(PathBuf::from(path))
    });
}

/// Tells the client `line`, where one is connected: queued for the writer, never waited on.
pub(crate) fn send(line: String) {
    if let Some(Some(served)) = SERVED.get() {
        let _ = served.out.lock().send(line);
    }
}

/// Binds the socket at `path` and starts its threads.
fn serve(path: PathBuf) -> Option<Served> {
    // A file left by an earlier run: a socket bound there again would be refused.
    let _ = std::fs::remove_file(&path);
    let listener = match UnixListener::bind(&path) {
        Ok(listener) => listener,
        Err(e) => {
            log::warn!(
                target: "reims_vgpu_window",
                "no control socket at {}: {e}",
                path.display()
            );
            return None;
        }
    };
    log::info!(
        target: "reims_vgpu_window",
        "the windows take requests on {}",
        path.display()
    );
    let client: Arc<Mutex<Option<UnixStream>>> = Arc::default();
    let (out, lines) = mpsc::channel::<String>();
    let writer_client = Arc::clone(&client);
    spawn_named(
        &OsThreads::without_hints(),
        "reims-vgpu.ctl-out",
        STACK,
        Box::new(()),
        move || {
            for line in lines {
                let stream = writer_client
                    .lock()
                    .as_ref()
                    .and_then(|s| s.try_clone().ok());
                if let Some(mut stream) = stream {
                    let _ = stream.write_all(line.as_bytes());
                }
            }
        },
    );
    spawn_named(
        &OsThreads::without_hints(),
        "reims-vgpu.ctl",
        STACK,
        Box::new(()),
        move || accept(&listener, &client),
    );
    Some(Served {
        out: Mutex::new(out),
    })
}

/// Accepts each client in turn: the one before is shut down, and the new one read.
fn accept(listener: &UnixListener, client: &Mutex<Option<UnixStream>>) {
    for stream in listener.incoming() {
        let stream = match stream {
            Ok(stream) => stream,
            Err(e) => {
                log::warn!(target: "reims_vgpu_window", "control socket: {e}");
                continue;
            }
        };
        let Ok(reader) = stream.try_clone() else {
            continue;
        };
        if let Some(old) = client.lock().replace(stream) {
            let _ = old.shutdown(std::net::Shutdown::Both);
        }
        log::info!(target: "reims_vgpu_window", "control socket: a client connected");
        spawn_named(
            &OsThreads::without_hints(),
            "reims-vgpu.ctl-in",
            STACK,
            Box::new(()),
            move || read(reader),
        );
    }
}

/// Reads a client's lines until it is gone, posting each request to the UI thread.
fn read(stream: UnixStream) {
    for line in BufReader::new(stream).lines() {
        let Ok(line) = line else {
            break;
        };
        if line.trim().is_empty() {
            continue;
        }
        let request = control::parse(&line).and_then(prepare);
        match request {
            Ok(request) => super::ui::post(move || super::sdl::control(request)),
            Err(why) => log::warn!(
                target: "reims_vgpu_window",
                "control socket: {line:?} refused: {why}"
            ),
        }
    }
}

/// `request` as the UI thread takes it: a toolbar's atlas read from its file, and checked
/// against its size ([`Atlas::new`]).
fn prepare(request: Request) -> Result<Control, String> {
    Ok(match request {
        Request::Toolbar { atlas, cell, ids } => {
            let count = u32::try_from(ids.len()).map_err(|_| "too many buttons".to_owned())?;
            let atlas = Atlas::new(cell, count, read_atlas(&atlas)?)?;
            Control::Toolbar {
                atlas: Arc::new(atlas),
                ids,
            }
        }
        Request::ToolbarOff => Control::ToolbarOff,
        Request::Rotate(rotation) => Control::Rotate(rotation),
    })
}

/// The bytes of the atlas at `path`.
fn read_atlas(path: &Path) -> Result<Vec<u8>, String> {
    std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))
}
