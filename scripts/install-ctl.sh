#!/usr/bin/env bash
#
# Builds and installs reims-ctl, and the QEMU it runs, with apple-gfx on this
# checkout's reims-vgpu.
#
#   scripts/install-ctl.sh [--yes] [--update] [--skip-qemu]
#
# macOS on arm64 and Linux on x86-64 (Ubuntu or Debian, Arch or CachyOS; other
# distributions get the list of what to install). It:
#   1. checks the build's dependencies, and offers to install those missing
#      (Homebrew on macOS; apt or pacman on Linux, through sudo), and Rust
#      through rustup.rs, whose installer asks its own questions;
#   2. fetches vendor/qemu, QEMU's source, without QEMU's own submodules (its
#      build fetches what it needs); cargo fetches metal2vulkan itself, at the
#      commit the library's Cargo.toml names;
#   3. builds the library (cargo xtask dist) and QEMU on it, in
#      ~/.cache/reims-ctl, and installs QEMU into
#      ~/.local/share/reims-ctl/qemu/<build>, linking qemu/current to it. A
#      build never replaces another: a running VM keeps the binary it started
#      with;
#   4. installs reims-ctl with cargo install (into ~/.cargo/bin, or
#      CARGO_INSTALL_ROOT's bin), signed on macOS with the virtualization
#      entitlement, which installing macOS from an IPSW needs;
#   5. on Linux, installs reims-util beside it, which makes a new x86-64 VM's
#      OpenCore image and downloads its macOS installer from Apple: reims-ctl
#      does (__record-install), at the commit it names
#      (crates/reims-ctl/src/reims_util.rs).
#
# --yes       install what is missing without asking (rustup with -y too)
# --update    pull this checkout first (fast-forward only), putting back the
#             committed Cargo.lock the build rewrote; where the remote branch
#             was force-pushed, the checkout's commit is kept on a branch
#             BRANCH-before-update-DATE and the checkout moves to the remote's;
#             reims-ctl update runs this
# --skip-qemu install reims-ctl alone, keeping the installed QEMU
#
# Environment: CARGO_TARGET_DIR, the library's build (default the checkout's
# target/); XDG_DATA_HOME and XDG_CACHE_HOME move the install and the build
# from ~/.local/share and ~/.cache, where they are on macOS too.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
YES=0 UPDATE=0 SKIP_QEMU=0
for arg in "$@"; do
  case "$arg" in
    --yes | -y) YES=1 ;;
    --update) UPDATE=1 ;;
    --skip-qemu) SKIP_QEMU=1 ;;
    -h | --help)
      sed -n '2,/^set -euo/p' "$0" | sed -e '$d' -e 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "unknown option: $arg (see --help)" >&2
      exit 2
      ;;
  esac
done

bold=$'\033[1m' red=$'\033[31m' yellow=$'\033[33m' green=$'\033[32m' off=$'\033[0m'
[[ -t 1 ]] || bold='' red='' yellow='' green='' off=''
say() { printf '%s==> %s%s\n' "$bold" "$*" "$off"; }
warn() { printf '%swarning:%s %s\n' "$yellow" "$off" "$*" >&2; }
die() {
  printf '%serror:%s %s\n' "$red" "$off" "$*" >&2
  exit 1
}
# ask QUESTION: yes with --yes; no without a terminal to ask on.
ask() {
  [[ $YES == 1 ]] && return 0
  [[ -t 0 ]] || return 1
  local a
  read -r -p "$1 [Y/n] " a
  [[ -z $a || $a == [yY]* ]]
}

OS="$(uname -s)" ARCH="$(uname -m)"
case "$OS/$ARCH" in
  Darwin/arm64)
    PLATFORM=macos-arm64
    QEMU_CONFIGURE=(--target-list=aarch64-softmmu --enable-hvf --disable-sdl --disable-gtk)
    QEMU_BINARY=qemu-system-aarch64
    ;;
  Linux/x86_64)
    PLATFORM=linux-x86_64
    QEMU_CONFIGURE=(--target-list=x86_64-softmmu --enable-kvm --disable-sdl --enable-gtk --enable-slirp)
    QEMU_BINARY=qemu-system-x86_64
    ;;
  *) die "reims-ctl supports macOS on arm64 and Linux on x86-64; this is $OS on $ARCH" ;;
esac

# The XDG directory VAR names, else ~/FALLBACK: unset on macOS, and ignored
# unless absolute (as reims-ctl does).
xdg() { # VAR FALLBACK
  local v="${!1:-}"
  if [[ $v == /* ]]; then printf '%s' "$v"; else printf '%s' "$HOME/$2"; fi
}
DATA="$(xdg XDG_DATA_HOME .local/share)/reims-ctl"
CACHE="$(xdg XDG_CACHE_HOME .cache)/reims-ctl"
mkdir -p "$DATA" "$CACHE"

# ------------------------------------------------------------------ update

# published BEFORE: whether HEAD is a commit the remote branch once held, as
# BEFORE (its remote-tracking ref before the fetch) or an earlier value of that
# ref contains it. The ref's reflog holds each value, an update's old one as
# well as its new: a clone logs none, so the first fetch's old one is its only
# record of what the clone took. A HEAD the new remote branch lacks was then
# taken away by a force push; local commits never were the remote's.
published() {
  local c log
  log=$(git -C "$ROOT" rev-parse --git-path "logs/$upref")
  [[ $log == /* ]] || log=$ROOT/$log
  while read -r c; do
    git -C "$ROOT" merge-base --is-ancestor HEAD "$c" 2>/dev/null && return 0
  done < <(echo "$1"; [[ -f $log ]] && awk '{ print $1; print $2 }' "$log")
  return 1
}

if [[ $UPDATE == 1 ]]; then
  say "Pulling $ROOT"
  if upref=$(git -C "$ROOT" rev-parse --symbolic-full-name '@{u}' 2>/dev/null); then
    # The build writes Cargo.lock again (cargo adds the git sources the
    # committed lock leaves out), and a changed lock stops a pull that changes
    # it too: the committed one goes back first.
    git -C "$ROOT" checkout -q HEAD -- Cargo.lock
    branch=$(git -C "$ROOT" symbolic-ref --short HEAD)
    remote=$(git -C "$ROOT" config "branch.$branch.remote")
    before=$(git -C "$ROOT" rev-parse "$upref")
    if [[ $remote != . ]]; then
      # Forced (+), so a remote branch that was force-pushed still moves its
      # remote-tracking ref, where a plain fetch refuses it.
      merge=$(git -C "$ROOT" config "branch.$branch.merge")
      git -C "$ROOT" fetch -q "$remote" "+$merge:$upref" || die "git fetch from $remote failed"
    fi
    if git -C "$ROOT" merge-base --is-ancestor HEAD "$upref"; then
      git -C "$ROOT" merge -q --ff-only "$upref" || die "the checkout's local changes stop the update: commit or set them aside, then run $0"
    elif published "$before"; then
      # The checkout's commit is kept on a branch of its own, made once the
      # move has worked; --keep refuses to move over a local change the move
      # would overwrite.
      old=$(git -C "$ROOT" rev-parse HEAD)
      git -C "$ROOT" reset -q --keep "$upref" || die "the checkout's local changes stop the update: commit or set them aside, then run $0"
      backup="$branch-before-update-$(date +%Y%m%d-%H%M%S)"
      git -C "$ROOT" branch "$backup" "$old"
      warn "${upref#refs/remotes/} was force-pushed: the checkout's earlier commit stays on the branch $backup"
    else
      die "the checkout has commits of its own that ${upref#refs/remotes/} lacks: update it yourself, then run $0"
    fi
  else
    warn "the checkout's branch tracks no remote branch: building it as it is"
  fi
fi

# ------------------------------------------------------------------ dependencies

missing=()

if [[ $OS == Darwin ]]; then
  xcode-select -p >/dev/null 2>&1 || die "Xcode's command line tools are missing: run xcode-select --install, then this again"
  command -v brew >/dev/null 2>&1 || die "Homebrew is missing (https://brew.sh): install it, then run this again"
  for f in ninja pkgconf glib pixman libslirp lld; do
    brew list --formula "$f" >/dev/null 2>&1 || missing+=("$f")
  done
  if ((${#missing[@]})); then
    say "Missing Homebrew packages: ${missing[*]}"
    ask "Install them with brew?" || die "install them (brew install ${missing[*]}), then run this again"
    brew install "${missing[@]}"
  fi
else
  distro_id="" distro_like=""
  if [[ -r /etc/os-release ]]; then
    # shellcheck disable=SC1091
    distro_id="$(. /etc/os-release && echo "${ID:-}")"
    # shellcheck disable=SC1091
    distro_like="$(. /etc/os-release && echo "${ID_LIKE:-}")"
  fi
  # The library (cargo xtask dist: clang, lld and binutils for the option ROM;
  # CMake and the X11, Wayland, xkbcommon and EGL headers for SDL3's video,
  # which it builds from source), and QEMU (glib, pixman, libslirp; GTK 3 for
  # its own window, which a VM on VMware SVGA shows its display in; pipewire,
  # pulse and ALSA for its audio backends; bzip2 unpacks its firmware). A new
  # x86-64 VM's OVMF comes from reims-util, with its OpenCore image.
  apt_pkgs=(build-essential git curl ca-certificates pkg-config python3 python3-venv ninja-build
    flex bison bzip2 cmake clang lld binutils libglib2.0-dev libpixman-1-dev libslirp-dev
    zlib1g-dev libfdt-dev libpipewire-0.3-dev libpulse-dev libasound2-dev libvulkan1
    libx11-dev libxext-dev libxrandr-dev libxcursor-dev libxi-dev libxfixes-dev libxss-dev
    libxtst-dev libxkbcommon-dev libwayland-dev libwayland-bin libdecor-0-dev
    libegl-dev libgl-dev libdrm-dev libgbm-dev libdbus-1-dev libudev-dev libgtk-3-dev)
  pacman_pkgs=(base-devel git curl python ninja pkgconf flex bison bzip2 cmake clang lld
    glib2 pixman libslirp zlib dtc pipewire libpulse alsa-lib vulkan-icd-loader
    libx11 libxext libxrandr libxcursor libxi libxfixes libxss libxtst libxkbcommon
    wayland libdecor mesa libdrm dbus systemd-libs gtk3)
  case " $distro_id $distro_like " in
    *" debian "* | *" ubuntu "*)
      for p in "${apt_pkgs[@]}"; do
        dpkg-query -W -f='${Status}' "$p" 2>/dev/null | grep -q 'install ok installed' || missing+=("$p")
      done
      if ((${#missing[@]})); then
        say "Missing packages: ${missing[*]}"
        ask "Install them with sudo apt-get?" || die "install them (sudo apt-get install ${missing[*]}), then run this again"
        sudo apt-get update
        sudo apt-get install -y "${missing[@]}"
      fi
      ;;
    *" arch "* | *" cachyos "*)
      for p in "${pacman_pkgs[@]}"; do
        pacman -Qq "$p" >/dev/null 2>&1 || pacman -Qqg "$p" >/dev/null 2>&1 || missing+=("$p")
      done
      if ((${#missing[@]})); then
        say "Missing packages: ${missing[*]}"
        ask "Install them with sudo pacman?" || die "install them (sudo pacman -S --needed ${missing[*]}), then run this again"
        sudo pacman -S --needed --noconfirm "${missing[@]}"
      fi
      ;;
    *)
      warn "$distro_id is neither Debian/Ubuntu nor Arch/CachyOS: install the equivalents of these Ubuntu packages yourself if the build stops: ${apt_pkgs[*]}"
      ;;
  esac
  if [[ ! -r /dev/kvm || ! -w /dev/kvm ]]; then
    warn "/dev/kvm is not usable by you: enable virtualization in the firmware and add yourself to the kvm group (sudo usermod -aG kvm \$USER; log in again), or VMs will not start"
  fi
  # A new VM gets half the host's memory, 4 to 16 GiB, and its views of guest
  # memory take about 64 mappings per MiB (reims-ctl's runtime::max_map_count).
  maps="$(cat /proc/sys/vm/max_map_count 2>/dev/null || echo 0)"
  vm_mib=$(($(awk '/^MemTotal:/ { print $2 }' /proc/meminfo) / 2 / 1048576 * 1024))
  vm_mib=$((vm_mib < 4096 ? 4096 : vm_mib > 16384 ? 16384 : vm_mib))
  if ((maps < vm_mib * 64 + 81920)); then
    warn "vm.max_map_count is $maps: a new VM's $((vm_mib / 1024)) GiB needs about $((vm_mib * 64 + 81920)) (sudo sysctl -w vm.max_map_count=2097152, and a file in /etc/sysctl.d to keep it)"
  fi
  # An AMD GPU (RADV) imports the guest's memory as dma-bufs made by udmabuf
  # (docs/qemu.md, "Linux hosts: AMD GPUs and udmabuf").
  if grep -qsx 0x1002 /sys/class/drm/*/device/vendor && [[ ! -r /dev/udmabuf || ! -w /dev/udmabuf ]]; then
    warn "/dev/udmabuf is not usable by you: a VM on an AMD GPU needs it for the guest's memory (add yourself to the group that owns it, kvm on Ubuntu: sudo usermod -aG kvm \$USER; log in again)"
  fi
fi

if ! command -v cargo >/dev/null 2>&1 && [[ -x "$HOME/.cargo/bin/cargo" ]]; then
  # shellcheck disable=SC1091
  . "$HOME/.cargo/env"
fi
if ! command -v rustup >/dev/null 2>&1; then
  if command -v cargo >/dev/null 2>&1; then
    die "cargo is installed without rustup: the build needs the toolchain rust-toolchain.toml pins, which rustup installs (https://rustup.rs)"
  fi
  say "Rust is not installed"
  ask "Install it from https://rustup.rs?" || die "install Rust (https://rustup.rs), then run this again"
  if [[ $YES == 1 ]]; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
  else
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  fi
  # shellcheck disable=SC1091
  . "$HOME/.cargo/env"
fi

# ------------------------------------------------------------------ sources

# vendor/qemu at the commit this checkout names: shallow, else whole (a server
# that refuses a shallow fetch of a commit by its hash).
fetch_qemu() {
  git -C "$ROOT" submodule update --init --depth 1 vendor/qemu ||
    git -C "$ROOT" submodule update --init vendor/qemu ||
    die "could not fetch vendor/qemu's commit $(git -C "$ROOT" ls-tree HEAD vendor/qemu | awk '{print $3}') from $(git -C "$ROOT" config -f .gitmodules --get submodule.vendor/qemu.url): check that you can read that repository; a commit it does not have was not pushed to it"
}

if [[ $SKIP_QEMU == 0 ]]; then
  # A linked worktree shares the main checkout's module repository: initialising
  # vendor/qemu here would rewrite that repository's core.worktree for every
  # worktree (docs/lessons.md).
  gitdir="$(git -C "$ROOT" rev-parse --absolute-git-dir)"
  commondir="$(cd "$ROOT" && cd "$(git rev-parse --git-common-dir)" && pwd)"
  if [[ ! -f "$ROOT/vendor/qemu/configure" ]]; then
    if [[ "$gitdir" != "$commondir" ]]; then
      die "this is a git worktree, and vendor/qemu is empty: run $0 in the main checkout, or clone vendor/qemu into this worktree yourself"
    fi
    say "Fetching QEMU's source (vendor/qemu, without its own submodules)"
    fetch_qemu
  elif [[ $UPDATE == 1 && "$gitdir" == "$commondir" ]]; then
    fetch_qemu
  fi
fi

# ------------------------------------------------------------------ the library and QEMU

# The files running processes execute, their links resolved, a line each: on
# Linux every process's /proc/PID/exe; on macOS the txt files lsof lists of
# those whose name starts with qemu (QEMU, qemu-img). A QEMU's executable names
# the build it started from, where its command line, and ps, show only
# qemu/current, through which reims-ctl runs QEMU, and which a new build moves.
qemu_executables() {
  if [[ -d /proc/self ]]; then
    readlink /proc/[0-9]*/exe 2>/dev/null || true
  else
    lsof -nP -a -c qemu -d txt -Fn 2>/dev/null | sed -n 's/^n//p' || true
  fi
}

# prune_qemu DIR: removes the QEMU builds in DIR but the three newest and those
# a running process executes a file of.
prune_qemu() {
  local real in_use old
  real="$(cd "$1" && pwd -P)"
  in_use="$(qemu_executables)"
  ls -1d "$1/"[0-9]*/ 2>/dev/null | sort -r | tail -n +4 | while read -r old; do
    old="${old%/}"
    grep -qF "$real/${old##*/}/" <<<"$in_use" || rm -rf "$old"
  done
}

COMMIT="$(git -C "$ROOT" rev-parse --short HEAD)"
TARGET="${CARGO_TARGET_DIR:-$ROOT/target}"
if [[ $OS == Darwin ]]; then DIST="$TARGET/dist"; else DIST="$TARGET/dist-vulkan"; fi

if [[ $SKIP_QEMU == 0 ]]; then
  say "Building the library (cargo xtask dist)"
  (cd "$ROOT" && CARGO_TARGET_DIR="$TARGET" cargo xtask dist)
  [[ -f "$DIST/lib/pkgconfig/reims-vgpu.pc" ]] || die "cargo xtask dist wrote no $DIST/lib/pkgconfig/reims-vgpu.pc"

  BUILD="$CACHE/qemu-build-$PLATFORM"
  PC="$DIST/lib/pkgconfig"
  # QEMU's install prefix: never used where it is, since QEMU finds its files
  # relative to its binary; the build is staged with DESTDIR and moved.
  PREFIX=/reims-ctl-qemu
  mkdir -p "$BUILD"
  # No docs and no test suite: users run QEMU, not its tests.
  CONFIGURE=("$ROOT/vendor/qemu/configure" --prefix="$PREFIX" "${QEMU_CONFIGURE[@]}"
    --enable-pvg --disable-docs --disable-tests --disable-werror)
  CONFIGURED="PKG_CONFIG_PATH=$PC ${CONFIGURE[*]}"
  # Configure again when the source, the options, the package's directory, or
  # the package itself changed: meson keeps the package's flags from configure
  # (docs/lessons.md); and when ninja's dry run fails, as it does once a system
  # update removed a library file build.ninja names (/usr/lib/libfdt.so.1.8.1).
  if [[ ! -f "$BUILD/build.ninja" ]] ||
    [[ "$(cat "$BUILD/configured-with" 2>/dev/null)" != "$CONFIGURED" ]] ||
    ! cmp -s "$PC/reims-vgpu.pc" "$BUILD/reims-vgpu.pc.configured" ||
    ! ninja -C "$BUILD" -n >/dev/null 2>&1; then
    say "Configuring QEMU"
    (cd "$BUILD" && PKG_CONFIG_PATH="$PC" "${CONFIGURE[@]}")
    cp "$PC/reims-vgpu.pc" "$BUILD/reims-vgpu.pc.configured"
    printf '%s\n' "$CONFIGURED" >"$BUILD/configured-with"
  fi
  say "Building QEMU"
  # A build that stops configures again on the next run: when ninja has meson
  # regenerate build.ninja, meson reuses the dependencies it found before, a
  # removed one too, which the dry run above does not catch.
  ninja -C "$BUILD" || { rm -f "$BUILD/configured-with"; die "QEMU's build stopped; the next run configures it again"; }
  STAGE="$CACHE/qemu-stage"
  rm -rf "$STAGE"
  DESTDIR="$STAGE" ninja -C "$BUILD" install >/dev/null
  [[ -x "$STAGE$PREFIX/bin/$QEMU_BINARY" ]] || die "QEMU's install has no bin/$QEMU_BINARY"
  find "$STAGE$PREFIX" -name ReimsVgpu.rom -type f | grep -q . || die "QEMU's install has no ReimsVgpu.rom: was it configured with reims-vgpu?"
  if [[ $OS == Darwin ]] && ! codesign -d --entitlements - "$STAGE$PREFIX/bin/$QEMU_BINARY" 2>/dev/null | grep -q com.apple.security.hypervisor; then
    codesign -s - -f --entitlements "$ROOT/vendor/qemu/accel/hvf/entitlements.plist" "$STAGE$PREFIX/bin/$QEMU_BINARY"
  fi
  ID="$(date +%Y%m%d-%H%M%S)-$COMMIT"
  mkdir -p "$DATA/qemu"
  mv "$STAGE$PREFIX" "$DATA/qemu/$ID"
  rm -rf "$STAGE"
  ln -sfn "$ID" "$DATA/qemu/current"
  prune_qemu "$DATA/qemu"
  QEMU_DIR="$DATA/qemu/$ID"
else
  QEMU_DIR="$(cd "$DATA/qemu/current" 2>/dev/null && pwd -P || true)"
  [[ -n "$QEMU_DIR" ]] || warn "no QEMU installed yet: run this without --skip-qemu"
fi

# ------------------------------------------------------------------ reims-ctl

say "Installing reims-ctl (cargo install)"
(cd "$ROOT" && CARGO_TARGET_DIR="$CACHE/ctl-target" cargo install --locked --force --quiet --path crates/reims-ctl)
BIN="${CARGO_INSTALL_ROOT:-${CARGO_HOME:-$HOME/.cargo}}/bin/reims-ctl"
[[ -x "$BIN" ]] || die "cargo install left no $BIN"
if [[ $OS == Darwin ]]; then
  ent="$CACHE/reims-ctl.entitlements"
  cat >"$ent" <<'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>com.apple.security.virtualization</key>
	<true/>
</dict>
</plist>
EOF
  codesign -s - -f --entitlements "$ent" "$BIN" 2>/dev/null
fi
"$BIN" __record-install --repo "$ROOT" --qemu "${QEMU_DIR:-}" --commit "$COMMIT"

echo
printf '%sreims-ctl is installed.%s\n' "$green$bold" "$off"
echo "  reims-ctl            the interactive interface: create, start and stop VMs"
echo "  reims-ctl help       the same from a shell"
[[ -n "${QEMU_DIR:-}" ]] && echo "  QEMU                 $QEMU_DIR"
echo
# By its full path, which works before PATH finds reims-ctl.
echo "Check this host:"
printf '  %q doctor\n' "$BIN"
if ! command -v reims-ctl >/dev/null 2>&1 || [[ "$(command -v reims-ctl)" != "$BIN" ]]; then
  echo
  warn "$(dirname "$BIN") is not first on your PATH: open a new terminal, or run: . \"\$HOME/.cargo/env\""
fi
