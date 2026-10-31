//! [`SwapchainSink`]: what draws a window display's frames on the Vulkan platform, on either OS,
//! into the window's swapchain on the runtime's own device, with no copy.
//!
//! The window's render thread asks the sink for a presenter at its first draw: the sink makes the
//! window's Vulkan surface on the runtime's instance (through the window's [`VulkanTarget`]: SDL's
//! window on Linux, the window's `CAMetalLayer` on macOS) and a swapchain over it, presented on a
//! presenting queue of the swapchain's own, apart from the guest's queues where the device has one
//! to spare (`reims_vgpu_vulkan::swapchain`). Each draw then takes the swapchain's next image, has
//! the display draw its current frame into the image's texture in a command buffer of the frame
//! queue, as any window frame ([`SinkDisplay::encode`]), with the image's semaphore signalled
//! after it, and presents the image once the runtime has scheduled that command buffer. The
//! image's wait is the render thread's alone: the frame's command buffer waits for nothing of the
//! compositor's, so the guest's frames never wait for the window. Where the swapchain's images are
//! not the window's size (a surface with a size of its own, a step ahead of the window's during a
//! resize), the frame's region is scaled to them ([`fit`]). On macOS the images are the layer's
//! drawables, each fetched on the render thread as the image is taken
//! (`reims_vgpu_vulkan::swapchain`), and no Metal call of the sink's draws or presents them.
//!
//! Where the window asks for a decoration (its control's rotation and button bar,
//! `reims_vgpu_window::decor`), the draw has the display draw it too
//! ([`SinkDisplay::encode_decorated`]): the sink's display is made to draw them, with textures
//! and a buffer of the platform's own ([`own_texture`], [`own_buffer`]).
//!
//! What each OS's window asks of the sink is its own (`FrameSink`): `linux.rs` and `macos.rs`
//! give it the presenter made here ([`SwapchainSink::make_presenter`]).
//!
//! **Lifetime.** The display is held weakly, as the Metal platform's sink holds it, and upgraded
//! for an encode and its commit alone. A presenter holds the frame queue, and with it the
//! display's device, until the window's render thread ends at the window's end: its swapchain and
//! surface go there, before the window under them.
//!
//! **Statistics.** With `REIMS_VGPU_STATS` on, each frame committed counts as a window frame, as
//! every sink's does, and its command buffer's GPU time is the window's ([`crate::stats`]).

use std::sync::{Arc, Weak};

use reims_vgpu_core::display::Display as CoreDisplay;
use reims_vgpu_core::platform::{Handle as _, Platform};
use reims_vgpu_vulkan::VulkanPlatform;
use reims_vgpu_vulkan::swapchain::{self, Acquired, Frame, Swapchain};
use reims_vgpu_window::{
    BarLayout, Decoration, Drawn, DrawnCursor, FrameSink, PixelRect, Presenter, VulkanTarget,
};

use crate::entry;
use crate::stats::DisplayStats;
use crate::window::{SinkDisplay, WindowFrame};

/// The platform's command queue.
type Queue = <VulkanPlatform as Platform>::CommandQueue;
/// Its command buffer.
type CommandBuffer = <VulkanPlatform as Platform>::CommandBuffer;
/// Its texture.
type Texture = <VulkanPlatform as Platform>::Texture;

/// The sink of a window display on `display`, drawn in command buffers of `queue`, the frame
/// queue of its device.
pub(crate) fn sink(
    platform: Arc<VulkanPlatform>,
    display: Weak<CoreDisplay<VulkanPlatform>>,
    queue: Queue,
) -> Option<Arc<dyn FrameSink>> {
    drop(platform);
    let port = display.upgrade().map(|d| d.port());
    let sink: Arc<dyn FrameSink> = Arc::new(SwapchainSink {
        display: Arc::new(
            SinkDisplay::new(display, queue.clone_ref(), own_texture)
                .with_decorations(own_texture, own_buffer),
        ),
        queue,
        port,
    });
    Some(sink)
}

/// The guest's cursor glyph's texture on the Vulkan platform: one of the platform's own, which
/// takes no resource ID and no range of Metal's space (`VulkanDevice::new_own_texture`), so no
/// address or ID the guest reads moves for it.
pub(in crate::window::vulkan) fn own_texture(
    device: &<VulkanPlatform as Platform>::Device,
    descriptor: &<VulkanPlatform as Platform>::TextureDescriptor,
) -> Option<Texture> {
    device.new_own_texture(descriptor)
}

/// A buffer of the platform's own holding `bytes`, for the window's decorations: one that takes
/// no range of Metal's space (`VulkanDevice::new_own_buffer`), so no address the guest reads moves
/// for it.
fn own_buffer(
    device: &<VulkanPlatform as Platform>::Device,
    bytes: &[u8],
) -> Option<<VulkanPlatform as Platform>::Buffer> {
    device.new_own_buffer(bytes)
}

/// What draws a display's current frame into its window on the Vulkan platform (module doc).
pub(crate) struct SwapchainSink {
    /// The display, held only for a draw, and the frame queue; shared with the presenters.
    display: Arc<SinkDisplay<VulkanPlatform>>,
    /// The frame queue, of whose device the swapchain makes its presenting queue.
    queue: Queue,
    /// The display's port, which names its window in the log.
    port: Option<u64>,
}

impl SwapchainSink {
    /// The guest's cursor position, as the display reads it now ([`SinkDisplay::cursor_position`]).
    pub(crate) fn cursor_position(&self) -> Option<(u16, u16)> {
        entry::run(|_| self.display.cursor_position())
    }

    /// The window's surface on the runtime's instance, and a swapchain over it; none where either
    /// is refused, which is logged, as are the present mode a swapchain takes, MAILBOX where the
    /// surface offers it, else FIFO, and whether its presents signal fences
    /// (`reims_vgpu_vulkan::swapchain`).
    pub(crate) fn make_presenter(&self, target: &dyn VulkanTarget) -> Option<Box<dyn Presenter>> {
        // SAFETY: the target makes the surface on the instance it is given, and hands it over.
        let made = entry::run(|_| unsafe {
            Swapchain::over(&self.queue, |get_instance_proc_addr, instance| {
                target.create_surface(get_instance_proc_addr, instance)
            })
        });
        match made {
            Ok(swapchain) => {
                log::info!(
                    target: "reims_vgpu::window",
                    "the window of the display on port {} presents with {}, {}",
                    self.port.map_or_else(|| "?".to_owned(), |p| p.to_string()),
                    if swapchain.is_mailbox() { "MAILBOX" } else { "FIFO" },
                    if swapchain.has_present_fences() {
                        "each present fenced"
                    } else {
                        "with no present fences"
                    }
                );
                Some(Box::new(SwapchainPresenter {
                    display: Arc::clone(&self.display),
                    swapchain,
                }))
            }
            Err(refused) => {
                log::warn!(
                    target: "reims_vgpu::window",
                    "no swapchain for the window ({refused}): it stays empty"
                );
                None
            }
        }
    }
}

/// A window's presenter: the display and the window's swapchain, on its render thread.
struct SwapchainPresenter {
    /// The display and the frame queue.
    display: Arc<SinkDisplay<VulkanPlatform>>,
    /// The swapchain.
    swapchain: Swapchain,
}

impl Presenter for SwapchainPresenter {
    /// The next image, the display's frame encoded into it with `cursor` over it, and its
    /// present (module doc).
    fn draw(
        &mut self,
        drawable: (u64, u64),
        region: PixelRect,
        cursor: Option<&DrawnCursor>,
    ) -> Drawn {
        self.draw_decorated(drawable, region, cursor, None)
    }

    /// [`Presenter::draw`], with `decoration` drawn around the frame where given (module doc).
    fn draw_decorated(
        &mut self,
        drawable: (u64, u64),
        region: PixelRect,
        cursor: Option<&DrawnCursor>,
        decoration: Option<&Decoration>,
    ) -> Drawn {
        entry::run(|_| {
            let frame = match self.swapchain.acquire(drawable) {
                Acquired::Image(frame) => frame,
                Acquired::Retry => return Drawn::Retry,
                Acquired::Failed => return Drawn::Ended,
            };
            let extent = self.swapchain.extent().unwrap_or(drawable);
            let region = fit(region, drawable, extent);
            let fitted;
            let decoration = match decoration {
                Some(d) if extent != drawable => {
                    fitted = fit_decoration(d, drawable, extent);
                    Some(&fitted)
                }
                d => d,
            };
            let mut image = ImageFrame {
                swapchain: &self.swapchain,
                frame,
                stats: self.display.stats(),
                ready: None,
            };
            if !self
                .display
                .encode_decorated(region, cursor, decoration, &mut image)
            {
                return Drawn::Nothing;
            }
            let Some(command_buffer) = image.ready else {
                return Drawn::Nothing;
            };
            self.swapchain.present(frame, command_buffer);
            Drawn::Presented
        })
    }
}

/// `region` of a drawable of `drawable` pixels, scaled to one of `extent` pixels and kept within
/// it: the swapchain's images can be another size than the window's ([`Swapchain::extent`]).
fn fit(region: PixelRect, drawable: (u64, u64), extent: (u64, u64)) -> PixelRect {
    if drawable == extent || drawable.0 == 0 || drawable.1 == 0 {
        return region;
    }
    // Pixel counts are far below 2^32, so the products fit, and the rounding is to the nearest.
    let scale = |v: u64, from: u64, to: u64| (v.saturating_mul(to) + from / 2) / from;
    let x = scale(region.x, drawable.0, extent.0).min(extent.0);
    let y = scale(region.y, drawable.1, extent.1).min(extent.1);
    PixelRect {
        x,
        y,
        width: scale(region.width, drawable.0, extent.0).min(extent.0 - x),
        height: scale(region.height, drawable.1, extent.1).min(extent.1 - y),
    }
}

/// `decoration`, laid out for a drawable of `drawable` pixels, scaled to images of `extent`
/// pixels as the frame's region is ([`fit`]).
fn fit_decoration(decoration: &Decoration, drawable: (u64, u64), extent: (u64, u64)) -> Decoration {
    let mut fitted = decoration.clone();
    if let Some(bar) = &mut fitted.bar {
        bar.layout = BarLayout {
            bar: fit(bar.layout.bar, drawable, extent),
            buttons: bar
                .layout
                .buttons
                .iter()
                .map(|&b| fit(b, drawable, extent))
                .collect(),
        };
    }
    fitted
}

/// One window frame into a swapchain image.
struct ImageFrame<'a> {
    /// The swapchain.
    swapchain: &'a Swapchain,
    /// The image taken.
    frame: Frame,
    /// The display's statistics, where they are on.
    stats: Option<&'a Arc<DisplayStats>>,
    /// The frame's command buffer, once the image's semaphore is signalled in it: the image is
    /// presented only then.
    ready: Option<CommandBuffer>,
}

impl WindowFrame<VulkanPlatform> for &mut ImageFrame<'_> {
    /// The image's texture.
    fn texture(&mut self) -> Option<&Texture> {
        self.swapchain.texture(self.frame)
    }

    /// Signals the image's semaphore after the frame, and has its GPU time reported.
    fn encoded(&mut self, command_buffer: &CommandBuffer) {
        if !self.swapchain.encode_ready(self.frame, command_buffer) {
            return;
        }
        if let Some(stats) = self.stats {
            let stats = Arc::clone(stats);
            swapchain::report_gpu_time(command_buffer, move |time| stats.window_gpu_time(time));
        }
        self.ready = Some(command_buffer.clone_ref());
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    /// The bytes of `texture`, `size` pixels of a 4-byte format, once the work committed before
    /// on `queue` has completed: a blit into a shared buffer, committed and waited for. Rows of
    /// `size.0 * 4` bytes.
    #[cfg(target_os = "macos")]
    pub(in crate::window::vulkan) fn read_back(
        queue: &Queue,
        texture: &Texture,
        size: (u64, u64),
    ) -> Vec<u8> {
        use reims_vgpu_core::platform::blit::{BlitCmd, BlitEncoder as _};
        use reims_vgpu_core::platform::command::{CommandBuffer as _, CommandQueue as _};
        use reims_vgpu_core::platform::device::GpuDevice as _;
        use reims_vgpu_core::platform::resource::Buffer as _;
        use reims_vgpu_core::platform::types::{MtlOrigin, MtlSize};
        use reims_vgpu_core::raise::OnFault;

        use crate::display::no_catch;

        entry::run(|_| {
            let length = size.0 * size.1 * 4;
            let buffer =
                no_catch(queue.device.new_buffer_of_length(OnFault::Die, length, 0)).unwrap();
            let copy = no_catch(queue.command_buffer(OnFault::Die)).unwrap();
            let blit = no_catch(copy.blit_encoder(OnFault::Die)).unwrap();
            no_catch(blit.encode(
                OnFault::Die,
                BlitCmd::CopyTextureToBuffer {
                    source_texture: texture,
                    source_slice: 0,
                    source_level: 0,
                    source_origin: MtlOrigin::new(0, 0, 0),
                    source_size: MtlSize::new(size.0, size.1, 1),
                    destination_buffer: &buffer,
                    destination_offset: 0,
                    destination_bytes_per_row: size.0 * 4,
                    destination_bytes_per_image: length,
                },
            ));
            no_catch(blit.encode(OnFault::Die, BlitCmd::EndEncoding));
            no_catch(copy.commit(OnFault::Die));
            no_catch(copy.wait_until_completed(OnFault::Die));
            // SAFETY: the buffer is shared, `length` bytes long, and the copy into it has
            // completed.
            unsafe {
                std::slice::from_raw_parts(
                    buffer.contents().as_ptr::<u8>(),
                    usize::try_from(length).unwrap(),
                )
            }
            .to_vec()
        })
    }

    /// Draws `display`'s current frame over `region` into the next image of `swapchain`, made
    /// for a window of `size` pixels, in a command buffer of `queue`, as a presenter's draw
    /// does, then reads the image back ([`read_back`]) before it is presented. The swapchain's
    /// pixels, for a test's comparison (`macos.rs`).
    #[cfg(target_os = "macos")]
    pub(in crate::window::vulkan) fn draw_and_read_back(
        display: Weak<CoreDisplay<VulkanPlatform>>,
        queue: &Queue,
        swapchain: &mut Swapchain,
        size: (u64, u64),
        region: PixelRect,
    ) -> Vec<u8> {
        entry::run(|_| {
            let Acquired::Image(frame) = swapchain.acquire(size) else {
                panic!("no image of the swapchain");
            };
            let sink = SinkDisplay::new(display, queue.clone_ref(), super::own_texture);
            let mut image = ImageFrame {
                swapchain,
                frame,
                stats: None,
                ready: None,
            };
            assert!(sink.encode(region, None, &mut image), "the frame is drawn");
            let command_buffer = image.ready.take().unwrap();
            let bytes = read_back(queue, swapchain.texture(frame).unwrap(), size);
            swapchain.present(frame, command_buffer);
            bytes
        })
    }

    /// A rectangle.
    fn rect(x: u64, y: u64, width: u64, height: u64) -> PixelRect {
        PixelRect {
            x,
            y,
            width,
            height,
        }
    }

    /// The region is the window's where the images are the window's size
    #[test]
    fn the_region_stays_where_the_images_are_the_windows_size() {
        let region = rect(10, 20, 300, 200);
        assert_eq!(fit(region, (320, 240), (320, 240)), region);
    }

    /// The region scales with the images where they are another size, and stays within them
    #[test]
    fn the_region_scales_to_images_of_another_size() {
        assert_eq!(
            fit(rect(100, 0, 800, 600), (1000, 600), (500, 300)),
            rect(50, 0, 400, 300)
        );
        assert_eq!(
            fit(rect(0, 0, 1000, 600), (1000, 600), (1001, 601)),
            rect(0, 0, 1001, 601)
        );
        let within = fit(rect(999, 599, 1, 1), (1000, 600), (10, 10));
        assert!(within.x + within.width <= 10 && within.y + within.height <= 10);
    }
}
