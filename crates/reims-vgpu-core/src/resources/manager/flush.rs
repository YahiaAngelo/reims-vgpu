//! The flush before a suspend: every resource whose host copy is ahead of the guest's is
//! written back.
//!
//! The walk sorts the cached resources, in cache order, into CPU, blit and compute evictions.
//! The CPU evictions run first, with no command buffer. Then, if there is any GPU eviction, one
//! command buffer from the task's paging queue takes a blit encoder with every blit eviction and
//! a compute encoder with every compute eviction, and is committed and waited for. An
//! eviction's raise is logged and the walk goes on; a raise of the queue, the command buffer or
//! an encoder leaves the flush. Only a suspend's flush and the GPU task's flush job, with no
//! handler, flush, so such a raise dies at its send.

use std::sync::Arc;

use crate::platform::Platform;
use crate::platform::blit::{BlitCmd, BlitEncoder};
use crate::platform::command::{CommandBuffer, CommandQueue};
use crate::platform::compute::{ComputeCmd, ComputeEncoder};
use crate::raise::{OnFault, RaiseResult};
use crate::resources::manager::ResourceManager;
use crate::resources::resource::Resource;
use crate::rt::Hold;

impl<P: Platform> ResourceManager<P> {
    /// Writes back every resource that needs it. The cache is walked under the leaf lock only.
    pub fn flush_resources(&self, hold: Hold<'_>) -> RaiseResult<()> {
        let resources: Vec<Arc<Resource<P>>> =
            self.with_caches(|c| c.resources.iter().map(|(_, r)| Arc::clone(r)).collect());
        let mut blit = Vec::new();
        let mut compute = Vec::new();
        let mut cpu = Vec::new();
        for resource in resources {
            if resource.needs_sync() {
                if resource.needs_cpu_paging() {
                    cpu.push(resource);
                } else if resource.needs_compute_paging() {
                    compute.push(resource);
                } else {
                    blit.push(resource);
                }
            }
        }

        for resource in &cpu {
            if let Err(raised) = resource.evict_with_cpu() {
                hold.land(raised);
                self.log_eviction_raise();
            }
        }

        if !blit.is_empty() || !compute.is_empty() {
            // A gone task or a nil queue answers nil all the way down
            let queue = self.task.upgrade().and_then(|t| t.paging_queue());
            let command_buffer = queue
                .as_ref()
                .map_or(Ok(None), |q| q.command_buffer(OnFault::Die))?;
            drop(queue);
            if !blit.is_empty() {
                let encoder = command_buffer
                    .as_ref()
                    .map_or(Ok(None), |cb| cb.blit_encoder(OnFault::Die))?;
                for resource in &blit {
                    let evicted = resource.evict_in_encoder(hold, encoder.as_ref());
                    if let Err(raised) = evicted {
                        self.log_eviction_raise();
                        hold.land(raised);
                    }
                }
                encoder
                    .as_ref()
                    .map_or(Ok(()), |e| e.encode(OnFault::Die, BlitCmd::EndEncoding))?;
            }
            if !compute.is_empty() {
                let encoder = command_buffer
                    .as_ref()
                    .map_or(Ok(None), |cb| cb.compute_encoder(OnFault::Die))?;
                for resource in &compute {
                    let evicted = resource.evict_in_compute_encoder(hold, encoder.as_ref());
                    if let Err(raised) = evicted {
                        self.log_eviction_raise();
                        hold.land(raised);
                    }
                }
                encoder
                    .as_ref()
                    .map_or(Ok(()), |e| e.encode(OnFault::Die, ComputeCmd::EndEncoding))?;
            }
            command_buffer
                .as_ref()
                .map_or(Ok(()), |cb| cb.commit(OnFault::Die))?;
            command_buffer
                .as_ref()
                .map_or(Ok(()), |cb| cb.wait_until_completed(OnFault::Die))?;
        }
        Ok(())
    }

    fn log_eviction_raise(&self) {
        log::error!("a resource raised while the task was evicted; evicting the rest");
    }
}

#[cfg(test)]
mod tests {
    use super::super::fixture::*;
    use super::*;
    use crate::raise::death::DeathKind;
    use crate::resources::manager::create::Resources;
    use crate::resources::texture::views::SerializerTextureResource;
    use crate::testkit::capture::catch_death;
    use crate::testkit::{Step, kind};

    /// The line of each eviction whose raise the flush caught.
    const EVICTION_RAISED: &str = "a resource raised while the task was evicted; evicting the rest";

    fn entries3() -> Vec<(u8, u32, u64)> {
        vec![(0, 0, 0), (6, 0x20, REC), (6, 0x10, REC + 0x100)]
    }

    fn texture_resource(fx: &Fx) -> Arc<Resource<P>> {
        let t = fx
            .platform
            .world()
            .new_obj::<kind::Texture>(Default::default());
        SerializerTextureResource::new(t, false)
    }

    fn stale_view(fx: &Fx, reference: u32) {
        let r = texture_resource(fx);
        r.set_is_host_valid(true);
        r.set_is_guest_valid(false);
        assert!(fx.m.register_object::<Resources>(reference, r));
    }

    #[test]
    fn nothing_to_evict_makes_no_command_buffer() {
        let fx = Fx::new(&entries3(), Placement::Off);
        let mark = fx.clear_log();
        fx.m.flush_resources(Hold::NONE).unwrap();
        assert!(fx.sels_since(mark).is_empty());
    }

    /// A guest-invalid, host-valid serializer texture needs a blit eviction: the paging queue's
    /// command buffer, a blit encoder, the eviction (which raises: caught, logged, and the walk
    /// goes on), `endEncoding`, `commit`, `waitUntilCompleted`.
    #[test]
    fn blit_evictions_share_one_command_buffer_and_a_raise_is_logged() {
        let fx = Fx::new(&entries3(), Placement::Off);
        fx.platform
            .world()
            .set_auto_complete(Some(crate::testkit::CompleteOn::Caller));
        stale_view(&fx, 1);
        stale_view(&fx, 2);
        // In sync: no eviction.
        assert!(fx.m.register_object::<Resources>(3, texture_resource(&fx)));
        let mark = fx.clear_log();
        let (r, lines) = fx.logged(|| fx.m.flush_resources(Hold::NONE));
        assert!(r.is_ok());
        assert_eq!(
            fx.sels_since(mark),
            vec![
                "paging_queue",
                "commandBuffer",
                "blitCommandEncoder",
                "endEncoding",
                "commit",
                "waitUntilCompleted"
            ]
        );
        assert_eq!(lines, vec![EVICTION_RAISED.to_owned(); 2]);
    }

    /// A CPU-paged resource (an ICB) is evicted before any command buffer: here its write-back
    /// cursor raises, which is caught and logged, and no Metal call follows.
    #[test]
    fn a_cpu_eviction_raise_is_logged_without_a_command_buffer() {
        let fx = Fx::new(&entries3(), Placement::Off);
        let r = crate::resources::icb::IcbResource::new(
            Arc::downgrade(&fx.task_dyn),
            crate::guest::records::GuestPages::HOST,
            &[0; 12],
            None,
            Default::default(),
            Default::default(),
        )
        .unwrap()
        .unwrap();
        r.set_is_host_valid(true);
        r.set_is_guest_valid(false);
        assert!(fx.m.register_object::<Resources>(1, r));
        fx.platform.script().push(
            "writable_cursor_from_virtual_offset",
            Step::raise("NSGenericException", "x"),
        );
        let mark = fx.clear_log();
        let (r, lines) = fx.logged(|| fx.m.flush_resources(Hold::NONE));
        assert!(r.is_ok());
        assert_eq!(
            fx.log()
                .calls_to("writable_cursor_from_virtual_offset")
                .len(),
            1
        );
        assert!(!fx.sels_since(mark).contains(&"commandBuffer"));
        assert_eq!(lines, vec![EVICTION_RAISED.to_owned()]);
    }

    /// A nil paging queue answers nil all the way down; the evictions still run.
    #[test]
    fn a_nil_paging_queue_sends_nothing_further() {
        let fx = Fx::new(&entries3(), Placement::Off);
        stale_view(&fx, 1);
        fx.platform.script().push("paging_queue", Step::Nil);
        let mark = fx.clear_log();
        let (r, lines) = fx.logged(|| fx.m.flush_resources(Hold::NONE));
        assert!(r.is_ok());
        assert_eq!(fx.sels_since(mark), vec!["paging_queue"]);
        assert_eq!(lines, vec![EVICTION_RAISED.to_owned()]);
    }

    /// A Metal raise outside the evictions' handlers (here `commandBuffer`) leaves the flush,
    /// whose callers have no handler: it dies at the send.
    #[test]
    fn a_metal_raise_escapes() {
        let fx = Fx::new(&entries3(), Placement::Off);
        stale_view(&fx, 1);
        fx.platform.script().push(
            "commandBuffer",
            Step::raise("NSInternalInconsistencyException", "x"),
        );
        let site = catch_death(|| fx.m.flush_resources(Hold::NONE)).unwrap_err();
        assert_eq!(
            (site.kind, site.file),
            (DeathKind::UncaughtObjc, "commandBuffer")
        );
    }
}
