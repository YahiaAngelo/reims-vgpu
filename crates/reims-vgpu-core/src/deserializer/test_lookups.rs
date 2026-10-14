//! The deserializer's real neighbour in its unit tests: a resource manager with no object list,
//! over a scripted task, whose lookup the deserializer holds as the product's does. A test puts
//! the objects a command names straight into the manager's caches ([`Lookups::provide`]); a
//! reference nothing was put under is the manager's failed lookup. Each lookup shows in the
//! call log as a `ManagerLookup` row (the lookup's test trace).

use std::sync::Arc;

use crate::platform::caps::DeviceCaps;
use crate::platform::device::GpuDevice as _;
use crate::resources::host::ResourceHost;
use crate::resources::icb::{IcbResource, IndirectCommandBufferFlags, IndirectCommandLayout};
use crate::resources::manager::create::Resources;
use crate::resources::manager::{
    ComputePipelineStates, DepthStencilStates, Fences, Functions, ManagedKind,
    RasterizationRateMaps, RenderPipelineStates, ResourceManager, SamplerStates,
};
use crate::resources::resource::Resource;
use crate::resources::texture::views::SerializerTextureResource;
use crate::rt::Hold;
use crate::seams::links::ObjectLookupRef;
use crate::tasks::object_lists::ObjectListKind;
use crate::tasks::object_lists::ObjectLists;
use crate::testkit::objects::{Fk, Kind, kind};
use crate::testkit::platform::FakePlatform;
use crate::testkit::resource_host::FakeResourceHost;

type P = FakePlatform;

/// Sets `p`'s device's capabilities, which the decoders read their limits from: its defaults,
/// changed by `f`.
pub(crate) fn set_caps(p: &P, f: impl FnOnce(&mut DeviceCaps)) {
    let device = p.device();
    let mut caps = DeviceCaps::clone(&device.caps());
    f(&mut caps);
    device.set_caps(caps);
}

/// The manager and the task it reads through.
pub(crate) struct Lookups {
    task_dyn: Arc<dyn ResourceHost<P>>,
    pub(crate) manager: Arc<ResourceManager<P>>,
}

impl Lookups {
    /// A manager on `p` with nothing cached and no object list.
    pub(crate) fn new(p: &Arc<P>) -> Lookups {
        let task_dyn: Arc<dyn ResourceHost<P>> = FakeResourceHost::new(p.clone());
        let lists = ObjectLists::new(
            ObjectListKind::MemoryMap,
            0x4_0000,
            false,
            Arc::downgrade(&task_dyn),
            crate::guest::records::GuestPages::HOST,
        );
        let manager = ResourceManager::new(
            p.clone(),
            Some(p.device()),
            Arc::downgrade(&task_dyn),
            lists,
            false,
        );
        Lookups { task_dyn, manager }
    }

    /// The manager's lookup, as a task hands it to its deserializer.
    pub(crate) fn lookup(&self) -> ObjectLookupRef<P> {
        self.manager.object_delegate()
    }

    /// Caches `h` under `reference`, replacing what was there: a buffer, texture or ICB as a
    /// resource over it, a state object as itself. A heap is not cached (its lookup fails).
    pub(crate) fn provide<K: Kind>(&self, reference: u32, h: &Fk<K>) {
        let any = h.erase();
        if let Some(b) = any.cast::<kind::Buffer>() {
            // A heap child: its heap is not kept, and its buffer is `b`.
            let heap = SerializerTextureResource::<P>::new(None, false);
            let r = crate::resources::buffer::child::HeapBufferResource::new(heap, b, false);
            self.put::<Resources>(reference, r);
        } else if let Some(t) = any.cast::<kind::Texture>() {
            self.put::<Resources>(reference, SerializerTextureResource::new(t, false));
        } else if let Some(i) = any.cast::<kind::Icb>() {
            {
                let r = IcbResource::new(
                    Arc::downgrade(&self.task_dyn),
                    crate::guest::records::GuestPages::HOST,
                    &[0; 12],
                    Some(i),
                    IndirectCommandLayout::default(),
                    IndirectCommandBufferFlags::default(),
                )
                .ok()
                .flatten()
                .map(|r: Arc<Resource<P>>| r);
                if let Some(r) = r {
                    self.put::<Resources>(reference, r);
                }
            }
        } else if let Some(o) = any.cast::<kind::Function>() {
            self.put::<Functions>(reference, o);
        } else if let Some(o) = any.cast::<kind::ComputePipeline>() {
            self.put::<ComputePipelineStates>(reference, o);
        } else if let Some(o) = any.cast::<kind::RenderPipeline>() {
            self.put::<RenderPipelineStates>(reference, o);
        } else if let Some(o) = any.cast::<kind::SamplerState>() {
            self.put::<SamplerStates>(reference, o);
        } else if let Some(o) = any.cast::<kind::DepthStencilState>() {
            self.put::<DepthStencilStates>(reference, o);
        } else if let Some(o) = any.cast::<kind::Fence>() {
            self.put::<Fences>(reference, o);
        } else if let Some(o) = any.cast::<kind::RateMap>() {
            self.put::<RasterizationRateMaps>(reference, o);
        }
    }

    fn put<K: ManagedKind<P>>(&self, reference: u32, object: K::Obj) {
        self.manager
            .delete_object::<K>(Hold::NONE, reference, false);
        assert!(self.manager.register_object::<K>(reference, object));
    }
}
