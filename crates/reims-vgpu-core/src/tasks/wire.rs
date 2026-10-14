//! The messages between a remote task and its service: [`ToService`] (17 variants) and
//! [`ToHost`] (4 variants).
//!
//! Byte strings are `Vec<u8>`, objects are platform handles, the memory map is the host's own,
//! and each reply is a [`Reply`] of its values. The transport (`tasks::isolation`) hands them
//! across queues in this process. Each variant's name ([`ToService::selector`],
//! [`ToHost::selector`]) is a label for diagnostics and tests only.

use std::sync::Arc;

use crate::guest::records::{GuestFeatures, GuestPages};
use crate::memory::map::MemoryMap;
use crate::platform::Platform;
use crate::platform::types::NsErrorText;

/// A message's reply: callable more than once (the service can reply twice). The host's
/// `ServiceLink` runs the host's reply at most once and then releases it (`ServiceLink::send`
/// states when).
pub struct Reply<T>(Arc<dyn Fn(T) + Send + Sync>);

impl<T> Reply<T> {
    /// A reply that runs `f` with the reply's values.
    pub fn new(f: impl Fn(T) + Send + Sync + 'static) -> Reply<T> {
        Reply(Arc::new(f))
    }

    /// Runs the reply.
    pub fn call(&self, v: T) {
        (self.0)(v)
    }
}

impl<T> Clone for Reply<T> {
    /// Another reference to the same reply.
    fn clone(&self) -> Self {
        Reply(Arc::clone(&self.0))
    }
}

impl<T> std::fmt::Debug for Reply<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Reply(..)")
    }
}

/// The handshake's configuration. The names are ours: the remote task fills the fields from the
/// device's `registryID` and `supportsArgumentBuffers`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct TaskConfiguration {
    /// `registryID`: the service opens the Metal device with it (`Platform::open_device`, the
    /// device of `MTLCopyAllDevices()` with that registry ID).
    pub registry_id: u64,
    /// `supportsArgumentBuffers`.
    pub supports_argument_buffers: bool,
    /// The device's guest pages (ours: Apple's service took its build's).
    pub guest_pages: GuestPages,
}

/// Host to service.
pub enum ToService<P: Platform> {
    /// sel: handshakeWithTaskRoot:taskSize:taskID:configuration:features:memoryMap:reply:
    /// reply `(uint32_t result, int pid)`.
    Handshake {
        /// `taskRoot`.
        task_root: u32,
        /// `taskSize`.
        task_size: u64,
        /// `taskID`.
        task_id: u32,
        /// `configuration`.
        configuration: TaskConfiguration,
        /// `features`: a copy of the device's, by value.
        features: GuestFeatures,
        /// `memoryMap`: the device's map; `None` is nil.
        memory_map: Option<Arc<MemoryMap<P>>>,
        /// `reply`.
        reply: Reply<(u32, i32)>,
    },
    /// sel: flushResourcesWithReply:
    FlushResources {
        /// `reply`.
        reply: Reply<u32>,
    },
    /// sel: unmapMemoryAtOffset:length:reply:
    UnmapMemory {
        /// `offset`.
        offset: u64,
        /// `length`.
        length: u64,
        /// `reply`.
        reply: Reply<u32>,
    },
    /// sel: setObjectListOffset:objectListLength:placementListOffset:placementListLength:reply:
    SetObjectList {
        /// `objectListOffset`.
        object_list_offset: u64,
        /// `objectListLength`.
        object_list_length: u32,
        /// `placementListOffset`.
        placement_list_offset: u64,
        /// `placementListLength`.
        placement_list_length: u32,
        /// `reply`.
        reply: Reply<u32>,
    },
    /// sel: getComputeInfo:maxKey:count:offset:reply:
    GetComputeInfo {
        /// `getComputeInfo:` (the pipeline state reference).
        pipeline_state_reference: u32,
        /// `maxKey`.
        max_key: u32,
        /// `count`.
        count: u32,
        /// `offset`.
        offset: u64,
        /// `reply`.
        reply: Reply<u32>,
    },
    /// sel: replacePhysical:reply:
    ReplacePhysical {
        /// `replacePhysical:`.
        physical: u32,
        /// `reply`.
        reply: Reply<u32>,
    },
    /// sel: invalidateResources:count:reply: — `resources` is the `NSData` of the records.
    InvalidateResources {
        /// `invalidateResources:`.
        resources: Vec<u8>,
        /// `count`.
        count: u32,
        /// `reply`.
        reply: Reply<u32>,
    },
    /// sel: synchronizeResources:count:discard:reply:
    SynchronizeResources {
        /// `synchronizeResources:`.
        resources: Vec<u8>,
        /// `count`.
        count: u32,
        /// `discard`.
        discard: bool,
        /// `reply`.
        reply: Reply<u32>,
    },
    /// sel: discardResources:count:reply:
    DiscardResources {
        /// `discardResources:`.
        resources: Vec<u8>,
        /// `count`.
        count: u32,
        /// `reply`.
        reply: Reply<u32>,
    },
    /// sel: heapTextureSizeAndAlignWithSerializedData:replyVirtualOffset:replyLength:reply:
    HeapTextureSizeAndAlign {
        /// `heapTextureSizeAndAlignWithSerializedData:`.
        data: Vec<u8>,
        /// `replyVirtualOffset`.
        reply_virtual_offset: u64,
        /// `replyLength`.
        reply_length: u64,
        /// `reply`.
        reply: Reply<u32>,
    },
    /// sel: deleteObjectWithSerializedData:reply:
    DeleteObject {
        /// `deleteObjectWithSerializedData:`.
        data: Vec<u8>,
        /// `reply`.
        reply: Reply<u32>,
    },
    /// sel: deleteResource:reply:
    DeleteResource {
        /// `deleteResource:`.
        resource: u32,
        /// `reply`.
        reply: Reply<u32>,
    },
    /// sel: execIndirect3WithData:cmdBufferCount:resourceCount:stampIndex:fifoEventValueIn:gpuCompletion:reply:
    /// — reply `(uint32_t, NSData *, uint32_t)`.
    ExecIndirect3 {
        /// `execIndirect3WithData:`.
        data: Vec<u8>,
        /// `cmdBufferCount`.
        cmd_buffer_count: u32,
        /// `resourceCount`.
        resource_count: u32,
        /// `stampIndex`.
        stamp_index: u32,
        /// `fifoEventValueIn`.
        fifo_event_value_in: u64,
        /// `gpuCompletion`.
        gpu_completion: u64,
        /// `reply`.
        reply: Reply<(u32, Option<Vec<u8>>, u32)>,
    },
    /// sel: infoIndirectWithData:cmdBufferCount:resourceCount:stampIndex:fifoEventValueIn:reply:
    InfoIndirect {
        /// `infoIndirectWithData:`.
        data: Vec<u8>,
        /// `cmdBufferCount`.
        cmd_buffer_count: u32,
        /// `resourceCount`.
        resource_count: u32,
        /// `stampIndex`.
        stamp_index: u32,
        /// `fifoEventValueIn`.
        fifo_event_value_in: u64,
        /// `reply`.
        reply: Reply<u32>,
    },
    /// sel: resetRasterizationRateMap:reply:
    ResetRasterizationRateMap {
        /// `resetRasterizationRateMap:`.
        map: u32,
        /// `reply`.
        reply: Reply<u32>,
    },
    /// sel: setFifoEvent:forIndex: — no reply.
    SetFifoEvent {
        /// `setFifoEvent:` (`MTLSharedEventHandle`), `None` for nil: the FIFO's event's new `+1`
        /// handle (`RemoteTask`'s `share_fifo_event`), unchecked, nil when the FIFO's event is
        /// nil. The message owns it.
        event: Option<P::SharedEventHandle>,
        /// `forIndex`.
        index: u32,
    },
    /// sel: fifoDeleted:reply:
    FifoDeleted {
        /// `fifoDeleted:`.
        stamp_index: u32,
        /// `reply`.
        reply: Reply<u32>,
    },
}

/// `ToService`'s labels, by index.
pub const TO_SERVICE_SELECTORS: [&str; 17] = [
    "handshakeWithTaskRoot:taskSize:taskID:configuration:features:memoryMap:reply:",
    "flushResourcesWithReply:",
    "unmapMemoryAtOffset:length:reply:",
    "setObjectListOffset:objectListLength:placementListOffset:placementListLength:reply:",
    "getComputeInfo:maxKey:count:offset:reply:",
    "replacePhysical:reply:",
    "invalidateResources:count:reply:",
    "synchronizeResources:count:discard:reply:",
    "discardResources:count:reply:",
    "heapTextureSizeAndAlignWithSerializedData:replyVirtualOffset:replyLength:reply:",
    "deleteObjectWithSerializedData:reply:",
    "deleteResource:reply:",
    "execIndirect3WithData:cmdBufferCount:resourceCount:stampIndex:fifoEventValueIn:gpuCompletion:reply:",
    "infoIndirectWithData:cmdBufferCount:resourceCount:stampIndex:fifoEventValueIn:reply:",
    "resetRasterizationRateMap:reply:",
    "setFifoEvent:forIndex:",
    "fifoDeleted:reply:",
];

impl<P: Platform> ToService<P> {
    /// The variant's index (0-16): its kind on the wire.
    pub fn index(&self) -> usize {
        match self {
            ToService::Handshake { .. } => 0,
            ToService::FlushResources { .. } => 1,
            ToService::UnmapMemory { .. } => 2,
            ToService::SetObjectList { .. } => 3,
            ToService::GetComputeInfo { .. } => 4,
            ToService::ReplacePhysical { .. } => 5,
            ToService::InvalidateResources { .. } => 6,
            ToService::SynchronizeResources { .. } => 7,
            ToService::DiscardResources { .. } => 8,
            ToService::HeapTextureSizeAndAlign { .. } => 9,
            ToService::DeleteObject { .. } => 10,
            ToService::DeleteResource { .. } => 11,
            ToService::ExecIndirect3 { .. } => 12,
            ToService::InfoIndirect { .. } => 13,
            ToService::ResetRasterizationRateMap { .. } => 14,
            ToService::SetFifoEvent { .. } => 15,
            ToService::FifoDeleted { .. } => 16,
        }
    }

    /// The variant's label.
    pub fn selector(&self) -> &'static str {
        TO_SERVICE_SELECTORS
            .get(self.index())
            .copied()
            .unwrap_or("")
    }
}

impl<P: Platform> std::fmt::Debug for ToService<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ToService({})", self.selector())
    }
}

/// Service to host.
pub enum ToHost<P: Platform> {
    /// sel: retainIOSurfaceWithMappingID:reply: — reply `(IOSurface *)`. The service blocks its
    /// queue until the host answers.
    RetainIoSurface {
        /// `retainIOSurfaceWithMappingID:`.
        mapping_id: u64,
        /// `reply`: the surface, or nil.
        reply: Reply<Option<P::Surface>>,
    },
    /// sel: releaseIOSurfaceWithMappingID: — no reply.
    ReleaseIoSurface {
        /// `releaseIOSurfaceWithMappingID:`.
        mapping_id: u64,
    },
    /// sel: completeAsyncTokenID:success:error: — no reply.
    CompleteAsyncToken {
        /// `completeAsyncTokenID:`.
        token_id: u64,
        /// `success`: the exec's status ([`ExecStatus`](crate::tasks::ExecStatus)'s value),
        /// unvalidated.
        success: u32,
        /// `error`: nil or an `NSError` (the host drops it: tokens signal with nil).
        error: Option<NsErrorText>,
    },
    /// sel: newSharedTextureHandleForID:reply: — reply `(MTLSharedTextureHandle *)`.
    NewSharedTextureHandle {
        /// `newSharedTextureHandleForID:`.
        id: u32,
        /// `reply`: the handle, or nil.
        reply: Reply<Option<P::SharedTextureHandle>>,
    },
}

/// `ToHost`'s labels, by index.
pub const TO_HOST_SELECTORS: [&str; 4] = [
    "retainIOSurfaceWithMappingID:reply:",
    "releaseIOSurfaceWithMappingID:",
    "completeAsyncTokenID:success:error:",
    "newSharedTextureHandleForID:reply:",
];

impl<P: Platform> ToHost<P> {
    /// The variant's index (0-3): its kind on the wire.
    pub fn index(&self) -> usize {
        match self {
            ToHost::RetainIoSurface { .. } => 0,
            ToHost::ReleaseIoSurface { .. } => 1,
            ToHost::CompleteAsyncToken { .. } => 2,
            ToHost::NewSharedTextureHandle { .. } => 3,
        }
    }

    /// The variant's label.
    pub fn selector(&self) -> &'static str {
        TO_HOST_SELECTORS.get(self.index()).copied().unwrap_or("")
    }
}

impl<P: Platform> std::fmt::Debug for ToHost<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ToHost({})", self.selector())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use parking_lot::Mutex;

    /// Reply::call runs the reply, twice if called twice (it is not FnOnce)
    #[test]
    fn reply_is_callable_more_than_once() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let s = seen.clone();
        let r: Reply<(u32, i32)> = Reply::new(move |v| s.lock().push(v));
        let r2 = r.clone();
        r.call((1, 42));
        r2.call((2, -1));
        assert_eq!(*seen.lock(), vec![(1, 42), (2, -1)]);
        assert_eq!(format!("{r:?}"), "Reply(..)");
    }

    /// The `Debug` forms name the message's selector (diagnostics only).
    #[test]
    fn debug_names_the_selector() {
        use crate::testkit::FakePlatform;
        let all: [ToHost<FakePlatform>; 4] = [
            ToHost::RetainIoSurface {
                mapping_id: 3,
                reply: Reply::new(|_| {}),
            },
            ToHost::ReleaseIoSurface { mapping_id: 3 },
            ToHost::CompleteAsyncToken {
                token_id: 1,
                success: 1,
                error: None,
            },
            ToHost::NewSharedTextureHandle {
                id: 2,
                reply: Reply::new(|_| {}),
            },
        ];
        for (m, sel) in all.iter().zip(TO_HOST_SELECTORS) {
            assert_eq!(format!("{m:?}"), format!("ToHost({sel})"));
        }
        let m: ToService<FakePlatform> = ToService::FlushResources {
            reply: Reply::new(|_| {}),
        };
        assert_eq!(format!("{m:?}"), "ToService(flushResourcesWithReply:)");
    }

    /// Every variant's index is its label's
    #[test]
    fn indices_name_the_labels() {
        let m: ToService<crate::testkit::FakePlatform> = ToService::FifoDeleted {
            stamp_index: 1,
            reply: Reply::new(|_| {}),
        };
        assert_eq!((m.index(), m.selector()), (16, "fifoDeleted:reply:"));
        let c = TaskConfiguration::default();
        assert_eq!((c.registry_id, c.supports_argument_buffers), (0, false));
    }
}
