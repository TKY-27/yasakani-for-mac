//! Counts live Rust heap bytes, including engine workers, without allocating.
use std::{
    alloc::{GlobalAlloc, Layout, System},
    sync::atomic::{AtomicUsize, Ordering},
};

pub struct Counting;
static LIVE: AtomicUsize = AtomicUsize::new(0);

pub fn live_bytes() -> usize {
    LIVE.load(Ordering::SeqCst)
}

// SAFETY: every operation forwards its original pointer/layout to System;
// accounting only uses nonallocating atomics after successful allocations.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            LIVE.fetch_add(layout.size(), Ordering::SeqCst);
        }
        ptr
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() {
            LIVE.fetch_add(layout.size(), Ordering::SeqCst);
        }
        ptr
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        LIVE.fetch_sub(layout.size(), Ordering::SeqCst);
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        let new = unsafe { System.realloc(ptr, layout, size) };
        if !new.is_null() {
            // Add before subtracting so concurrent reads cannot see underflow.
            LIVE.fetch_add(size, Ordering::SeqCst);
            LIVE.fetch_sub(layout.size(), Ordering::SeqCst);
        }
        new
    }
}
