//! Facts about this machine, read at run time with read-only `sysctlbyname`
//! queries (no setting is changed). A missing name is `None`, never an error:
//! callers choose a fallback or fail closed (D23).

/// What this process can learn about the machine it runs on.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Machine {
    /// `machdep.cpu.brand_string`, e.g. "Apple M5".
    pub chip: Option<String>,
    /// `kern.osproductversion`, e.g. "27.2.1".
    pub macos_version: Option<String>,
    /// Physical cores of each performance level, fastest first
    /// (`hw.perflevel{i}.physicalcpu` for i < `hw.nperflevels`); empty when unknown.
    pub core_levels: Vec<usize>,
    /// `hw.physicalcpu`.
    pub physical_cores: Option<usize>,
    /// CPUs this process may use (`std::thread::available_parallelism`).
    pub available_cpus: Option<usize>,
    /// `hw.memsize` in bytes.
    pub memory_bytes: Option<u64>,
    /// `hw.pagesize` in bytes.
    pub page_size: Option<u64>,
}

impl Machine {
    pub fn detect() -> Self {
        let levels = sysctl_u64("hw.nperflevels").unwrap_or(0);
        let core_levels: Option<Vec<usize>> = (0..levels)
            .map(|level| {
                sysctl_u64(&format!("hw.perflevel{level}.physicalcpu"))
                    .and_then(|n| usize::try_from(n).ok())
                    .filter(|&n| n > 0)
            })
            .collect();
        Self {
            chip: sysctl_string("machdep.cpu.brand_string"),
            macos_version: sysctl_string("kern.osproductversion"),
            // One missing level makes the split unknown rather than wrong.
            core_levels: core_levels.unwrap_or_default(),
            physical_cores: sysctl_u64("hw.physicalcpu")
                .and_then(|n| usize::try_from(n).ok())
                .filter(|&n| n > 0),
            available_cpus: std::thread::available_parallelism().ok().map(|n| n.get()),
            memory_bytes: sysctl_u64("hw.memsize").filter(|&n| n > 0),
            page_size: sysctl_u64("hw.pagesize").filter(|&n| n > 0),
        }
    }

    /// One CPU worker per physical core of every performance level (on the M5,
    /// 4 + 6 = 10, the count D10 measured best), falling back to `hw.physicalcpu`,
    /// then to the CPUs available to the process, then to 1; never more than the
    /// process may use.
    pub fn default_threads(&self) -> usize {
        let cores = if self.core_levels.is_empty() {
            self.physical_cores.or(self.available_cpus).unwrap_or(1)
        } else {
            self.core_levels.iter().sum()
        };
        cores.min(self.available_cpus.unwrap_or(cores)).max(1)
    }
}

/// An integer sysctl (4 or 8 bytes, little-endian), `None` when it does not exist.
pub fn sysctl_u64(name: &str) -> Option<u64> {
    let bytes = sysctl_bytes(name)?;
    match bytes.len() {
        4 => Some(u64::from(u32::from_le_bytes(bytes.try_into().ok()?))),
        8 => Some(u64::from_le_bytes(bytes.try_into().ok()?)),
        _ => None,
    }
}

/// A string sysctl without its terminating NUL, `None` when it does not exist.
pub fn sysctl_string(name: &str) -> Option<String> {
    let bytes = sysctl_bytes(name)?;
    let text = bytes.split(|&b| b == 0).next()?;
    let text = String::from_utf8(text.to_vec()).ok()?;
    (!text.is_empty()).then_some(text)
}

#[cfg(target_os = "macos")]
fn sysctl_bytes(name: &str) -> Option<Vec<u8>> {
    use std::ffi::{CString, c_char, c_int, c_void};
    unsafe extern "C" {
        fn sysctlbyname(
            name: *const c_char,
            oldp: *mut c_void,
            oldlenp: *mut usize,
            newp: *mut c_void,
            newlen: usize,
        ) -> c_int;
    }
    let name = CString::new(name).ok()?;
    let mut len = 0_usize;
    // SAFETY: `name` is NUL-terminated; a null `oldp` asks only for the size,
    // written to the live `len`. `newp` is null with length 0: nothing is set.
    let status = unsafe {
        sysctlbyname(
            name.as_ptr(),
            std::ptr::null_mut(),
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    if status != 0 || len == 0 {
        return None;
    }
    let mut buffer = vec![0_u8; len];
    // SAFETY: `buffer` has `len` writable bytes and `len` says so; the kernel
    // writes at most that many and stores the actual length. Read-only as above.
    let status = unsafe {
        sysctlbyname(
            name.as_ptr(),
            buffer.as_mut_ptr().cast(),
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    if status != 0 {
        return None;
    }
    buffer.truncate(len);
    Some(buffer)
}

#[cfg(not(target_os = "macos"))]
fn sysctl_bytes(_name: &str) -> Option<Vec<u8>> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_threads_follow_core_levels_with_fallbacks_for_missing_names() {
        let machine = |levels: &[usize], physical, available| Machine {
            core_levels: levels.to_vec(),
            physical_cores: physical,
            available_cpus: available,
            ..Machine::default()
        };
        // The measured M5: 4 performance + 6 efficiency cores.
        assert_eq!(machine(&[4, 6], Some(10), Some(10)).default_threads(), 10);
        assert_eq!(machine(&[12, 4], Some(16), Some(16)).default_threads(), 16);
        assert_eq!(machine(&[24, 8], Some(32), Some(32)).default_threads(), 32);
        // A process limited to fewer CPUs never gets more workers.
        assert_eq!(machine(&[12, 4], Some(16), Some(6)).default_threads(), 6);
        // Missing hw.perflevel names: physical cores, then available CPUs, then 1.
        assert_eq!(machine(&[], Some(8), Some(8)).default_threads(), 8);
        assert_eq!(machine(&[], None, Some(12)).default_threads(), 12);
        assert_eq!(machine(&[], None, None).default_threads(), 1);
    }

    #[test]
    fn this_machine_reports_consistent_facts() {
        let machine = Machine::detect();
        println!("{machine:?}");
        assert!(machine.default_threads() >= 1);
        if cfg!(target_os = "macos") {
            assert!(machine.memory_bytes.is_some_and(|n| n > 0));
            if let (false, Some(physical)) =
                (machine.core_levels.is_empty(), machine.physical_cores)
            {
                assert_eq!(machine.core_levels.iter().sum::<usize>(), physical);
            }
        }
        assert_eq!(sysctl_u64("yskn.no.such.name"), None);
        assert_eq!(sysctl_string("yskn.no.such.name"), None);
    }
}
