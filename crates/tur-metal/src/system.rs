#![forbid(unsafe_code)]
//! What macOS reports about memory outside this process's device allocations.

use tur_core::{Result, execution::invalid};

/// One reading of the system's memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SystemMemory {
    /// Bytes macOS can hand out without swapping: free and purgeable pages, and file-backed
    /// ones among those it would reclaim first. Inactive anonymous pages do not count:
    /// reclaiming them means compressing or swapping them.
    pub spare_bytes: usize,
    /// Pages written to swap since boot.
    pub swapouts: u64,
}

impl SystemMemory {
    /// Read `vm_stat`. Spawns a process, so call it away from token paths.
    pub fn read() -> Result<Self> {
        let output = std::process::Command::new("vm_stat").output()?;
        Self::parse(&String::from_utf8_lossy(&output.stdout))
            .ok_or_else(|| invalid("cannot read spare memory from vm_stat"))
    }

    fn parse(report: &str) -> Option<Self> {
        let page: usize = report
            .split("page size of ")
            .nth(1)?
            .split_whitespace()
            .next()?
            .parse()
            .ok()?;
        let count = |key: &str| -> Option<usize> {
            let line = report
                .lines()
                .find(|line| line.trim_start().starts_with(key))?;
            line.rsplit(' ').next()?.trim_end_matches('.').parse().ok()
        };
        let reclaimable = count("Pages inactive:")?.checked_add(count("Pages speculative:")?)?;
        Some(Self {
            spare_bytes: count("Pages free:")?
                .checked_add(count("Pages purgeable:")?)?
                .checked_add(reclaimable.min(count("File-backed pages:")?))?
                .checked_mul(page)?,
            swapouts: count("Swapouts:")? as u64,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spare_memory_counts_only_pages_macos_reclaims_without_swapping() {
        let report = "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n\
            Pages free:                                  10.\n\
            Pages active:                              1000.\n\
            Pages inactive:                              20.\n\
            Pages speculative:                            3.\n\
            Pages wired down:                          1000.\n\
            Pages purgeable:                              7.\n\
            File-backed pages:                            5.\n\
            Swapouts:                                    42.\n";
        // Free and purgeable, and at most the file-backed share of inactive pages.
        assert_eq!(
            SystemMemory::parse(report),
            Some(SystemMemory {
                spare_bytes: 22 * 16384,
                swapouts: 42
            })
        );
        assert_eq!(SystemMemory::parse("no page size here"), None);
    }
}
