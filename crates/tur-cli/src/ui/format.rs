//! Numbers for people.

/// A byte size in binary units.
pub(crate) fn bytes(value: f64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut value = value.max(0.0);
    let mut unit = 0;
    while value >= 1024.0 && unit + 1 < UNITS.len() {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{value:.0} B")
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}

/// A duration, from milliseconds to days.
pub(crate) fn seconds(value: f64) -> String {
    let whole = value as u64;
    match whole {
        _ if value < 1.0 => format!("{:.0} ms", value * 1000.0),
        0..60 => format!("{value:.1} s"),
        60..3600 => format!("{}m {:02}s", whole / 60, whole % 60),
        3600..86400 => format!("{}h {:02}m", whole / 3600, whole % 3600 / 60),
        _ => format!("{}d {:02}h", whole / 86400, whole % 86400 / 3600),
    }
}

/// A count, shortened past four digits.
pub(crate) fn count(value: f64) -> String {
    if value < 10_000.0 {
        format!("{value:.0}")
    } else if value < 1e6 {
        format!("{:.1}k", value / 1e3)
    } else {
        format!("{:.1}M", value / 1e6)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_numbers_pick_a_readable_unit() {
        assert_eq!(bytes(512.0), "512 B");
        assert_eq!(bytes(1536.0), "1.5 KiB");
        assert_eq!(bytes(3.0 * (1u64 << 30) as f64), "3.0 GiB");
        assert_eq!(seconds(0.25), "250 ms");
        assert_eq!(seconds(1.5), "1.5 s");
        assert_eq!(seconds(125.0), "2m 05s");
        assert_eq!(seconds(3720.0), "1h 02m");
        assert_eq!(seconds(90_000.0), "1d 01h");
        assert_eq!(count(9999.0), "9999");
        assert_eq!(count(12_300.0), "12.3k");
        assert_eq!(count(2.5e6), "2.5M");
    }
}
