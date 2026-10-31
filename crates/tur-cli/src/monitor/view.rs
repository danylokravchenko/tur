//! Draws a snapshot's sections as a full-screen dashboard.
use super::snapshot::{Health, Rates, Row, Section, Snapshot};
use ratatui::{
    Frame,
    layout::{Constraint, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Sparkline},
};

/// Columns below which sections stack instead of sitting side by side.
const TWO_COLUMNS: u16 = 100;
const LABEL_WIDTH: usize = 12;
const METER_WIDTH: usize = 16;

/// What one frame shows.
pub(super) struct View<'a> {
    pub(super) url: &'a str,
    pub(super) snapshot: Option<&'a Snapshot>,
    pub(super) rates: &'a Rates,
    /// Why the snapshot may be old.
    pub(super) problem: Option<&'a str>,
    pub(super) logs: Option<&'a [String]>,
    pub(super) keys: &'a str,
}

fn dim() -> Style {
    Style::new().fg(Color::DarkGray)
}

fn meter_color(fraction: f64) -> Color {
    if fraction >= 0.9 {
        Color::Red
    } else if fraction >= 0.75 {
        Color::Yellow
    } else {
        Color::Green
    }
}

fn row_line(row: &Row) -> Line<'static> {
    let mut spans = vec![Span::styled(format!("{:LABEL_WIDTH$} ", row.label), dim())];
    if let Some(fraction) = row.meter {
        let fraction = fraction.clamp(0.0, 1.0);
        let filled = (fraction * METER_WIDTH as f64).round() as usize;
        spans.push(Span::styled(
            "█".repeat(filled),
            Style::new().fg(meter_color(fraction)),
        ));
        spans.push(Span::styled("░".repeat(METER_WIDTH - filled), dim()));
        spans.push(Span::raw(" "));
    }
    let style = if row.alert {
        Style::new().fg(Color::Yellow)
    } else {
        Style::new()
    };
    spans.push(Span::styled(row.value.clone(), style));
    Line::from(spans)
}

fn block(title: &str) -> Block<'static> {
    Block::new()
        .borders(Borders::ALL)
        .border_style(dim())
        .title(Span::styled(
            format!(" {title} "),
            Style::new().add_modifier(Modifier::BOLD),
        ))
}

fn section(frame: &mut Frame<'_>, area: Rect, section: &Section) {
    let lines: Vec<Line<'static>> = section.rows.iter().map(row_line).collect();
    frame.render_widget(Paragraph::new(lines).block(block(&section.title)), area);
}

fn header(view: &View<'_>) -> Line<'static> {
    let (state, color) = match (view.problem, view.snapshot.map(Snapshot::health)) {
        (Some(_), _) | (None, None) => ("● unreachable", Color::Red),
        (None, Some(Health::Ready)) => ("● ready", Color::Green),
        (None, Some(Health::Draining)) => ("● draining", Color::Yellow),
        (None, Some(Health::WorkerStopped)) => ("● worker stopped", Color::Red),
    };
    let version = view.snapshot.map_or("", Snapshot::version).to_owned();
    let uptime = view
        .snapshot
        .map_or_else(String::new, |snapshot| format!("up {}", snapshot.uptime()));
    Line::from(vec![
        Span::styled(" tur ", Style::new().add_modifier(Modifier::BOLD)),
        Span::styled(format!("{version}  "), dim()),
        Span::styled(state, Style::new().fg(color)),
        Span::raw(format!("  {}  ", view.url)),
        Span::styled(uptime, dim()),
    ])
}

fn height(section: &Section) -> u16 {
    section.rows.len() as u16 + 2
}

fn chart(frame: &mut Frame<'_>, area: Rect, rates: &Rates) {
    let inner = usize::from(area.width.saturating_sub(2));
    let skip = rates.history.len().saturating_sub(inner);
    let data: Vec<u64> = rates.history.iter().skip(skip).copied().collect();
    frame.render_widget(
        Sparkline::default()
            .block(block("Tokens/s"))
            .style(Style::new().fg(Color::Cyan))
            .data(&data),
        area,
    );
}

fn logs(frame: &mut Frame<'_>, area: Rect, lines: &[String]) {
    let lines: Vec<Line<'_>> = lines.iter().map(|line| Line::raw(line.as_str())).collect();
    frame.render_widget(Paragraph::new(lines).block(block("Logs")), area);
}

/// Draw `view` over the whole frame: sections in columns when they fit, the chart beneath
/// them, and what height is left for logs.
pub(super) fn draw(frame: &mut Frame<'_>, view: &View<'_>) {
    let [top, body, bottom] = Layout::vertical([
        Constraint::Length(1),
        Constraint::Min(0),
        Constraint::Length(1),
    ])
    .areas(frame.area());
    frame.render_widget(Paragraph::new(header(view)), top);
    let footer = view.problem.map_or_else(
        || Span::styled(format!(" {}", view.keys), dim()),
        |problem| Span::styled(format!(" {problem}"), Style::new().fg(Color::Yellow)),
    );
    frame.render_widget(Paragraph::new(Line::from(footer)), bottom);
    let sections = view
        .snapshot
        .map_or_else(Vec::new, |snapshot| snapshot.sections(Some(view.rates)));
    let remainder = if body.width >= TWO_COLUMNS {
        columns(frame, body, &sections)
    } else {
        stacked(frame, body, &sections)
    };
    let [graph, rest] =
        Layout::vertical([Constraint::Length(5), Constraint::Min(0)]).areas(remainder);
    chart(frame, graph, view.rates);
    if let Some(lines) = view.logs {
        let shown = usize::from(rest.height.saturating_sub(2));
        let skip = lines.len().saturating_sub(shown);
        logs(frame, rest, lines.get(skip..).unwrap_or(lines));
    }
}

/// Lay `sections` into two balanced columns and return the area beneath them.
fn columns(frame: &mut Frame<'_>, area: Rect, sections: &[Section]) -> Rect {
    let [left, right] =
        Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)]).areas(area);
    let (mut left_used, mut right_used) = (0u16, 0u16);
    for item in sections {
        let (column, used) = if left_used <= right_used {
            (left, &mut left_used)
        } else {
            (right, &mut right_used)
        };
        let rect = Rect {
            x: column.x,
            y: column.y + *used,
            width: column.width,
            height: height(item).min(column.height.saturating_sub(*used)),
        };
        section(frame, rect, item);
        *used += rect.height;
    }
    let used = left_used.max(right_used).min(area.height);
    Rect {
        y: area.y + used,
        height: area.height - used,
        ..area
    }
}

/// Stack `sections` and return the area beneath them.
fn stacked(frame: &mut Frame<'_>, area: Rect, sections: &[Section]) -> Rect {
    let mut used = 0u16;
    for item in sections {
        let rect = Rect {
            y: area.y + used,
            height: height(item).min(area.height - used),
            ..area
        };
        section(frame, rect, item);
        used += rect.height;
    }
    Rect {
        y: area.y + used,
        height: area.height - used,
        ..area
    }
}

#[cfg(test)]
#[allow(clippy::panic_in_result_fn, reason = "tests assert drawn output")]
mod tests {
    use super::super::snapshot::SAMPLE;
    use super::*;
    use ratatui::{Terminal, backend::TestBackend};
    use tur::{Result, TurError};

    fn screen(width: u16, height: u16, view: &View<'_>) -> Result<String> {
        let failed = |error: &dyn std::fmt::Display| TurError::Other(error.to_string());
        let mut terminal =
            Terminal::new(TestBackend::new(width, height)).map_err(|e| failed(&e))?;
        terminal
            .draw(|frame| draw(frame, view))
            .map_err(|e| failed(&e))?;
        Ok(terminal.backend().to_string())
    }

    #[test]
    fn test_dashboard_fits_any_terminal_and_shows_logs() -> Result<()> {
        let snapshot = Snapshot::parse(SAMPLE)?;
        let rates = Rates::default();
        let logs = vec!["older line".to_owned(), "newest line".to_owned()];
        let view = View {
            url: "http://x:1",
            snapshot: Some(&snapshot),
            rates: &rates,
            problem: None,
            logs: Some(&logs),
            keys: "q close",
        };
        let wide = screen(140, 50, &view)?;
        for expected in [
            "worker stopped",
            "http://x:1",
            "State pool",
            "Tokens/s",
            "newest line",
            "q close",
        ] {
            assert!(wide.contains(expected), "missing {expected}:\n{wide}");
        }
        assert!(screen(70, 60, &view)?.contains("K/V pages"));
        // Too small to show much, but it must not panic on its arithmetic.
        for (width, height) in [(1, 1), (20, 4), (120, 6), (60, 8)] {
            screen(width, height, &view)?;
        }
        Ok(())
    }

    #[test]
    fn test_unreachable_server_says_so_and_keeps_the_last_reading() -> Result<()> {
        let snapshot = Snapshot::parse(r#"{"draining": false}"#)?;
        let rates = Rates::default();
        let view = View {
            url: "http://x:1",
            snapshot: Some(&snapshot),
            rates: &rates,
            problem: Some("no Tur server at http://x:1"),
            logs: None,
            keys: "q quit",
        };
        let text = screen(120, 20, &view)?;
        assert!(text.contains("unreachable") && text.contains("no Tur server"));
        Ok(())
    }
}
