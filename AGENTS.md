# AGENTS.md - AI Assistant Context for Tur

## How to Use This File

**For AI Coding Agents**: This document provides an overview and references to detailed coding standards for the Logrange project. You MUST:

1. Read and understand the rules before generating any code
2. Verify existing patterns by reading relevant files before creating new code
3. Never deviate from the safety rules under any circumstances
4. Ask for clarification if a pattern is unclear rather than inventing new approaches

## Project Overview

**Tur** is a high-performance, production-grade Rust inference engine and generation pipeline - vLLM alternative. It utilizes state-of-the-art optimizations for inference engines strict performance, reliability, and safety requirements.

## Core Characteristics

- **Language**: Rust edition 2024
- **Runtime**: Built on Candle library for ML/DL
- **Architecture**: Modular design
- **Performance**: Zero-copy where possible, lock-free data structures, optimized for throughput
- **Safety**: No unsafe code, no panics, comprehensive error handling
- **Testing**: High coverage (>80%) using cargo-nextest

## Agent Workflow for Code Changes

### Before Writing Code

1. **Understand the request**: Identify which workspace member(s) are affected
2. **Read existing code**: Use tools to read relevant files in the target module
3. **Identify patterns**: Look for similar implementations to follow
4. **Check dependencies**: Verify what types/traits are available
5. **Plan the change**: Outline the approach before implementation

## Agent Rules for Code Generation

### Allowed

- Modify existing files when explicitly instructed
- Create new modules within the appropriate workspace member
- Use only existing crates and modules unless asked to add new dependencies
- Follow existing error and concurrency patterns strictly
- Generated code should be performance-critical and optimized for throughput
- Rust idiomatic conventions

### Not Allowed

- Do not invent new **foundational** structs, traits, or modules that duplicate existing functionality
- Do not create new workspace crates without explicit approval
- Do not introduce unsafe code, unwrap/expect, or panic
- Do not modify shared foundational types (Config, Errors) unless explicitly requested
- Do not change public APIs without confirmation

## Documentation Standards

### When to Comment

- **DO**: Explain "why" for non-obvious decisions
- **DO**: Document complex algorithms
- **DO**: Add doc comments for public APIs
- **DON'T**: Restate what code clearly shows
- **DON'T**: Add redundant comments

## Testing Rules

**Priority**: HIGH - Ensures code quality and reliability

- **Unit tests**: Every module with `#[cfg(test)]`
- **Integration tests**: In `tests/` directory
- **Benchmarks**: For performance-critical paths in `benches/` directory

## Test Naming

- Test names: `test_<behavior>`
- Use descriptive names that explain what is being tested

## Adding Dependencies

Agents must not add new crates unless:

1. They appear in the project root `Cargo.toml`, OR
2. The user explicitly instructs to add a new dependency

All new dependencies must:

- Use the latest tested version
- Be lightweight and production-safe
- Support Rust 2024

## Build and Development

### Quick Commands

```bash
cargo test --lib    # Run unit tests
cargo check         # Check for errors and warnings
cargo build         # Build the project
```

## Quick Reference Card for Agents

### ✅ Always Do

- Use `?` for error propagation
- Add doc comments to public APIs explaining the method behavior
- Write tests for new functionality

### ❌ Never Do

- `unwrap()`, `expect()`, `panic!()`
- `unsafe { }`
- Useless comments
- Repetitions
- Hold locks across await points
- Clone unnecessarily
- Add dependencies without approval

### 🔍 Before Coding

1. Read existing similar code
2. Identify the correct workspace member
3. Check existing code for reusable types
4. Verify error handling patterns

### 📝 Code Review Checklist

- [ ] No unwrap/expect/panic
- [ ] Proper error handling with `?`
- [ ] No locks held across await
- [ ] Tests added/updated
- [ ] Follows existing patterns

---

**Last Updated**: 2026-05-18  
**Maintainers**: Danylo Kravchenko
**License**: Apache-2.0
