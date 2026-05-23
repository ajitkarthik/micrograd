# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run

The project is configured with CMake; an out-of-tree `build/` directory already exists.

```sh
cmake --build build          # incremental build
./build/micrograd            # run the executable
```

If `build/` is missing or `CMakeLists.txt` changed:

```sh
cmake -S . -B build
```

There is no test suite; verification is done by running `micrograd` and inspecting stdout (the `print` method dumps the expression graph with `data`, `grad`, and `op`).

## Code Conventions

- C++23 (`CMAKE_CXX_STANDARD 23`), compiled with `-Wall -Wextra -g -O0`.
- Formatting: Google style, 4-space indent, 100-column limit (`.clang-format`).

## Architecture

This is a C++ port of Karpathy's [micrograd](https://github.com/karpathy/micrograd) — a tiny reverse-mode autograd engine over scalar values. Everything lives in `src/main.cpp`.

The key design split is `Value` (value-semantic handle) vs. `Value::Node` (the shared graph node):

- `Value::Node` holds the actual autograd state: `data`, `grad`, the list of parent `Node`s in `prev`, an `Op` enum tag (`LEAF`, `ADD`, `MUL`, `TANH`), and a human-readable `label`/`opstring`. Nodes live behind `std::shared_ptr<Node>` so the DAG is kept alive as long as any `Value` references it.
- `Value` is a thin wrapper holding one `std::shared_ptr<Node>`. Operator overloads (`operator+`, `operator*`, `tanh()`) construct a new `Node` whose `prev` points to the operand nodes, and return a new `Value` wrapping it. Copying a `Value` shares the underlying node — it does **not** clone the graph.

Backward pass: `Value::backward()` delegates to `Node::backward()`, which dispatches on the `Op` tag to add local-gradient × upstream-grad into each parent's `grad`, then recurses into `prev`. The caller is expected to seed `grad = 1.0` on the output node before calling `backward()` (see the `main` examples). There is no topological-sort deduplication, so a node reached via multiple paths will be visited and updated more than once per traversal; expressions that re-use a sub-`Value` (e.g. `a + a`) rely on this to accumulate the correct gradient.

When adding new ops: add an enum variant to `Node::Op`, an operator/method on `Value` that constructs the node with the right `prev`, and a `case` in `Node::backward()` for the local gradient.
