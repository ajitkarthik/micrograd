# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run

The project is configured with CMake; an out-of-tree `build/` directory already exists.

```sh
cmake --build build          # incremental build
./build/neuron               # run the example executable
```

If `build/` is missing or `CMakeLists.txt` changed:

```sh
cmake -S . -B build
```

There is no test suite; verification is done by running an example (currently `neuron`) and inspecting stdout — streaming a `Value` to `std::ostream` (via `operator<<`) walks the graph and prints each node's `data`, `grad`, `op`, and `opstring`.

## Code Conventions

- C++23 (`CMAKE_CXX_STANDARD 23`).
- Both the `micrograd` library and example targets compile with `-Wall -Wextra -g -O0 -Wimplicit-fallthrough` (set per-target in `CMakeLists.txt`).
- Formatting: Google style, 4-space indent, 100-column limit (`.clang-format`).

## Architecture

This is a C++ port of Karpathy's [micrograd](https://github.com/karpathy/micrograd) — a tiny reverse-mode autograd engine over scalar values.

Layout:

- `include/value.hpp` — public `Value` API (declarations only).
- `src/value.cpp` — `Value` and `Value::Node` definitions; compiled into a `micrograd` static library.
- `examples/neuron.cpp` — example program (the `neuron` executable) that links against `micrograd`. New examples should be added here and wired up with their own `add_executable` + `target_link_libraries(... PRIVATE micrograd)` in `CMakeLists.txt`.

The key design split is `Value` (value-semantic handle) vs. `Value::Node` (the shared graph node, defined as a private nested struct):

- `Value::Node` holds the actual autograd state: `data`, `grad`, the list of parent `Node`s in `prev`, an `Op` enum tag (`LEAF`, `ADD`, `MUL`, `TANH`), and human-readable `label`/`opstring` fields. Nodes live behind `std::shared_ptr<Node>` so the DAG is kept alive as long as any `Value` references it.
- `Value` is a thin wrapper holding one `std::shared_ptr<Node>`. Operator overloads (`operator+`, `operator*`, `tanh()`) construct a new `Node` whose `prev` points to the operand nodes, and return a new `Value` wrapping it. Copying a `Value` shares the underlying node — it does **not** clone the graph.

Backward pass: `Value::backward()` delegates to `Node::backward()`, which is a driver that:

1. Runs a DFS from the output node (`Node::DFSVisit`) to build a post-order topological sort, pushing each node onto a `std::stack<Node*>` after its parents have been visited.
2. Pops the stack and calls `Node::backward_local()` on each node. `backward_local()` dispatches on the `Op` tag and adds local-gradient × this-node's-grad into each parent's `grad` — it does **not** recurse.

Splitting `backward()` (the topo-driver) from `backward_local()` (the per-op gradient rule) is what guarantees a node's `grad` is fully accumulated by all downstream consumers before it propagates upward. Without that split, diamond DAGs (a non-leaf `Value` reused by multiple downstream ops) would produce wrong gradients.

The caller is expected to seed `grad = 1.0` on the output node before calling `backward()` (see `examples/neuron.cpp`). The current topo-sort implementation is intentionally simplified for graphs with a single output root (e.g. an MLP with one output neuron) — see the comment block above `DFSVisit` in `value.cpp`. Multiple-output graphs would need a separate driver that seeds and traverses from each root.

`Node::print` independently dedupes via its own `visited` set; that's purely a pretty-printing concern, unrelated to the backward pass.

When adding new ops: add an enum variant to `Node::Op`, an operator/method on `Value` in `value.hpp` + `value.cpp` that constructs the node with the right `prev`, and a `case` in `Node::backward_local()` for the local gradient. Do **not** add recursion there — the topo driver in `Node::backward()` owns visit order.
