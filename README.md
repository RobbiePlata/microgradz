# microgradz

Implementation of [Micrograd](https://github.com/karpathy/micrograd) by Andrej Karpathy utilizing [Ziglang](https://ziglang.org/). Micrograd is a simpler scalar-value based autograd engine which creates a topological graph of operations and compute gradients using backpropagation.

This is my first project after completing the [ziglings](https://codeberg.org/ziglings/exercises/#ziglings) excercises hence the 'z' in "microgradz".

# Usage
```zig
const std = @import("microgradz").Value;

const a = Value.init(std.heap.page_allocator, 2.0);
const b = Value.init(std.heap.page_allocator, 3.0);
const c = a.mul(b);
const d = c.add(a);
const z = d.relu();
defer for ([_]*Value{ a, b, c, d, z }) |v| v.deinit();

// or

const a = Value.init(std.heap.page_allocator, 2.0);
const b = Value.init(std.heap.page_allocator, 3.0);
const z = (a.mul(b).add(a)).relu();
defer for ([_]*Value{ a, b, z }) |v| v.deinit(); // frees leafs and nodes

// yes we exist
const d_ptr = z.prev[0];
const c_ptr = d_ptr.?.*.prev[0];

a.data == 2.0;
b.data == 3.0;
c_ptr.?.data == 6.0;
d_ptr.?.data == 8.0;
z.data == 8.0;

_ = z.backward();
```
