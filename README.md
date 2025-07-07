# microgradz

Implementation of [Micrograd](https://github.com/karpathy/micrograd) by Andrej Karpathy utilizing [Ziglang](https://ziglang.org/). Micrograd is a simpler scalar-value based autograd engine which creates a topological graph of operations and compute gradients using backpropagation.

This is my first project after completing the [ziglings](https://codeberg.org/ziglings/exercises/#ziglings) excercises hence the 'z' in "microgradz".

# Usage
```zig
var g = Graph.init(std.heap.page_allocator);
defer g.deinit();

const a = g.value(2.0);
const b = g.value(3.0);
const z = a.mul(b).add(a).relu();

z.backward();
const d = z.prev[0].?;
const c = d.prev[0].?;

1.0 == z.grad
1.0 == d.grad
1.0 == c.grad
2.0 == b.grad
4.0 == a.grad
```
