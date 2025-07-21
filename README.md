# microgradz

Implementation of [Micrograd](https://github.com/karpathy/micrograd) by Andrej Karpathy utilizing [Ziglang](https://ziglang.org/). Micrograd is a simpler scalar-value based autograd engine which creates a topological graph of operations and compute gradients using backpropagation.

This is my first project after completing the [ziglings](https://codeberg.org/ziglings/exercises/#ziglings).

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

# Tests
```
zig build test --summary all
```
# Run

```
zig build run
```

Ensure the following directory structure relative to your project root
```
microgradz.exe
weights/
  ├── mnist_weights.txt
  └── makemoons_weights.txt
data/
  ├── mnist/
  │   ├── train-images-idx3-ubyte 
  │   │   └── train-images-idx3-ubyte
  │   │── train-labels-idx1-ubyte    
  │   │   └── train-labels-idx1-ubyte   
  └── makemoons/
      └── make_moons.csv
```

Training is **very** slow, no vectorization or parallelization is utilized for any operations!