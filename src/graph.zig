const std = @import("std");

pub const Op = enum { None, Add, Sub, Mul, Div, Relu, Log, Exp };

fn op_repr(op: Op) []const u8 {
    return switch (op) {
        .Add => "+",
        .Sub => "-",
        .Mul => "*",
        .Div => "/",
        .Relu => "relu",
        .Log => "log",
        .Exp => "exp",
        .None => "noop",
    };
}

pub const Graph = struct {
    arena: std.heap.ArenaAllocator,

    pub fn init(alloc: std.mem.Allocator) Graph {
        return .{ .arena = std.heap.ArenaAllocator.init(alloc) };
    }

    pub fn deinit(self: *Graph) void {
        self.arena.deinit();
    }

    pub fn clear(self: *Graph) void {
        _ = self.arena.reset(.retain_capacity);
    }

    pub fn allocator(self: *Graph) std.mem.Allocator {
        return self.arena.allocator();
    }

    pub fn value(self: *Graph, data: f64) *Value {
        const v = self.allocator().create(Value) catch @panic("OOM");
        v.* = .{ .graph = self, .data = data };
        return v;
    }
};

pub const Value = struct {
    graph: *Graph,
    data: f64,
    grad: f64 = 0,
    operation: Op = .None,
    prev: [2]?*Value = .{ null, null },

    fn spawn(self: *Value, data: f64, op: Op, prev: [2]?*Value) *Value {
        const v = self.graph.allocator().create(Value) catch @panic("OOM");
        v.* = .{
            .graph = self.graph,
            .data = data,
            .operation = op,
            .prev = prev,
        };
        return v;
    }

    pub fn add(self: *Value, other: *Value) *Value {
        return self.spawn(self.data + other.data, .Add, .{ self, other });
    }

    pub fn sub(self: *Value, other: *Value) *Value {
        return self.spawn(self.data - other.data, .Sub, .{ self, other });
    }

    pub fn mul(self: *Value, other: *Value) *Value {
        return self.spawn(self.data * other.data, .Mul, .{ self, other });
    }

    pub fn div(self: *Value, other: *Value) !*Value {
        if (other.data == 0.0) {
            return error.DivisionByZero;
        }
        return self.spawn(self.data / other.data, .Div, .{ self, other });
    }

    pub fn relu(self: *Value) *Value {
        return self.spawn(@max(self.data, 0.0), .Relu, .{ self, null });
    }

    pub fn log(self: *Value) !*Value {
        const epsilon: f64 = 1e-10;
        const clipped: f64 = if (self.data <= epsilon) epsilon else self.data;
        return self.spawn(std.math.log(f64, std.math.e, clipped), .Log, .{ self, null });
    }

    pub fn exp(self: *Value) *Value {
        return self.spawn(std.math.exp(self.data), .Exp, .{ self, null });
    }

    pub fn zero_grad(self: *Value) void {
        self.grad = 0.0;
    }

    fn dfs_topo(
        node: *Value,
        list: *std.ArrayList(*Value),
        seen: *std.AutoHashMap(usize, void),
    ) void {
        const Item = struct { node: *Value, is_visited: bool };
        var stack = std.ArrayList(Item).init(seen.allocator);
        defer stack.deinit();

        stack.append(.{ .node = node, .is_visited = false }) catch @panic("OOM");

        while (stack.items.len > 0) {
            const idx = stack.items.len - 1;
            var item_ptr = &stack.items[idx];

            if (item_ptr.is_visited) {
                _ = stack.pop();
                list.append(item_ptr.node) catch @panic("OOM");
                continue;
            }
            item_ptr.is_visited = true;

            const current_node = item_ptr.node;
            const key = @intFromPtr(current_node);

            if (seen.contains(key)) {
                _ = stack.pop();
                continue;
            }
            seen.put(key, {}) catch @panic("OOM");

            if (current_node.prev[1]) |p1| {
                stack.append(.{ .node = p1, .is_visited = false }) catch @panic("OOM");
            }
            if (current_node.prev[0]) |p0| {
                stack.append(.{ .node = p0, .is_visited = false }) catch @panic("OOM");
            }
        }
    }

    pub fn backward(self: *Value) void {
        var arena = std.heap.ArenaAllocator.init(self.graph.allocator());
        defer arena.deinit();
        const al = arena.allocator();

        var topo = std.ArrayList(*Value).init(al);
        var seen = std.AutoHashMap(usize, void).init(al);
        dfs_topo(self, &topo, &seen);

        for (topo.items) |n| n.grad = 0.0;
        self.grad = 1.0;

        var i = topo.items.len;
        while (i > 0) : (i -= 1) {
            const n = topo.items[i - 1];
            switch (n.operation) {
                .Add => {
                    const p0 = n.prev[0].?;
                    const p1 = n.prev[1].?;
                    p0.grad += n.grad;
                    p1.grad += n.grad;
                },
                .Sub => {
                    const p0 = n.prev[0].?;
                    const p1 = n.prev[1].?;
                    p0.grad += n.grad;
                    p1.grad -= n.grad;
                },
                .Mul => {
                    const p0 = n.prev[0].?;
                    const p1 = n.prev[1].?;
                    p0.grad += n.grad * p1.data;
                    p1.grad += n.grad * p0.data;
                },
                .Div => {
                    const p0 = n.prev[0].?;
                    const p1 = n.prev[1].?;
                    p0.grad += n.grad / p1.data;
                    p1.grad -= n.grad * p0.data / (p1.data * p1.data);
                },
                .Relu => {
                    const p0 = n.prev[0].?;
                    if (n.data > 0.0) p0.grad += n.grad;
                },
                .Log => {
                    const p0 = n.prev[0].?;
                    p0.grad += n.grad / p0.data;
                },
                .Exp => {
                    const p0 = n.prev[0].?;
                    p0.grad += n.grad * n.data;
                },
                .None => {},
            }
        }
    }

    pub fn fmt_debug(self: *Value) void {
        std.debug.print(
            "[data:{d}, grad:{d}, op:{s}, prev:[{?*},{?*}]] @{*}\n",
            .{
                self.data,
                self.grad,
                op_repr(self.operation),
                self.prev[0],
                self.prev[1],
                self,
            },
        );
    }
};

test "forward/backward" {
    var g = Graph.init(std.heap.page_allocator);
    defer g.deinit();

    const a = g.value(2.0);
    const b = g.value(3.0);
    const z = a.mul(b).add(a).relu();

    try std.testing.expectEqual(6.0, z.prev[0].?.prev[0].?.data);
    try std.testing.expectEqual(8.0, z.prev[0].?.data);
    try std.testing.expectEqual(8.0, z.data);

    z.backward();
    const d = z.prev[0].?;
    const c = d.prev[0].?;

    try std.testing.expectEqual(1.0, z.grad);
    try std.testing.expectEqual(1.0, d.grad);
    try std.testing.expectEqual(1.0, c.grad);
    try std.testing.expectEqual(2.0, b.grad); // a
    try std.testing.expectEqual(4.0, a.grad); // b + 1
}
