const std = @import("std");

pub const Op = enum { None, Add, Sub, Mul, Div, Relu };

fn op_repr(op: Op) []const u8 {
    return switch (op) {
        .Add => "+",
        .Sub => "-",
        .Mul => "*",
        .Div => "/",
        .Relu => "relu",
        .None => "noop",
    };
}

pub const Value = struct {
    data: f64,
    grad: f64 = 0.0,
    operation: Op = Op.None,
    prev: [2]?*Value = .{ null, null },
    allocator: std.mem.Allocator,
    ref_count: usize = 1,

    pub fn zero_grad(self: *Value) void {
        self.grad = 0.0;
    }

    pub fn init(allocator: std.mem.Allocator, input: ?f64) *Value {
        const data = if (input) |v| v else 0.0;
        const self_ptr = allocator.create(Value) catch {
            std.debug.panic("Failed to allocate memory for Value. Out of memory.", .{});
        };

        self_ptr.* = Value{
            .data = data,
            .allocator = allocator,
        };
        return self_ptr;
    }

    pub fn hold(self: *Value) void {
        self.ref_count += 1;
    }

    pub fn deinit(self: *Value) void {
        self.ref_count -= 1;
        if (self.ref_count == 0) {
            // std.debug.print("{?*} (ref_count={})\n", .{ self, self.ref_count });
            if (self.prev[0]) |p0| p0.deinit();
            if (self.prev[1]) |p1| p1.deinit();
            self.allocator.destroy(self);
        }
    }

    pub fn add_backward(self: *Value) void {
        const p0 = self.prev[0] orelse return;
        const p1 = self.prev[1] orelse return;
        p0.grad += self.grad;
        p1.grad += self.grad;
    }

    pub fn sub_backward(self: *Value) void {
        const p0 = self.prev[0] orelse return;
        const p1 = self.prev[1] orelse return;
        p0.grad += self.grad;
        p1.grad -= self.grad;
    }

    pub fn mul_backward(self: *Value) void {
        const p0 = self.prev[0] orelse return;
        const p1 = self.prev[1] orelse return;
        p0.grad += self.grad * p1.data;
        p1.grad += self.grad * p0.data;
    }

    pub fn div_backward(self: *Value) void {
        const p0 = self.prev[0] orelse return;
        const p1 = self.prev[1] orelse return;
        p0.grad += self.grad / p1.data;
        p1.grad -= (self.grad * p0.data) / (p1.data * p1.data);
    }

    pub fn relu_backward(self: *Value) void {
        const p0 = self.prev[0] orelse return;
        if (self.data > 0) {
            p0.grad += self.grad;
        }
    }

    fn dfs_topological(
        node: *Value,
        list: *std.ArrayList(*Value),
        seen: *std.AutoHashMap(usize, void),
    ) void {
        const key = @intFromPtr(node);
        if (seen.contains(key)) return;
        seen.put(key, {}) catch std.debug.panic("Failed to put in seen map. Out of memory.", .{});
        if (node.prev.len == 2) {
            if (node.prev[0]) |p0| dfs_topological(p0, list, seen);
            if (node.prev[1]) |p1| dfs_topological(p1, list, seen);
        }
        list.append(node) catch {
            std.debug.panic("Failed to append to topological list. Out of memory.", .{});
        };
    }

    pub fn backward(self: *Value) f64 {
        var arena = std.heap.ArenaAllocator.init(self.allocator);
        defer arena.deinit();
        const al = arena.allocator();

        var topo = std.ArrayList(*Value).init(al);
        var seen = std.AutoHashMap(usize, void).init(al);
        dfs_topological(self, &topo, &seen);

        for (topo.items) |n| n.grad = 0.0;
        self.grad = 1.0;

        var i: usize = topo.items.len;
        while (i > 0) : (i -= 1) {
            const n = topo.items[i - 1];
            switch (n.operation) {
                .Add => n.add_backward(),
                .Sub => n.sub_backward(),
                .Mul => n.mul_backward(),
                .Div => n.div_backward(),
                .Relu => n.relu_backward(),
                .None => {},
            }
        }

        return self.grad;
    }

    fn perform_op(self: *Value, other: *Value, op: Op, val: f64) *Value {
        const res = self.allocator.create(Value) catch {
            std.debug.panic("Failed to allocate memory for Value: Out of memory.", .{});
        };
        self.hold();
        other.hold();
        res.* = .{
            .data = val,
            .operation = op,
            .prev = .{ self, other },
            .allocator = self.allocator,
        };
        return res;
    }

    pub fn add(self: *Value, other: *Value) *Value {
        return self.perform_op(other, .Add, self.data + other.data);
    }

    pub fn sub(self: *Value, other: *Value) *Value {
        return self.perform_op(other, .Sub, self.data - other.data);
    }

    pub fn mul(self: *Value, other: *Value) *Value {
        return self.perform_op(other, .Mul, self.data * other.data);
    }

    pub fn div(self: *Value, other: *Value) *Value {
        if (other.data == 0.0) {
            std.debug.panic("Division by zero in Value.div");
        }
        return self.perform_op(other, .Div, self.data / other.data);
    }

    pub fn relu(self: *Value) *Value {
        const res = self.allocator.create(Value) catch {
            std.debug.panic("Failed to allocate memory for Value: Out of memory.", .{});
        };
        self.hold();
        res.* = .{
            .data = @max(self.data, 0.0),
            .operation = .Relu,
            .prev = .{ self, null },
            .allocator = self.allocator,
        };
        return res;
    }

    pub fn fmt_debug(self: *Value) void {
        std.debug.print(
            "[data:{d} grad:{d} op:{s} prev:[{?*},{?*}]]@{*}\n",
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

test "forward" {
    std.debug.print("Testing Forwards..\n", .{});
    const a = Value.init(std.heap.page_allocator, 2.0);
    const b = Value.init(std.heap.page_allocator, 3.0);
    const c = a.mul(b);
    const d = c.add(a);
    const z = d.relu();
    defer for ([_]*Value{ a, b, c, d, z }) |v| v.deinit();

    try std.testing.expectEqual(a.data, 2.0);
    try std.testing.expectEqual(b.data, 3.0);
    try std.testing.expectEqual(c.data, 6.0);
    try std.testing.expectEqual(d.data, 8.0);
    try std.testing.expectEqual(z.data, 8.0);

    const e = Value.init(std.heap.page_allocator, 2.0);
    const f = Value.init(std.heap.page_allocator, 3.0);
    const g = (e.mul(f).add(e)).relu();
    defer for ([_]*Value{ e, f, g }) |v| v.deinit();

    try std.testing.expectEqual(e.data, 2.0);
    try std.testing.expectEqual(f.data, 3.0);
    try std.testing.expectEqual(g.prev[0].?.prev[0].?.data, 6.0);
    try std.testing.expectEqual(g.prev[0].?.data, 8.0);
    try std.testing.expectEqual(g.data, 8.0);
}

test "backward" {
    std.debug.print("Testing Backwards..\n", .{});
    const a = Value.init(std.heap.page_allocator, 2.0);
    const b = Value.init(std.heap.page_allocator, 3.0);
    const c = a.mul(b); // c = a * b
    const d = c.add(a); // d = c + a
    defer for ([_]*Value{ a, b, c, d }) |v| v.deinit();

    try std.testing.expectEqual(0, a.grad);
    try std.testing.expectEqual(0, b.grad);
    try std.testing.expectEqual(0, c.grad);
    try std.testing.expectEqual(0, d.grad);

    _ = d.backward();

    try std.testing.expectEqual(4.0, a.grad); // d/da = b + 1
    try std.testing.expectEqual(2.0, b.grad); // d/db = a
    try std.testing.expectEqual(1.0, c.grad); // d/dc = 1
    try std.testing.expectEqual(1.0, d.grad); // d/dd = 1

    d.zero_grad();
    const z = d.relu();
    defer z.deinit();

    try std.testing.expectEqual(8.0, z.data);

    _ = z.backward();
    try std.testing.expectEqual(1.0, z.grad); // d/dz = 1
    try std.testing.expectEqual(1.0, d.grad); // d/dd = 1
    try std.testing.expectEqual(1.0, c.grad); // d/dc = 1
    try std.testing.expectEqual(4.0, a.grad); // d/da = b + 1
    try std.testing.expectEqual(2.0, b.grad); // d/db = a

    const negative_value = Value.init(std.heap.page_allocator, -1.0);
    const result = negative_value.relu();
    defer negative_value.deinit();
    defer result.deinit();
    _ = result.backward();

    try std.testing.expectEqual(0.0, negative_value.grad);
    try std.testing.expectEqual(1.0, result.grad);
}
