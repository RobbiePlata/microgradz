const std = @import("std");
const Graph = @import("value.zig").Graph;

pub fn main() !void {
    std.debug.print("Run `zig build test --summary all` to execute unit tests.\n", .{});
    const allocator = std.heap.page_allocator;
    const g = Graph.init(allocator);
    defer g.deinit();

    const a = g.value(1.0);
    const b = g.value(2.0);
    const c = a.mul(b);

    c.backward();
}
