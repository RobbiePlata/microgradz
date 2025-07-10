const std = @import("std");
const Graph = @import("graph.zig").Graph;

pub fn main() !void {
    std.debug.print("Run `zig build test --summary all` to execute unit tests.\n", .{});
    var g = Graph.init(std.heap.page_allocator);
    defer g.deinit();

    const a = g.value(1.0);
    const b = g.value(2.0);
    const c = a.mul(b);

    c.backward();
}
