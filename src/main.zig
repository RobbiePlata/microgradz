const std = @import("std");
const Value = @import("value.zig").Value;

pub fn main() !void {
    std.debug.print("Run `zig build test --summary all` to execute unit tests.\n", .{});
    const allocator = std.heap.page_allocator;

    const a = Value.init(allocator, 1.0);
    defer a.deinit();

    const b = Value.init(allocator, 2.0);
    defer b.deinit();

    const c = a.mul(b);
    defer c.deinit();

    c.backward();
}
