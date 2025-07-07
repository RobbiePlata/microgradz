const std = @import("std");
const Value = @import("graph.zig").Value;

pub fn generate_test_weights(allocator: std.mem.Allocator, params: []*Value, filepath: []const u8) !void {
    var data = try allocator.alloc(f64, params.len);
    defer allocator.free(data);
    for (0..params.len) |i| {
        data[i] = params[i].data;
    }

    const file = try std.fs.cwd().createFile(filepath, .{ .truncate = true });
    defer file.close();

    for (data) |weight| {
        const str = try std.fmt.allocPrint(allocator, "{}\n", .{weight});
        defer allocator.free(str);
        try file.writeAll(str);
    }
}

pub fn expect_weights_equal(allocator: std.mem.Allocator, params: []*Value, filepath: []const u8) !void {
    const size = (try std.fs.cwd().statFile(filepath)).size;
    const raw = try std.fs.cwd().readFileAlloc(allocator, filepath, size);
    defer allocator.free(raw);

    var lines = std.mem.splitScalar(u8, raw, '\n');
    var i: usize = 0;
    while (lines.next()) |line| {
        const trimmed = std.mem.trim(u8, line, " \r");
        if (trimmed.len == 0) continue;
        try std.testing.expectEqual(
            try std.fmt.parseFloat(f64, trimmed),
            params[i].data,
        );
        i += 1;
    }
}
