const std = @import("std");

const Point = struct {
    x: f64,
    y: f64,
};

pub const MakeMoons = struct {
    arena: std.heap.ArenaAllocator,
    X: []Point,
    Y: []const u8,

    pub fn init(allocator: std.mem.Allocator, sub_path: []const u8, filename: []const u8) !MakeMoons {
        var arena = std.heap.ArenaAllocator.init(allocator);
        errdefer arena.deinit();

        var arena_allocator = arena.allocator();

        var dir = try std.fs.cwd().openDir(sub_path, .{});
        defer dir.close();

        const file = try dir.openFile(filename, .{});
        defer file.close();

        const csv_data = try file.readToEndAlloc(arena_allocator, 1024 * 1024);
        defer arena_allocator.free(csv_data);

        var lines_iter = std.mem.splitSequence(u8, csv_data, "\n");

        var point_arr = std.ArrayList(Point).init(arena_allocator);
        var label_arr = std.ArrayList(u8).init(arena_allocator);

        while (lines_iter.next()) |line| {
            if (line.len == 0) continue;
            const trimmed_line = std.mem.trimRight(u8, line, "\r");
            var parts_iter = std.mem.splitSequence(u8, trimmed_line, ",");
            const x_str = parts_iter.next() orelse continue;
            const y_str = parts_iter.next() orelse continue;
            const label_str = parts_iter.next() orelse continue;
            if (parts_iter.next() != null) continue;
            const x_val: f64 = std.fmt.parseFloat(f64, x_str) catch continue;
            const y_val: f64 = std.fmt.parseFloat(f64, y_str) catch continue;
            try point_arr.append(Point{ .x = x_val, .y = y_val });
            const label_val: u8 = std.fmt.parseUnsigned(u8, label_str, 10) catch continue;
            try label_arr.append(label_val);
        }

        return MakeMoons{
            .arena = arena,
            .X = point_arr.items,
            .Y = label_arr.items,
        };
    }

    pub fn deinit(self: *MakeMoons) void {
        self.arena.deinit();
    }

    pub fn to_string(self: *MakeMoons) []const u8 {
        var arena = self.arena;
        const arena_allocator = arena.allocator();
        var string_buffer = std.ArrayList(u8).init(arena_allocator);
        errdefer string_buffer.deinit();
        var writer = string_buffer.writer();
        const len = self.X.len;
        for (0..len) |i| {
            const point = self.X[i];
            const y_val = self.Y[i];
            writer.print("x:{d}y:{d},{d}\n", .{ point.x, point.y, y_val }) catch return &[_]u8{};
        }
        return string_buffer.toOwnedSlice() catch return &[_]u8{};
    }
};
