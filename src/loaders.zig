const std = @import("std");

pub fn csv(comptime data: [:0]const u8) type {
    const header_slice = std.mem.sliceTo(data, '\n');
    const header = std.mem.trimRight(u8, header_slice, "\r");

    var count: usize = 0;
    var iter = std.mem.splitScalar(u8, header, ',');
    while (iter.next()) |_| count += 1;

    iter = std.mem.splitScalar(u8, header, ',');
    var field_names: [count][:0]const u8 = undefined;
    var index: usize = 0;
    while (iter.next()) |tok| : (index += 1) {
        const trimmed = std.mem.trim(u8, tok, " \t");
        field_names[index] = (trimmed ++ .{0})[0..trimmed.len :0];
    }

    var fields: [count]std.builtin.Type.StructField = undefined;
    for (field_names, 0..) |name, i| {
        fields[i] = .{
            .name = name,
            .type = []const u8,
            .default_value_ptr = null,
            .is_comptime = false,
            .alignment = @alignOf([]const u8),
        };
    }

    const RowType = @Type(.{ .@"struct" = .{
        .layout = .auto,
        .fields = &fields,
        .decls = &.{},
        .is_tuple = false,
    } });

    return struct {
        pub const Row = RowType;
        const file_data: []const u8 = data;

        pub fn load(allocator: std.mem.Allocator) ![]Row {
            var lines = std.mem.splitScalar(u8, file_data, '\n');
            _ = lines.next();

            var rows = std.ArrayList(Row).init(allocator);
            errdefer rows.deinit();

            while (lines.next()) |line| {
                if (line.len == 0) continue;
                const trimmed_line = std.mem.trimRight(u8, line, "\r");
                var cols = std.mem.splitScalar(u8, trimmed_line, ',');

                var p: Row = undefined;
                var col_index: usize = 0;
                inline for (@typeInfo(Row).@"struct".fields) |fieldInfo| {
                    if (cols.next()) |tok| {
                        @field(p, fieldInfo.name) = std.mem.trim(u8, tok, " \t");
                        col_index += 1;
                    } else {
                        return error.MissingColumn;
                    }
                }
                if (cols.next() != null) return error.ExtraColumn;
                try rows.append(p);
            }
            return try rows.toOwnedSlice();
        }
    };
}
