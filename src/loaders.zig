const std = @import("std");

pub fn csv(comptime data: [:0]const u8, comptime Row: type) type {
    const row_type_info = @typeInfo(Row);
    if (row_type_info != .@"struct") @compileError("RowType must be a struct");

    const fields = row_type_info.@"struct".fields;
    const count = fields.len;

    const header_slice = std.mem.sliceTo(data, '\n');
    const header = std.mem.trimRight(u8, header_slice, "\r");

    var iter = std.mem.splitScalar(u8, header, ',');
    var index: usize = 0;
    while (iter.next()) |tok| : (index += 1) {
        if (index >= count) @compileError("Header has more columns than struct fields");
        const trimmed = std.mem.trim(u8, tok, " \t");
        if (!std.mem.eql(u8, trimmed, fields[index].name)) {
            @compileError("Header field '" ++ trimmed ++ "' does not match struct field '" ++ fields[index].name ++ "'");
        }
    }
    if (index != count) @compileError("Header has fewer columns than struct fields");

    inline for (fields) |f| {
        const ti = @typeInfo(f.type);
        switch (ti) {
            .int, .float, .bool => {},
            .pointer => |pi| {
                if (pi.size != .Slice or pi.child != u8 or !pi.is_const or pi.sentinel != null) {
                    @compileError("Unsupported pointer type for field " ++ f.name ++ ": only []const u8 is supported for strings.");
                }
            },
            else => @compileError("Unsupported type for field " ++ f.name ++ ": " ++ @typeName(f.type)),
        }
    }

    return struct {
        const file_data: []const u8 = data;

        fn parse(comptime T: type, s: []const u8) !T {
            const trimmed = std.mem.trim(u8, s, " \t");
            const type_info = @typeInfo(T);
            return switch (type_info) {
                .int => |int_info| {
                    if (int_info.signedness == .signed) {
                        return try std.fmt.parseInt(T, trimmed, 0);
                    } else {
                        return try std.fmt.parseUnsigned(T, trimmed, 0);
                    }
                },
                .float => try std.fmt.parseFloat(T, trimmed),
                .bool => {
                    if (std.ascii.eqlIgnoreCase(trimmed, "true")) return true;
                    if (std.ascii.eqlIgnoreCase(trimmed, "false")) return false;
                    return error.InvalidBoolean;
                },
                .pointer => |ptr_info| {
                    if (ptr_info.size == .Slice and ptr_info.child == u8 and ptr_info.is_const and ptr_info.sentinel == null) {
                        return trimmed;
                    } else {
                        @compileError("Unsupported pointer type: only []const u8 is supported for strings.");
                    }
                },
                else => @compileError("Unsupported type: " ++ @typeName(T)),
            };
        }

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
                inline for (@typeInfo(Row).@"struct".fields) |field_info| {
                    if (cols.next()) |tok| {
                        @field(p, field_info.name) = try parse(field_info.type, tok);
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
