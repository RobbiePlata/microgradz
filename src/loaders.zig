const std = @import("std");

pub fn parseCsv(allocator: std.mem.Allocator, data: []const u8, comptime Row: type) ![]Row {
    const row_type_info = @typeInfo(Row);
    if (row_type_info != .@"struct") return error.RowTypeMustBeStruct;

    const fields = row_type_info.@"struct".fields;

    inline for (fields) |f| {
        const ti = @typeInfo(f.type);
        switch (ti) {
            .int, .float, .bool => {},
            .pointer => |pi| {
                if (pi.size != .Slice or pi.child != u8 or !pi.is_const or pi.sentinel != null) {
                    return error.UnsupportedPointerType;
                }
            },
            else => return error.UnsupportedFieldType,
        }
    }

    const header_slice = std.mem.sliceTo(data, '\n');
    const header = std.mem.trimRight(u8, header_slice, "\r");

    var iter = std.mem.splitScalar(u8, header, ',');
    inline for (fields) |field| {
        if (iter.next()) |tok| {
            const trimmed = std.mem.trim(u8, tok, " \t");
            if (!std.mem.eql(u8, trimmed, field.name)) {
                return error.HeaderFieldMismatch;
            }
        } else {
            return error.HeaderHasMissingColumns;
        }
    }
    if (iter.next() != null) return error.HeaderHasExtraColumns;

    var lines = std.mem.splitScalar(u8, data, '\n');
    _ = lines.next();

    var rows = std.ArrayList(Row).init(allocator);
    errdefer rows.deinit();

    while (lines.next()) |line| {
        if (line.len == 0) continue;
        const trimmed_line = std.mem.trimRight(u8, line, "\r");
        var cols = std.mem.splitScalar(u8, trimmed_line, ',');

        var p: Row = undefined;
        inline for (fields) |field_info| {
            if (cols.next()) |tok| {
                @field(p, field_info.name) = try parse(field_info.type, tok);
            } else {
                return error.MissingColumn;
            }
        }
        if (cols.next() != null) return error.ExtraColumn;
        try rows.append(p);
    }
    return rows.toOwnedSlice();
}

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
                return error.UnsupportedPointerType;
            }
        },
        else => error.UnsupportedType,
    };
}
