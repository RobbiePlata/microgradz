// const eval_make_moons = @import("./makemoons/makemoons.zig").eval_make_moons;
const train_make_moons = @import("./makemoons/makemoons.zig").train_make_moons;
const NonLinear = @import("microgradz").NonLinear;
const std = @import("std");

pub fn main() !void {
    const shape = [_]usize{ 2, 16, 16, 1 };
    const nonlin = [_]NonLinear{ NonLinear.relu, NonLinear.relu, NonLinear.none };

    try train_make_moons(.{
        .layer_spec = .{
            .shape = shape[0..],
            .nonlin = nonlin[0..],
        },
        .checkpoint = "make_moons_weights.txt",
    });
}
