const std = @import("std");
const mnist = @import("./mnist/mnist.zig").mnist;
const make_moons = @import("./makemoons/makemoons.zig").make_moons;
const NonLinear = @import("microgradz").NonLinear;
const LayerSpec = @import("microgradz").LayerSpec;

pub fn main() !void {
    // const shape = [_]usize{ 2, 16, 16, 1 };
    // const nonlin = [_]NonLinear{ NonLinear.relu, NonLinear.relu, NonLinear.none };
    // try make_moons(.{
    //     .epochs = 50,
    //     .load_checkpoint = "make_moons_weights.txt",
    //     .save_checkpoint = "make_moons_weights.txt",
    //     .train = true,
    //     .layer_spec = .{
    //         .shape = shape[0..],
    //         .nonlin = nonlin[0..],
    //     },
    // });

    // const shape = ([_]usize{ 784, 128, 10 })[0..];
    // const nonlin = ([_]NonLinear{ NonLinear.relu, NonLinear.none })[0..];
    // try mnist(.{
    //     .epochs = 1,
    //     .load_checkpoint = "mnist_weights.txt",
    //     .save_checkpoint = "mnist_weights.txt",
    //     .train = false,
    //     .layer_spec = .{
    //         .shape = shape[0..],
    //         .nonlin = nonlin[0..],
    //     },
    // });
}
