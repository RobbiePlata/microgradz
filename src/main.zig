const std = @import("std");
const Graph = @import("graph.zig").Graph;
const MakeMoons = @import("utils/makemoons.zig").MakeMoons;
const MLP = @import("mlp.zig").MLP;
const Value = @import("graph.zig").Value;
const LayerSpec = @import("mlp.zig").LayerSpec;
const NonLinear = @import("neuron.zig").NonLinear;

pub fn main() !void {
    const allocator = std.heap.page_allocator;
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    var makemoons = try MakeMoons.init(allocator, "./", "make_moons.csv");
    defer makemoons.deinit();

    var input_graph = Graph.init(allocator);
    defer input_graph.deinit();

    var weight_graph = Graph.init(allocator);
    defer weight_graph.deinit();

    const sizes = [_]usize{ 2, 10, 10, 1 };
    const non_linear = [_]NonLinear{ NonLinear.relu, NonLinear.relu, NonLinear.none };
    const layer_spec: LayerSpec = .{ .sizes = sizes[0..], .non_linear = non_linear[0..] };

    var mlp = try MLP.init(&weight_graph, layer_spec, 99999);

    const params = mlp.parameters();
    std.debug.print("Parameters: {d}\n", .{params.len});

    const N: i64 = @intCast(makemoons.X.len);
    const f_N: f64 = @floatFromInt(N);
    const epochs = 150;
    const alpha = 1e-4;

    for (0..epochs) |epoch| {
        var correct: i32 = 0;
        input_graph.clear();

        var total_loss = input_graph.value(0.0);
        for (makemoons.X, 0..) |pt, i| {
            const x0 = input_graph.value(pt.x);
            const x1 = input_graph.value(pt.y);
            var inputs: [2]*Value = .{ x0, x1 };
            const outputs = try mlp.forward(inputs[0..]);
            const score = outputs[0];
            const y_val = input_graph.value(if (makemoons.Y[i] == 0) -1.0 else 1.0);

            const neg_y_score = y_val.mul(score).mul(input_graph.value(-1.0));
            const margin = input_graph.value(1.0).add(neg_y_score).relu();

            total_loss = total_loss.add(margin);

            if ((score.data > 0.0) == (makemoons.Y[i] == 1)) {
                correct += 1;
            }
        }

        total_loss = total_loss.mul(input_graph.value(1.0 / f_N));

        total_loss.backward();

        for (params) |p| {
            p.grad += 2.0 * alpha * p.data;
        }

        const f_epoch: f64 = @floatFromInt(epoch);
        const f_epochs: f64 = @floatFromInt(epochs);
        const l_r = 0.1 - 0.09 * f_epoch / f_epochs;
        for (params) |p| {
            p.data -= l_r * p.grad;
            p.zero_grad();
        }

        const acc: f64 = @as(f64, @floatFromInt(correct)) / f_N;
        std.debug.print(
            "Epoch {d}/{d} - loss={d:.4}, acc={d:.2}\n",
            .{ epoch + 1, epochs, total_loss.data, acc },
        );
    }
}
