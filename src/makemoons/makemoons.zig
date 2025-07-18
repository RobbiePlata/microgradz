const std = @import("std");
const Graph = @import("microgradz").Graph;
const MLP = @import("microgradz").MLP;
const Value = @import("microgradz").Value;
const LayerSpec = @import("microgradz").LayerSpec;
const NonLinear = @import("microgradz").NonLinear;
const csv = @import("../loaders.zig").csv;

const MakeMoons = csv(@embedFile("make_moons.csv"));

const TrainOptions = struct {
    layer_spec: LayerSpec,
    epochs: usize = 100,
    overwrite: bool = false,
    checkpoint: []const u8 = "make_moons_weights.txt",
    seed: u64 = 123,
};

pub fn train_make_moons(
    options: TrainOptions,
) !void {
    const checkpoint = options.checkpoint;

    const allocator = std.heap.page_allocator;
    const makemoons = try MakeMoons.load(allocator);
    defer allocator.free(makemoons);

    var input_graph = Graph.init(allocator);
    defer input_graph.deinit();

    var weight_graph = Graph.init(allocator);
    defer weight_graph.deinit();

    var mlp = try MLP.init(&weight_graph, options.layer_spec, options.seed);

    mlp.load(allocator, checkpoint) catch |err| {
        std.debug.print("Failed to load weights: {}\n", .{err});
        return err;
    };

    const params = mlp.parameters();
    std.debug.print("Parameters: {d}\n", .{params.len});

    const N: i64 = @intCast(makemoons.len);
    const f_N: f64 = @floatFromInt(N);

    const f_epochs: f64 = @floatFromInt(options.epochs);

    const alpha = 1e-4;

    for (0..options.epochs) |epoch| {
        var correct: i32 = 0;
        input_graph.clear();

        var total_loss = input_graph.value(0.0);
        for (makemoons) |row| {
            const f_X: f64 = try std.fmt.parseFloat(f64, row.X);
            const f_Y: f64 = try std.fmt.parseFloat(f64, row.Y);
            const x0 = input_graph.value(f_X);
            const x1 = input_graph.value(f_Y);
            const label: i8 = try std.fmt.parseInt(i8, row.label, 10);

            var inputs: [2]*Value = .{ x0, x1 };
            const outputs = try mlp.forward(inputs[0..]);
            const score = outputs[0];

            const y_val = input_graph.value(if (label == 0) -1.0 else 1.0);
            const neg_y_score = y_val.mul(score).mul(input_graph.value(-1.0));
            const margin = input_graph.value(1.0).add(neg_y_score).relu();

            total_loss = total_loss.add(margin);

            if ((score.data > 0.0) == (label == 1)) {
                correct += 1;
            }
        }
        total_loss = total_loss.mul(input_graph.value(1.0 / f_N));
        total_loss.backward();

        for (params) |p| {
            p.grad += 2.0 * alpha * p.data;
        }

        const f_epoch: f64 = @floatFromInt(epoch);
        const l_r = 0.1 - 0.09 * f_epoch / f_epochs;
        for (params) |p| {
            p.data -= l_r * p.grad;
            p.zero_grad();
        }

        const acc: f64 = @as(f64, @floatFromInt(correct)) / f_N;
        std.debug.print(
            "Epoch {d}/{d} - loss={d:.4}, acc={d:.2}\n",
            .{ epoch + 1, options.epochs, total_loss.data, acc },
        );
    }

    if (options.overwrite) {
        try mlp.save(checkpoint);
    }
}

const EvalParams = struct { layer_spec: LayerSpec, checkpoint: []const u8 = "make_moons_weights.txt" };

pub fn eval_make_moons(
    options: EvalParams,
) !void {
    const allocator = std.heap.page_allocator;

    const makemoons = try MakeMoons.load(allocator);
    defer allocator.free(makemoons);

    var input_graph = Graph.init(allocator);
    defer input_graph.deinit();

    var weight_graph = Graph.init(allocator);
    defer weight_graph.deinit();

    var mlp = try MLP.init(&weight_graph, options.layer_spec, 123);

    mlp.load(allocator, options.checkpoint) catch |err| {
        std.debug.print("Failed to load weights: {}\n", .{err});
        return err;
    };

    const params = mlp.parameters();
    std.debug.print("Parameters: {d}\n", .{params.len});
    var correct: i32 = 0;
    input_graph.clear();

    for (makemoons) |row| {
        const f_X: f64 = try std.fmt.parseFloat(f64, row.X);
        const f_Y: f64 = try std.fmt.parseFloat(f64, row.Y);
        const label: i8 = try std.fmt.parseInt(i8, row.label, 10);

        const x0 = input_graph.value(f_X);
        const x1 = input_graph.value(f_Y);

        var inputs: [2]*Value = .{ x0, x1 };
        const outputs = try mlp.forward(inputs[0..]);
        const score = outputs[0];

        if ((score.data > 0.0) == (label == 1)) {
            correct += 1;
        }
    }
    const N: i64 = @intCast(makemoons.len);
    const f_N: f64 = @floatFromInt(N);
    const acc: f64 = @as(f64, @floatFromInt(correct)) / f_N;
    std.debug.print("Accuracy: {d:.2}\n", .{acc});
}
