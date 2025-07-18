const std = @import("std");
const Graph = @import("microgradz").Graph;
const MLP = @import("microgradz").MLP;
const Value = @import("microgradz").Value;
const LayerSpec = @import("microgradz").LayerSpec;
const NonLinear = @import("microgradz").NonLinear;
const csv = @import("../loaders.zig").csv;

const MakeMoonsRow = struct {
    X: f64,
    Y: f64,
    label: i8,
};

const MakeMoons = csv(@embedFile("make_moons.csv"), MakeMoonsRow);

const Options = struct {
    load_checkpoint: ?[]const u8 = null,
    epochs: usize = 100,
    layer_spec: LayerSpec,
    save_checkpoint: ?[]const u8 = null,
    seed: ?u64 = null,
    train: bool = false,
};

pub fn make_moons(
    options: Options,
) !void {
    const load_checkpoint = options.load_checkpoint;
    const epochs: usize = if (options.train) options.epochs else 1;
    const layer_spec = options.layer_spec;
    const save_checkpoint = options.save_checkpoint;
    const seed = options.seed;
    const train = options.train;

    const allocator = std.heap.page_allocator;
    const makemoons = try MakeMoons.load(allocator);
    defer allocator.free(makemoons);

    var input_graph = Graph.init(allocator);
    defer input_graph.deinit();

    var weight_graph = Graph.init(allocator);
    defer weight_graph.deinit();

    var mlp = try MLP.init(&weight_graph, layer_spec, seed);

    if (load_checkpoint) |chk| {
        mlp.load(allocator, chk) catch |err| {
            std.debug.print("Failed to load weights: {}\n", .{err});
            return err;
        };
    }

    const params = mlp.parameters();
    std.debug.print("Parameters: {d}\n", .{params.len});

    const alpha = 1e-4;

    for (0..epochs) |epoch| {
        var correct: f64 = 0.0;
        input_graph.clear();

        var total_loss = input_graph.value(0.0);
        for (makemoons) |row| {
            var inputs: [2]*Value = .{ input_graph.value(row.X), input_graph.value(row.Y) };
            const outputs = try mlp.forward(inputs[0..]);
            const score = outputs[0];

            if ((score.data > 0.0) == (row.label == 1)) {
                correct += 1.0;
            }

            if (train) {
                const y_val = input_graph.value(if (row.label == 0) -1.0 else 1.0);
                const neg_y_score = y_val.mul(score).mul(input_graph.value(-1.0));
                const margin = input_graph.value(1.0).add(neg_y_score).relu();
                total_loss = total_loss.add(margin);
            } else {
                std.debug.print(
                    "X: {d}, Y: {d}, label: {d}, score: {d:.4}\n",
                    .{ row.X, row.Y, row.label, score.data },
                );
            }
        }

        if (train) {
            total_loss = total_loss.mul(input_graph.value(1.0 / @as(f64, @floatFromInt(makemoons.len))));
            total_loss.backward();

            for (params) |p| {
                p.grad += 2.0 * alpha * p.data;
            }

            const l_r = 0.1 - 0.09 * @as(f64, @floatFromInt(epoch)) / @as(f64, @floatFromInt(options.epochs));
            for (params) |p| {
                p.data -= l_r * p.grad;
                p.zero_grad();
            }

            const acc: f64 = @as(f64, correct) / @as(f64, @floatFromInt(makemoons.len));
            std.debug.print(
                "Epoch {d}/{d} - loss={d:.4}, acc={d:.2}\n",
                .{ epoch + 1, epochs, total_loss.data, acc },
            );
        }
    }

    if (save_checkpoint) |save_chk| {
        try mlp.save(save_chk);
        std.debug.print("Saving weights to '{s}'\n", .{save_chk});
    }
}
