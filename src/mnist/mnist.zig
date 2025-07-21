const std = @import("std");
const LayerSpec = @import("microgradz").LayerSpec;
const NonLinear = @import("microgradz").NonLinear;
const MLP = @import("microgradz").MLP;
const Graph = @import("microgradz").Graph;
const Value = @import("microgradz").Value;
const softmax = @import("microgradz").softmax;

const MnistData = struct {
    num_images: u32,
    rows: u32,
    cols: u32,
    image_size: usize,
    data: []u8,

    // this has to be relative to the exe
    fn load_images(allocator: std.mem.Allocator, relative_path: []const u8) !MnistData {
        var buffer: [std.fs.max_path_bytes]u8 = undefined;
        const exe_dir = try std.fs.selfExeDirPath(&buffer);

        const full_path = try std.fs.path.join(allocator, &[_][]const u8{ exe_dir, relative_path });
        defer allocator.free(full_path);

        var file = try std.fs.openFileAbsolute(full_path, .{});
        defer file.close();

        var reader = file.reader();

        const magic = try reader.readInt(u32, .big);
        if (magic != 2051) return error.InvalidMagicNumber;

        const num_images = try reader.readInt(u32, .big);
        const rows = try reader.readInt(u32, .big);
        const cols = try reader.readInt(u32, .big);

        const image_size: usize = @as(usize, rows) * @as(usize, cols);
        const total_size: usize = @as(usize, num_images) * image_size;

        const data = try allocator.alloc(u8, total_size);
        const bytes_read = try reader.readAll(data);
        if (bytes_read != total_size) return error.IncompleteData;

        return .{
            .num_images = num_images,
            .rows = rows,
            .cols = cols,
            .image_size = image_size,
            .data = data,
        };
    }

    fn load_labels(allocator: std.mem.Allocator, relative_path: []const u8) ![]u8 {
        var buffer: [std.fs.max_path_bytes]u8 = undefined;
        const exe_dir = try std.fs.selfExeDirPath(&buffer);

        const full_path = try std.fs.path.join(allocator, &[_][]const u8{ exe_dir, relative_path });
        defer allocator.free(full_path);

        var file = try std.fs.openFileAbsolute(full_path, .{});
        defer file.close();

        var reader = file.reader();

        const magic = try reader.readInt(u32, .big);
        if (magic != 2049) return error.InvalidMagicNumber;

        const num_labels = try reader.readInt(u32, .big);

        const data = try allocator.alloc(u8, num_labels);
        const bytes_read = try reader.readAll(data);
        if (bytes_read != num_labels) return error.IncompleteData;

        return data;
    }
};

const Options = struct {
    load_checkpoint: ?[]const u8 = null,
    epochs: usize = 100,
    layer_spec: LayerSpec,
    save_checkpoint: ?[]const u8 = null,
    seed: ?u64 = null,
    train: bool = false,
};

pub fn mnist(options: Options) !void {
    const load_checkpoint = options.load_checkpoint;
    const epochs: usize = if (options.train) options.epochs else 1;
    const layer_spec = options.layer_spec;
    const save_checkpoint = options.save_checkpoint;
    const seed = options.seed;
    const train = options.train;

    const allocator = std.heap.page_allocator;
    const flat_images = try MnistData.load_images(allocator, "data/train-images-idx3-ubyte/train-images-idx3-ubyte");
    const labels = try MnistData.load_labels(allocator, "data/train-labels-idx1-ubyte/train-labels-idx1-ubyte");
    defer allocator.free(flat_images.data);
    defer allocator.free(labels);

    var weight_graph = Graph.init(allocator);
    defer weight_graph.deinit();

    const mlp = try MLP.init(&weight_graph, layer_spec, seed);
    if (load_checkpoint) |chk| {
        mlp.load(allocator, chk) catch |err| {
            std.debug.print("Error loading checkpoint: {s}\n", .{@errorName(err)});
            return err;
        };
    }

    const params = mlp.parameters();

    std.debug.print("Num parameters: {d}\n", .{params.len});
    for (params, 0..) |p, i| {
        if (!std.math.isFinite(p.data)) {
            std.debug.print("Invalid parameter {d}: {d}\n", .{ i, p.data });
            return error.InvalidParameter;
        }
    }

    for (0..epochs) |_| {
        var loss_sum: f64 = 0.0;
        var correct: u32 = 0;

        var avg_loss: f64 = 0.0;
        img_loop: for (0..flat_images.num_images) |i| {
            var untracked_graph = Graph.init(allocator);
            defer untracked_graph.deinit();

            const start = i * flat_images.image_size;
            const end = start + flat_images.image_size;

            var input: [784]*Value = undefined;
            for (flat_images.data[start..end], 0..) |val, index| {
                const f_val: f64 = @as(f64, @floatFromInt(val)) / 255.0;
                input[index] = untracked_graph.value(f_val);
            }

            const output = try mlp.forward(input[0..]);

            const probabilities = softmax(&untracked_graph, output) catch |err| {
                std.debug.print("Error in softmax: {s}\n", .{@errorName(err)});
                for (params) |p| p.zero_grad();
                continue :img_loop;
            };

            const true_label = labels[i];
            var log_probabilities = probabilities[true_label].log() catch |err| {
                std.debug.print("Error in log probabilities: {s}\n", .{@errorName(err)});
                for (params) |p| p.zero_grad();
                continue :img_loop;
            };

            var predicted: usize = 0;
            var max_prob: f64 = -std.math.inf(f64);

            for (probabilities, 0..) |prob, k| {
                if (prob.data > max_prob) {
                    max_prob = prob.data;
                    predicted = k;
                }
            }

            if (predicted == true_label) {
                correct += 1;
            }

            if (train) {
                var loss = log_probabilities.mul(untracked_graph.value(-1.0));
                loss.backward();
                loss_sum += loss.data;

                avg_loss = loss_sum / @as(f64, @floatFromInt(i + 1));
                std.debug.print("Image {d}: Loss = {d}, Predicted = {d}, True Label = {d}\n", .{ i, avg_loss, predicted, true_label });

                for (params) |p| {
                    p.data -= 0.001 * p.grad;
                    p.zero_grad();
                }
            } else {
                const accuracy: f64 = (@as(f64, @floatFromInt(correct)) / @as(f64, @floatFromInt(i + 1))) * 100.0;
                std.debug.print("Image: {d} Predicted = {d}, Label = {d}, Accuracy = {d}%\n", .{ i, predicted, true_label, accuracy });
            }
        }
    }
    if (save_checkpoint) |save_chk| {
        try mlp.save(save_chk);
        std.debug.print("Saving weights to '{s}'\n", .{save_chk});
    }
}
