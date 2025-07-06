const std = @import("std");
const Value = @import("value.zig").Value;
const Layer = @import("neuron.zig").Layer;

pub const MLP = struct {
    layers: []*Layer,
    allocator: std.mem.Allocator,

    pub fn init(opts: struct { allocator: std.mem.Allocator, layer_sizes: []const usize, seed: ?u64 }) !*MLP {
        const allocator = opts.allocator;
        const layer_sizes = opts.layer_sizes;
        const seed = opts.seed;

        if (layer_sizes.len < 2) return error.InvalidLayerSizes;
        const mlp = try allocator.create(MLP);
        var layers = try allocator.alloc(*Layer, layer_sizes.len - 1);
        for (0..layer_sizes.len - 1) |i| {
            layers[i] = try Layer.init(.{ .allocator = allocator, .n_inputs = layer_sizes[i], .n_outputs = layer_sizes[i + 1], .seed = seed });
        }
        mlp.* = .{
            .layers = layers,
            .allocator = allocator,
        };
        return mlp;
    }

    pub fn deinit(self: *MLP) void {
        for (self.layers) |l| l.deinit();
        self.allocator.free(self.layers);
        self.allocator.destroy(self);
    }

    pub fn forward(self: *MLP, inputs: []*Value) ![]*Value {
        var current = inputs;
        var is_input = true;

        for (self.layers) |layer| {
            const next = try layer.forward(current);
            if (!is_input) {
                for (current) |v| v.deinit();
                self.allocator.free(current);
            }
            current = next;
            is_input = false;
        }
        return current;
    }

    pub fn parameters(self: *MLP) []*Value {
        var params = std.ArrayList(*Value).init(self.allocator);
        defer params.deinit();
        for (self.layers) |l| {
            const layer_params = l.parameters();
            for (layer_params) |p| params.append(p) catch {};
            self.allocator.free(layer_params);
        }
        return params.toOwnedSlice() catch &[_]*Value{};
    }
};

const test_utils = @import("./test-utils.zig");

test "MLP" {
    std.debug.print("Testing MLP..\n", .{});
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const layer_sizes = [_]usize{ 3, 4, 4, 1 };
    const mlp = try MLP.init(.{ .allocator = allocator, .layer_sizes = &layer_sizes, .seed = 1337 });
    defer mlp.deinit();

    var inputs = try allocator.alloc(*Value, 3);
    defer {
        for (inputs) |v| v.deinit();
        allocator.free(inputs);
    }
    inputs[0] = Value.init(allocator, 2.0);
    inputs[1] = Value.init(allocator, 3.0);
    inputs[2] = Value.init(allocator, -1.0);

    const outputs = try mlp.forward(inputs);
    defer {
        for (outputs) |v| v.deinit();
        allocator.free(outputs);
    }

    for (outputs) |out| {
        _ = out.backward();
    }

    const learning_rate: f64 = 0.01;
    const params = mlp.parameters();
    defer allocator.free(params);
    for (params) |p| {
        p.data -= learning_rate * p.grad;
        p.zero_grad();
    }

    // try test_utils.generate_test_weights(allocator, params, "test.weights.txt");
    try test_utils.expect_weights_equal(allocator, params, "test.weights.txt");
}
