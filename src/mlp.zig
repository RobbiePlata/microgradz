const std = @import("std");
const Value = @import("graph.zig").Value;
const Graph = @import("graph.zig").Graph;
const Layer = @import("neuron.zig").Layer;
const test_utils = @import("test-utils.zig");
const NonLinear = @import("neuron.zig").NonLinear;

pub const LayerSpec = struct {
    sizes: []usize,
    non_linear: []NonLinear,
};

pub const MLP = struct {
    layers: []*Layer,
    graph: *Graph,

    pub fn init(
        graph: *Graph,
        layer_specs: LayerSpec,
        seed: ?u64,
    ) !*MLP {
        const layer_sizes = layer_specs.sizes;
        const nonlinears = layer_specs.non_linear;

        if (layer_sizes.len < 2) return error.InvalidLayerSizes;
        if (nonlinears.len != layer_sizes.len - 1) return error.InvalidNonlinearsLength;

        const allocator = graph.allocator();
        const mlp = try allocator.create(MLP);
        const layers = try allocator.alloc(*Layer, layer_sizes.len - 1);
        for (0..layer_sizes.len - 1) |i| {
            layers[i] = try Layer.init(graph, layer_sizes[i], layer_sizes[i + 1], seed, nonlinears[i]);
        }

        mlp.* = .{
            .layers = layers,
            .graph = graph,
        };
        return mlp;
    }

    pub fn forward(self: *MLP, inputs: []*Value) ![]*Value {
        var current = inputs;
        for (self.layers) |layer| {
            current = try layer.forward(current);
        }
        return current;
    }

    pub fn parameters(self: *MLP) []*Value {
        const allocator = self.graph.allocator();
        var list = std.ArrayList(*Value).init(allocator);
        for (self.layers) |l| {
            const ps = l.parameters();
            for (ps) |p| list.append(p) catch {};
        }
        return list.toOwnedSlice() catch &[_]*Value{};
    }
};

test "MLP with arena graph" {
    std.debug.print("Testing MLP with arena..\n", .{});
    var g = Graph.init(std.heap.page_allocator);
    defer g.deinit();

    const layer_sizes = [_]usize{ 3, 4, 4, 1 };
    const mlp = try MLP.init(&g, &layer_sizes, 1337);

    const inputs = try std.heap.page_allocator.alloc(*Value, layer_sizes[0]);
    inputs[0] = g.value(2.0);
    inputs[1] = g.value(3.0);
    inputs[2] = g.value(-1.0);

    const outputs = try mlp.forward(inputs);

    for (outputs) |out| {
        _ = out.backward();
    }

    const learning_rate = 0.01;
    const params = mlp.parameters();
    for (params) |p| {
        p.data -= learning_rate * p.grad;
        p.zero_grad();
    }

    try test_utils.expect_weights_equal(g.allocator(), params, "test.weights.txt");

    g.clear();
}
