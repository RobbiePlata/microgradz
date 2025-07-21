const std = @import("std");
const Value = @import("graph.zig").Value;
const Graph = @import("graph.zig").Graph;
const Layer = @import("neuron.zig").Layer;
const test_utils = @import("test-utils.zig");
const NonLinear = @import("neuron.zig").NonLinear;

const LoadError = error{InvalidWeightsLength};

pub const LayerSpec = struct {
    shape: []const usize,
    nonlin: []const NonLinear,
};

pub const MLP = struct {
    layers: []*Layer,
    graph: *Graph,

    pub fn init(
        graph: *Graph,
        layer_specs: LayerSpec,
        seed: ?u64,
    ) !*MLP {
        if (layer_specs.shape.len < 2) return error.InvalidLayerShape;
        if (layer_specs.nonlin.len != layer_specs.shape.len - 1) return error.InvalidNonlinearsLength;

        const allocator = graph.allocator();
        const mlp = try allocator.create(MLP);
        const layers = try allocator.alloc(*Layer, layer_specs.shape.len - 1);
        for (0..layer_specs.shape.len - 1) |i| {
            layers[i] = try Layer.init(
                graph,
                layer_specs.shape[i],
                layer_specs.shape[i + 1],
                layer_specs.nonlin[i],
                seed,
            );
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

    pub fn save(self: *MLP, path: []const u8) !void {
        const file = try std.fs.cwd().createFile(path, .{ .truncate = true });
        defer file.close();

        var writer = file.writer();
        const ps = self.parameters();
        for (ps) |param| {
            try writer.print("{d}\n", .{param.data});
        }
    }

    pub fn load(
        self: *MLP,
        allocator: std.mem.Allocator,
        path: []const u8,
    ) !void {
        const cwd = std.fs.cwd();
        var file = try cwd.openFile(path, .{ .mode = std.fs.File.OpenMode.read_only });
        defer file.close();

        var reader = file.reader();
        const params = self.parameters();
        const paramCount = params.len;

        var idx: usize = 0;
        while (true) {
            const maybe_line = try reader.readUntilDelimiterOrEofAlloc(allocator, '\n', 1024 * 1024);
            if (maybe_line) |line| {
                defer allocator.free(line);
                if (line.len == 0) continue;
                if (idx >= paramCount) {
                    return LoadError.InvalidWeightsLength;
                }
                params[idx].data = try std.fmt.parseFloat(f64, line);
                idx += 1;
            } else {
                break;
            }
        }

        if (idx != paramCount) {
            return LoadError.InvalidWeightsLength;
        }
    }
};

pub fn softmax(graph: *Graph, logits: []*Value) ![]*Value {
    if (logits.len == 0) @panic("Softmax requires at least one logit");

    const alloc = graph.allocator();
    var max_logit: *Value = logits[0];
    for (logits[1..]) |logit| {
        if (logit.data > max_logit.data) {
            max_logit = logit;
        }
    }

    var shifted_exps = try alloc.alloc(*Value, logits.len);
    var sum_exp: *Value = graph.value(0.0);

    for (logits, 0..) |logit, i| {
        const shifted = logit.sub(max_logit);

        const e = shifted.exp();
        shifted_exps[i] = e;
        sum_exp = sum_exp.add(e);
    }

    var probs = try alloc.alloc(*Value, logits.len);
    for (shifted_exps, 0..) |e, i| {
        probs[i] = try e.div(sum_exp);
    }

    return probs;
}

test "MLP with arena graph" {
    std.debug.print("Testing MLP with arena..\n", .{});
    var g = Graph.init(std.heap.page_allocator);
    defer g.deinit();

    const layer_sizes = [_]usize{ 3, 4, 4, 1 };
    const nonlinears = [_]NonLinear{ NonLinear.none, NonLinear.none, NonLinear.none };

    const layer_spec = LayerSpec{
        .shape = layer_sizes[0..],
        .nonlin = nonlinears[0..],
    };
    const mlp = try MLP.init(&g, layer_spec, 1337);

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
