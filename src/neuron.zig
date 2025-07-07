const std = @import("std");
const value_mod = @import("value.zig");
const Value = value_mod.Value;
const Graph = value_mod.Graph;

pub const Neuron = struct {
    weights: []*Value,
    bias: *Value,
    graph: *Graph,

    pub fn init(g: *Graph, n_inputs: usize, seed: ?u64) !*Neuron {
        const allocator = g.allocator();
        const neuron = try allocator.create(Neuron);
        var weights = try allocator.alloc(*Value, n_inputs);

        var prng = if (seed) |s| std.Random.DefaultPrng.init(s) else std.Random.DefaultPrng.init(@intCast(std.time.milliTimestamp()));
        const rand = prng.random();

        for (0..n_inputs) |i| {
            weights[i] = g.value(rand.float(f64) * 2 - 1);
        }

        const b = g.value(0.0);

        neuron.* = .{
            .weights = weights,
            .bias = b,
            .graph = g,
        };
        return neuron;
    }

    pub fn forward(self: *Neuron, inputs: []*Value) !*Value {
        if (inputs.len != self.weights.len) return error.InvalidInputSize;

        var sum = self.bias;
        for (0..inputs.len) |i| {
            const wx = self.weights[i].mul(inputs[i]);
            sum = sum.add(wx);
        }
        return sum.relu();
    }

    pub fn parameters(self: *Neuron) []*Value {
        const allocator = self.graph.allocator();
        var list = std.ArrayList(*Value).init(allocator);
        for (self.weights) |w| list.append(w) catch {};
        list.append(self.bias) catch {};
        return list.toOwnedSlice() catch &[_]*Value{};
    }
};

pub const Layer = struct {
    neurons: []*Neuron,
    graph: *Graph,

    pub fn init(g: *Graph, n_inputs: usize, n_outputs: usize, seed: ?u64) !*Layer {
        const allocator = g.allocator();
        const layer = try allocator.create(Layer);
        const neurons = try allocator.alloc(*Neuron, n_outputs);
        for (0..n_outputs) |i| {
            neurons[i] = try Neuron.init(g, n_inputs, seed);
        }
        layer.* = .{
            .neurons = neurons,
            .graph = g,
        };
        return layer;
    }

    pub fn forward(self: *Layer, inputs: []*Value) ![]*Value {
        const allocator = self.graph.allocator();
        var outputs = try allocator.alloc(*Value, self.neurons.len);
        for (0..self.neurons.len) |i| {
            outputs[i] = try self.neurons[i].forward(inputs);
        }
        return outputs;
    }

    pub fn parameters(self: *Layer) []*Value {
        const allocator = self.graph.allocator();
        var list = std.ArrayList(*Value).init(allocator);
        for (self.neurons) |n| {
            const ps = n.parameters();
            for (ps) |p| list.append(p) catch {};
        }
        return list.toOwnedSlice() catch &[_]*Value{};
    }
};
