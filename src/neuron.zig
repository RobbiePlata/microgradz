const std = @import("std");
const Value = @import("value.zig").Value;
const Op = @import("value.zig").Op;

pub const NeuronOptions = struct {};

pub const Neuron = struct {
    weights: []*Value,
    bias: *Value,
    allocator: std.mem.Allocator,

    pub fn init(opts: struct {
        allocator: std.mem.Allocator,
        n_inputs: usize,
        seed: ?u64,
    }) !*Neuron {
        const allocator = opts.allocator;
        const n_inputs = opts.n_inputs;
        const seed = opts.seed;

        const neuron = try allocator.create(Neuron);
        var weights = try allocator.alloc(*Value, n_inputs);

        var prng = if (seed) |s| std.Random.DefaultPrng.init(s) else std.Random.DefaultPrng.init(@intCast(std.time.milliTimestamp()));
        const rand = prng.random();

        for (0..n_inputs) |i| {
            const w = Value.init(allocator, (rand.float(f64) * 2 - 1));
            weights[i] = w;
        }
        const b = Value.init(allocator, 0.0);
        neuron.* = .{
            .weights = weights,
            .bias = b,
            .allocator = allocator,
        };
        return neuron;
    }

    pub fn deinit(self: *Neuron) void {
        for (self.weights) |w| w.deinit();
        self.allocator.free(self.weights);
        self.bias.deinit();
        self.allocator.destroy(self);
    }

    pub fn forward(self: *Neuron, inputs: []*Value) !*Value {
        if (inputs.len != self.weights.len) {
            return error.InvalidInputSize;
        }

        var sum = self.bias;
        for (0..inputs.len) |i| {
            const w = self.weights[i];
            const x = inputs[i];
            const wx = w.mul(x);
            defer wx.deinit();
            const new_sum = sum.add(wx);
            if (sum != self.bias) sum.deinit();
            sum = new_sum;
        }

        const out = sum.relu();
        if (sum != self.bias) sum.deinit();
        return out;
    }

    pub fn parameters(self: *Neuron) []*Value {
        var params = std.ArrayList(*Value).init(self.allocator);
        defer params.deinit();
        for (self.weights) |w| params.append(w) catch {};
        params.append(self.bias) catch {};
        return params.toOwnedSlice() catch &[_]*Value{};
    }
};

pub const Layer = struct {
    neurons: []*Neuron,
    allocator: std.mem.Allocator,

    pub fn init(opts: struct {
        allocator: std.mem.Allocator,
        n_inputs: usize,
        n_outputs: usize,
        seed: ?u64,
    }) !*Layer {
        const allocator = opts.allocator;
        const n_inputs = opts.n_inputs;
        const n_outputs = opts.n_outputs;
        const seed = opts.seed;

        const layer = try allocator.create(Layer);
        var neurons = try allocator.alloc(*Neuron, n_outputs);
        for (0..n_outputs) |i| {
            neurons[i] = try Neuron.init(.{ .allocator = allocator, .n_inputs = n_inputs, .seed = seed });
        }
        layer.* = .{
            .neurons = neurons,
            .allocator = allocator,
        };
        return layer;
    }

    pub fn deinit(self: *Layer) void {
        for (self.neurons) |n| n.deinit();
        self.allocator.free(self.neurons);
        self.allocator.destroy(self);
    }

    pub fn forward(self: *Layer, inputs: []*Value) ![]*Value {
        var outputs = self.allocator.alloc(*Value, self.neurons.len) catch {
            std.debug.panic("Failed to allocate memory for Layer outputs: Out of memory.", .{});
        };
        for (0..self.neurons.len) |i| {
            outputs[i] = try self.neurons[i].forward(inputs);
        }
        return outputs;
    }

    pub fn parameters(self: *Layer) []*Value {
        var params = std.ArrayList(*Value).init(self.allocator);
        defer params.deinit();
        for (self.neurons) |n| {
            const neuron_params = n.parameters();
            for (neuron_params) |p| params.append(p) catch {};
            self.allocator.free(neuron_params);
        }
        return params.toOwnedSlice() catch &[_]*Value{};
    }
};
