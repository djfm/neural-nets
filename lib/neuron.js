var _                   = require('underscore');
var activationFunctions = require('./activation-functions');

function Neuron (options) {

    var inputNeuron = false;

    options = _.defaults(options || {}, {
        activationFunction: activationFunctions.phi
    });

    this.inputs             = [];
    this.outputs            = [];
    this.value              = null;
    this.lastComputedValue  = null;
    this.lastComputedInput  = null;

    this.activationFunction = options.activationFunction;

    this.setValue = function setValue (value) {
        inputNeuron = true;
        this.value = value;
        return this;
    };

    this.compute = function compute () {
        if (this.isInputNeuron()) {
            this.lastComputedValue = this.value;
        } else {
            this.lastComputedInput = _.reduce(this.inputs, function (sum, input) {
                return sum + input.connection.weight * input.neuron.compute();
            }, 0);

            this.lastComputedValue = this.activationFunction(
                this.lastComputedInput
            );
        }

        return this.lastComputedValue;
    };

    this.backPropagate = function backPropagate (error, learningRate, momentum) {
        // this.lastComputedValue is passed to the derivative to allow phi.derivative to optimize
        var derivative = this.activationFunction.derivative(this.lastComputedInput, this.lastComputedValue);

        if (this.outputs.length === 0) {
            this.deltaError = error * derivative;
        } else {
            this.deltaError = _.reduce(this.outputs, function (sum, output) {
                return sum + output.neuron.deltaError * output.connection.weight;
            }, 0) * derivative;
        }

        _.each(this.inputs, function (input) {
            input.connection.deltaError = this.deltaError * input.neuron.lastComputedValue;
            input.neuron.backPropagate(undefined, learningRate, momentum);

            if (undefined !== learningRate) {
                var delta = learningRate * input.connection.deltaError;

                if (undefined !== momentum && undefined !== input.connection.lastDeltaWeight) {
                    delta += momentum * input.connection.lastDeltaWeight;
                }

                input.connection.weight -= delta;
                input.connection.lastDeltaWeight = delta;
            }

        }, this);

        return this;
    };

    this.isInputNeuron = function isInputNeuron () {
        return inputNeuron;
    };
}

Neuron.connect = function connectNeurons (a, b, weight) {

    weight = +(weight || 0);

    var connection = {
        weight: weight,
        deltaError: 0,
    };

    b.inputs.push({
        neuron: a,
        connection: connection
    });

    a.outputs.push({
        neuron: b,
        connection: connection
    });

    return Neuron;
};

module.exports = Neuron;
