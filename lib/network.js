var _ = require('underscore');

var Neuron = require('./neuron');

function Network (options /*, input neurons count, hidden layer 1 count, ..., output layer count */) {

    var layerCounts = Array.prototype.slice.call(arguments, typeof options === 'object' ? 1 : 0);
    options = options || {};
    var inputLayer, outputLayer = [];

    this.layers = this.neurons = [];

    var optionsForNeuron = _.pick(options, 'activationFunction');

    _.each(layerCounts, function (layerCount) {
        var layer = [];

        for (var i = 0; i < layerCount; ++i) {
            var neuron = new Neuron(optionsForNeuron);
            layer.push(neuron);
            _.each(outputLayer, function (inputNeuron) {

                var weight = options.initialWeightingFunction ?
                                options.initialWeightingFunction() :
                                Math.random()
                ;

                Neuron.connect(inputNeuron, neuron, weight);
            });
        }

        outputLayer = layer;

        if (!inputLayer) {
            inputLayer = layer;
        }

        this.layers.push(layer);
    }, this);

    var canBackPropagate = false;

    this.addBias = function addBias () {
        var bias = new Neuron(optionsForNeuron).setValue(1);
        for (var l = 1, len = this.layers.length; l < len; ++l) {
            _.each(this.layers[l], function (neuron) {
                Neuron.connect(bias, neuron);
            });
        }
        return this;
    };

    this.train = function train (inputVector, targetOutputVector, learningRate, momentum) {
        this.compute(inputVector);
        return this.backPropagate(targetOutputVector, learningRate, momentum);
    };

    this.batchTrain = function batchTrain (dataSet, learningRate, momentum) {
        var error = 0;
        var batchSize = dataSet.length;
        for (var i = 0; i < batchSize; ++i) {
            error += this.train(dataSet[i][0], dataSet[i][1], learningRate, momentum);
        }
        return Math.sqrt(error) / batchSize;
    };

    this.compute = function compute (inputVector) {
        canBackPropagate = true;

        for (var i = 0, len = inputVector.length; i < len; ++i) {
            inputLayer[i].setValue(inputVector[i]);
        }

        return _.map(outputLayer, function runComputeOnNeuron (neuron) {
            return neuron.compute();
        });
    };

    this.backPropagate = function backPropagate (targetOutputVector, learningRate) {

        if (!canBackPropagate) {
            throw new Error(
                'The `compute` method must be called on the network before backpropagation of errors is possible.'
            );
        } else {
            canBackPropagate = false;
        }

        /**
         * Compute error at output layer first.
         */

        var totalError = 0;

        for (var j = 0, layerLen = targetOutputVector.length; j < layerLen; ++j) {
            var outputNeuron = outputLayer[j];
            var error = outputNeuron.lastComputedValue - targetOutputVector[j];
            totalError += error * error;
            outputNeuron.backPropagate(error, learningRate);
        }

        return totalError;
    };
}

module.exports = Network;
