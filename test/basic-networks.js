require('chai').should();
var _ = require('underscore');

/* global describe, it */

var Network = require('../lib/network');

describe('A textbook example network from https://www4.rgu.ac.uk/files/chapter3%20-%20bp.pdf', function () {

    var network = new Network(2, 2, 1);

    network.neurons[0][0].outputs[0].connection.weight = 0.1;
    network.neurons[0][0].outputs[1].connection.weight = 0.4;
    network.neurons[0][1].outputs[0].connection.weight = 0.8;
    network.neurons[0][1].outputs[1].connection.weight = 0.6;
    network.neurons[1][0].outputs[0].connection.weight = 0.3;
    network.neurons[1][1].outputs[0].connection.weight = 0.9;

    var prec = 0.001;

    it('do a forward pass and check output', function () {
        network.compute([0.35, 0.9])[0].should.be.closeTo(0.69, prec);
    });

    it('train the network towards a target and check the weights were updated correctly', function () {
        network.train([0.35, 0.9], [0.5], 1);
        network.neurons[2][0].deltaError.should.be.closeTo(0.0406, prec);

        network.neurons[0][0].outputs[0].connection.weight.should.be.closeTo(0.09916    , prec);
        network.neurons[0][0].outputs[1].connection.weight.should.be.closeTo(0.3972     , prec);
        network.neurons[0][1].outputs[0].connection.weight.should.be.closeTo(0.7978     , prec);
        network.neurons[0][1].outputs[1].connection.weight.should.be.closeTo(0.5928     , prec);
        network.neurons[1][0].outputs[0].connection.weight.should.be.closeTo(0.272392   , prec);
        network.neurons[1][1].outputs[0].connection.weight.should.be.closeTo(0.87305    , prec);
    });

    it('now do another forward pass to see if we\'re closer to the target', function () {
        (network.compute([0.35, 0.9])[0] - 0.5).should.be.closeTo(0.18205, prec);
    });
});

describe('A XOR Neural Network', function () {
    it('should model the xor function', function () {
        var network = new Network(2, 2, 1).addBias();

        var dataSet = [
            [[0, 1], [1]],
            [[1, 0], [1]],
            [[0, 0], [0]],
            [[1, 1], [0]],
        ];


        function check () {
            _.each(dataSet, function (sample) {
                var res = network.compute(sample[0])[0];
                res.should.be.closeTo(sample[1][0], 0.1);
            });
        }

        var learningRate = 0.3;
        var momentum = 0.9;

        for (var i = 1; i <= 10000; ++i) {

            var error = 0;

            _.each(dataSet, function (sample) {
                var input   = sample[0];
                var output  = sample[1];
                error += network.train(input, output, learningRate, momentum);
            });

            error = Math.sqrt(error) / dataSet.length;

            if (error < 0.04) {
                break;
            }
        }

        check();
    });
});
