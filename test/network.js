var chai = require('chai');
chai.should();

/* global describe, it */

var Network = require('../lib/network');

describe('The Network constructor', function () {
    it('with no options, should build a network with as many layers as arguments', function () {
        var network = new Network(1, 2, 3);
        network.layers.length.should.equal(3);
        network.layers[0].length.should.equal(1);
        network.layers[1].length.should.equal(2);
        network.layers[2].length.should.equal(3);
    });

    it('with options, should still build a network with as many layers as arguments after the options hash', function () {
        var network = new Network({some: 'options'}, 1, 2, 3);
        network.layers.length.should.equal(3);
        network.layers[0].length.should.equal(1);
        network.layers[1].length.should.equal(2);
        network.layers[2].length.should.equal(3);
    });
});

describe('A Neural Network', function () {
    it('should compute its output as a vector', function () {
        var network = new Network({
            activationFunction: function identity (x) {
                return x;
            },
            initialWeightingFunction: function sameWeight () {
                return 2;
            }
        }, 2, 1);

        network.compute([1, 2]).should.deep.equal([6]);
    });

    function makeSimpleNetwork () {
        function identity (x) {
            return x;
        }

        identity.derivative = function unit () {
            return 1;
        };

        return new Network({
            activationFunction: identity,
            initialWeightingFunction: function sameWeight () {
                return 2;
            }
        }, 2, 2, 1);
    }

    it('should compute the derivative of the error function with respect to each weight', function () {

        var network = makeSimpleNetwork();

        network.compute([1, 2]).should.deep.equal([24]);

        network.backPropagate([0], 0.01);

        network.layers[2][0].inputs[0].connection.deltaError.should.equal(144);
        network.layers[2][0].inputs[1].connection.deltaError.should.equal(144);
        network.layers[1][0].inputs[0].connection.deltaError.should.equal(48);
        network.layers[1][0].inputs[1].connection.deltaError.should.equal(96);
        network.layers[1][1].inputs[0].connection.deltaError.should.equal(48);
        network.layers[1][1].inputs[1].connection.deltaError.should.equal(96);

        network.compute([1, 2])[0].should.be.below(24);
    });


    it('should refuse to backpropagate if compute was not called', function () {
        var network = makeSimpleNetwork();
        chai.expect(function () {
            network.backPropagate([0], 0.01);
        }).to.throw();
    });

    it('should refuse to backpropagate twice if compute was not called', function () {
        var network = makeSimpleNetwork();

        network.compute([1, 2]);

        network.backPropagate([0], 0.01);

        chai.expect(function () {
            network.backPropagate([0], 0.01);
        }).to.throw();
    });
});
