var chai = require('chai');
var _    = require('underscore');
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

    it('should not compute a neuron\'s output more times than necessary', function () {
        var network = new Network(2, 2, 2, 1);

        var callsTo_compute_noCache = 0;

        _.each(network.layers, function (layer) {
            _.each(layer, function (neuron) {
                var _compute_noCache = neuron._compute_noCache.bind(neuron);
                neuron._compute_noCache = function (pass) {

                    ++callsTo_compute_noCache;

                    var result = _compute_noCache(pass);

                    this._compute_noCache = function shouldNotBeCalledAgain () {
                        throw new Error('_compute_noCache should not have been called twice in the same pass for any neuron');
                    };

                    return result;
                };
            });
        });

        network.compute([1,1]);
        callsTo_compute_noCache.should.equal(2+2+2+1);
    });
});
