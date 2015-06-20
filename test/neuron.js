require('chai').should();

/* global describe, it */

var Neuron = require('../lib/neuron');

describe('An input neuron', function () {
    it('has a fixed value', function () {
        var neuron = new Neuron();
        neuron.setValue(3.14).compute().should.equal(3.14);
    });
});

describe('A hidden neuron', function () {
    it('gets its value from a computation based on its input neurons', function () {
        var inputs = [new Neuron().setValue(1), new Neuron().setValue(2)];

        var hidden = new Neuron({
            activationFunction: function square (x) {
                return Math.pow(x, 2);
            }
        });

        Neuron
            .connect(inputs[0], hidden, 0.5)
            .connect(inputs[1], hidden, 0.5)
        ;

        hidden.compute().should.equal(Math.pow(1 *  0.5 + 2 * 0.5, 2));
    });
});
