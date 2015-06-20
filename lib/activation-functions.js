function phi (x) {
    return 1 / (1 + Math.exp(-x));
}

phi.derivative = function phiPrime (x, phiOfX) {
    return phiOfX * (1 - phiOfX);
};

exports.phi = phi;
