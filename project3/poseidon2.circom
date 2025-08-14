pragma circom 2.0.0;
include "circomlib/poseidon2.circom";

template Poseidon2Hash() {
    signal input preimage[2];  // 隐私输入（哈希原像，2个元素）
    signal input hash;         // 公共输入（Poseidon2哈希值）

    // Poseidon2哈希参数 (256, 3, 5)
    component poseidon = Poseidon2(2);

    for (var i = 0; i < 2; i++) {
        poseidon.inputs[i] <== preimage[i];
    }

    poseidon.out === hash;
}

component main = Poseidon2Hash();


