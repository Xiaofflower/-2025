import subprocess
import os
import json
from poseidon_hash import poseidon_hash  # 我们会写一个纯 Python Poseidon2 实现

def run(cmd):
    print(f"\n[+] 运行: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        raise RuntimeError(f"命令执行失败: {cmd}")

if __name__ == "__main__":
  
    preimage = [123456789, 987654321]
    print(f"[+] 原像: {preimage}")

  
    hash_val = poseidon_hash(preimage)
    print(f"[+] 计算得到哈希值: {hash_val}")

    # 写入 input.json
    input_data = {
        "preimage": [str(x) for x in preimage],
        "hash": str(hash_val)
    }
    with open("input.json", "w") as f:
        json.dump(input_data, f)
    print("[+] 已写入 input.json")

 
    if not os.path.exists("powersOfTau28_hez_final_10.ptau"):
        run("wget https://hermez.s3-eu-west-1.amazonaws.com/powersOfTau28_hez_final_10.ptau")

  
    run("circom poseidon2.circom --r1cs --wasm --sym")

  
    run("snarkjs groth16 setup poseidon2.r1cs powersOfTau28_hez_final_10.ptau poseidon2_0000.zkey")
    run('snarkjs zkey contribute poseidon2_0000.zkey poseidon2_final.zkey --name="First contribution" -v -e="random text"')

  
    run("snarkjs zkey export verificationkey poseidon2_final.zkey verification_key.json")

  
    run("node poseidon2_js/generate_witness.js poseidon2_js/poseidon2.wasm input.json witness.wtns")
    run("snarkjs groth16 prove poseidon2_final.zkey witness.wtns proof.json public.json")

 
    run("snarkjs groth16 verify verification_key.json public.json proof.json")

    print("\n 全流程完成，证明验证成功！")
