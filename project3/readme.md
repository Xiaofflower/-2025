
### **1. 电路功能**

* 实现 **Poseidon2** 哈希函数，参数为 `(n, t, d) = (256, 3, 5)` 或 `(256, 2, 5)`，对应 256 位域、t 元状态和 5 次幂 S-box。
* 电路接受：

  * **隐私输入**：原像（preimage）
  * **公开输入**：Poseidon2 哈希值

### **2. 主要模块结构**

1. **Poseidon2Round**

   * 实现单轮 Poseidon2 的 MDS 混合、S-box 变换。
   * 使用 `pow` 模运算实现 S-box $x^5$。
   * 常量参数（如 MDS 矩阵、round constants）直接硬编码。

2. **Poseidon2Circuit**

   * 将多轮 `Poseidon2Round` 串联，实现完整哈希计算。
   * 输入为 `preimage`（隐私输入），输出为哈希值。
   * 与 `publicHash` 比较，作为 Groth16 验证条件。

3. **主模板** (`template main`)

   * 声明公开输入 `publicHash` 与隐私输入 `preimage`。
   * 调用 `Poseidon2Circuit` 完成哈希计算。
   * 生成约束确保 `hashResult == publicHash`。


### **3. 零知识证明流程**

代码可以与 `snarkjs` 配合执行以下流程：

1. 编译 Circom 电路 (`circom poseidon2.circom`)
2. 生成 Groth16 的 proving key & verification key
3. 给定原像和哈希，生成证明
4. 验证证明（外部可用 verification key 验证，不泄露原像）


### **4. 优化特点**

* 使用 Circom 数组操作与常量预计算减少约束。
* 单 block 输入，避免多 block 复杂度。
* 在约束层面只保留必要的哈希计算与等式验证，减少电路规模。

