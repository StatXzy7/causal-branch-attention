# causal-branch-attention

Linux CPU 上的三分支因果注意力引擎（C++17，仅依赖标准库，第三方依赖 0 项）：

- **压缩分支 (compression)**：把键/值按连续块取均值。块按序列起点对齐，
  长度为 `block_len = b`；查询位置 `t` 处末尾不足一整块的可见部分单独成块，
  且只按可见行数取均值（不按整块长度归一化）。对块均值键做稳定的缩放点积
  softmax（缩放因子 `1/sqrt(d)`），输出为块均值值的加权和。
- **选择分支 (selection)**：以压缩分支的**归一化**相关性（softmax 概率）
  对可见块评分，取最高的 `select_count = n` 个块；分数相同按原始块索引升序
  排列，返回的索引也按升序给出。`n` 超过可见块数时截断；`n = 0` 或没有候选
  块时该分支输出为零、门为 0。在被选块的**原始**（非均值）键/值上再做一次
  稳定缩放点积注意力。
- **窗口分支 (window)**：只看最近 `window = w` 个可见位置
  `[max(0,t-w+1), t]`，同样做稳定缩放点积注意力。

三个分支各自独立做 softmax 归一化，各自配一个独立 sigmoid 门：

```
g_c = sigmoid(max 压缩分支缩放logit)
g_s = sigmoid(max 选择分支缩放logit)   (无候选时 g_s = 0)
g_w = sigmoid(max 窗口分支缩放logit)
out = g_c * O_c + g_s * O_s + g_w * O_w
```

三个门只相加、**不再归一化**。softmax 与 sigmoid 均为数值稳定实现
（减最大值 / 按符号分段）。

## 因果性

查询位置 `t` 只读取位置 `[0, t]`。逐条追加时，引擎只保存已追加的键/值；
在 `t1` 得到的输出与之后追加的未来键值完全无关。

## 两种调用方式，结果一致

- 整段：`causal3::forward(params, q, k, v, T, dim, &out)`；
- 逐条：`Engine::create(params)` 后反复 `append(q, k, v, dim)`，用 `last()`
  取当前位置结果。整段接口内部走同一条 append 路径，两种模式逐位一致
  （浮点逐字节相同）。

朴素参考实现见 `src/reference.cpp`（独立循环、double 累加），供自动化测试
交叉核对。

## 输入校验

- 拒绝非有限数（NaN/Inf，覆盖 q/k/v 全部元素）；
- 拒绝形状冲突（`dim` 与配置不符、空指针、`T <= 0`）；
- 拒绝非正 `block_len` / `window`，拒绝负 `select_count`；
- `append` 在任何状态修改前完成全部校验，失败时已有状态与上一条输出不变。

## 构建与测试

```bash
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
```

测试覆盖：选块排序/平分决胜/截断/无候选、末尾不足整块的均值、窗口可见范围、
三路门控数值（手算）、整段 vs 追加逐位一致、引擎 vs 朴素参考、因果性（未来
大值键值不影响过去输出）、非法输入以及追加失败后状态不变。

**零用例即失败**：CTest 用例在 configure 阶段从 `REGISTER_TEST` 声明解析，
解析不到任何用例时 configure 直接报错；另外注册了两个运行时守卫
（`__test_case_list_guard`、`__ctest_nonempty_guard`），二进制广告的用例数
不足或 `ctest -N` 发现用例数不足都会让 CTest 失败。

## 目录

- `include/causal3/engine.h`：引擎接口、`Params`、`StepResult`、
  `SequenceResult`；
- `include/causal3/reference.h`、`src/reference.cpp`：朴素参考；
- `src/engine.cpp`：整段与逐条实现；
- `tests/test_engine.cpp`：自包含测试运行器（`--list` 列出用例）；
- `cmake/`：CTest 非空守卫脚本。
