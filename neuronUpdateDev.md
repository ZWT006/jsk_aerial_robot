# Neuron Servo Task 更新开发说明

## 1. 文档目的

本文档记录将 `neuron_g4` 的 Dynamixel servo task 重构为类似 spinal direct-servo 驱动的开发需求、已经确认的设计决策、可直接复用的代码、必须适配的差异、已知风险和验收方法。

目标读者是后续继续实现该功能的开发者或 agent。开始修改前，应先阅读本文档以及下列核心文件：

- `aerial_robot_nerve/neuron/neuron_g4/Core/Src/main.c`
- `aerial_robot_nerve/neuron/neuronlib/Servo/servo.h`
- `aerial_robot_nerve/neuron/neuronlib/Servo/servo.cpp`
- `aerial_robot_nerve/neuron/neuronlib/Servo/Dynamixel/dynamixel_serial.h`
- `aerial_robot_nerve/neuron/neuronlib/Servo/Dynamixel/dynamixel_serial.cpp`
- `aerial_robot_nerve/spinal/mcu_project/lib/Jsk_Lib/servo/drivers/Dynamixel/dynamixel_serial.h`
- `aerial_robot_nerve/spinal/mcu_project/lib/Jsk_Lib/servo/drivers/Dynamixel/dynamixel_serial.cpp`

本文只针对 `neuron_g4` 的新控制路径。`neuron_f1`、`neuron_f4` 的旧行为不应在第一轮重构中改变。

## 2. 已确认需求

### 2.1 servoTask 调度

1. `servoTask` 必须自己维护稳定的 1 ms 调度 tick，不再以 CAN target 到达作为 Dynamixel control cycle 的直接时钟源。
2. 1 ms 是软件调度周期，不代表每 1 ms 完成一次完整 Dynamixel 读写。
3. Dynamixel 高频闭环周期暂时保持当前配置：
   - `CONTROL_CYCLE_DU = 5 ms`
   - 闭环频率为 200 Hz
4. 应使用绝对时间调度，例如 FreeRTOS `vTaskDelayUntil()`，避免 `update()` 执行时间累积到下一周期。

### 2.2 CAN target

1. CAN RX 只负责解析并更新最新 target：position、velocity、current、torque enable、valid flags、sequence 和接收时间。
2. 一个新的 target frame 还可以产生轻量的 phase-sync event，用于锁定 neuron 本地 5 ms control cycle 的相位。该 event 不等于执行一次 UART cycle。
3. CAN RX 不直接执行 Dynamixel UART 操作。
4. CAN RX 不再调用 `startTargetCycle()` 触发完整读写。
5. servoTask 在自己的 5 ms control cycle 中消费最新 target。
6. 如果多个 CAN target 在一个 control cycle 内到达，应采用 latest-wins，不得把过期 target 全部排入 instruction queue。

### 2.3 CAN service/config request

1. neuron 的配置请求同样来自 CAN，但这不妨碍复用 spinal 的 `INST_SERVICE_REQUEST` 设计。
2. 与 spinal ROS service 不同，CAN RX task 不应等待 semaphore 获得同步结果。
3. 推荐保留“最多一个 outstanding config request”的限制：
   - CAN RX 将请求内容复制到 service request slot。
   - 将 `INST_SERVICE_REQUEST` 放入 servo instruction queue。
   - servoTask 执行请求。
   - 完成后由 servoTask 调用 `sendConfigResponse()` 异步回复 CAN。
   - 如果 slot 忙或 queue 满，立即回复 BUSY/QUEUE_FULL。
4. spinal 的 `service_completion_`、`callback_waiting` 等同步等待字段不需要原样移植到 neuron。

### 2.4 Mode-aware goal write

每个 5 ms control cycle 应先读取最新 CVP，再根据 servo mode 和最新 CAN target 生成 goal write：

| Neuron mode | 读后计算/安全行为 | Dynamixel 写入 |
| --- | --- | --- |
| DP | 使用最新 position target | Goal Position |
| DEP | 使用最新 position target | Goal Position |
| DV | 使用最新 velocity；watchdog timeout 后写 0 | Goal Velocity |
| DC | 使用最新 current；watchdog timeout 后写 0 | Goal Current |
| MIT | 使用最新 CVP、position/velocity/current target 计算电流 | Goal Current |
| CbP | 使用 position 和 current magnitude/limit | 先 Goal Position，再 Goal Current |
| EM | 不发送 goal | 无 |

高频路径推荐和 spinal 一样只入队一个逻辑事件 `INST_SET_GOAL_STATE`。不同 mode 的寄存器选择应由 `cmdBulkWriteGoalStates()` 完成，而不是把每个 mode 拆成独立 UART event。

原因是一个 neuron 可能同时连接不同 mode 的 servo。Bulk Write 可以在一个 packet 中混合写不同 ID、不同地址；如果按 mode 拆 event，会增加 packet 数量并降低 5 ms 周期的余量。

### 2.5 CbP 写入顺序

neuron 当前开发分支已将 CbP 顺序修改为：

```text
packet 1: Goal Position
packet 2: Goal Current
```

位置：

```text
aerial_robot_nerve/neuron/neuronlib/Servo/Dynamixel/dynamixel_serial.cpp
DynamixelSerial::cmdBulkWriteGoalStates()
```

不要在从 spinal 移植 `cmdBulkWriteGoalStates()` 时覆盖掉这个顺序。当前 spinal 实现仍然是 Goal Current 在前、Goal Position 在后。

CbP 必须保持两个 packet，因为 Dynamixel Protocol 2.0 Bulk Write 中同一个 ID 不能重复出现，而且 Goal Current 与 Goal Position 中间存在 profile registers，不能安全地作为一段连续寄存器写入。

## 3. 核心结论：能否几乎复用 spinal

可以。Dynamixel driver 的大部分调度和状态机都应直接参考 spinal，而不是在 neuron 中重新设计另一套框架。

### 3.1 可以直接参考或移植的部分

- `ServoInstruction`
- `PendingTransaction`
- `RingBufferDx<ServoInstruction, 64>`
- `enqueueInstruction()`
- `enqueueInstructionBatch()`
- `popInstruction()`
- `schedulePeriodicCommands()`
- `processInstructionQueue()`
- `startPendingTransaction()`
- `completePendingTransaction()` 的总体状态机
- `activeServoMask()` 和 bus group mask
- 高频 control-cycle batch 的生成顺序
- temperature、moving、hardware error 的低频调度
- `INST_SET_GOAL_STATE` 和 mode-aware Bulk Write
- target watchdog
- UART 错误恢复和 deadline 比较方法
- 最后一组 CVP 完成后产生 fresh-state 标志的机制

### 3.2 不能原样复制的部分

| 项目 | Spinal | Neuron 需要的适配 |
| --- | --- | --- |
| Target 来源 | ROS callback/直接 servo API | CAN RX，latest-wins target buffer |
| Service 来源 | ROS service，调用者同步等待 | CAN service，异步回复，不阻塞 canRxTask |
| Feedback 出口 | 设置 ROS fresh flag，由 publish 流程消费 | 设置 CAN feedback fresh flag，唤醒 canTxTask |
| Servo 数据 | spinal direct servo 数据结构 | neuron 还包含 external encoder、pulley skip、CAN sequence |
| Task 周期要求 | 正常时约 1 ms，故障时可能 busy-wait | 目标是故障时也不因 UART RX 等待阻塞 5 ms |
| Flash config | service 内同步处理 | 需要明确配置期间是否允许破坏 1 ms deadline |

因此合理策略是：先移植 spinal 的 instruction/scheduler 框架，再针对 CAN service 和 feedback 出口做薄适配。

## 4. Spinal 的 instruction queue 行为

### 4.1 队列容量

spinal 使用：

```cpp
RingBufferDx<ServoInstruction, 64> instruction_buffer_;
```

ring buffer 为了区分 empty 和 full 会保留一个空槽，所以实际容量是 63 条 instruction，而不是 64 条。

### 4.2 批量入队

`enqueueInstructionBatch()` 的行为：

1. 校验所有 servo mask。
2. 获取 `queue_mutex_`。
3. 计算剩余槽位。
4. 如果空间不足，整批拒绝，不会只压入一部分。
5. 空间足够时一次压入整个 batch。

高频周期最大生成四个 instruction：

```text
RS485 CVP read
TTL CVP read
RS485 goal write
TTL goal write
```

非 mixed bus 时通常只有两个：

```text
CVP read
goal write
```

### 4.3 Queue full 时 spinal 如何处理

周期调度函数即使入队失败，也会推进下一次 deadline。其结果是：

- 当前过载周期被丢弃。
- 不会在后续周期补排过期 control event。
- 不会形成 catch-up burst。
- 当前代码没有专门记录 dropped/missed cycle counter。

低频 temperature/moving/error event 的入队失败同样会被丢弃，等待下一个周期。

service request 如果入队失败，会返回 `SERVO_RESULT_QUEUE_FULL`。

torque-off 不依赖普通 FIFO，而是使用 atomic pending mask 并在普通队列之前处理。这一安全行为应保留。

### 4.4 Neuron 的采用方案

建议 neuron 复用同样的 all-or-none batch 和 drop-current-cycle 策略，并补充以下计数器：

```cpp
uint32_t control_cycle_enqueue_fail_count_;
uint32_t periodic_event_drop_count_;
uint16_t instruction_queue_high_watermark_;
```

CAN target 本身不进入 instruction queue，因此 CAN burst 不会直接把队列塞满。队列只保存 UART transaction/service event。

## 5. 建议的 ServoInstruction 结构

第一轮尽量与 spinal 保持一致：

```cpp
struct ServoInstruction
{
  uint8_t command;
  ServoMask servo_mask;

  // 非零只用于一个 control cycle 中最后一组 CVP read。
  ServoMask feedback_cycle_mask;
};
```

暂时不要把 target 数值复制进每个 instruction。position/velocity/current 使用独立的 latest target snapshot；执行 `INST_SET_GOAL_STATE` 时读取该 control cycle 已冻结的 snapshot。

如果后续需要对 CAN sequence 和实际写入值做精确关联，可增加：

```cpp
uint8_t target_sequence;
uint32_t target_generation;
```

但不应在第一轮重构中扩大状态机复杂度。

## 6. 推荐的 1 ms update 顺序

`DynamixelSerial::update()` 每 1 ms 调用一次，顺序建议与 spinal 接近：

```text
1. 更新 target watchdog
2. 如果存在 pending UART transaction，尝试完成它
3. 处理紧急 torque-off pending mask
4. 应用一份完整的 latest CAN target snapshot
5. schedulePeriodicCommands()
6. processInstructionQueue()，最多启动一个新的 UART transaction
7. 返回本 tick 是否产生完整 fresh feedback
```

不能再使用 neuron 当前的：

```cpp
while (popInstruction(...)) {
  // drain whole queue
}
```

应改成 spinal 的语义：跳过无效 instruction 后，每次 `update()` 最多执行一个有效 UART 操作。

## 7. 200 Hz control-cycle batch

`schedulePeriodicCommands()` 应以绝对 deadline 每 5 ms 原子加入一组 ordered instructions。

非 mixed bus：

```text
INST_GET_PRESENT_CVP(active_mask, feedback_cycle_mask=active_mask)
INST_SET_GOAL_STATE(goal_mask)
```

TTL/RS485 mixed bus：

```text
INST_GET_PRESENT_CVP(rs485_feedback_mask, feedback_cycle_mask=0)
INST_GET_PRESENT_CVP(ttl_feedback_mask, feedback_cycle_mask=full_feedback_mask)
INST_SET_GOAL_STATE(rs485_goal_mask)
INST_SET_GOAL_STATE(ttl_goal_mask)
```

如果只有 RS485 feedback，RS485 read 自己就是 final feedback group；TTL-only 同理。

如果上一周期尚未处理完成，不要继续积累新的周期。可以用 queue 中是否已有 high-rate batch，或显式 `control_cycle_pending_` 标志判断，并记录 missed cycle。

## 8. Mode-aware Goal State 是否可以直接参考 spinal

可以，并且这是推荐方案。

spinal 高频 scheduler 实际入队的是统一事件：

```cpp
INST_SET_GOAL_STATE
```

spinal 虽然定义了 `INST_SET_GOAL_POS`、`INST_SET_GOAL_VEL`、`INST_SET_GOAL_CUR`，但当前 200 Hz 高频路径没有按 mode 分别入这些 event；真正的 mode dispatch 在 `cmdBulkWriteGoalStates()` 中。

neuron 应保留同样结构：

1. instruction 表示一次 bus-level goal state transaction。
2. `cmdBulkWriteGoalStates(mask)` 遍历 mask 中 servo。
3. 根据 `servo.mode_` 选择寄存器和数值。
4. 对同一个 bus 构建尽可能少的 Bulk Write packet。
5. CbP 额外发送第二个 packet，并保持 Position 在 Current 之前。

需要保留 neuron 特有逻辑：

- `getInternalGoalPosition()` 的 resolution ratio/internal offset 转换。
- external encoder 和 pulley skip 行为。
- CbP current 使用绝对值并受 `current_limit_` 限制。
- MIT 使用 neuron 的 gain scale 和最新 CVP。
- DV/DC/MIT watchdog timeout 后的零输出。
- 未配置 servo 和 EM servo 不发送 control goal。

## 9. Spinal 的 5 ms response timeout 行为

### 9.1 实际状态机

spinal 收到 read instruction 时：

1. 发送 Fast Sync Read packet。
2. 建立 `PendingTransaction`，deadline 为当前时间加 5 ms。
3. 当前 `update()` 返回。
4. 下一次 `update()` 调用 `completePendingTransaction()`。
5. `completePendingTransaction()` 调用 `readFastSyncStatusPacket()`。

这个设计将“发送请求”和“开始接收处理”分到了两个 servoTask tick。

### 9.2 仍然存在的问题

spinal 的 `readUntil()` 在 DMA buffer 没有新数据时使用 busy loop：

```cpp
while (true) {
  if (dma_has_data) return HAL_OK;
  if (deadline_expired) return HAL_TIMEOUT;
}
```

因此：

- 正常情况下，Dynamixel response 在下一次 1 ms tick 前已经进入 DMA，解析很快，问题不明显。
- servo 掉线、bus 错误或 response 不完整时，servoTask 可能 busy-wait 到剩余 deadline。
- 最坏情况下会占用接近 5 ms，spinal 的 `osDelay(1)` loop 并不保证故障时仍保持严格 1 ms。
- spinal servoTask 是 Realtime priority，因此 busy-wait 还可能增加其他同优先级/低优先级 task 的调度压力。

### 9.3 Neuron 应如何处理

有两个实施层级：

#### 方案 A：先完全复用 spinal

优点：

- 修改量小。
- 行为与 spinal 一致。
- 容易先验证功能正确性。

缺点：

- Dynamixel 故障时不能保证稳定 1 ms loop。

如果项目对“稳定 1 ms”的要求只针对正常运行，可以先采用该方案完成第一轮移植。

#### 方案 B：将 pending transaction 改成完全非阻塞

这是最终推荐方案。

每次 `update()`：

1. 读取 DMA write pointer。
2. 如果完整 packet 尚未到达，保持 pending 并立即返回。
3. 如果 packet 完整，解析并完成 transaction。
4. 如果 deadline 到达，记录 timeout、执行 `recoverRx()` 并结束 transaction。

Fast Sync Read 的期望 packet 长度可以由 `servo_mask` 和寄存器长度提前计算，因此可以先判断 DMA 中是否已经积累完整 packet。

普通单 servo service read 可以：

- 使用一个增量 parser state；或
- 先检查最小 header/length，再等待 DMA 中出现完整 length。

不能在 critical section 或 queue mutex 持有期间等待 UART response。

此外，`HAL_UART_Transmit(..., timeout=10)` 本身也是阻塞 API。正常 1 Mbps、当前 packet 长度下通常远小于 1 ms，但必须实测并记录 max execution time。如果写 packet 可能超过 1 ms，后续应改用 TX DMA。

## 10. Feedback 发送时序是否可以参考 spinal

可以，推荐直接参考 spinal 的 fresh-state 语义。

### 10.1 Spinal 的行为

1. control cycle 的最后一组 CVP read instruction 携带完整的 `feedback_cycle_mask`。
2. 每组 CVP 成功后累积 `feedback_received_mask_`。
3. 最后一组完成时，只有全部期望 servo 都成功更新才设置 fresh flag。
4. ROS publish 流程检查 fresh flag 和 feedback frequency。
5. 发布后清除 fresh flag。

注意：spinal 的 fresh flag 在 CVP read 完成时设置，不等待后续 goal write 完成。这个语义表示“有一组新的测量”，不是“整个 read/write cycle 已成功”。

### 10.2 Neuron 的对应实现

建议增加：

```cpp
ServoMask feedback_received_mask_;
bool feedback_fresh_;
```

最后一组 CVP 成功后：

```cpp
feedback_fresh_ =
  (feedback_received_mask_ & expected_mask) == expected_mask;
```

随后：

1. `Servo::update()` 返回/提取 `feedback_fresh_`。
2. `servoTaskCallback()` release `canTxSemHandle`。
3. `canTxTask` 调用 `servo_.sendData()`。
4. `sendData()` 继续使用现有 `servo_feedback_frequency_hz_` 做 downsample。
5. 每个 servo 继续用 `pvc_update_count_` 生成 `SERVO_STATE_PVC_FRESH`。

当前 `CANDeviceManager::userReceiveMessagesCallback()` 在收到 target 后立即 release `canTxSemHandle`。该逻辑应删除，否则 canTxTask 的优先级高于 servoTask，反馈通常会在新 CVP 完成前发送，得到上一周期数据。

如果需要区分：

- 新 feedback 已读取；
- goal write 已完成；

应使用两个不同 flag。不要改变 spinal fresh-state 的含义。

## 11. Target 数据并发

neuron 的 CAN RX task 是 Realtime priority，servoTask 当前是 Normal priority。CAN RX 可能在 servoTask 构建 Bulk Write packet 时更新 target 字段。

即使 Cortex-M4 对齐的 32-bit 写入通常是原子的，一帧中的 position、velocity、current 和 valid flags 仍可能来自不同 CAN generation。

推荐使用双缓冲或短 critical section：

```cpp
struct ServoTargetSnapshot
{
  int32_t position;
  int32_t velocity;
  int16_t current;
  uint8_t valid_flags;
  uint8_t sequence;
  uint32_t receive_tick;
};
```

CAN RX 更新 pending snapshot；servoTask 在 control cycle 开始前一次性复制到 active snapshot。critical section 只覆盖内存复制，绝不能覆盖 UART 读写。

Watchdog 应使用 CAN 实际接收时间，而不是 servoTask 应用 snapshot 的时间。

## 12. CAN Goal 与本地 Control Cycle 的相位同步

### 12.1 这个需求有意义

spinal 不是在同一个 tick 给所有 neuron 发送 goal。当前 `Spine::send()` 在 1 kHz 主循环中使用：

```cpp
const uint8_t servo_slot = now % CAN::CANFD_MAX_NEURON_NUM;
neuron_.at(servo_slot).can_servo_.sendData();
```

`CANFD_MAX_NEURON_NUM` 是 5，因此同一个 neuron 每 5 ms 收到一次 target，不同 neuron 的 target 约相差 1 ms。这是已有的 CAN 时隙安排。

当前 neuron 在 target 到达后设置 `target_cycle_pending_`，随后直接运行一整次 Dynamixel read/write。因此它虽然没有独立的 phase scheduler，但确实“顺便”保留了 spinal 的 1 ms 错峰。

如果重构后让所有 neuron 在上电后立即按各自的 `HAL_GetTick() % 5` 自由运行，不同板的 local tick 没有共同 epoch，上电次序将决定相位。多个 neuron 可能在同一时刻完成 CVP 并集中发送 CAN feedback。CAN 仲裁会避免数据冲突，但会增加队列、延迟和 jitter。

所以应保留“CAN 提供全局相位，neuron 本地维持周期”的混合设计。

### 12.2 当前实现的不足

当前只有：

```cpp
bool target_cycle_pending_;
bool target_cycle_complete_;
```

它们表示“来了一帧，执行一次”，没有分离以下概念：

- target data generation；
- CAN callback 记录的启动 tick；
- 本地 control-cycle deadline；
- scheduler 是否已经同步；
- CAN 断流后是否保持本地 loop。

target frame 的 `data[1]` 已经是每个 neuron 独立递增的 sequence，并且 feedback 会在 `data[2]` 回显 `last_target_sequence_`。该 sequence 可用于识别新帧、丢帧和主机重启。

### 12.3 推荐的简化实现

不修改 CAN protocol，不实现额外的锁相环。直接在 CAN callback 中使用 `HAL_GetTick()`，只保留以下状态：

```cpp
bool phase_start_pending_;
uint32_t phase_start_tick_;
bool control_scheduler_started_;
uint32_t next_control_tick_;
```

CAN target callback 在完整 frame 校验和 latest target 更新后：

```cpp
if (!control_scheduler_started_) {
  phase_start_tick_ = HAL_GetTick();
  phase_start_pending_ = true;
}
```

这几个共享字段的读写应包在很短的 critical section 内。在 scheduler 启动前，后到的 target 可以覆盖 `phase_start_tick_`，使 servoTask 使用最近一帧的启动 tick。随后 callback 立即返回，不访问 Dynamixel UART，不直接压入 read/write instruction。

servoTask 从上电开始仍以 1 ms 固定周期运行。high-rate scheduler 初始不启动，在空闲边界消费第一个 `phase_start_pending_`：

```cpp
if (!control_scheduler_started_ && phase_start_pending_) {
  next_control_tick_ = phase_start_tick_ + 1U;
  control_scheduler_started_ = true;
  phase_start_pending_ = false;
}
```

固定增加 1 ms 是为了让 CAN callback 先提交完整 target snapshot。所有 neuron 使用同一延迟，因此 spinal 的 1 ms goal slot 错峰得以保留。

后续只使用本地绝对 deadline：

```cpp
if (timeReached(now, next_control_tick_)) {
  freezeLatestTargetSnapshot();
  enqueueControlCycleBatch();
  next_control_tick_ += CONTROL_CYCLE_DU;  // 5 ms
}
```

必须使用 `next += 5`，不能使用 `next = now + 5`。当前周期未完成时，丢弃本次新周期并继续推进 deadline，不做 catch-up burst。

后续每帧 CAN target 仅更新 latest target 和 watchdog 时间，不重置 `next_control_tick_`。target sequence 继续用于诊断丢帧和在 feedback 中回显，不参与复杂的相位计算。

单帧或短时 CAN 丢包时，本地 5 ms loop 继续运行；超过 watchdog 后 DV/DC/MIT 写安全零输出。如果 CAN 长时间中断后希望重新与 spinal 错峰，可在 bus 空闲时同时清除 `control_scheduler_started_` 和 `phase_start_pending_`，由下一帧 target 重新执行上述一次性启动。

`WAIT_PHASE_SYNC` 期间仍可以处理 config、torque-off 和初始化，但不启动周期性 goal write。

### 12.4 Feedback 时序与局限

feedback 仍应在完整 CVP group 成功后置 fresh，然后唤醒 canTxTask。由于每个 neuron 的 CVP read 起点已按 CAN goal slot 错峰，feedback 通常也会保持类似错峰。

但是不同 neuron 的 servo 数量、TTL/RS485 group 数和 UART timeout 不同，所以“起始相位错峰”不等于“feedback 永远不重叠”。CAN 会依靠仲裁正确处理偶发重叠。如果实测仍有明显的 TX queue 积压，应在 spinal 分配明确的 feedback window/token，而不是在各 neuron 中使用更多本地 delay 猜测时隙。

### 12.5 建议增加的诊断项

```cpp
uint32_t phase_start_count;
uint32_t target_sequence_drop_count;
uint32_t control_holdover_cycle_count;
uint32_t feedback_tx_wakeup_delay_max_us;
```

## 13. Service/config 对 1 ms 周期的影响

neuron 的部分 `configure()` 路径会执行 Flash erase/write，其他路径还包含 Dynamixel write 后 readback 验证。这些操作可能明显超过 1 ms。

需要明确运行保证：

- 推荐定义为：正常 control mode 下 servoTask 维持 1 ms；配置/Flash maintenance 期间允许临时超期。
- config 只允许 torque disabled 时执行现有安全检查。
- service event 的优先级低于 torque-off，但高低于/高于 control cycle 需要明确。
- 推荐在完整 control-cycle batch 之间执行一个 service request，不能插入到 CVP read 和对应 goal write 中间。
- Flash erase/write 若必须做到完全不影响 control loop，应进一步拆到独立 maintenance task；这不属于第一轮 spinal parity 移植范围。

## 14. 具体修改清单

### 14.1 `neuronlib/Servo/Dynamixel/dynamixel_serial.h`

- 增加 `ServoInstruction`。
- 增加 `PendingTransaction`。
- 将 instruction buffer 类型改为 `RingBuffer<ServoInstruction, 64>`。
- 增加 `queue_mutex_`；若所有 periodic event 只由 servoTask 产生，仍可为 service producer 保留 mutex。
- 增加 `feedback_received_mask_` 和 fresh flag。
- 增加 queue/drop/timeout instrumentation counters。
- 增加 `phase_start_pending_`、`phase_start_tick_`、`control_scheduler_started_` 和 `next_control_tick_`。
- 增加 `schedulePeriodicCommands()`、`processInstructionQueue()`。
- 增加仅提交 phase-start flag 的 `notifyTargetFrame()` 或等价 API；该 API 不执行 UART。
- 增加 start/complete pending transaction API。
- 删除或逐步废弃 `target_cycle_pending_`、`target_cycle_complete_`。

### 14.2 `neuronlib/Servo/Dynamixel/dynamixel_serial.cpp`

- 参考 spinal 移植 queue batch/pop 实现。
- 将 G4 的 direct target-cycle block 替换成内部 5 ms scheduler。
- scheduler 初始处于 `WAIT_PHASE_SYNC`，由首帧 CAN goal 锁定 epoch，丢包时进入本地 holdover。
- 启动后的 CAN goal 只更新 latest target 和 watchdog，不得每帧直接重启 cycle。
- 将 instruction queue 从一次全部 drain 改为每 tick 一个有效 bus operation。
- 参考 spinal 处理 mixed TTL/RS485 groups。
- 参考 spinal 移植 mode-aware `INST_SET_GOAL_STATE`。
- 保留 neuron 的 CbP Position -> Current 顺序。
- 第一轮可以复用 spinal pending read；第二轮将 RX completion 改为非阻塞。
- 保留 neuron external encoder、pulley skip、watchdog 和 CAN-specific state。

### 14.3 `neuronlib/Servo/servo.h/.cpp`

- 增加 CAN target pending/active snapshot 或等价同步机制。
- CAN target callback 更新 target，同时提交独立 phase-sync event，但不直接触发 Dynamixel cycle。
- 使用 target sequence 识别 duplicate/drop/restart，并继续在 feedback 中回显已应用的 target generation。
- 将 config slot 接入 `INST_SERVICE_REQUEST`。
- config 完成后异步调用 `sendConfigResponse()`。
- `Servo::update()` 返回是否出现一组新的完整 feedback。
- 删除对 `startTargetCycle()` 的调用。

### 14.4 `neuron_g4/Core/Src/main.c`

- `servoTaskCallback()` 使用 `vTaskDelayUntil()` 保持绝对 1 ms tick。
- servo update 产生 fresh feedback 时 release `canTxSemHandle`。
- 删除 CAN target receive callback 中立即 release `canTxSemHandle` 的逻辑。
- 第一轮保持 servoTask priority 为 Normal。
- 如果最终调整 task priority，同步更新 `neuron_g4.ioc`，避免 CubeMX 再生成时覆盖。

## 15. 推荐实施阶段

### 阶段 A：Spinal parity

- 移植 `ServoInstruction`、queue mutex、batch enqueue、scheduler 和 process queue。
- 使用 spinal 当前 pending transaction 行为。
- 保持 target 数据结构暂时不变，但将 CAN-triggered cycle 拆成“首帧锁定相位 + 本地 5 ms scheduler”。
- 确认 200 Hz read/write 和各 mode 正常。

### 阶段 B：CAN 适配

- service request 改为异步 CAN response。
- target 改为 latest snapshot。
- 增加 target sequence 诊断和丢包时的本地 holdover。
- feedback 改为 CVP-complete 驱动 CAN TX。

### 阶段 C：实时性加固

- `osDelay(1)` 改为 `vTaskDelayUntil()`。
- UART RX 改为 nonblocking pending transaction。
- 增加 deadline miss、queue high watermark、UART timeout 等计数器。
- 根据测量决定是否使用 UART TX DMA 或调整 task priority。

分阶段的原因是可以先验证与 spinal 行为一致，再把故障情况下的 busy-wait 替换掉；不要同时重写调度、UART parser、CAN protocol 和 service path。

## 16. 验收标准

### 16.1 功能

- CAN target 不直接触发 Dynamixel transaction。
- high-rate scheduler 在首帧有效 CAN target 之前处于 `WAIT_PHASE_SYNC`。
- 不同 neuron 即使以不同次序上电，control cycle 仍保持 spinal goal slot 提供的约 1 ms 错峰。
- 丢失一帧 CAN target 时本地 5 ms loop 继续，不会停止或连续补做 cycle。
- CAN 长时间中断后，可在 UART/control-batch 空闲边界清除 started flag，由下一帧 target 重新启动。
- 无论 CAN target 到达频率如何，servoTask 都保持 1 ms 调度 tick。
- Dynamixel CVP/goal control cycle 正常时稳定为 5 ms。
- DP、DEP、DV、DC、MIT、CbP、EM 行为与 mode 定义一致。
- CbP 每次都是 Goal Position packet 在 Goal Current packet 之前。
- DV/DC/MIT target timeout 后发送安全零输出。
- torque-off 不会被普通 queue backlog 延迟。
- config request 能异步回复成功、失败、BUSY 和 QUEUE_FULL。

### 16.2 Queue

- high-rate batch 只能整体入队或整体失败。
- queue full 时不会积累过期 control cycle。
- service request 不会破坏正在进行的 read/write control-cycle 顺序。
- mixed TTL/RS485 最多四个 high-rate events，能够在下一个 5 ms deadline 前完成。

### 16.3 Feedback

- 只有完整 CVP feedback group 成功后才标记 fresh。
- CAN feedback 不再由 target receive 立即触发。
- 验证五个 neuron 的 feedback 在时间上分散，并记录偶发仲裁、TX queue 深度和最大发送延迟。
- `SERVO_STATE_PVC_FRESH` 与 `pvc_update_count_` 一致。
- feedback frequency downsample 继续有效。

### 16.4 实时性与故障

建议至少记录：

```cpp
servo_loop_max_us
servo_loop_deadline_miss_count
control_cycle_count
control_cycle_enqueue_fail_count
instruction_queue_high_watermark
uart_timeout_count
uart_crc_error_count
target_generation
applied_target_generation
```

测试场景包括：

- 正常四 servo、单 bus。
- mixed TTL/RS485。
- 不同 mode 混合。
- 高频 CAN target。
- neuron 以不同顺序上电和中途重启。
- 单帧 target 丢失、连续 target 丢失、spinal 重启和 sequence 回绕。
- CAN target 停止触发 watchdog。
- 单个 Dynamixel 掉线。
- Dynamixel response 截断或 CRC 错误。
- queue 人为压满。
- config request 与 control cycle 同时发生。
- torque-off 与 queue backlog 同时发生。

## 17. 当前验证状态

- neuron CbP Position -> Current 的源代码修改已完成。
- `git diff --check` 已通过。
- 使用 host `g++ -fsyntax-only -fpermissive` 对修改后的 Dynamixel source 做过语法检查并通过。
- 当前开发环境缺少 `arm-none-eabi-g++`，尚未完成 neuron_g4 的正式交叉编译。

后续 agent 在实现结构性移植前，不应修改用户当前已有的以下无关工作区变更：

- `aerial_robot_nerve/spinal/scripts/servo_scale_converter.py`
- `robots/dragon/config/quad/v1_5_202601/RLServo.yaml`
- `robots/dragon/launch/servo_response.launch`

## 18. 仍需确认的设计问题

1. “稳定 1 ms”是否要求在 Dynamixel 掉线和 Flash config 期间也成立？
   - 如果是，必须完成 nonblocking RX，并将 Flash maintenance 移出 servoTask。
2. fresh feedback 应在最后一组 CVP read 后产生，还是要求 goal write 也成功？
   - 推荐沿用 spinal：CVP read 完成即表示 fresh measurement；另设 control-write status。
3. service request 是否允许延迟一个或多个 control cycle？
   - 推荐不打断已入队的 read -> write batch，在 batch 边界执行。
4. mixed bus 和四 servo 的最坏 UART 时间是否能稳定小于 5 ms？
   - 必须用逻辑分析仪或 cycle counter 实测。
5. 第一轮是否先接受 spinal 的 busy-wait timeout 行为？
   - 推荐先做 parity，再单独提交 nonblocking RX，便于定位回归。
6. 1 ms goal 起始错峰在实机上是否已足以避免 CAN TX queue 积压？
   - 先量测；若不足，由 spinal 引入显式 feedback window/token，不在 neuron 内叠加固定 delay。
