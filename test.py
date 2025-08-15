import os, pickle
import numpy as np
import matplotlib.pyplot as plt
import types

from main import (
    MultiRoadDataProcessor, MultiRoadDrivingEnvironment, DDPGAgent
)

# === 2. 参数 ===
ROAD_FOLDER = "data"
ACTOR_WEIGHTS = "multi_road_ddpg_actor_5sec.weights.h5"
CRITIC_WEIGHTS= "multi_road_ddpg_critic_5sec.weights.h5"
TIME_STEP     = 0.1

# === 3. 加载道路 & 环境 ===
multi_road = MultiRoadDataProcessor(ROAD_FOLDER)
env        = MultiRoadDrivingEnvironment(multi_road)

# 1) 与训练对齐的关键参数
env.profile_res  = 0.05
env.preview_time = 6.0
env.max_speed    = 30.0
env.profile_len  = 1024      # ← 改成当时训练用的值
# env.vd_series.fill(15.0)     # 或者注释掉看真实限速

# 2) 加载模型
state, _ = env.reset(episode_num=0, road_strategy="sequential")
state_dim = len(state)
print("state_dim =", state_dim)  # 现在应 ~1032
agent = DDPGAgent(state_dim)
if os.path.exists(ACTOR_WEIGHTS):
    agent.actor.load_weights(ACTOR_WEIGHTS)
agent.noise_scale = 0.0

env.time_step = TIME_STEP
# 兜底初始化：有就用原来的，没有就补上
if not hasattr(env, 'global_step'):
    env.global_step = 0
if not hasattr(env, 'last_pred_global_step'):
    env.last_pred_global_step = -10**9  # 确保一开始就会刷新预测
if not hasattr(env, 'use_pred'):
    env.use_pred = True
if not hasattr(env, 'pred_every'):
    # 每隔 0.2s 刷新一次预测
    env.pred_every = max(1, int(0.2 / env.time_step))
if not hasattr(env, '_pred_cache'):
    env._pred_cache = (0.0, 0.0, 0.0)

# 可选：把 max_position 绑定到当前路长，避免尾段插值拉平
env.max_position = float(env.current_road.y.max() - 1.0)
# 确保 reset 后第一次就有预测（可选，但稳妥）
env._pred_cache = env.predict_comfort(env.speed, horizon_s=env.preview_time)



###############################################################################
def step_nolimit(self, action):
    # 计数（补回）
    self.episode_steps += 1
    self.global_step   += 1

    # —— 预测刷新（补回）——
    if self.use_pred and (self.global_step - self.last_pred_global_step >= self.pred_every):
        self._pred_cache = self.predict_comfort(self.speed, horizon_s=self.preview_time)
        self.last_pred_global_step = self.global_step

    action = np.clip(action, -1, 1)
    a_cmd = action * 3.0
    jerk = (a_cmd - self.prev_acceleration) / self.time_step

    pos0, v0 = self.position, self.speed
    self.acceleration = a_cmd

    # 无“vd”上界，仅最小/最大速度约束
    v1   = np.clip(v0 + self.acceleration * self.time_step, self.min_speed, self.max_speed)
    v_dyn = 0.5 * (v0 + v1)

    vibration = self.quarter_car.step(self.current_road, pos0, v_dyn, self.time_step)

    # 记录 zr（与你写法一致）
    N = self.quarter_car.integration_steps
    sub_dt = self.time_step / N
    wheel_pos_last = pos0 + v_dyn * sub_dt * (N - 0.5)
    zr_last = self.current_road.get_height_at_position(wheel_pos_last)
    self.history['zr'].append(zr_last)
    self.history['zr_pos'].append(wheel_pos_last)

    # 步后更新
    self.speed = v1
    self.position += v_dyn * self.time_step
    self.current_time += self.time_step
    self.prev_acceleration = self.acceleration

    running_rms = self.quarter_car.get_running_rms()
    wrmsa = 0

    # 记录历史
    self.history['position'].append(self.position)
    self.history['speed'].append(self.speed)
    self.history['acceleration'].append(self.acceleration)
    self.history['vibration'].append(abs(vibration))
    self.history['wrmsa'].append(wrmsa)
    self.history['running_rms'].append(running_rms)
    self.history['jerk'].append(abs(jerk))

    # 奖励仍参考 vd_now（基于 _pred_cache）
    vd_now = self._current_vd()
    reward = self._calculate_reward(vibration, self.speed, self.acceleration,
                                    jerk, running_rms, vd_now)

    next_state = self._get_state()
    done = (self.position >= self.max_position or
            self.episode_steps >= self.max_episode_steps)
    return next_state, reward, done, {'current_road_index': self.current_road_index}

# 绑定覆盖
env.step = types.MethodType(step_nolimit, env)
# 3) 按路逐条测试
for road_idx in range(multi_road.get_num_roads()):
    state, _ = env.reset(episode_num=road_idx, road_strategy="sequential")
    zs_list, zr_list, zr_pos_list = [], [], []
    done = False
    while not done:
        pos0, v0 = env.position, env.speed
    
        # 1) 先由策略给出动作
        action = agent.act(state, add_noise=False)
        # 2) 用当前速度做前瞻预测→必要时在 step 之前修改动作
        rms_pred, rms_max1s, peak_pred = env.predict_comfort(env.speed, horizon_s=env.preview_time)
        RMS_LIM, PEAK_LIM = 0.9, 1.2
        if (rms_max1s > RMS_LIM) or (peak_pred > PEAK_LIM):
            # 简易 governor：触发时给温和制动；也可按超标幅度自适应
            action = np.clip(action - 0.5, -1.0, 1.0)

        # 3) 推进（已无限速夹紧）
        next_state, reward, done, info = env.step(action)
        zs_list.append(env.quarter_car.state[0])
        state = next_state

        # 复现实用 v_dyn 的子步 zr
        N = env.quarter_car.integration_steps
        sub_dt = env.time_step / N
        # ⚠️ 与 step 保持一致：无上界
        v1 = max(v0 + env.acceleration * env.time_step, env.min_speed)
        v_dyn = 0.5 * (v0 + v1)
        for i in range(N):
            wheel_pos = pos0 + v_dyn * sub_dt * (i + 0.5)
            zr_list.append(env.current_road.get_height_at_position(wheel_pos))
            zr_pos_list.append(wheel_pos)
            
    # === 6. 取出历史数据 ===
    s = env.history  # 简化书写
    pos   = np.array(s['position'])     # 已是 m
    speed = np.array(s['speed'])        # m/s
    rms   = np.array(s['running_rms'])  # m/s²
    acc   = np.array(s['acceleration']) # m/s²
    zs    = np.array(zs_list)           # m
    
    # 手动获取道路高度（因为history中可能没有记录）
    road = np.array([env.current_road.get_height_at_position(p) for p in pos])
    
    # === 7. 画 4‑subplot 图 ===
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f"Road {road_idx+1}  •  1 km Test", fontsize=16)
    
    # 真·原始路面
    y_raw = env.current_road.y
    z_raw = env.current_road.z

    zr_pos = np.array(zr_pos_list)
    zr     = np.array(zr_list)
    zs     = np.array(zs_list)

    axes[0].plot(y_raw, z_raw, lw=0.5, label='Raw road (y_raw, z_raw)')  # ✅ 原始
    axes[0].plot(zr_pos, zr, lw=1.5, label='zr used by dynamics')        # ✅ 真 zr
    # axes[0].plot(pos, zs, alpha=0.5, label='Sprung mass z_s')             # 可选：响应
    axes[0].set_ylabel('Height  (m)')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    
    # (2) 速度
    axes[1].plot(pos, speed)
    axes[1].set_ylabel('Speed  (m/s)')
    axes[1].grid(True, alpha=0.3)
    
    # (3) 垂向舒适度 (1 s RMS)
    axes[2].plot(pos, rms)
    axes[2].set_ylabel('Running RMS  (m/s²)')
    axes[2].grid(True, alpha=0.3)
    
    # (4) 动作（纵向加速度指令）
    axes[3].plot(pos, acc)
    axes[3].set_ylabel('Action  (m/s²)')
    axes[3].set_xlabel('Distance along road  (m)')
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

print("✓ All test roads finished.")