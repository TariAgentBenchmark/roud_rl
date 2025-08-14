

##################
import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque
import matplotlib.pyplot as plt
import random
from scipy import signal
from scipy.fft import fft, fftfreq
import pickle
import os
import glob
import time
from datetime import timedelta
# from utils import post_process      

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)
h_thr   = 0.01   # m，高度阈值（特征里用）
vib_thr = 0.9    # m/s^2，垂向加速度峰值阈值（奖励里用）
peak_w  = 1    # 峰值惩罚权重
w_rms   = 1.0

LEAD_DIST      = 10.0   # 提前判定距离
DECEL_THR      = -0.2  # 判定为“减速”的加速度阈值 (m/s^2)
# ===================== 1. Multi-Road Data Processor (unchanged) =====================
class MultiRoadDataProcessor:
    def __init__(self, road_folder):
        """Load and process multiple road profile data files"""
        self.road_folder = road_folder
        self.road_processors = []
        self.road_files = []
        
        # Find all .npy files in the folder
        npy_files = glob.glob(os.path.join(road_folder, "*.npy"))
        
        if len(npy_files) == 0:
            raise FileNotFoundError(f"No .npy files found in {road_folder}")
        
        print(f"🛣️ Loading {len(npy_files)} road files...")
        
        # Load each road file
        for i, npy_file in enumerate(npy_files):
            try:
                processor = RoadDataProcessor(npy_file)
                self.road_processors.append(processor)
                self.road_files.append(os.path.basename(npy_file))
                print(f"   ✅ Road {i+1}: {os.path.basename(npy_file)}")
            except Exception as e:
                print(f"   ❌ Failed to load {os.path.basename(npy_file)}: {e}")
        
        if len(self.road_processors) == 0:
            raise RuntimeError("No valid road files could be loaded")
        
        print(f"📊 Successfully loaded {len(self.road_processors)} roads")
        
        # Analyze road characteristics
        self._analyze_road_characteristics()
    
    def _analyze_road_characteristics(self):
        """Analyze and display characteristics of all roads"""
        print("\n📈 Road Characteristics Analysis:")
        print("-" * 50)
        
        for i, (processor, filename) in enumerate(zip(self.road_processors, self.road_files)):
            road_length = processor.y.max() - processor.y.min()
            height_std = processor.z.std() * 1000  # mm
            height_range = (processor.z.max() - processor.z.min()) * 1000  # mm
            
            print(f"Road {i+1} ({filename[:20]}...):")
            print(f"   Length: {road_length:.1f}m")
            print(f"   Height Std: {height_std:.2f}mm")
            print(f"   Height Range: {height_range:.2f}mm")
    
    def get_random_road(self):
        """Get a random road processor"""
        return random.choice(self.road_processors)
    
    def get_road_by_index(self, index):
        """Get a specific road processor by index"""
        if 0 <= index < len(self.road_processors):
            return self.road_processors[index]
        else:
            raise IndexError(f"Road index {index} out of range [0, {len(self.road_processors)-1}]")
    
    def get_num_roads(self):
        """Get the number of available roads"""
        return len(self.road_processors)


class RoadDataProcessor:
    def __init__(self, npy_file):
        """Load and process road profile data"""
        self.road_data = np.load(npy_file)
        self.filename = os.path.basename(npy_file)
        
        # Extract positions and heights
        self.y = self.road_data[:, 1]  # Longitudinal position
        self.z = self.road_data[:, 2]  # Height
        
        # Sort by position
        sort_idx = np.argsort(self.y)
        self.y = self.y[sort_idx]
        self.z = self.z[sort_idx]
        self.build_adaptive_thresholds()
    
    def build_adaptive_thresholds(self, q: int = 96):
        samples = np.arange(self.y.min(), self.y.max(), 0.15)
        stds, slopes = [], []
        for p in samples:
            h = self.get_road_segment(p - 0.15, 0.3, 0.01)
            if len(h):
                stds.append(np.std(h))
                slopes.append(np.max(np.abs(np.gradient(h, 0.01))))
        # 97 % 分位，再加下限保护
        self.std_thr   = max(np.percentile(stds, q),   0.006)   # ≥6 mm
        self.slope_thr = max(np.percentile(slopes, q), 1.0)     # ≥1.0


    def get_height_at_position(self, position):
        """Get interpolated height at any position with dense interpolation"""
        position = np.clip(position, self.y.min(), self.y.max())
        return np.interp(position, self.y, self.z)
    
    def get_road_roughness_at(self, position, window_size=2.0):
        """Get road roughness (std deviation) around a position"""
        start_pos = max(position - window_size/2, self.y.min())
        end_pos = min(position + window_size/2, self.y.max())
        
        # Get heights in the window
        mask = (self.y >= start_pos) & (self.y <= end_pos)
        if np.any(mask):
            heights = self.z[mask]
            return np.std(heights)
        else:
            return 0.0
    
    def get_road_segment(self, start_pos, length=30, resolution=0.01):
        """Get road heights for a segment with high resolution"""
        start_pos = np.clip(start_pos, self.y.min(), self.y.max() - length)
        num_points = int(length / resolution)
        positions = np.linspace(start_pos, start_pos + length, num_points)
        heights = np.interp(positions, self.y, self.z)
        return heights
    
    def _local_peak2peak(seg, dy, small_win):
            win_pts = max(1, int(small_win/dy))
            if win_pts<3: return np.ptp(seg)
            # 滑窗 peak2peak 的最大值
            p2ps = []
            for i in range(0, len(seg)-win_pts+1, win_pts//2):
                p2ps.append(seg[i:i+win_pts].ptp())
            return max(p2ps) if p2ps else seg.ptp()
    
    def is_irregular(self, position,
                     win_len = 0.1,
                     res     = 0.01):
        """
        判断指定位置是否属于“不平整”。
        使用自适应阈值：self.std_thr / self.slope_thr
        """
        # 若阈值丢失（极少见），即时再算一次
        if not hasattr(self, 'std_thr'):
            self.build_adaptive_thresholds()

        h = self.get_road_segment(position - win_len/2, win_len, res)
        if len(h) == 0:
            return False

        h_std   = np.std(h)
        max_slp = np.max(np.abs(np.gradient(h, res)))

        return (h_std > self.std_thr) or (max_slp > self.slope_thr)
    
    def get_irregular_mask(self, resolution=0.01):
        """离线生成整条路的 irregular 布尔序列（缓存一次即可）"""
        if hasattr(self, '_irregular_mask'):
            return self._irregular_mask, self._mask_positions

        positions = np.arange(self.y.min(), self.y.max(), resolution)
        mask = np.array([self.is_irregular(p) for p in positions])

        # ---- 新增后处理 ----
        mask = post_process(mask, min_len=2)       # <<<  至少 2 连点才算 bump

        self._irregular_mask = mask
        self._mask_positions = positions
        return mask, positions


    def extract_window_features(self, start_pos, window_length, num_segments=5):
        """Extract statistical features from road ahead with dense sampling"""
        # Use dense resolution for feature extraction (0.01m = 1cm)
        heights = self.get_road_segment(start_pos, window_length, resolution=0.01)
        dy = 0.01  # 你的分辨率
    
        seg_len = len(heights) // num_segments
        feats = []
        for i in range(num_segments):
            s = i*seg_len
            e = (i+1)*seg_len if i<num_segments-1 else len(heights)
            seg = heights[s:e]
            if len(seg)==0:
                feats.extend([0,0,0,0])
                continue
            
            seg_d = seg - seg.mean()
            rms = np.sqrt(np.mean(seg_d**2))
            peak = np.max(np.abs(seg_d))
            crest = peak/(rms+1e-8)
            max_slope = np.max(np.abs(np.gradient(seg, dy)))
            over_thr_ratio = np.mean(np.abs(seg_d) > h_thr)
            #p2p_small = _local_peak2peak(seg, dy, small_win)
            
            # 你可以选其中两个替换掉 std / range
            feats.extend([
                seg.mean(),          # 原 mean
                crest,               # 新: 尖峰因子
                max_slope,           # 新: 最大坡度
                over_thr_ratio       # 或 p2p_small，二选一
            ])
        return np.array(feats)

    

# ===================== 2. Quarter-Car Model with Running RMS =====================
class QuarterCarModel:
    def __init__(self):
        """Initialize quarter-car suspension model with running RMS"""
        self.ms = 360.0    # Sprung mass (kg)
        self.mu = 45.0     # Unsprung mass (kg)
        self.ks = 15000.0  # Suspension stiffness (N/m)
        self.kt = 190000.0 # Tire stiffness (N/m)
        self.cs = 2500.0   # Suspension damping (N⋅s/m)
        self.state = np.zeros(4)  # [zs, zs_dot, zu, zu_dot]
        self.dt = 0.01
        self.integration_steps = 20
        self.sub_dt = self.dt / self.integration_steps
        
        # Running RMS parameters (1 second window = 10 steps)
        self.running_rms_buffer = deque(maxlen=10)
        
        # # WRMSA parameters (keep for reference)
        # self.sampling_rate = int(1 / self.dt)
        # self.acceleration_buffer = []
        # self.buffer_size = 100
        
    #     self.setup_iso_wrmsa_parameters()

    # def setup_iso_wrmsa_parameters(self):
    #     """Setup ISO 2631-1 WRMSA calculation parameters"""
    #     self.center_frequencies = np.array([
    #         0.5, 0.63, 0.8, 1.0, 1.25, 1.6, 2.0, 2.5, 3.15, 4.0, 
    #         5.0, 6.3, 8.0, 10.0, 12.5, 16.0, 20.0, 25.0, 31.5, 
    #         40.0, 50.0, 63.0, 80.0
    #     ])
    
    #     self.weighting_factors = np.array([
    #         0.5, 0.56, 0.63, 0.71, 0.8, 0.9, 1.0, 1.0, 1.0, 1.0,
    #         1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 0.63, 0.5, 0.4,
    #         0.315, 0.25, 0.2
    #     ])
        
    #     self.freq_bands = []
    #     for fc in self.center_frequencies:
    #         fl = fc / (2**(1/6))
    #         fu = fc * (2**(1/6))
    #         self.freq_bands.append((fl, fu))

    def get_running_rms(self):
        """Get running RMS from 1-second window"""
        if len(self.running_rms_buffer) < 10:
            return 0.0
        return np.sqrt(np.mean([acc**2 for acc in self.running_rms_buffer]))

    # def calculate_standard_wrmsa(self):
    #     """Standard ISO 2631-1 WRMSA calculation"""
    #     if len(self.acceleration_buffer) < 15:
    #         return 0.0
        
    #     window_size = min(60, len(self.acceleration_buffer))
    #     data = np.array(self.acceleration_buffer[-window_size:])
    #     data = data - np.mean(data)
        
    #     try:
    #         frequencies, psd = signal.welch(
    #             data, 
    #             fs=self.sampling_rate,
    #             window='hann',
    #             nperseg=min(16, len(data)//2),
    #             noverlap=min(8, len(data)//4),
    #             detrend='constant'
    #         )
    #     except:
    #         return np.sqrt(np.mean(data**2))
        
    #     weighted_sum = 0.0
        
    #     for i, (fl, fu) in enumerate(self.freq_bands):
    #         if i >= len(self.weighting_factors):
    #             break
                
    #         freq_mask = (frequencies >= fl) & (frequencies <= fu)
            
    #         if np.any(freq_mask):
    #             band_psd = psd[freq_mask]
    #             band_freq = frequencies[freq_mask]
                
    #             if len(band_psd) > 1:
    #                 integral = np.trapezoid(band_psd, band_freq)
    #             elif len(band_psd) == 1:
    #                 integral = band_psd[0] * (fu - fl)
    #             else:
    #                 integral = 0.0
                
    #             weighted_sum += (self.weighting_factors[i]**2) * integral
        
    #     return np.sqrt(max(weighted_sum, 0.0))

    def dynamics(self, state, zr):
        """Calculate state derivatives"""
        zs, zs_dot, zu, zu_dot = state
        fs = self.ks * (zu - zs) + self.cs * (zu_dot - zs_dot)
        ft = self.kt * (zr - zu)
        zs_ddot = fs / self.ms
        zu_ddot = (ft - fs) / self.mu
        return np.array([zs_dot, zs_ddot, zu_dot, zu_ddot])
    
    def step(self, road_proc, vehicle_pos, speed, dt=0.02):
        """Simplified step function with numerical stability"""
        if dt != self.dt:                       # ← ★ 关键：检测外部传入 dt
            self.dt = dt
            self.sub_dt = dt / self.integration_steps
            # 重新定义 1 s running RMS 缓冲区长度
            self.running_rms_buffer = deque(maxlen=max(1, int(1.0 / self.dt)))
            # 采样率也更新，供 WRMSA 频谱用
            self.sampling_rate = int(1 / self.dt)
        accumulated_acceleration = 0.0
        
        for i in range(self.integration_steps):
            wheel_pos = vehicle_pos + speed * self.sub_dt * (i + 0.5)
            zr = road_proc.get_height_at_position(wheel_pos)   # ★ 轮胎实时读取路面
            derivatives = self.dynamics(self.state, zr)
            
            max_accel = 2.5
            
            def soft_limit(x, limit):
                return limit * np.tanh(x / limit)
            
            derivatives[1] = soft_limit(derivatives[1], max_accel)
            derivatives[3] = soft_limit(derivatives[3], max_accel)
            
            self.state += derivatives * self.sub_dt
            accumulated_acceleration += derivatives[1]
        
        average_acceleration = accumulated_acceleration / self.integration_steps
        
        self.state[0] = np.clip(self.state[0], -0.1, 0.1)
        self.state[1] = np.clip(self.state[1], -4.0, 4.0)
        self.state[2] = np.clip(self.state[2], -0.08, 0.08)
        self.state[3] = np.clip(self.state[3], -4.0, 4.0)
        
        # Update running RMS buffer
        self.running_rms_buffer.append(average_acceleration)
        
        # # Update WRMSA buffer (keep for reference)
        # self.acceleration_buffer.append(average_acceleration)
        # if len(self.acceleration_buffer) > self.buffer_size:
        #     self.acceleration_buffer.pop(0)
        
        return average_acceleration
    
    def reset(self):
        """Complete reset of all states"""
        self.state = np.zeros(4)
        self.running_rms_buffer.clear()
        self.acceleration_buffer = []

# ===================== 3. Multi-Road Driving Environment with 5-second Time Window =====================
class MultiRoadDrivingEnvironment:
    def __init__(self, multi_road_processor):
        self.multi_road = multi_road_processor
        self.shadow_car = QuarterCarModel()   # 用于前瞻预测
        self.current_road = None
        self.current_road_index = -1
        self.h_thr   = 0.01   # m，高度阈值（特征里用）
        self.vib_thr = 0.9    # m/s^2，垂向加速度峰值阈值（奖励里用）
        self.peak_w  = 3.0    # 峰值惩罚权重
        self.w_rms   = 1.0
        # ---- Reward 3 项权重（方便外部调参）----
        self.w_eff   = 3.0      # driving‑efficiency
        self.w_vert  = 2.0      # vertical‑comfort
        self.w_long  = 1.0      # longitudinal‑comfort
        
        # ---- 是否使用“5 s 预测舒适度” ----
        self.use_pred = True
        # Environment parameters - CHANGED TO TIME-BASED
        
        # Constraints
        self.min_speed = 2.0
        self.max_speed = 30.0
        self.preview_time = 6.0  # 6 seconds preview time
        self.spatial_step = 6.0  # Keep original 6.0m
        self.time_step = 0.1
        self.pred_every  = max(1, int(0.2 / self.time_step))
        # self.pred_every  = 1#int(0.5 / self.time_step)   # 0.5 s = N steps
        # --- 用全局步数管理预测刷新（跨 episode 不丢失节奏） ---
        self.global_step = 0
        self.last_pred_global_step = -10**9  # 保证一开始能刷新
        self._pred_cache = (0., 0., 0.)                # (rms_mean, rms_max1s, peak)
        self.quarter_car = QuarterCarModel()
        self.quarter_car.integration_steps = 20
        self.shadow_car.integration_steps  = 20
        self.prev_acceleration = 0

        self.lead_dist      = LEAD_DIST
        self.decel_thr      = DECEL_THR
        
        # State variables
        self.position = 0
        self.speed = 15.0
        self.acceleration = 0
        # 固定分辨率采样：始终每 0.02 m 取一个样本
        # 预瞄距离 = preview_time × speed ，在最高速下计算最大可能长度，
        # 这样就能把 profile “截断 / 填充” 到统一维度，网络仍用 Dense。
        # ------------------------------------------------------------------
        self.profile_res = 0.05      # 3 cm 采样分辨率
        self.profile_len = 1024
        self.max_profile_len = int(  # 30 m/s × 10 s / 0.02 m ≈ 15000
            np.ceil(self.max_speed * self.preview_time / self.profile_res)
        )
        
        
        
        
        # Episode management
        self.max_position = 990
        self.episode_steps = 0
        self.max_episode_steps = 3000
        # 固定基础限速 (m/s)
        self.vd_base = 15.0
        self.debug_pred  = True
        self.debug_every = 2500
        # 预测与风险的调试缓存（便于 step() 打印）
        self._dbg = {
            "vd_base": self.vd_base,
            "vd_risk": self.vd_base,
            "risk": 0.0,
            "rms_pred": 0.0,
            "rms1s_pred": 0.0,
            "peak_pred": 0.0,
        }
        # Action history tracking
        self.action_history = []
        self.max_action_history = 10
        
        # Training statistics
        self.road_usage_count = [0] * self.multi_road.get_num_roads()
        self.total_episodes = 0
        

        
    def get_current_window_length(self):
        """Calculate current window length based on speed and preview time"""
        return self.speed * self.preview_time
        
    def predict_comfort(self, current_speed,
                        horizon_s=6.0,
                        win_len_s=1.0,
                        slide_step_s=0.5):
        # """
        # 返回 (rms_mean, rms_max1s, peak_acc):
        #     rms_mean   : 5 s 整段均方根
        #     rms_max1s  : 前瞻 5 s 内滑窗(1 s) 的最大 RMS
        #     peak_acc   : 5 s 内最大 |a|
        # """
        # 修复版：
        #    - 每次预测前 reset 影子车
        #    - 用当前真实四分之一车状态作为影子车起点
           
         # ---- 保存影子车进入前的状态（可选但更稳） ----
        _pre_state   = self.shadow_car.state.copy()
        _pre_dt      = self.shadow_car.dt
        _pre_sub_dt  = self.shadow_car.sub_dt
 
        # ---- 清空历史并对齐起点 ----
        self.shadow_car.reset()
        self.shadow_car.state = self.quarter_car.state.copy()
 
        # ---- 时间步：与空间分辨率一致，保证数值稳定 ----
        dt = max(self.profile_res / max(current_speed, 0.1), 0.01)
        if self.shadow_car.dt != dt:
            self.shadow_car.dt     = dt
            self.shadow_car.sub_dt = dt / self.shadow_car.integration_steps
 
        # ---- 向前推演 horizon_s ----
        n_tot  = int(np.ceil(horizon_s / dt))
        acc_buf = []
        for i in range(1, n_tot + 1):
            pos = self.position + current_speed * dt * i
            acc = self.shadow_car.step(self.current_road, pos, current_speed, dt)
            acc_buf.append(acc)
 
        acc_arr  = np.asarray(acc_buf, dtype=np.float32)
        rms_mean = float(np.sqrt(np.mean(acc_arr ** 2)))
 
        # ---- 1 s 滑窗最大 RMS ----
        win_pts   = max(1, int(win_len_s   / dt))
        slide_pts = max(1, int(slide_step_s/ dt))
        rms_max1s = 0.0
        for s in range(0, len(acc_arr) - win_pts + 1, slide_pts):
            seg = acc_arr[s:s+win_pts]
            rms_max1s = max(rms_max1s, float(np.sqrt(np.mean(seg**2))))

        peak_acc = float(np.max(np.abs(acc_arr)))
 
        # ---- 恢复影子车（消除副作用） ----
        self.shadow_car.state  = _pre_state
        self.shadow_car.dt     = _pre_dt
        self.shadow_car.sub_dt = _pre_sub_dt
        self.shadow_car.running_rms_buffer.clear()
        self.shadow_car.acceleration_buffer = []
 
        return rms_mean, rms_max1s, peak_acc
    
    def estimate_comfort_from_roughness(self, road_roughness, speed):
        """Estimate running RMS comfort from road roughness and speed"""
        # Simple empirical model: comfort correlates with roughness and speed
        k = 15.0  # scaling factor
        base_comfort = k * road_roughness * np.sqrt(speed)
        
        # Add some non-linearity for high roughness
        if road_roughness > 0.02:
            base_comfort *= (1.0 + road_roughness * 10)
        
        return base_comfort
    
    def aggregate_comfort_predictions(self, predictions):
        """Aggregate 5-second comfort predictions into a single value"""
        if len(predictions) == 0:
            return 0.0
            
        # Strategy: Combination of worst case and average
        worst_comfort = max(predictions)
        avg_comfort = np.mean(predictions)
        
        # 60% weight on worst case, 40% on average
        combined_comfort = 0.6 * worst_comfort + 0.4 * avg_comfort
        
        return combined_comfort
        
    def select_road_for_episode(self, episode_num=None, strategy="single"):
        """Select road for current episode with different strategies"""
        num_roads = self.multi_road.get_num_roads()
        
        if strategy == "sequential":
            road_index = episode_num % num_roads if episode_num is not None else 0
            selection_method = "sequential cycle"
            
        elif strategy == "balanced":
            min_usage = min(self.road_usage_count)
            candidates = [i for i, count in enumerate(self.road_usage_count) if count == min_usage]
            road_index = candidates[0]
            selection_method = "balanced (least used)"
            
        elif strategy == "single":
            road_index = 0
            selection_method = "single road only"
            
        elif strategy == "difficulty":
            if not hasattr(self, 'road_difficulty_order'):
                self._calculate_road_difficulty_order()
            
            total_episodes = 400
            if episode_num is not None:
                if episode_num < total_episodes // 3:
                    difficulty_level = 0
                elif episode_num < 2 * total_episodes // 3:
                    difficulty_level = 1
                else:
                    difficulty_level = 2
                
                roads_in_level = len(self.road_difficulty_order[difficulty_level])
                if roads_in_level > 0:
                    level_episode = episode_num % roads_in_level
                    road_index = self.road_difficulty_order[difficulty_level][level_episode]
                else:
                    road_index = 0
            else:
                road_index = 0
            
            selection_method = f"difficulty progression (level {difficulty_level})"
            
        elif strategy == "random":
            road_index = random.randint(0, num_roads - 1)
            selection_method = "random"
            
        else:
            road_index = episode_num % num_roads if episode_num is not None else 0
            selection_method = "default sequential"
        
        # 设置当前路面
        self.current_road = self.multi_road.get_road_by_index(road_index)
        self.current_road_index = road_index
        self.road_usage_count[road_index] += 1
        
        # 只在前10个episode或每50个episode显示详细信息
        if episode_num is not None and (episode_num < 10 or episode_num % 50 == 0):
            print(f"🎯 Episode {episode_num}: Road {road_index+1} ({selection_method})")
        
        return road_index
    
    def _calculate_road_difficulty_order(self):
        """Calculate road difficulty based on roughness and classify into levels"""
        road_roughness = []
        
        for processor in self.multi_road.road_processors:
            roughness = np.std(processor.z) * 1000  # mm
            road_roughness.append(roughness)
        
        sorted_indices = np.argsort(road_roughness)
        
        num_roads = len(sorted_indices)
        easy_count = max(1, num_roads // 3)
        medium_count = max(1, num_roads // 3)
        
        self.road_difficulty_order = {
            0: sorted_indices[:easy_count].tolist(),
            1: sorted_indices[easy_count:easy_count+medium_count].tolist(),
            2: sorted_indices[easy_count+medium_count:].tolist()
        }
        
        print(f"\n📊 Road Difficulty Classification:")
        for level, roads in self.road_difficulty_order.items():
            level_names = ["Easy", "Medium", "Hard"]
            road_list = [f"Road{i+1}" for i in roads]
            print(f"   {level_names[level]}: {road_list}")
    
    def reset(self, episode_num=None, road_strategy="single"):
        """Reset environment with configurable road selection strategy"""
        # Select road for this episode
        selected_road_index = self.select_road_for_episode(episode_num, road_strategy)
        
        # Reset basic states
        self.position = 0
        self.speed = 15
        self.acceleration = 0
        self.episode_steps = 0
        self.prev_acceleration = 0
        # 让 reset 后立刻能触发一次预测（可选但推荐）
        self.last_pred_global_step = self.global_step - self.pred_every
        # Reset vehicle model
        self.quarter_car.reset()
        self.shadow_car.reset()
        
        
        # Reset action history
        self.action_history = []
        
        self.history = {
            'position': [], 'speed': [], 'acceleration': [], 'vibration': [],
            'wrmsa': [], 'running_rms': [], 'jerk': [],
            'road_height': [], 'time': [],'zr': [], 'zr_pos': []
        }

        # episode-level reward分量缓存
        self.ep_r_eff   = []
        self.ep_r_vert  = []
        self.ep_r_long  = []
        self.ep_r_energy = []
        self.ep_r_total = []
        self.current_time = 0
        
        self.total_episodes += 1
        
        # Get initial state
        initial_state = self._get_state()
        
        return initial_state, selected_road_index
    
    # NEW ▸ 采样前方固定长度原始路面高度序列
    def _sample_ahead_profile(self):
        # 固定长度降维版（仍然端到端）：
        #    1) 取前瞻距离 look_ahead = speed * preview_time
        #    2) 先按 profile_res 细采样一遍，避免漏掉尖峰
        #    3) 去均值
        #    4) 等距重采样/填充到固定长度 self.profile_len
        #  备注：启用此函数后，state 维度会变化，需重训模型
        look_ahead = max(0.5, float(self.speed * self.preview_time))
        n_raw = max(1, int(look_ahead / self.profile_res))
        pos_vec = self.position + np.linspace(0.0, look_ahead, n_raw, endpoint=False, dtype=np.float32)
        # 用路面原始 (y,z) 做线性插值
        heights = np.interp(pos_vec, self.current_road.y, self.current_road.z)
        # 去均值（保留形状）
        heights = heights - heights.mean()
        # 固定长度
        if len(heights) >= self.profile_len:
            idx = np.linspace(0, len(heights) - 1, self.profile_len, dtype=np.float32)
            heights = np.interp(idx, np.arange(len(heights), dtype=np.float32), heights)
        else:
            heights = np.pad(heights, (0, self.profile_len - len(heights)), mode='constant')
        return heights.astype(np.float32)

    def _in_bump(self, pos):
        return self.current_road.is_irregular(pos)

    def _dist_to_next_bump(self, pos, max_ahead=30.0, step=0.1):
        # 沿车前方扫描，返回最近 irregular 距离
        cluster_len = 3                       # NEW ➜ 至少 3 连续 True 才算 bump
        scan_pos = np.arange(pos, pos + max_ahead, step)
        irr = [self.current_road.is_irregular(p) for p in scan_pos]

        # --- 简单聚簇过滤 ---
        run = 0
        for i, flag in enumerate(irr):
            run = run + 1 if flag else 0
            if run >= cluster_len:            # 找到第一个满足簇
                return scan_pos[i - run + 1] - pos
        return np.inf


    def _get_state(self):
        """Get normalized state vector"""
        # 1Vehicle state
        vehicle_state = np.array([
            self.speed / 20.0,
            self.acceleration / 3.0,
            np.clip(self.quarter_car.state[0], -1, 1),
            np.clip(self.quarter_car.state[1] / 5.0, -1, 1), 
            self._current_vd() / 25.0
        ])
        #2前瞻路面高度序列
        road_profile = self._sample_ahead_profile()

        # 3️⃣ 未来 10 s 舒适度预测（3 维；若禁用用 0 填充）
        if self.use_pred:
            
            rms_pred, rms_max1s, peak_pred = self._pred_cache

            # 简单归一化到 0-1；阈值同 reward 里配置
            rms_norm  = np.clip((rms_pred   - 0.315) / (2 - 0.315), 0., 1.)
            max1s_norm= np.clip((rms_max1s - 0.315) / (2 - 0.315), 0., 1.)
            peak_norm = np.clip((peak_pred  - 0.9  ) / (2.5 - 0.9  ), 0., 1.)
            pred_feats = np.array([rms_norm, max1s_norm, peak_norm], dtype=np.float32)
        else:
            pred_feats = np.zeros(3, dtype=np.float32)
        return np.concatenate([vehicle_state, road_profile, pred_feats]).astype(np.float32)
    
    def _build_dynamic_speed_limit(self, road_length,
                                   base_limit=13.9, noise_std=1.5, seed=42):
        np.random.seed(seed)
        n_pts = int(road_length / (base_limit * self.time_step)) + 500
        noise = np.random.normal(0, noise_std, n_pts)
        noise = np.convolve(noise, np.ones(20)/20, mode='same')  # 2 s 平滑
        vd = np.clip(base_limit + noise, 5.0, 25.0)              # [5,25] m s-1
        return vd.astype(np.float32)

    def _current_vd(self):
        """固定基础限速 + 风险动态降速；并把中间量缓存到 self._dbg 供日志打印。"""
        vd_base = float(self.vd_base)

        if not self.use_pred:
            self._dbg.update(vd_base=vd_base, vd_risk=vd_base, risk=0.0,
                             rms_pred=0.0, rms1s_pred=0.0, peak_pred=0.0)
            return vd_base


        rms_pred, rms1s_pred, peak_pred = self._pred_cache

        # 风险分量（0~1）
        rms_thr, rms_cap = 0.315, 2.0
        risk_rms  = np.clip((rms_pred   - rms_thr)/(rms_cap - rms_thr), 0.0, 1.0)
        risk_rms1 = np.clip((rms1s_pred - rms_thr)/(rms_cap - rms_thr), 0.0, 1.0)
        risk_peak = np.clip((peak_pred  - 0.9     )/(2.5 - 0.9),        0.0, 1.0)
        risk = 0.5 * risk_rms1 + 0.3 * risk_rms + 0.2 * risk_peak

        # 风险=1 时降到 50%（可调）
        vd_risk = vd_base * (1.0 - 0.5 * risk)
        vd_risk = max(self.min_speed, vd_risk)

        # 缓存到调试结构
        self._dbg.update(
            vd_base=vd_base, vd_risk=vd_risk, risk=float(risk),
            rms_pred=float(rms_pred), rms1s_pred=float(rms1s_pred), peak_pred=float(peak_pred)
        )
        return vd_risk
    
    def step(self, action):
        """Execute one time step"""
        self.episode_steps += 1
        self.global_step   += 1
        # === 只在 step() 里刷新一次预测 ===
         # --- 用全局步数判断是否刷新预测 ---
        if self.use_pred and (self.global_step - self.last_pred_global_step >= self.pred_every):
            speed_for_pred = max(self.speed, 0.1)
            dt_pred_live   = max(self.profile_res / speed_for_pred, 0.01)
            horizon_m_live = speed_for_pred * self.preview_time
            n_pred_live    = int(np.ceil(self.preview_time / dt_pred_live))

            # 真正的预测（影子车仿真）
            self._pred_cache = self.predict_comfort(self.speed, horizon_s=self.preview_time)
            self.last_pred_global_step = self.global_step

            # （可选）标记一下方便肉眼确认确实在刷新
            # print(f"[PRED REFRESH] gstep={self.global_step} pos={self.position:.2f} v={self.speed:.2f}")

            if hasattr(self, "_dbg"):
                self._dbg.update(horizon_m=float(horizon_m_live),
                                 dt_pred=float(dt_pred_live),
                                 n_pred=int(n_pred_live))
        # 1) 动作与jerk
        action = np.clip(action, -1, 1)
        a_cmd = action * 3.0
        jerk = (a_cmd - self.prev_acceleration) / self.time_step

        # 2) 记录步前状态
        pos0 = self.position
        v0   = self.speed
        self.acceleration = a_cmd
        vd_now = self._current_vd()

        # 3) 先算新速度，再取“动力学用速度”（建议用梯形平均）
        v1 = np.clip(v0 + self.acceleration * self.time_step,
                    self.min_speed, self.max_speed)
        v_dyn = 0.5 * (v0 + v1)  # 用中点速度近似本步恒速

        # 4) 用“步前位置 + v_dyn”做动力学积分 → 采到的就是本步路径
        vibration = self.quarter_car.step(
            self.current_road,  # road_proc
            pos0,               # ★ 步前基准位置
            v_dyn,              # ★ 本步用于积分的速度
            self.time_step
        )
        # ★ 新增：记录最后一个子步的 zr 和对应位置
        N = self.quarter_car.integration_steps
        sub_dt = self.time_step / N
        wheel_pos_last = pos0 + v_dyn * sub_dt * (N - 0.5)
        zr_last = self.current_road.get_height_at_position(wheel_pos_last)
        self.history['zr'].append(zr_last)
        self.history['zr_pos'].append(wheel_pos_last)
        # 5) 再更新位移/时间到步后
        self.speed = v1
        self.position = pos0 + v_dyn * self.time_step
        self.current_time += self.time_step

        self.prev_acceleration = self.acceleration
        
        # Calculate running RMS and WRMSA
        running_rms = self.quarter_car.get_running_rms()
        wrmsa = 0#self.quarter_car.calculate_standard_wrmsa()
        
        
        # pred_flag = 1 if (dist2bump < self.lead_dist and self.acceleration < self.decel_thr) else 0
        # ── 统一的调试打印：速度/限速/预测/风险/路面 ──
        if self.debug_pred and (self.episode_steps % self.debug_every == 0):
            delta_s = self.time_step / self.quarter_car.integration_steps * v_dyn
            d2b = self._dist_to_next_bump(self.position, max_ahead=self.lead_dist)
            # --- 实时计算（打印用），避免显示滞后 ---
            horizon_m_live = self.speed * self.preview_time
            dt_pred_live   = max(self.profile_res / max(self.speed, 0.1), 0.01)
            n_pred_live    = int(np.ceil(self.preview_time / dt_pred_live))
            dbg = self._dbg
            print(f"  [LOOKAHEAD] horizon={horizon_m_live:.1f} m  "
                  f"dt_pred={dt_pred_live:.3f} s  n_pred={n_pred_live}")
            print(f"[t={self.current_time:6.2f}s] pos={self.position:7.2f}m  v={self.speed:5.2f} m/s  "
                  f"Δs_sub={delta_s:.3f} m")
            print(f"  [VD] base={dbg['vd_base']:5.2f}  risk_vd={dbg['vd_risk']:5.2f}  risk={dbg['risk']:.2f}")
            print(f"  [PRED] rms={dbg['rms_pred']:.3f}  rms1s={dbg['rms1s_pred']:.3f}  peak={dbg['peak_pred']:.2f}  "
                  f"d2b={d2b if np.isfinite(d2b) else float('inf'):.1f} m")
            print(f"  [MEAS] runningRMS={running_rms:.3f}  |a|={abs(vibration):.3f}  |jerk|={abs(jerk):.2f}")
            print("-" * 40)
        # 记录
        
        self.history['position'].append(self.position)
        self.history['speed'].append(self.speed)
        self.history['acceleration'].append(self.acceleration)
        self.history['vibration'].append(abs(vibration))
        # self.history['wrmsa'].append(wrmsa)
        self.history['running_rms'].append(running_rms)
        self.history['jerk'].append(abs(jerk))
        
        
        # Calculate simplified reward
        
        reward = self._calculate_reward(vibration, self.speed, self.acceleration,
                                        jerk, running_rms,vd_now)
        
        # ★ 每 50 step 打印一次 4 个 reward 分量
        if self.episode_steps % 50 == 0:
            eff, vert, lon, ene = self._dbg_r    # 四个分量
            tot = reward
            print(f"[DBG-50]  Eff={eff:+.2f}  Vert={vert:+.2f}  "
                f"Long={lon:+.2f}  Ene={ene:+.2f}  Tot={tot:+.2f}")
        # Get next state
        next_state = self._get_state()
        
        # Check termination
        done = (self.position >= self.max_position or 
                self.episode_steps >= self.max_episode_steps)
        
        info = {'current_road_index': self.current_road_index}
        
        return next_state, reward, done, info
    
    def _calculate_reward(self, vibration, speed, acceleration, jerk,
                          running_rms,vd_now):
        dt = self.time_step

        # ------------------------------------------------------------------ #
        # 1. Driving efficiency  (纯量：当前速度偏差)    ☆ 不乘 dt
        # ------------------------------------------------------------------ #
        if self.use_pred:
            rms_pred, rms1s_pred, peak_pred = self._pred_cache
            risk_rms1 = np.clip((rms1s_pred - 0.315)/(2.0-0.315), 0.0, 1.0)
            risk_rms  = np.clip((rms_pred   - 0.315)/(2.0-0.315), 0.0, 1.0)
            risk_peak = np.clip((peak_pred  - 0.9  )/(2.5-0.9),   0.0, 1.0)
            risk = 0.5*risk_rms1 + 0.3*risk_rms + 0.2*risk_peak
        else:
            risk = 0.0

        # 虚拟限速（最多降 50%）：风险越高，目标越低
        vd_target = vd_now    # 预测风险降速后的目标速度
        r_driving_efficiency = - ((speed - vd_target) / vd_target) ** 2
        

        # 2. Vertical comfort 归一化到 [-1,0]
        # ------------------------------------------------------------------ #
        rms_thr, rms_cap = 0.315, 2.0     # 放宽 cap (原1.5→2.0)
        vib_cap = 2.5
        thr_peak = 0.9                    # 放宽预测峰值阈 (原0.8→1.2)

        # (A) 实测 running RMS → [0,1]
        rms_norm = 0.0 if running_rms <= rms_thr else np.clip((running_rms - rms_thr)/(rms_cap - rms_thr), 0.0, 1.0)

        # (B) 实测峰值 → [0,1]
        peak_norm_meas = 0.0
        peak_excess = abs(vibration) - self.vib_thr
        if peak_excess > 0:
            peak_norm_meas = np.clip(peak_excess / (vib_cap - self.vib_thr), 0.0, 1.0)

        # (C) 预测项 → [0,1]
        pred_rms_norm, pred_peak_norm = 0.0, 0.0
        if self.use_pred:
            rms_pred, rms_max1s_pred, peak_pred = self._pred_cache

            pred_rms_norm   = np.clip((rms_pred       - rms_thr)/(rms_cap - rms_thr), 0.0, 1.0)
            pred_rms1s_norm = np.clip((rms_max1s_pred - rms_thr)/(rms_cap - rms_thr), 0.0, 1.0)
            if peak_pred > thr_peak:
                pred_peak_norm = np.clip((peak_pred - thr_peak) / (vib_cap - thr_peak), 0.0, 1.0)

        # 前瞻比重更高，并显式纳入 1s窗口
        w_rms_meas = 0.35; w_peak_meas = 0.10
        w_rms_pred = 0.30; w_rms1s_pred = 0.15; w_peak_pred = 0.10
        mix = (w_rms_meas*rms_norm +
               w_peak_meas*peak_norm_meas +
               w_rms_pred*pred_rms_norm +
               w_rms1s_pred*pred_rms1s_norm +
               w_peak_pred*pred_peak_norm)

        # 负号并裁剪到 [-1,0]
        r_vertical_comfort = -np.clip(mix, 0.0, 1.0)
        # 3. Longitudinal comfort
        jerk_penalty = -(jerk ** 2) / 3600
        acceleration_penalty = -(acceleration ** 2) / 90
        r_longitudinal_comfort = jerk_penalty + acceleration_penalty
        # 如果你希望效率分量也是“按秒累计”，就乘同样的 scale；
        # 4. Energy   Re  ≈ - VSP/1000  (kW t-1)
        vsp = speed * (1.1 * acceleration + 0.132)
        r_energy = - vsp / 1000.0 * self.time_step
        

        # Combine
        total_reward = (self.w_eff  * r_driving_efficiency +
                        self.w_vert * r_vertical_comfort * dt   +
                        self.w_long * r_longitudinal_comfort * dt + 
                        1 * r_energy)

        
         # 记录分量
        self.ep_r_eff.append(r_driving_efficiency * dt)
        self.ep_r_vert.append(r_vertical_comfort   * dt)
        self.ep_r_long.append(r_longitudinal_comfort * dt)
        self.ep_r_energy.append(r_energy)          # 已含 dt
        self.ep_r_total.append(total_reward)
        self._dbg_r = (r_driving_efficiency, r_vertical_comfort, r_longitudinal_comfort,r_energy)
        return total_reward
    
    def get_training_statistics(self):
        """Get training statistics across all roads"""
        stats = {
            'total_episodes': self.total_episodes,
            'road_usage_count': self.road_usage_count.copy(),
            'road_usage_percentage': [(count/max(1, self.total_episodes))*100 
                                    for count in self.road_usage_count]
        }
        return stats

# ===================== 4. DDPG Agent (unchanged) =====================
class DDPGAgent:
    def __init__(self, state_dim, action_dim=1):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Hyperparameters
        self.batch_size = 512
        self.gamma = 0.99
        self.tau = 0.001
        
        self.initial_noise = 0.3
        self.min_noise = 0.01
        self.noise_scale = 0.3
        self.noise_decay = 0.995
        
        # Action smoothing
        self.action_smoothing = 0.5
        self.prev_raw_action = 0.0
        
        # Networks
        self.actor = self._build_actor()
        self.critic = self._build_critic()
        self.target_actor = self._build_actor()
        self.target_critic = self._build_critic()
        
        # Initialize targets
        self.update_targets(tau=1.0)
        
        # Replay buffer
        self.buffer = deque(maxlen=500000)
        
    def _build_actor(self):
        """Build actor network"""
        inputs = keras.Input(shape=(self.state_dim,))
        
        x = keras.layers.LayerNormalization()(inputs)
        
        x = keras.layers.Dense(128, activation='relu',
                              kernel_initializer=keras.initializers.HeNormal(),
                              kernel_regularizer=keras.regularizers.l2(0.01))(x)
        x = keras.layers.LayerNormalization()(x)
        x = keras.layers.Dense(64, activation='relu',
                              kernel_initializer=keras.initializers.HeNormal(),
                              kernel_regularizer=keras.regularizers.l2(0.01))(x)
        x = keras.layers.LayerNormalization()(x)
        x = keras.layers.Dense(32, activation='relu',
                              kernel_initializer=keras.initializers.HeNormal())(x)
        
        outputs = keras.layers.Dense(self.action_dim, activation='tanh',
                                   kernel_initializer=keras.initializers.RandomUniform(-0.001, 0.001))(x)
        
        model = keras.Model(inputs, outputs)
        model.compile(optimizer=keras.optimizers.Adam(1e-4, clipnorm=0.5))
        return model
    
    def _build_critic(self):
        """Build critic network"""
        state_input = keras.Input(shape=(self.state_dim,))
        action_input = keras.Input(shape=(self.action_dim,))
        
        state_h = keras.layers.LayerNormalization()(state_input)
        state_h = keras.layers.Dense(128, activation='relu',
                                    kernel_initializer=keras.initializers.HeNormal(),
                                    kernel_regularizer=keras.regularizers.l2(0.01))(state_h)
        state_h = keras.layers.LayerNormalization()(state_h)
        state_h = keras.layers.Dense(64)(state_h)
        
        action_h = keras.layers.Dense(64)(action_input)
        
        concat = keras.layers.Concatenate()([state_h, action_h])
        concat_h = keras.layers.Dense(64, activation='relu',
                                     kernel_initializer=keras.initializers.HeNormal())(concat)
        concat_h = keras.layers.Dense(32, activation='relu',
                                     kernel_initializer=keras.initializers.HeNormal())(concat_h)
        
        outputs = keras.layers.Dense(1,
                                   kernel_initializer=keras.initializers.RandomUniform(-0.003, 0.003))(concat_h)
        
        model = keras.Model([state_input, action_input], outputs)
        model.compile(optimizer=keras.optimizers.Adam(1e-3, clipnorm=1.0), loss='mse')
        return model
    
    def act(self, state, add_noise=True):
        """Action selection with smoothing"""
        state = np.array(state).reshape(1, -1).astype(np.float32)
        raw_action = self.actor.predict(state, verbose=0)[0]
        
        # Action smoothing
        smoothed_action = (self.action_smoothing * self.prev_raw_action + 
                          (1 - self.action_smoothing) * raw_action)
        
        if add_noise:
            noise = np.random.normal(0, self.noise_scale, self.action_dim)
            smoothed_action = np.clip(smoothed_action + noise, -1, 1)
        
        # Limit action change
        max_change = 0.3
        action_change = smoothed_action[0] - self.prev_raw_action
        if abs(action_change) > max_change:
            direction = 1 if action_change > 0 else -1
            smoothed_action[0] = self.prev_raw_action + direction * max_change
        
        self.prev_raw_action = smoothed_action[0]
        return smoothed_action[0]
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience with filtering"""
        
        self.buffer.append((state, action, reward, next_state, done))
    
    def train(self):
        """Train networks"""
        if len(self.buffer) < self.batch_size:
            return
        
        # Sample batch
        batch = random.sample(self.buffer, self.batch_size)
        states = np.array([e[0] for e in batch]).astype(np.float32)
        actions = np.array([e[1] for e in batch]).reshape(-1, 1).astype(np.float32)
        rewards = np.array([e[2] for e in batch]).astype(np.float32)
        next_states = np.array([e[3] for e in batch]).astype(np.float32)
        dones = np.array([e[4] for e in batch]).astype(np.float32)
        
        # Reward clipping
        rewards = np.clip(rewards, -50, 50)
        
        # Train critic
        target_actions = self.target_actor.predict(next_states, verbose=0)
        target_q = self.target_critic.predict([next_states, target_actions], verbose=0).flatten()
        target_q = np.clip(target_q, -100, 100)
        targets = rewards + self.gamma * target_q * (1 - dones)
        targets = np.clip(targets, -100, 100)
        
        self.critic.train_on_batch([states, actions], targets)
        
        # Train actor
        with tf.GradientTape() as tape:
            states_tensor = tf.convert_to_tensor(states, dtype=tf.float32)
            actions_pred = self.actor(states_tensor, training=True)
            q_values = self.critic([states_tensor, actions_pred], training=True)
            actor_loss = -tf.reduce_mean(q_values)
        
        if not tf.math.is_nan(actor_loss):
            actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
            actor_grads = [tf.clip_by_norm(g, 0.5) if g is not None else g for g in actor_grads]
            self.actor.optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        
        # Update targets
        self.update_targets()
    
    def update_targets(self, tau=None):
        """Soft update target networks"""
        if tau is None:
            tau = self.tau
        
        for target, source in zip(self.target_actor.weights, self.actor.weights):
            target.assign(tau * source + (1 - tau) * target)
        for target, source in zip(self.target_critic.weights, self.critic.weights):
            target.assign(tau * source + (1 - tau) * target)
    
    def decay_noise(self):
        """Decay exploration noise"""
        self.noise_scale *= self.noise_decay
        self.noise_scale = max(self.noise_scale, self.min_noise)
    
    def reset_action_history(self):
        """Reset action history"""
        self.prev_raw_action = 0.0

# ===================== 5. Simplified Visualization Functions =====================
def visualize_multi_road_training_stats(training_stats, road_files):
    """Visualize training statistics across multiple roads"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Road usage count
    road_names = [f"Road {i+1}" for i in range(len(road_files))]
    usage_counts = training_stats['road_usage_count']
    usage_percentages = training_stats['road_usage_percentage']
    
    bars1 = ax1.bar(road_names, usage_counts, color='skyblue', alpha=0.7)
    ax1.set_ylabel('Episode Count')
    ax1.set_title('Road Usage Distribution (Episodes)')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, count in zip(bars1, usage_counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}', ha='center', va='bottom')
    
    # Road usage percentage
    bars2 = ax2.bar(road_names, usage_percentages, color='lightcoral', alpha=0.7)
    ax2.set_ylabel('Usage Percentage (%)')
    ax2.set_title('Road Usage Distribution (Percentage)')
    ax2.grid(True, alpha=0.3)
    
    # Add percentage labels
    for bar, pct in zip(bars2, usage_percentages):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{pct:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed statistics
    print(f"\n📊 Multi-Road Training Statistics:")
    print(f"   Total Episodes: {training_stats['total_episodes']}")
    for i, (count, pct) in enumerate(zip(usage_counts, usage_percentages)):
        filename = road_files[i][:30] + "..." if len(road_files[i]) > 30 else road_files[i]
        print(f"   Road {i+1} ({filename}): {count} episodes ({pct:.1f}%)")

def visualize_training_rewards_with_roads(episode_rewards, road_indices):
    """Visualize training rewards with separate lane plots for each road"""
    num_roads = max(road_indices) + 1
    
    # Create subplots 
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Training Progress: Episode Rewards by Road (5-second Time Window)', fontsize=16)
    
    # Flatten axes for easier indexing
    axes = axes.flatten()
    
    # Colors for each road
    colors = plt.cm.Set1(np.linspace(0, 1, num_roads))
    
    # Separate episodes and rewards for each road
    road_data = [[] for _ in range(num_roads)]
    road_episodes = [[] for _ in range(num_roads)]
    
    for episode, (reward, road_idx) in enumerate(zip(episode_rewards, road_indices)):
        road_data[road_idx].append(reward)
        road_episodes[road_idx].append(episode)
    
    # Plot each road in its own subplot
    for road_idx in range(num_roads):
        ax = axes[road_idx]
        
        if len(road_data[road_idx]) > 0:
            episodes = road_episodes[road_idx]
            rewards = road_data[road_idx]
            
            # Main line plot
            ax.plot(episodes, rewards, color=colors[road_idx], linewidth=1.5, 
                   alpha=0.7, label=f'Road {road_idx+1} Episodes')
            
            # Add moving average if enough data points
            window_size = min(10, len(rewards) // 5)
            if window_size > 1 and len(rewards) > window_size:
                moving_avg = []
                smoothed_episodes = []
                for i in range(len(rewards)):
                    if i >= window_size - 1:
                        start_idx = max(0, i - window_size + 1)
                        moving_avg.append(np.mean(rewards[start_idx:i+1]))
                        smoothed_episodes.append(episodes[i])
                
                ax.plot(smoothed_episodes, moving_avg, color=colors[road_idx], 
                       linewidth=3, alpha=0.9, label=f'Moving Average ({window_size})')
            
            # Fill area under the curve for better visualization
            ax.fill_between(episodes, rewards, alpha=0.2, color=colors[road_idx])
            
            # Statistics for this road
            mean_reward = np.mean(rewards)
            std_reward = np.std(rewards)
            min_reward = np.min(rewards)
            max_reward = np.max(rewards)
            
            # Add horizontal line for mean
            ax.axhline(mean_reward, color=colors[road_idx], linestyle='--', 
                      alpha=0.8, linewidth=2, label=f'Mean: {mean_reward:.1f}')
            
            # Set labels and title
            ax.set_title(f'Road {road_idx+1}\n'
                        f'Episodes: {len(rewards)} | Mean: {mean_reward:.1f}±{std_reward:.1f}\n'
                        f'Range: [{min_reward:.1f}, {max_reward:.1f}]', 
                        fontsize=12, pad=10)
            
            ax.set_xlabel('Episode')
            ax.set_ylabel('Reward')
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(True, alpha=0.3)
            
            # Improve y-axis range
            y_margin = (max_reward - min_reward) * 0.1
            ax.set_ylim(min_reward - y_margin, max_reward + y_margin)
            
        else:
            # If no data for this road
            ax.text(0.5, 0.5, f'Road {road_idx+1}\nNo Data', 
                   ha='center', va='center', transform=ax.transAxes, 
                   fontsize=14, color='gray')
            ax.set_title(f'Road {road_idx+1} - No Episodes')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Reward')
    
    # Hide extra subplots if fewer than 6 roads
    for road_idx in range(num_roads, 6):
        axes[road_idx].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print(f"\n📊 Training Summary by Road:")
    print("-" * 60)
    for road_idx in range(num_roads):
        if len(road_data[road_idx]) > 0:
            episodes_count = len(road_data[road_idx])
            mean_reward = np.mean(road_data[road_idx])
            std_reward = np.std(road_data[road_idx])
            latest_reward = road_data[road_idx][-1] if road_data[road_idx] else 0
            
            # Calculate trend (improvement over time)
            if episodes_count > 10:
                first_half = np.mean(road_data[road_idx][:episodes_count//2])
                second_half = np.mean(road_data[road_idx][episodes_count//2:])
                trend = "📈 Improving" if second_half > first_half else "📉 Declining"
                trend_value = second_half - first_half
            else:
                trend = "📊 Insufficient data"
                trend_value = 0
            
            print(f"Road {road_idx+1}: {episodes_count:3d} episodes | "
                  f"Mean: {mean_reward:6.1f}±{std_reward:5.1f} | "
                  f"Latest: {latest_reward:6.1f} | "
                  f"{trend} ({trend_value:+.1f})")
        else:
            print(f"Road {road_idx+1}:   0 episodes | No data available")
    
    print("-" * 60)



# ===================== 6. Main Execution =====================
if __name__ == "__main__":
    print("🛣️ Multi-Road DDPG Training System (5-Second Time Window + Simplified Reward)")
    print("=" * 80)
    
    # Road data folder path
    road_folder = xx
    
    try:
        # Load multiple roads
        multi_road = MultiRoadDataProcessor(road_folder)
        
        # Create environment and agent
        env = MultiRoadDrivingEnvironment(multi_road)
        
        # 重新调用 reset() 拿到带预测特征的新 state
        test_state, _ = env.reset()

        # 用“新长度”直接初始化 agent
        agent = DDPGAgent(len(test_state))
        print(f"🔍 State dimension (with pred feats): {len(test_state)}")
        
        # Apply fixes
        print("🔧 Applying system improvements...")
        agent.buffer.clear()
        # agent.noise_scale = 0.02
        agent.update_targets(tau=0.1)
        print("✅ Improvements applied:")
        print("   - 5-second time-based preview window")
        print("   - Simplified 3-component reward system")
        print("   - Current + predicted vertical comfort")
        print("   - Running RMS for real-time comfort monitoring")
        
        # Training with configurable road selection strategy
        episode_rewards = []
        road_indices = []
        
        # Road selection strategy options
        print(f"\n🎛️ Available road selection strategies:")
        print(f"   1. 'sequential' - 顺序循环使用所有路面")
        print(f"   2. 'balanced' - 总是选择使用次数最少的路面") 
        print(f"   3. 'single' - 只使用第一条路面（单路面训练）")
        print(f"   4. 'difficulty' - 按难度递进（简单→中等→困难）")
        print(f"   5. 'random' - 随机选择路面")
        
        # Select road strategy here
        ROAD_STRATEGY = "single"  # 👈 Change this to modify strategy
        
        print(f"🎯 Selected strategy: {ROAD_STRATEGY}")
        print(f"🎯 Training agent on {multi_road.get_num_roads()} roads...")
        episode_durations = []      # 记录每集耗时
        est_window = 5              # 取最近 5 集平均来估计
        total_episodes_planned = 120   # 训练上面 range 的终止值，保持一致
        
        for episode in range(total_episodes_planned):
            start_time = time.time()
            state, road_index = env.reset(episode_num=episode, road_strategy=ROAD_STRATEGY)
            road_indices.append(road_index)
            agent.reset_action_history()
            episode_reward = 0
            done = False
            
            # Monitor first few episodes
            if episode < 5:
                print(f"\n📋 Episode {episode} (Road {road_index+1}):")
                print(f"   Initial: pos={env.position}, speed={env.speed}")
                print(f"   Initial window: {env.get_current_window_length():.1f}m")
                print(f"   Noise: {agent.noise_scale:.3f}")
            
            while not done:
                action = agent.act(state, add_noise=True)
                next_state, reward, done, info = env.step(action)
                agent.remember(state, action, reward, next_state, done)
                
                # Training
                if len(agent.buffer) >= agent.batch_size and env.episode_steps % 3 == 0:
                    agent.train()
                
                episode_reward += reward
                state = next_state

            agent.decay_noise()
            print(f"[EP {episode}]  ΣEff={sum(env.ep_r_eff):6.1f}  "
                  f"ΣVert={sum(env.ep_r_vert):6.1f}  "
                  f"ΣLong={sum(env.ep_r_long):6.1f}  "
                  f"ΣTotal={sum(env.ep_r_total):6.1f}")
            episode_rewards.append(episode_reward)
            
            # Progress reporting
            stats = env.get_training_statistics()
            print(f"Episode {episode}: Reward={episode_reward:.1f}, "
                    f"Road={road_index+1}, Steps={env.episode_steps}, "
                    f"Noise={agent.noise_scale:.3f}")
            

            # ----------- 统计本集耗时 & 预计剩余 -----------
            ep_time = time.time() - start_time
            episode_durations.append(ep_time)

            # 最近 est_window 集平均
            recent = episode_durations[-est_window:]
            avg_recent = sum(recent) / len(recent)

            episodes_left = total_episodes_planned - (episode + 1)
            eta_seconds = episodes_left * avg_recent
            eta = timedelta(seconds=int(eta_seconds))

            print(f"⏱  Episode time: {ep_time:.1f}s"
                f" | Avg({len(recent)})={avg_recent:.1f}s"
                f" | ETA ≈ {str(eta)}")
            # ----------------------------------------------
            
            if episode > 0 and episode % 50 == 0:
                print(f"   📊 Road usage: {stats['road_usage_count']}")

        # Save model
        print("\n" + "="*80)
        print("💾 Saving multi-road trained model...")
        
        try:
            agent.actor.save_weights("multi_road_ddpg_actor_5sec.weights.h5")
            agent.critic.save_weights("multi_road_ddpg_critic_5sec.weights.h5")
            
            # Save enhanced training info
            training_info = {
                'episode_rewards': episode_rewards,
                'road_indices': road_indices,
                'road_files': multi_road.road_files,
                'training_stats': env.get_training_statistics(),
                'final_noise_scale': agent.noise_scale,
                'buffer_size': len(agent.buffer),
                'total_episodes': len(episode_rewards),
                'best_reward': max(episode_rewards),
                'final_reward': episode_rewards[-1],
                'num_roads': multi_road.get_num_roads(),
                'road_strategy': ROAD_STRATEGY,
                'preview_time': env.preview_time,  # New field
                'reward_components': ['driving_efficiency', 'vertical_comfort', 'longitudinal_comfort'],  # New field
                'comfort_prediction': '5_second_detailed_sampling'  # New field
            }
            
            with open('multi_road_training_info_5sec.pkl', 'wb') as f:
                pickle.dump(training_info, f)
            
            print("✅ Model saved successfully!")
            print(f"   📁 Actor: multi_road_ddpg_actor_10sec.weights.h5")
            print(f"   📁 Critic: multi_road_ddpg_critic_10sec.weights.h5") 
            print(f"   📁 Training info: multi_road_training_info_10sec.pkl")
            print(f"   🔧 5-second time-based preview window")
            print(f"   🔧 Simplified 3-component reward system")
            
        except Exception as e:
            print(f"❌ Save failed: {e}")
        
        # Simplified visualizations
        print("\n=== Multi-Road Training Analysis (5-Second Window) ===")
        
        # 1. Training statistics
        print("\n📊 Training statistics...")
        final_stats = env.get_training_statistics()
        visualize_multi_road_training_stats(final_stats, multi_road.road_files)
        
        # 2. Enhanced reward visualization
        print("\n📈 Training rewards with road tracking...")
        visualize_training_rewards_with_roads(episode_rewards, road_indices)
        
        print("\n✅ Multi-road training complete!")
        print(f"📈 Trained on {multi_road.get_num_roads()} different roads")
        print(f"🎯 Total episodes: {len(episode_rewards)}")
        print(f"🏆 Best reward: {max(episode_rewards):.1f}")
        print(f"🔧 5-second time-based preview window")
        print(f"🔧 Simplified reward: efficiency + vertical + longitudinal comfort")
        print(f"🔧 Current + predicted comfort integration")
        
    except FileNotFoundError:
        print(f"❌ Error: Road folder not found or no .npy files in folder.")
        print("Please check the road_folder path.")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()