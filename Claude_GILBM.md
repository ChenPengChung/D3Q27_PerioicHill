# D3Q19 ISLBM → GILBM 實作計劃
**Periodic Hill with Curvilinear Coordinates**

**最後更新**: 2026-02-17
**狀態**: 準備實作
**核心方法**: Imamura 2005 GILBM
**預計時程**: 4-6 週（穩步推進）

---

## Context: 為什麼要改造

### 問題背景

現有的 D3Q19 ISLBM Periodic Hill 求解器（位於 `/Users/yetianzhong/Desktop/4.GitHub/D3Q27_PeriodicHill/`）在 **Cartesian 座標系**中運行，面臨以下挑戰：

1. **複雜的 BFL 邊界條件**
   - 12 個邊界判斷函數處理山丘表面
   - 56+ 個插值權重陣列需要預計算和存儲
   - 壁面附近精度下降

2. **非均勻網格的複雜性**
   - Z 方向使用 tanh 拉伸，但仍在物理空間插值
   - 插值權重計算繁瑣，記憶體開銷大

3. **精度需求**
   - 紊流分析需要 6 階空間精度
   - 現有 6 階 Lagrange 插值已實作，但在 Cartesian 框架下複雜

### 解決方案：GILBM

**GILBM** (Generalized Interpolation-Supplemented LBM, Imamura et al. 2005) 通過**座標變換**將問題簡化：

- **物理空間** (x,y,z): 非均勻、複雜邊界
- **計算空間** (η,ξ,ζ): **均勻網格**、貼體邊界

**核心優勢**：
1. ✅ 插值在均勻的 (η,ξ,ζ) 空間進行，簡化計算
2. ✅ 壁面邊界簡化為 bounce-back（網格貼體）
3. ✅ 消除 BFL 的所有複雜性
4. ✅ 保留 collision-streaming 範式（與現有 MRT 兼容）
5. ✅ 可加入 local time step 加速穩態收斂 70-80%

---

## 核心數學公式

### 1. 座標變換

**物理空間 → 計算空間**：
```
η(x) = x / LX                              (展向，均勻)
ξ(y) = y / LY                              (主流向，均勻)
ζ(y,z) = [z - H(y)] / [LZ - H(y)]         (壁面法向，貼體)
```

其中 `H(y)` 是山丘高度函數（12 段三次多項式）。

### 2. 度量項 (Metric Terms)

```
ζy = ∂ζ/∂y = -H'(y) / [LZ - H(y)]
ζz = ∂ζ/∂z = 1 / [LZ - H(y)]
J = LZ - H(y)  (Jacobian)
```

**關鍵**: `H'(y)` 需要解析求導（12 段導數）。

### 3. 逆變速度 (Contravariant Velocities)

對於 D3Q19 的每個方向 α，計算計算空間的速度：

```
ẽα_η = eα_x / LX                          (常數)
ẽα_ξ = eα_y / LY                          (常數)
ẽα_ζ = [eα_z + ζy·eα_y] · ζz             (空間變化！)
     = [eα_z - H'(y)·eα_y/(LZ-H)] / (LZ-H)
```

### 4. RK2 上風點追蹤（關鍵！）

Imamura 2005 強調：**必須用 RK2，一階 Euler 不足**。

```cuda
// Step 1: Euler 預測到中點
η_mid  = η - 0.5*dt*ẽα_η
ξ_mid  = ξ - 0.5*dt*ẽα_ξ
ζ_mid  = ζ - 0.5*dt*ẽα_ζ(η,ξ,ζ)

// Step 2: 在中點重新計算逆變速度
ẽα_ζ_mid = ComputeAtMidpoint(ξ_mid, ζ_mid)

// Step 3: RK2 校正
η_up  = η - dt*ẽα_η
ξ_up  = ξ - dt*ẽα_ξ
ζ_up  = ζ - dt*ẽα_ζ_mid  // 使用中點速度！
```

### 5. GILBM Streaming-Collision

```cuda
// 1. 計算當前點的逆變速度
ContravariantVelocities(metrics, alpha, ẽη, ẽξ, ẽζ);

// 2. RK2 計算上風點
RK2_UpwindPosition(η, ξ, ζ, ẽη, ẽξ, ẽζ, dt, metrics_field, η_up, ξ_up, ζ_up);

// 3. 在均勻計算空間插值（6階 Lagrange）
F_in[α] = Interpolate6thOrder(f_old[α], η_up, ξ_up, ζ_up);

// 4. MRT 碰撞（保持不變）
MRT_Collision(F_in, F_out, rho, u, v, w);

// 5. 寫入新值
f_new[α] = F_out[α];
```

---

## 實作階段

### Phase 0: 準備工作 (1-2 天)

**目標**: 實作 `HillFunctionDerivative`，建立基準數據。

**任務**:
1. 創建 `model_derivative.h`
   - 解析求導 12 段三次多項式
   - `H'(y) = 28a₁ + 56a₂y + 2352a₃y²` (對每段)
   - CPU + GPU 版本

2. 單元測試
   ```cuda
   double dH_analytic = HillFunctionDerivative(1.5);
   double dH_numeric = [HillFunction(1.5+ε) - HillFunction(1.5-ε)] / 2ε;
   assert(|dH_analytic - dH_numeric| < 1e-4);
   ```

3. 運行現有 ISLBM 至收斂，保存基準數據
   - `baseline_islbm/velocity_*.vtk`
   - `baseline_islbm/checkrho.dat`
   - 記錄分離泡位置 (x_sep, x_reatt)

**交付物**: `model_derivative.h`, 基準數據

---

### Phase 1: GILBM 框架 (2階插值) (7-10 天)

**目標**: 實作完整 GILBM 框架，先用 2 階插值驗證正確性。

#### 任務 1.1: 座標變換與度量項 (2天)

**新建檔案**: `gilbm/gilbm_transform.h`

```cuda
struct MetricTerms {
    double zeta_y;  // -H'(y)/(LZ-H(y))
    double zeta_z;  // 1/(LZ-H(y))
    double J;       // LZ-H(y)
};

__global__ void ComputeMetricTerms(
    MetricTerms *metrics,
    double *y_coords,
    double *z_coords,
    double *dHdy,  // 預計算的 H'(y)
    int NX, int NY, int NZ
);

__device__ void ContravariantVelocities(
    const MetricTerms &metric,
    const int alpha,  // 0-18
    double &e_tilde_eta,
    double &e_tilde_xi,
    double &e_tilde_zeta
);
```

**驗證**: 手算幾個點的度量項，對比 GPU 輸出。

#### 任務 1.2: RK2 上風點追蹤 (2-3天)

**新建檔案**: `gilbm/gilbm_rk2_upwind.h`

```cuda
__device__ void RK2_UpwindPosition(
    const double eta, const double xi, const double zeta,
    const double e_tilde_eta,
    const double e_tilde_xi,
    const double e_tilde_zeta,
    const double dt,
    const MetricTerms *metrics_field,
    const int i, const int j, const int k,
    const int NX, const int NY, const int NZ,
    double &eta_up, double &xi_up, double &zeta_up
);
```

**關鍵實作**:
- 中點度量項插值（2D 雙線性）
- 週期性邊界處理 (η, ξ 方向)
- 壁面截斷 (ζ ∈ [0,1])

**驗證**: 對比 RK2 vs Euler（精度測試）。

#### 任務 1.3: 2階插值 (1天)

**新建檔案**: `gilbm/interpolationGILBM_order2.h`

```cuda
__device__ double Interpolate_Order2_3D(
    double *f_field,
    double eta_up, double xi_up, double zeta_up,
    int NX, int NY, int NZ
);
```

三線性插值（8 個點）。

#### 任務 1.4: 壁面邊界條件 (2天)

**新建檔案**: `gilbm/boundary_conditions.h`

根據您的需求，實作**兩種方法**並支持切換：

```cuda
// 方法 1: Half-Way Bounce Back (主要)
__device__ void HalfwayBounceBack(
    double *f_in,
    int k,
    int NZ6
) {
    if (k == 3) {  // 下壁面
        // f5 ↔ f6 (Z 向上 ↔ Z 向下)
        double temp = f_in[5];
        f_in[5] = f_in[6];
        f_in[6] = temp;
        // ... 其他方向
    }
    // 類似處理上壁面
}

// 方法 2: Chapman-Enskog 展開 (可選)
__device__ void ChapmanEnskogBC(
    double *f_in,
    double *f_eq,
    double *velocity_gradient,  // ∂u/∂z
    double omega,
    double dt,
    int k
) {
    // fα|wall = fα^eq [1 - ωΔt·(3Ui,aUi,b/c² - δab)·∂ua/∂xb]
    // ... 實作
}

// 切換開關
#define BOUNDARY_METHOD 1  // 1: HalfwayBB, 2: ChapmanEnskog
```

**驗證**: 平板 Poiseuille 流（有解析解）。

#### 任務 1.5: 整合到 evolution kernel (2天)

**新建檔案**: `evolution_gilbm.h`

```cuda
__global__ void stream_collide_GILBM_Order2(
    double *f0_old, ..., double *f18_old,
    double *f0_new, ..., double *f18_new,
    MetricTerms *metrics,
    double *rho_d, double *u, double *v, double *w,
    double *Force, double *rho_modify
) {
    // ... 計算當前點的 (η, ξ, ζ)

    double F_in[19];
    F_in[0] = f0_old[index];  // 靜止方向不變

    for (int alpha = 1; alpha < 19; alpha++) {
        // 1. 逆變速度
        ContravariantVelocities(metrics[index], alpha, e_eta, e_xi, e_zeta);

        // 2. RK2 上風點
        RK2_UpwindPosition(..., eta_up, xi_up, zeta_up);

        // 3. 2階插值
        F_in[alpha] = Interpolate_Order2_3D(...);
    }

    // 4. 壁面 BC
    #if BOUNDARY_METHOD == 1
        HalfwayBounceBack(F_in, k, NZ6);
    #elif BOUNDARY_METHOD == 2
        ChapmanEnskogBC(F_in, ...);
    #endif

    // 5. MRT 碰撞（複製現有代碼）
    MRT_Collision(F_in, F_out, rho, u, v, w);

    // 6. 寫入
    for (int alpha = 0; alpha < 19; alpha++) {
        f_alpha_new[alpha][index] = F_out[alpha];
    }
}
```

#### 任務 1.6: 修改 main.cu (1天)

```cuda
// 使用條件編譯保留 ISLBM
#define USE_GILBM 1  // 0: ISLBM, 1: GILBM

#if USE_GILBM
    // 預計算度量項
    ComputeHillDerivative<<<grid, block>>>(dHdy_d, y_d, NY6);
    ComputeMetricTerms<<<grid, block>>>(metrics_d, y_d, z_d, dHdy_d, NX6, NY6, NZ6);

    // 主循環
    for (step = 0; step < loop; step++) {
        Launch_CollisionStreaming_GILBM(ft, fd);
        // ...
    }
#else
    // 原有 ISLBM 代碼
    Launch_CollisionStreaming(ft, fd);
#endif
```

#### 任務 1.7: Phase 1 驗證 (1天)

**測試指標**:
1. ✅ 質量守恆: `|ρ_avg - 1.0| < 1e-6`
2. ✅ 程式不崩潰，運行至 50,000 步
3. ✅ 速度場定性正確（有分離泡）
4. ✅ 與 ISLBM 基準相對誤差 < 20%（2階插值精度有限）

**交付物**: 可運行的 GILBM 框架（2階插值版）

---

### Phase 2: 升級到 6階插值 (4-5 天)

**目標**: 達到與現有 ISLBM 相同或更高的精度。

#### 任務 2.1: 6階 Lagrange 插值 (2天)

**新建檔案**: `gilbm/interpolationGILBM_order6.h`

```cuda
__device__ void Lagrange6Weights(double s, double w[7]) {
    // s ∈ [-3, +3]，7-point stencil
    double s_nodes[7] = {-3, -2, -1, 0, 1, 2, 3};
    for (int i = 0; i < 7; i++) {
        w[i] = 1.0;
        for (int j = 0; j < 7; j++) {
            if (i != j) {
                w[i] *= (s - s_nodes[j]) / (s_nodes[i] - s_nodes[j]);
            }
        }
    }
}

__device__ double Interpolate_Order6_3D(
    double *f_field,
    double eta_up, double xi_up, double zeta_up,
    int NX, int NY, int NZ
) {
    // 7×7×7 = 343 點插值
    // 分離式: wi[7] × wj[7] × wk[7]
    // ...
}
```

#### 任務 2.2: 更新 evolution kernel (0.5天)

將 `Interpolate_Order2_3D` 替換為 `Interpolate_Order6_3D`。

#### 任務 2.3: 性能優化 (1-2天)

**策略**:
- Shared memory 緩存局部數據
- Texture memory（GPU 特性）
- 預計算 Lagrange 權重表

#### 任務 2.4: 精度驗證 (網格收斂性測試) (1-2天)

**測試方案**:
```
粗網格: 16×64×32
中網格: 32×128×64  (原始)
細網格: 64×256×128

計算精度階數:
p = log(E_coarse - E_medium) / log(E_medium - E_fine) / log(2)

預期: p ≈ 6
```

**交付物**: 6階精度 GILBM，與 ISLBM 精度相當或更優

---

### Phase 3: Local Time Step 加速 (3-5 天)

**目標**: 穩態收斂加速 70-80%。

#### 任務 3.1: 實作空間變化的時間步 (2天)

**新建檔案**: `gilbm/gilbm_local_timestep.h`

```cuda
__global__ void ComputeLocalTimeStep(
    double *dt_local,
    MetricTerms *metrics,
    int NX, int NY, int NZ
) {
    // 計算每個格點的最大逆變速度
    double u_max_contravariant = 0.0;
    for (int alpha = 0; alpha < 19; alpha++) {
        ContravariantVelocities(metrics[index], alpha, e_eta, e_xi, e_zeta);
        double u_contra = sqrt(e_eta*e_eta + e_xi*e_xi + e_zeta*e_zeta);
        u_max_contravariant = fmax(u_max_contravariant, u_contra);
    }

    // CFL 條件
    double dx_min = min(dx_eta, min(dx_xi, dx_zeta));
    dt_local[index] = CFL * dx_min / (u_max_contravariant + 1e-10);
}
```

#### 任務 3.2: Re-estimation 機制 (1天)

當相鄰格點時間步不同時，需修正非平衡項（Imamura Eq. 36）：

```cuda
// 從上風格點 B 獲取 f 時
f_tilde = f_eq[B] + (f[B] - f_eq[B]) * (omega_A * dt_A) / (omega_B * dt_B);
```

#### 任務 3.3: 驗證與對比 (1-2天)

**測試**:
- Global time step: 500,000 iterations → steady
- Local time step: ~100,000 iterations → steady (5× speedup)
- 穩態解一致性: 相對誤差 < 1%

**交付物**: 完整 GILBM（含 local time step 加速）

---

## 關鍵檔案清單

### 新建檔案

| 檔案 | Phase | 功能 |
|------|-------|------|
| `model_derivative.h` | 0 | HillFunctionDerivative (12段導數) |
| `gilbm/gilbm_transform.h` | 1 | 度量項、逆變速度 |
| `gilbm/gilbm_rk2_upwind.h` | 1 | RK2 上風點追蹤 |
| `gilbm/interpolationGILBM_order2.h` | 1 | 2階插值 |
| `gilbm/boundary_conditions.h` | 1 | Half-Way BB + Chapman-Enskog |
| `evolution_gilbm.h` | 1 | GILBM streaming-collision kernel |
| `gilbm/interpolationGILBM_order6.h` | 2 | 6階插值 |
| `gilbm/gilbm_local_timestep.h` | 3 | 局部時間步 |

### 修改檔案

| 檔案 | 修改內容 |
|------|---------|
| `main.cu` | 添加 `#ifdef USE_GILBM`，調用 GILBM kernels |
| `variables.h` | 添加 GILBM 開關宏 |
| `memory.h` | 分配度量項、H'(y) 記憶體 |
| `initialization.h` | 註釋 BFL 初始化 |

### 保留不變

- `MRT_Matrix.h`, `MRT_Process.h` (MRT 碰撞)
- `communication.h` (MPI)
- `monitor.h`, `statistics.h` (輸出)
- `evolution.h` (保留作為 ISLBM 基準)

---

## 驗證策略

### 分層驗證

1. **單元測試** (每個函數)
   - `HillFunctionDerivative` vs 數值差分
   - 度量項手算驗證
   - RK2 vs Euler 精度對比

2. **子系統測試**
   - 單方向 streaming 測試
   - 簡單流場驗證 (Poiseuille 流)

3. **系統測試**
   - Periodic Hill Re=200
   - 與 ISLBM 基準對比
   - 與文獻數據對比 (Mellen 2000)

### 關鍵監測指標

| 指標 | 頻率 | 目標 |
|------|------|------|
| 質量守恆 | 每步 | `\|ρ_avg - 1.0\| < 1e-6` |
| 動量守恆 | 每 1000 步 | 殘差 < 1e-4 |
| 壁面剪應力 | 收斂後 | 與文獻值誤差 < 10% |
| 網格收斂性 | Phase 2 | p ≥ 5 |

### 對比基準

1. **與 ISLBM 對比** (Phase 1-2)
   - 分離泡位置 (x_sep, x_reatt)
   - 速度剖面 U(z) at x=0.5, 2.0, 4.5
   - 壁面剪應力分佈 τ_wall(x)

2. **與文獻對比** (Phase 2)
   - Mellen et al. (2000) DNS 數據
   - Breuer et al. (2009) 實驗數據

---

## 風險管理

### 風險 1: RK2 不穩定

**症狀**: 質量不守恆、NaN

**降級方案**:
```cuda
#define USE_EULER_UPWIND 1  // 回退到一階 Euler
#define CFL 0.3              // 減小時間步
```

### 風險 2: 插值精度不足

**症狀**: 網格收斂性 p < 4

**降級方案**: 保留 Phase 1 的 2 階插值版本，或使用 4 階插值折衷。

### 風險 3: 壁面邊界精度不足

**症狀**: 壁面剪應力誤差 > 20%

**降級方案**: 切換到 Chapman-Enskog 邊界條件：
```cuda
#define BOUNDARY_METHOD 2
```

### 風險 4: 性能下降

**症狀**: GILBM > 2× ISLBM 運行時間

**優化措施**:
- Shared memory
- Texture memory
- 降低插值階數

---

## 時程規劃 (4-6 週，穩步推進)

### 第 1 週
- Phase 0: 準備工作 (1-2 天)
- Phase 1.1-1.2: 座標變換 + RK2 (4-5 天)

### 第 2 週
- Phase 1.3-1.5: 插值 + 邊界 + 整合 (5-6 天)

### 第 3 週
- Phase 1.6-1.7: 驗證測試 (2 天)
- Phase 2.1-2.2: 6階插值實作 (3 天)

### 第 4 週
- Phase 2.3-2.4: 性能優化 + 精度驗證 (5-6 天)

### 第 5-6 週
- Phase 3: Local time step (3-5 天)
- 最終測試與文檔 (3-5 天)

**緩衝時間**: 預留 20%（約 1 週）處理意外問題。

---

## 成功標準

### Phase 1 (最小可行產品)
- ✅ 程式穩定運行至 50,000 步
- ✅ 質量守恆 < 1e-6
- ✅ 速度場定性正確
- ✅ 與 ISLBM 誤差 < 20%

### Phase 2 (生產級產品)
- ✅ 網格收斂性 p ≥ 5
- ✅ 與 ISLBM 誤差 < 5%
- ✅ 與文獻誤差 < 10%
- ✅ 運行時間 ≤ 2× ISLBM

### Phase 3 (優化版)
- ✅ 穩態收斂加速 ≥ 3×
- ✅ 穩態解一致性 < 1%

---

## 參考文獻

1. **Imamura, T., et al. (2005)**. "Acceleration of steady-state lattice Boltzmann simulations on non-uniform mesh using local time step method". *Journal of Computational Physics*, 202(2), 645-663.
   - 📄 `/Users/yetianzhong/Desktop/4.GitHub/LBM-PaperReView/曲線坐標系的處理/5.Acceleration...pdf`

2. **Mellen, C. P., et al. (2000)**. "Large Eddy Simulation of the flow over periodic hills". *Proc. ERCOFTAC Workshop on DNS and LES*.
   - Re=200 DNS 數據（驗證基準）

3. **Breuer, M., et al. (2009)**. "Flow over periodic hills - Numerical and experimental study in a wide range of Reynolds numbers". *Computers & Fluids*, 38, 433-457.

---

## 下一步行動

1. **確認計劃**: 用戶批准後開始實作
2. **創建開發分支**: `git checkout -b feature/gilbm-implementation`
3. **Phase 0 啟動**: 實作 `HillFunctionDerivative`
4. **建立測試框架**: 單元測試腳本

**準備開始實作！**
