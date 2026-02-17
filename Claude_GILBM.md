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
   - 56+ 個插值權重陣列需要預計算和存儲（~3.4 MB）
   - 壁面附近精度下降

2. **非均勻網格的複雜性**
   - Z 方向使用 tanh 拉伸：`z_h[j*NZ6+k] = tanhFunction(...) + H(y_j)`
   - 插值權重計算繁瑣，記憶體開銷大（Xi 方向佔 2.43 MB）

3. **精度需求**
   - 紊流分析需要 6 階空間精度
   - 現有 6 階 Lagrange 插值已實作，但在 Cartesian 框架下複雜

### 解決方案：GILBM

**GILBM** (Generalized Interpolation-Supplemented LBM, Imamura et al. 2005) 通過**座標變換**將問題簡化：

- **物理空間** (x,y,z): 非均勻、複雜邊界
- **計算空間** (η,ξ,ζ): **均勻網格**、貼體邊界

**核心優勢**：
1. ✅ 插值在均勻的 (η,ξ,ζ) 空間進行，可用**全局權重表**（記憶體減少 130×）
2. ✅ 壁面邊界簡化（網格貼體），但需用 **Non-Equilibrium Extrapolation**（非 Half-Way BB）
3. ✅ 消除 BFL 的複雜幾何判斷
4. ✅ 保留 collision-streaming 範式（與現有 MRT 兼容）
5. ✅ 可加入 local time step 加速穩態收斂 70-80%

---

## 🔑 關鍵技術修正（基於深度代碼分析）

### **修正 1：度量項必須用離散 Jacobian**

**原計劃錯誤**：只對 HillFunction 解析求導 H'(y)

**實際情況**（從 `initialization.h` 發現）：
```cpp
z_h[j*NZ6+k] = tanhFunction(total, minSize, a, k-3, NZ6-7) + HillFunction(y_j)
```
- 每個格點 (j,k) 的物理座標 z 都不同
- tanh 拉伸使座標映射變成**隱函數**

**正確方法**：
```cuda
∂ζ/∂z ≈ (xi_h[k+1] - xi_h[k-1]) / (z_h[j*NZ6+k+1] - z_h[j*NZ6+k-1])
∂ζ/∂y ≈ 數值差分計算（考慮整個映射）
```

**影響**：Phase 0 改為實作 `discrete_jacobian.h`，不需要 `model_derivative.h`

---

### **修正 2：插值權重可用全局表優化**

**原計劃不精確**：說"無需預計算"，實際上**仍需優化**

**實際情況**（從 `memory.h` 發現）：
- 現有 ISLBM：56+ 個陣列，每個 NYD6 × NZ6 × 7 doubles
- 總記憶體：~3.4 MB（Xi 方向佔 2.43 MB）

**GILBM 優勢**：
- 計算空間 (η,ξ,ζ) **均勻** → 分數位置可預計算
- **全局權重表**：256 個離散位置 × 7 權重 × 3 方向 = ~14 KB
- **記憶體減少 130×**！

**影響**：Phase 2 新增 `weight_table.h`，預計算全局表

---

### **修正 3：Wet Node 不能用 Half-Way Bounce Back**

**原計劃錯誤**：直接用 `f5 ↔ f6` 交換（從 Cartesian 類比）

**實際情況**（理論分析）：
- GILBM 中，ζ=0 處的格點是 **Wet Node**（壁面在格點上）
- 逆變速度 **空間變化**：`ẽ_α_ζ = e_αy·(∂ζ/∂y) + e_αz·(∂ζ/∂z)`
- **不再是標準晶格** → Half-Way BB 的距離條件不滿足
- **需要 BC 的方向不再固定**：取決於 `sign(ẽ_α_ζ)`，隨壁面位置 (i,j) 變化

**正確方法（漸進式開發）**：

- **Phase 1: Non-Equilibrium Extrapolation**（框架驗證用）
  ```
  f_α|wall = f_α^eq|wall + (1-ω)·(f_α - f_α^eq)|fluid
  ```
  - 搭配**動態方向判別** `NeedsBoundaryCondition(alpha, metric)`
  - 適用於低 Re (≤200)，快速跑通框架
  - 理論局限：f^neq 的外推忽略了度量項的空間變化

- **Phase 2: Chapman-Enskog BC**（生產級方法，理論更嚴謹）
  ```
  f_α|wall = f_α^eq · [1 - ωΔt·Q_αij·S_ij / (2c_s⁴)]
  ```
  - 顯式使用物理空間應變率張量 S_ij，通過度量項正確轉換
  - 與 GILBM 的 C-E 理論框架**自洽**（Imamura 2005 核心方法）
  - 不依賴 f^neq 外推假設，適用於高 Re 紊流

**影響**：Phase 1.4 使用 NEE + 動態方向判別；Phase 2.5（新增）升級到 C-E BC

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

**關鍵修正**：必須使用**離散 Jacobian**（數值微分），而非解析導數！

```
∂ζ/∂z ≈ (xi_h[k+1] - xi_h[k-1]) / (z_h[j*NZ6+k+1] - z_h[j*NZ6+k-1])
∂ζ/∂y ≈ (ζ_{j+1,k} - ζ_{j-1,k}) / (y_h[j+1] - y_h[j-1])
J = LZ - H(y_j)  (每層不同)
```

**原因**：
- 物理座標 `z_h[j*NZ6+k]` 依賴於 (j, k)（tanh 拉伸）
- 解析 H'(y) 無法捕捉整個座標映射的複雜性
- 必須基於**實際網格座標**計算度量項

### 3. 逆變速度 (Contravariant Velocities)

對於 D3Q19 的每個方向 α，根據**鏈式法則**計算計算空間的速度：

```
ẽα_η = eα_x · ∂η/∂x + eα_y · ∂η/∂y + eα_z · ∂η/∂z
     = eα_x / LX                          (常數)

ẽα_ξ = eα_x · ∂ξ/∂x + eα_y · ∂ξ/∂y + eα_z · ∂ξ/∂z
     = eα_y / LY                          (常數)

ẽα_ζ = eα_x · ∂ζ/∂x + eα_y · ∂ζ/∂y + eα_z · ∂ζ/∂z
     = eα_y · (∂ζ/∂y) + eα_z · (∂ζ/∂z)  (空間變化！)
```

**實際計算**（每個格點 (i,j,k)）：
```cuda
double dzeta_dy = (xi_h[k] - xi_h_at_j_minus_1) / (y_h[j] - y_h[j-1]);
double dzeta_dz = (xi_h[k+1] - xi_h[k-1]) / (z_h[j*NZ6+k+1] - z_h[j*NZ6+k-1]);

ẽα_ζ = eα_y * dzeta_dy + eα_z * dzeta_dz;
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

**目標**: 實作**離散 Jacobian 計算**，建立基準數據。

**任務**:
1. ~~創建 `model_derivative.h`~~（不需要解析導數）

2. **創建 `gilbm/discrete_jacobian.h`**
   - 基於現有網格座標 `z_h[j*NZ6+k]`, `y_h[j]` 計算度量項
   - 數值微分（2 階中心差分或 6 階 Lagrange）：
     ```cuda
     ∂ζ/∂z = (xi_h[k+1] - xi_h[k-1]) / (z_h[j*NZ6+k+1] - z_h[j*NZ6+k-1])
     ∂ζ/∂y = ComputeNumerically(...)
     ```
   - CPU + GPU 版本

3. 單元測試
   ```cuda
   // 驗證度量項在已知點的數值
   double dzeta_dz = ComputeMetric_Z(j, k, z_h, xi_h);
   double dzeta_dy = ComputeMetric_Y(j, k, z_h, y_h, xi_h);
   // 檢查 Jacobian > 0（流體區域）
   ```

4. 運行現有 ISLBM 至收斂，保存基準數據
   - `baseline_islbm/velocity_*.vtk`
   - `baseline_islbm/checkrho.dat`
   - 記錄分離泡位置 (x_sep, x_reatt)

**交付物**: `discrete_jacobian.h`, 基準數據

---

### Phase 1: GILBM 框架 (2階插值) (7-10 天)

**目標**: 實作完整 GILBM 框架，先用 2 階插值驗證正確性。

#### 任務 1.1: 座標變換與度量項 (2-3天)

**新建檔案**: `gilbm/gilbm_transform.h`

```cuda
struct MetricTerms {
    double dzeta_dy;  // ∂ζ/∂y（數值微分）
    double dzeta_dz;  // ∂ζ/∂z（數值微分）
    double J;         // Jacobian（用於體積修正）
};

__global__ void ComputeMetricTerms(
    MetricTerms *metrics,
    double *y_h,       // Y 座標陣列 [NYD6]
    double *z_h,       // Z 座標陣列 [NYD6*NZ6]，按行存儲
    double *xi_h,      // 標準化 ξ 座標 [NZ6]
    int NX, int NY, int NZ
) {
    int j = ..., k = ...;

    // 數值微分計算度量項
    // ∂ζ/∂z：Z 方向（固定 j）
    double dxi_dz = (xi_h[k+1] - xi_h[k-1]) /
                    (z_h[j*NZ6+(k+1)] - z_h[j*NZ6+(k-1)]);

    // ∂ζ/∂y：Y 方向（固定 k）
    double dxi_dy = (xi_h[k] - ...) / (y_h[j] - y_h[j-1]);
    // 需要考慮 Y 變化導致的 Z 變化

    metrics[index].dzeta_dy = dxi_dy;
    metrics[index].dzeta_dz = dxi_dz;
    metrics[index].J = ...;  // 計算 Jacobian
}

__device__ void ContravariantVelocities(
    const MetricTerms &metric,
    const int alpha,  // 0-18
    const double *e_physical,  // D3Q19 標準速度
    double &e_tilde_eta,
    double &e_tilde_xi,
    double &e_tilde_zeta
) {
    e_tilde_eta = e_physical[0] / LX;
    e_tilde_xi  = e_physical[1] / LY;
    e_tilde_zeta = e_physical[1] * metric.dzeta_dy +
                   e_physical[2] * metric.dzeta_dz;
}
```

**驗證**:
1. 手算幾個點的度量項，對比 GPU 輸出
2. 檢查 Jacobian > 0（流體區域有效性）
3. 對比數值微分 vs 有限差分精度

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

#### 任務 1.4: 壁面邊界條件 — NEE + 動態方向判別 (2天)

**新建檔案**: `gilbm/boundary_conditions.h`

**關鍵修正**：Wet Node（ζ=0）+ 逆變速度變形 → **不能用 Half-Way Bounce Back**！

**Phase 1 實作 NEE**（框架驗證用，Phase 2.5 再升級到 C-E BC）：

```cuda
// ===== 動態方向判別（Phase 1 & 2 共用）=====
// 在 GILBM 中，需要 BC 的方向不再固定！
// 取決於逆變速度 ẽ_α_ζ 的符號，隨壁面位置 (i,j) 變化
__device__ bool NeedsBoundaryCondition(
    int alpha,                // 離散速度方向 (0-18)
    const MetricTerms &metric, // 壁面度量項
    bool is_bottom_wall       // ζ=0 (bottom) or ζ=1 (top)
) {
    // 計算 ζ 方向逆變速度
    double e_tilde_zeta = e[alpha][1] * metric.dzeta_dy
                        + e[alpha][2] * metric.dzeta_dz;

    // 上風點在壁外 → 需要邊界條件
    if (is_bottom_wall) return (e_tilde_zeta > 0);
    else                return (e_tilde_zeta < 0);
}

// ===== Phase 1: Non-Equilibrium Extrapolation（框架驗證）=====
// 理論局限：f^neq 外推忽略度量項空間變化，低 Re 下可接受
__device__ void NonEquilibriumExtrapolation_GILBM(
    double *f_wall,           // 壁面分佈函數 (output)
    double *f_eq_wall,        // 壁面平衡態 (u_wall=0)
    double *f_fluid,          // ζ=Δζ 最近流體格點分佈函數
    double *f_eq_fluid,       // 流體格點平衡態
    double omega,
    const MetricTerms &metric // 壁面度量項
) {
    for (int alpha = 0; alpha < 19; alpha++) {
        if (NeedsBoundaryCondition(alpha, metric, true)) {
            // 只對「上風點在壁外」的方向施加 NEE
            f_wall[alpha] = f_eq_wall[alpha] +
                            (1.0 - omega) * (f_fluid[alpha] - f_eq_fluid[alpha]);
        }
        // 其他方向保持正常 GILBM streaming 結果
    }
}

// 切換開關（Phase 1 預設 NEE，Phase 2 切換到 C-E）
#define BOUNDARY_METHOD 1  // 1: NEE (Phase 1), 2: C-E (Phase 2)
```

**理論依據（為何 NEE 在 Phase 1 可用）**：
- 在 GILBM 中，ζ=0 處的格點是 **Wet Node**
- 逆變速度 `ẽ_α_ζ` **空間變化** → 必須**動態判別**需要 BC 的方向
- NEE 的 f^neq 外推在低 Re (≤200) 和平滑座標變換下是合理的近似
- **Phase 2.5 將升級到 Chapman-Enskog BC**（理論自洽，適用高 Re）

**驗證**:
1. 平板 Poiseuille 流（解析解）
2. 檢查壁面無滑移條件：`|u_wall| < 1e-6`
3. **方向判別驗證**：山丘最高點（H'=0）→ 退化為標準 5 方向；斜面 → 額外方向

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

        // 2. 壁面判斷：上風點是否在域外？
        bool at_bottom_wall = (k == 3);
        bool at_top_wall = (k == NZ6 - 4);
        bool needs_bc = false;

        if (at_bottom_wall)
            needs_bc = NeedsBoundaryCondition(alpha, metrics[index], true);
        if (at_top_wall)
            needs_bc = NeedsBoundaryCondition(alpha, metrics[index], false);

        if (needs_bc) {
            // 壁面 BC（上風點在域外，不能插值）
            #if BOUNDARY_METHOD == 1
                // Phase 1: NEE
                F_in[alpha] = f_eq_wall[alpha] +
                    (1.0 - omega) * (f_fluid[alpha] - f_eq_fluid[alpha]);
            #elif BOUNDARY_METHOD == 2
                // Phase 2: C-E BC（設置所有壁面方向）
                F_in[alpha] = f_CE_wall[alpha];
            #endif
        } else {
            // 正常 GILBM streaming
            // 3. RK2 上風點
            RK2_UpwindPosition(..., eta_up, xi_up, zeta_up);

            // 4. 插值
            F_in[alpha] = Interpolate_Order2_3D(...);
        }
    }

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

**1. 全局權重表（關鍵優化）**
```cuda
// 預計算不同分數位置的 Lagrange 權重
__constant__ double LagrangeWeightTable[256][7];  // 256 個離散位置

void PrecomputeWeightTable() {
    for (int s_idx = 0; s_idx < 256; s_idx++) {
        double s = -0.5 + s_idx / 256.0;  // 分數位置
        ComputeLagrange6thWeights(s, LagrangeWeightTable[s_idx]);
    }
}

// 運行時快速查表
__device__ void Interpolate6th_Fast(double frac_pos) {
    int idx = (int)((frac_pos + 0.5) * 256);
    // 使用 LagrangeWeightTable[idx][0..6]
}
```

**記憶體優化**：
- **ISLBM**：56 個陣列 × NYD6 × NZ6 × 7 = ~3.4 MB
- **GILBM**：1 個全局表 × 256 × 7 × 3 方向 = ~14 KB
- **減少 130×**！

**2. Shared memory 緩存局部數據**

**3. Texture memory（GPU 特性）**

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

#### 任務 2.5: Chapman-Enskog 壁面邊界條件升級 (2天)

**目標**: 將壁面 BC 從 Phase 1 的 NEE 升級到理論自洽的 C-E 方法。

**理論基礎**：
- Imamura (2005) 基於 Chapman-Enskog 展開推導 GILBM 宏觀方程等價性
- 邊界條件也應在同一理論框架下 → C-E BC 是自洽選擇
- NEE 的 f^neq 外推忽略度量項空間變化，高 Re 下精度不足

**更新 `gilbm/boundary_conditions.h`**:

```cuda
// ===== Phase 2: Chapman-Enskog BC（生產級）=====
__device__ void ChapmanEnskogBC_GILBM(
    double *f_wall,           // 壁面分佈函數 (output)
    double rho_wall,          // 壁面密度（從 ζ=Δζ 外推）
    const MetricTerms &metric, // 壁面度量項
    double *du_dzeta,         // ∂u_i/∂ζ|ζ=0（計算空間速度梯度，單側差分）
    double *du_dxi,           // ∂u_i/∂ξ|ζ=0（主流向速度梯度）
    double omega,
    double dt
) {
    // 1. 轉換速度梯度到物理空間
    //    ∂u/∂z = (∂u/∂ζ) · (∂ζ/∂z)
    //    ∂u/∂y = (∂u/∂ξ) · (∂ξ/∂y) + (∂u/∂ζ) · (∂ζ/∂y)
    double S[3][3];  // 物理空間應變率張量
    S[0][2] = 0.5 * du_dzeta[0] * metric.dzeta_dz;  // ∂u/∂z
    S[1][2] = 0.5 * (du_dxi[1] / LY + du_dzeta[1] * metric.dzeta_dz);
    S[2][2] = du_dzeta[2] * metric.dzeta_dz;
    // ... 完整 S_ij（含對稱項和交叉項）

    // 2. 計算壁面平衡態 (u_wall = 0)
    double f_eq[19];
    double u_wall[3] = {0.0, 0.0, 0.0};
    ComputeEquilibrium(rho_wall, u_wall, f_eq);

    // 3. C-E 修正：所有 19 個方向
    double cs2 = 1.0 / 3.0;
    for (int alpha = 0; alpha < 19; alpha++) {
        double Qij_Sij = 0.0;
        for (int a = 0; a < 3; a++)
            for (int b = 0; b < 3; b++)
                Qij_Sij += (e[alpha][a]*e[alpha][b] - cs2*(a==b)) * S[a][b];

        f_wall[alpha] = f_eq[alpha] * (1.0 - omega * dt * Qij_Sij / (2.0 * cs2 * cs2));
    }
}
```

**壁面速度梯度計算**（單側 2 階差分）：
```cuda
// ∂u/∂ζ|ζ=0 ≈ (-3u|k=3 + 4u|k=4 - u|k=5) / (2Δζ)
// 其中 u|k=3 = 0（無滑移），所以：
// ∂u/∂ζ|wall ≈ (4u|k=4 - u|k=5) / (2Δζ)
```

**驗證**:
1. Poiseuille 流：C-E BC vs NEE 精度對比（預期 C-E 更接近解析解）
2. Periodic Hill Re=200 壁面剪應力 τ_wall(y)：與 Mellen 2000 DNS 對比
3. 方向判別一致性：C-E BC 對所有 19 方向設置（不需要方向判別），結果應自洽

**切換預設**：
```cuda
#define BOUNDARY_METHOD 2  // Phase 2 起預設使用 C-E BC
```

**交付物**: C-E BC 壁面條件，壁面剪應力精度提升

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
| `gilbm/discrete_jacobian.h` | 0 | **離散 Jacobian**（數值微分度量項） |
| `gilbm/gilbm_transform.h` | 1 | 逆變速度計算 |
| `gilbm/gilbm_rk2_upwind.h` | 1 | RK2 上風點追蹤 |
| `gilbm/interpolationGILBM_order2.h` | 1 | 2階插值 |
| `gilbm/boundary_conditions.h` | 1→2 | Phase 1: **NEE** + 動態方向判別；Phase 2.5: **C-E BC**（生產級） |
| `gilbm/weight_table.h` | 2 | **全局 Lagrange 權重表**（記憶體優化） |
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
| **度量項合理性** | Phase 0 | Jacobian > 0，∂ζ/∂z > 0 |
| 質量守恆 | 每步 | `\|ρ_avg - 1.0\| < 1e-6` |
| 動量守恆 | 每 1000 步 | 殘差 < 1e-4 |
| **壁面無滑移** | 每 1000 步 | `\|u_wall\| < 1e-6` |
| 壁面剪應力 | 收斂後 | 與文獻值誤差 < 10% |
| 網格收斂性 | Phase 2 | p ≥ 5 |
| **記憶體使用** | Phase 2 | ≤ 500 KB（權重相關） |

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

**症狀**:
- 壁面剪應力誤差 > 20%
- 壁面速度不為零（無滑移條件失效）
- 方向判別遺漏（部分方向未施加 BC）

**診斷**:
1. 檢查 `NeedsBoundaryCondition()` 動態方向判別是否正確
   - 在 H'(y)=0 處應退化為標準 5 方向
   - 在山丘斜面打印實際需要 BC 的方向集合
2. 驗證度量項數值微分精度（∂ζ/∂y, ∂ζ/∂z）
3. Phase 1 (NEE): 檢查流體格點（ζ=Δζ）的 f^neq 是否合理
4. Phase 2 (C-E): 檢查壁面速度梯度單側差分精度

**升級方案**（Phase 1 → Phase 2）:
- 如果 NEE 壁面剪應力誤差過大，提前切換到 C-E BC：
```cuda
#define BOUNDARY_METHOD 2  // 升級到 Chapman-Enskog BC
```
- C-E BC 不依賴 f^neq 外推，直接從速度梯度重建壁面分佈

### 風險 4: 度量項計算錯誤

**症狀**:
- Jacobian < 0（非物理）
- 質量不守恆
- 速度場異常發散

**診斷**:
1. 檢查數值微分的格點索引
2. 驗證週期邊界處理
3. 對比解析導數（在簡單區域）

**降級方案**:
- 增加數值微分精度（2 階 → 6 階 Lagrange）
- 使用更密的網格

### 風險 4: 性能下降

**症狀**: GILBM > 2× ISLBM 運行時間

**優化措施**:
- Shared memory
- Texture memory
- 降低插值階數

---

## 📊 修正前後對比總結

| 項目 | 原計劃（錯誤） | 修正後（正確） | 依據來源 |
|------|--------------|--------------|---------|
| **度量項計算** | 解析導數 H'(y) | **離散 Jacobian**（數值微分） | `initialization.h` 的 tanh 拉伸 |
| **插值權重** | "無需預計算" | **全局權重表**（14 KB） | `memory.h` 的 56 個陣列分析 |
| **邊界條件** | Half-Way Bounce Back | Phase 1: **NEE + 動態方向判別**；Phase 2: **C-E BC** | Wet Node + 逆變速度變形 + C-E 自洽性 |
| **方向判別** | 固定 5 方向 | **動態 `NeedsBoundaryCondition()`**（基於 ẽ_α_ζ） | 逆變速度隨空間變化 |
| **Phase 0 檔案** | `model_derivative.h` | `discrete_jacobian.h` | 基於實際網格座標 |
| **記憶體優化** | 未量化 | **減少 130×**（3.4 MB → 14 KB） | 權重陣列統計 |
| **邊界精度** | 依賴標準晶格 | Phase 2: **C-E BC 顯式應變率張量** | Imamura 2005 C-E 理論框架 |

### 關鍵發現來源

1. **度量項修正**：
   - 文件：`/Users/yetianzhong/Desktop/4.GitHub/D3Q27_PeriodicHill/initialization.h:110-136`
   - 關鍵代碼：`z_h[j*NZ6+k] = tanhFunction(...) + HillFunction(y_j)`

2. **插值權重分析**：
   - 文件：`/Users/yetianzhong/Desktop/4.GitHub/D3Q27_PeriodicHill/memory.h:81-143`
   - 統計：8 個方向 × 7 個權重陣列 × (NYD6×NZ6) = 2.43 MB

3. **邊界條件**：
   - 文件：`/Users/yetianzhong/Desktop/4.GitHub/D3Q27_PeriodicHill/evolution.h:147-160`
   - 現有：k=3 的 Half-Way BB（Wet Node，但 Cartesian 框架，固定 5 方向）
   - GILBM Phase 1：NEE + **動態方向判別** `NeedsBoundaryCondition(alpha, metric)`
   - GILBM Phase 2：**C-E BC**（理論自洽，顯式應變率張量，不依賴 f^neq 外推）
   - 關鍵新增：`NeedsBoundaryCondition()` 基於 `sign(ẽ_α_ζ)` 動態判斷

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
- Phase 2.3-2.4: 性能優化 + 精度驗證 (4-5 天)
- Phase 2.5: **C-E BC 壁面邊界升級** (2 天)

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
3. **Phase 0 啟動**: 實作 `discrete_jacobian.h`
4. **建立測試框架**: 單元測試腳本

**準備開始實作！**

---

## 理論討論記錄

### Q: Wet-Node NEE 在扭曲晶格下是否成立？

**結論**：NEE 在 GILBM 扭曲晶格下**有條件成立**，但 C-E BC 是理論上更嚴謹的選擇。

**分析要點**：
1. NEE 的 f^neq 外推忽略了度量項的空間變化 → 低 Re 可接受，高 Re 誤差顯著
2. 需要 BC 的方向不再固定（取決於逆變速度 ẽ_α_ζ）→ 需動態判別
3. C-E BC 與 GILBM 的 Chapman-Enskog 理論框架自洽（Imamura 2005）
4. C-E BC 顯式使用物理空間應變率張量，通過度量項自然處理座標變換

**決策**：漸進式開發 — Phase 1 用 NEE 跑通框架，Phase 2.5 升級到 C-E BC。



＊每一次對話都輸出((((()))))

