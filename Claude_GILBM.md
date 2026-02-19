# Wet-Node Boundary Condition 在 GILBM 扭曲晶格下的理論分析與實作建議

## Context

在 GILBM (Generalized Interpolation-Supplemented LBM) 中，計算空間 (η,ξ,ζ) 是均勻的，但物理空間是扭曲的。壁面位於 ζ=0，是一個 **wet node**（壁面恰在格點上）。

核心問題：**Non-Equilibrium Extrapolation (NEE) 是在傳統 Cartesian LBM 下發展的方法，在晶格系統被座標變換扭曲後，它是否仍然成立？還是應該直接採用 Imamura (2005) 文獻中基於 Chapman-Enskog 展開的邊界條件？**

---

## 一、理論分析：NEE 在扭曲晶格下的有效性

### 1.1 NEE 的物理本質

Non-Equilibrium Extrapolation 的公式為：
```
f_α|wall = f_α^eq(ρ_wall, u_wall) + (1-ω)·(f_α - f_α^eq)|fluid_neighbor
```

其物理意義是：
- **平衡部分** `f^eq`：由壁面宏觀量（ρ, u=0）決定 → 這部分與座標系無關，永遠正確
- **非平衡部分** `f^neq = f - f^eq`：從最近流體節點「外推」到壁面 → **這是爭議所在**

### 1.2 NEE 在 GILBM 中面臨的三個根本問題

#### 問題 A：「哪些方向需要邊界條件？」不再是固定的

在 Cartesian LBM 中：
- 底壁 k=0：`e_z > 0` 的方向（f5, f11, f12, f15, f16）需要 BC，這是**固定的 5 個方向**

在 GILBM 中：
- 底壁 ζ=0：需要 BC 的條件是 **逆變速度** `ẽ_α_ζ > 0`（上風點在壁外）
- `ẽ_α_ζ = e_α_y · (∂ζ/∂y) + e_α_z · (∂ζ/∂z)` → **隨空間位置變化**
- 在山丘斜面上，∂ζ/∂y 較大，某些原本不需要 BC 的 y-方向速度（如 f3, f4）可能也需要 BC
- **需要 BC 的方向集合是 (i,j) 的函數**，不是固定的

→ NEE 需要做方向判別修改，但這不是根本性障礙，只需在每個壁面點動態判斷 `ẽ_α_ζ` 的符號。

#### 問題 B：f^neq 的張量結構被座標變換改變（核心問題）

Chapman-Enskog 分析告訴我們，在 Cartesian LBM 中：
```
f_α^neq ≈ -τ · f_α^eq · Q_αij · S_ij / (2·c_s⁴)
```
其中 `Q_αij = e_αi·e_αj - c_s²·δ_ij`，`S_ij = (∂u_i/∂x_j + ∂u_j/∂x_i)/2`

NEE 假設 `f^neq|wall ≈ f^neq|fluid_neighbor`，即非平衡應力張量在壁面-流體之間「平滑」。

**在 GILBM 中的問題**：
- Streaming 是沿計算空間的特徵線進行的，不是沿物理空間的晶格方向
- 碰撞仍在物理空間（MRT 不變），但 streaming 後的分佈函數 `f_α` 包含了座標變換的效應
- `f^neq` 不再**只**反映物理應力張量，還混入了座標變換的度量項效應
- 從流體節點（ζ=Δζ）直接外推 `f^neq` 到壁面（ζ=0），忽略了度量項在 ζ 方向的變化

**嚴重程度評估**：
- 在 Periodic Hill 中，ζ 方向的度量項 ∂ζ/∂z 主要由 tanh 拉伸決定
- 壁面附近（ζ→0），tanh 拉伸使得物理格距最小 → ∂ζ/∂z 最大
- 從 ζ=Δζ 到 ζ=0，度量項變化是**平滑但非零**的
- 對於低 Re (200) 的層流/弱紊流，f^neq 本身較小，外推誤差是二階小量
- **但對於高 Re 紊流（未來目標），壁面 f^neq 梯度陡峭，外推誤差會顯著增大**

#### 問題 C：鬆弛參數 ω 的含義改變

NEE 中的 `(1-ω)` 因子來自 BGK/MRT 碰撞的鬆弛。在 GILBM 中：
- 碰撞仍在物理空間，ω 的定義不變
- 但如果使用 Local Time Step（Phase 3），不同格點的有效 ω 不同
- Re-estimation 機制（Imamura Eq. 36）修正了相鄰格點間的 f^neq 傳遞
- NEE 的 `(1-ω)` 因子需要考慮 local dt 的影響

### 1.3 Chapman-Enskog BC 的理論優勢

Imamura (2005) 文獻中使用的壁面條件基於 C-E 展開：
```
f_α|wall = f_α^eq(ρ_wall, u_wall) · [1 - ω·Δt/(2·c_s⁴) · Q_αij · S_ij]
```

其中 `S_ij = (∂u_i/∂x_j + ∂u_j/∂x_i)/2` 是**物理空間**的應變率張量。

**優勢**：
1. 顯式使用速度梯度 → 可通過度量項正確轉換：`∂u/∂z = (∂u/∂ζ)·(∂ζ/∂z)`
2. 不依賴 f^neq 的外推假設 → 不受座標變換影響
3. 與 GILBM 的 C-E 理論框架**自洽**（Imamura 正是用 C-E 推導了 GILBM 的宏觀方程等價性）
4. 壁面法向速度梯度 `∂u/∂ζ` 可用單側差分從流體內部計算

**劣勢**：
1. 需要計算壁面處的速度梯度（增加計算量）
2. 速度梯度的數值精度影響邊界條件精度
3. 實作複雜度較高（需要度量項參與）

---

## 二、結論與建議

### 2.1 核心結論

**Chapman-Enskog BC 是 GILBM 下理論上更正確的選擇。**

理由：
1. **自洽性**：GILBM 本身就是基於 C-E 展開推導的（Imamura 2005 的核心貢獻），邊界條件也應在同一理論框架下
2. **度量項處理**：C-E BC 通過顯式的應變率張量自然包含座標變換效應，而 NEE 的 f^neq 外推忽略了這一點
3. **文獻一致性**：Imamura (2005) 文獻中明確使用的是 C-E 分析推導的壁面條件，不是 NEE
4. **可擴展性**：未來提高 Re 時，C-E BC 不會因為壁面 f^neq 梯度增大而降級

### 2.2 NEE 並非完全無效，但有適用條件

NEE **仍可作為 Phase 1 的簡化實作**，條件是：
- Re 較低（≤200），壁面附近 f^neq 梯度溫和
- 座標變換平滑（Periodic Hill 的 tanh 拉伸滿足此條件）
- 用作調試工具，快速驗證 GILBM 框架的其他部分（逆變速度、RK2、插值）

但 **Phase 2 及以後必須切換到 C-E BC** 作為生產級方法。

### 2.3 實作策略：漸進式開發（用戶決策）

```
Phase 1.4 壁面邊界條件 — NEE (2天)
├── 動態判斷需要 BC 的方向：sign(ẽ_α_ζ) > 0
├── f_α|wall = f_α^eq(ρ_wall, 0) + (1-ω)·f^neq|ζ=Δζ
├── 驗證：Poiseuille 流，壁面速度 < 1e-6
└── 目的：快速跑通 GILBM 框架，隔離其他 bug

Phase 2.5 壁面邊界條件升級 — Chapman-Enskog BC (2天，Phase 2 新增)
├── 計算壁面速度梯度：∂u/∂ζ|ζ=0 (單側 2 階差分)
├── 轉換到物理空間：∂u/∂z = (∂u/∂ζ)·(∂ζ/∂z)|ζ=0
├── 完整應變率張量 S_ij（含交叉項 ∂u/∂y）
├── f_α|wall = f_α^eq · [1 - ωΔt·Q_αij·S_ij / (2c_s⁴)]
├── 驗證：Poiseuille 解析解精度提升 + Periodic Hill 壁面剪應力
└── 切換預設：#define BOUNDARY_METHOD 2

兩個方法共用：
├── NeedsBoundaryCondition() — 動態逆變速度方向判別
└── boundary_conditions.h — 雙方法編譯開關
```

---

## 三、修改 Claude_GILBM.md 的具體內容

**用戶決策：採用漸進式策略 — Phase 1 先用 NEE 跑通框架，Phase 2 再加入 C-E BC**

### 修改 1: Phase 1.4 邊界條件章節

**舊** (`Claude_GILBM.md:329-378`):
- 列出 NEE 為"主要"方法，C-E 為"Phase 2 可選"
- `#define BOUNDARY_METHOD 1` (預設 NEE)

**新**:
- Phase 1: NEE 為框架驗證方法（`BOUNDARY_METHOD 1`）
- Phase 2: C-E BC 為生產級方法（`BOUNDARY_METHOD 2`，升為預設）
- 兩個 Phase 都需要動態方向判別邏輯 `sign(ẽ_α_ζ)`
- Phase 2 增加壁面速度梯度計算（需度量項）

### 修改 2: `boundary_conditions.h` — 分階段實作

#### Phase 1: NEE (框架驗證)
```cuda
__device__ void NonEquilibriumExtrapolation_GILBM(
    double *f_wall,
    double rho_wall,
    double *f_fluid,         // ζ=Δζ 的流體節點
    double *f_eq_fluid,      // 流體節點平衡態
    double *f_eq_wall,       // 壁面平衡態 (u=0)
    double omega,
    const MetricTerms &metric,
    int alpha_start, int alpha_end
) {
    for (int alpha = alpha_start; alpha < alpha_end; alpha++) {
        // 動態判斷：只對需要 BC 的方向施加
        if (NeedsBoundaryCondition(alpha, metric, /*is_bottom*/true)) {
            f_wall[alpha] = f_eq_wall[alpha]
                          + (1.0 - omega) * (f_fluid[alpha] - f_eq_fluid[alpha]);
        }
    }
}
```

#### Phase 2: C-E BC (生產級) — 在 NEE 驗證框架正確後加入

```cuda
__device__ void ChapmanEnskogBC_GILBM(
    double *f_wall,          // 壁面分佈函數 (output)
    double rho_wall,         // 壁面密度 (從流體外推)
    double *u_wall,          // 壁面速度 = {0, 0, 0}
    double *du_dxi,          // ∂u_i/∂ξ_j 計算空間速度梯度 [3×3]
    const MetricTerms &metric, // 度量項
    double omega,            // 鬆弛參數
    double dt
) {
    // 1. 轉換速度梯度到物理空間
    //    ∂u/∂z = (∂u/∂ζ) · (∂ζ/∂z)
    //    ∂u/∂y = (∂u/∂ξ) · (∂ξ/∂y) + (∂u/∂ζ) · (∂ζ/∂y)
    double S[3][3]; // 物理空間應變率張量
    ComputeStrainRate(du_dxi, metric, S);

    // 2. 計算壁面平衡態
    double f_eq[19];
    ComputeEquilibrium(rho_wall, u_wall, f_eq);

    // 3. C-E 修正
    for (int alpha = 0; alpha < 19; alpha++) {
        double Qij_Sij = 0.0;
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                Qij_Sij += (e[alpha][i]*e[alpha][j] - cs2*delta(i,j)) * S[i][j];

        f_wall[alpha] = f_eq[alpha] * (1.0 - omega*dt*Qij_Sij / (2.0*cs2*cs2));
    }
}
```

### 修改 3: 動態方向判別

```cuda
__device__ bool NeedsBoundaryCondition(
    int alpha,               // 離散速度方向
    const MetricTerms &metric, // 壁面度量項
    bool is_bottom_wall      // ζ=0 (bottom) or ζ=1 (top)
) {
    double e_tilde_zeta = e[alpha][1] * metric.dzeta_dy
                        + e[alpha][2] * metric.dzeta_dz;

    if (is_bottom_wall) return (e_tilde_zeta > 0);  // 上風點在壁外
    else                return (e_tilde_zeta < 0);   // 上風點在壁外
}
```

### 修改 4: 整合到 evolution_gilbm.h

```cuda
// 在 GILBM streaming-collision kernel 中
for (int alpha = 1; alpha < 19; alpha++) {
    if (k == 3 && NeedsBoundaryCondition(alpha, metrics[idx], true)) {
        // 使用 C-E BC
        F_in[alpha] = f_CE_wall[alpha];
    } else if (k == NZ6-4 && NeedsBoundaryCondition(alpha, metrics[idx], false)) {
        F_in[alpha] = f_CE_wall_top[alpha];
    } else {
        // 正常 GILBM streaming
        F_in[alpha] = Interpolate(..., eta_up, xi_up, zeta_up);
    }
}
```

---

## 四、關鍵檔案修改清單

| 檔案 | Phase | 修改內容 |
|------|-------|---------|
| `Claude_GILBM.md:86-103` | 立即 | 修正 3 章節：NEE → Phase 1 驗證，C-E → Phase 2 生產級 |
| `Claude_GILBM.md:329-378` | 立即 | Phase 1.4：NEE + 動態方向判別；新增 Phase 2.5：C-E BC |
| `Claude_GILBM.md:411-413` | 立即 | evolution kernel 中 BC 分 Phase 調用邏輯 |
| `gilbm/boundary_conditions.h` (新建) | Phase 1 | NEE 實作 + `NeedsBoundaryCondition()` 動態判別 |
| `gilbm/boundary_conditions.h` (更新) | Phase 2 | 加入 C-E BC，切換預設 `BOUNDARY_METHOD 2` |
| `gilbm/gilbm_transform.h` (新建) | Phase 1 | 逆變速度 + `NeedsBoundaryCondition()` |

---

## 五、main.cu 變數宣告審查（用戶已新增代碼）

### 發現的問題

用戶在 `main.cu:63-110` 新增了 GILBM 變數宣告，有以下問題需修正：

#### 問題 1：語法錯誤 — 多餘的星號
```cuda
// 錯誤（目前代碼）：
double *deta_dy_h, *deta_dy_d *, *deta_dz_h, *deta_dz_d *,
       *dxi_dy_h,  *dxi_dy_d *, *dxi_dz_h,  *dxi_dz_d *;
//                           ^^^ 這些多餘的 * 會導致編譯失敗

// 修正：
double *deta_dy_h, *deta_dy_d, *deta_dz_h, *deta_dz_d,
       *dxi_dy_h,  *dxi_dy_d,  *dxi_dz_h,  *dxi_dz_d;
```

#### 問題 2：命名不一致 — `deta/dxi` vs 計劃中的 `dk_dy/dk_dz`

計劃中的座標系統：
- 計算座標 = 網格索引 (i, j, k)
- η 方向 = x 方向 (i)
- ξ 方向 = y 方向 (j)
- ζ 方向 = z 方向 (k)

**用戶宣告的 `deta_dy, deta_dz, dxi_dy, dxi_dz` 有歧義**：
- 如果 η=x方向，那 ∂η/∂y = 0（恆等），不需要存
- 實際需要的是 **ζ 方向的度量項**：`∂ζ/∂z` 和 `∂ζ/∂y`

**建議統一命名**：
```cuda
// 度量項（只有 ζ 方向是非平凡的）
double *dzeta_dz_h, *dzeta_dz_d,  // ∂ζ/∂z = 1/(∂z/∂k)  [NYD6*NZ6]
       *dzeta_dy_h, *dzeta_dy_d;  // ∂ζ/∂y 座標耦合項    [NYD6*NZ6]

// 或者更直觀的名稱：
double *dk_dz_h, *dk_dz_d,  // 同 ∂ζ/∂z
       *dk_dy_h, *dk_dy_d;  // 同 ∂ζ/∂y
```

為什麼只需要 2 個而不是 4 個？因為：
- ∂η/∂x = 1/dx = **常數**，不需要陣列
- ∂η/∂y = ∂η/∂z = 0
- ∂ξ/∂y = 1/dy = **常數**，不需要陣列
- ∂ξ/∂x = ∂ξ/∂z = 0
- **∂ζ/∂z** 和 **∂ζ/∂y** 是唯二隨空間變化的度量項

#### 問題 3：逆變速度不需要 19×2 = 38 個獨立陣列

用戶宣告了 `e_0_eta, e_0_xi, ..., e_18_eta, e_18_xi`（38 個指標）。

**分析**：
```
ẽ_α_η = e_αx / dx → 常數（對每個 α 只有一個值）
ẽ_α_ξ = e_αy / dy → 常數（對每個 α 只有一個值）
ẽ_α_ζ = e_αy * dk_dy + e_αz * dk_dz → 隨空間變化！
```

- η 和 ξ 分量是**常數**（只取決於格距 dx, dy），用 `double e_tilde_eta[19]` 和 `double e_tilde_xi[19]`（共 38 個 double）就夠了，不需要陣列
- **只有 ζ 分量需要 [NYD6*NZ6] 陣列**，但也不需要 19 個獨立陣列：因為 `ẽ_α_ζ = e_αy * dk_dy[j][k] + e_αz * dk_dz[j][k]`，可以在 kernel 中即時計算（只需存 `dk_dy` 和 `dk_dz`）

**建議**：不需要 38 個陣列。度量項 `dk_dz` + `dk_dy` 已經足夠在 GPU kernel 中即時計算任何方向的逆變速度。

#### 問題 4：RK2 上風點座標也不需要 19×2 = 38 個陣列

用戶宣告了 `pos_0_eta, pos_0_xi, ..., pos_18_eta, pos_18_xi`（38 個指標）。

**分析**：上風點座標是 streaming 步驟中的**中間變量**，在每個格點的 for-loop 中計算然後用掉，不需要全場存儲。

```cuda
// 在 kernel 中：
for (int alpha = 1; alpha < 19; alpha++) {
    double eta_up = i - dt * e_tilde_eta[alpha];   // 局部變量
    double xi_up  = j - dt * e_tilde_xi[alpha];    // 局部變量
    double zeta_up = k - dt * e_tilde_zeta;         // 局部變量
    F_in[alpha] = Interpolate(eta_up, xi_up, zeta_up);
}
```

**建議**：刪除所有 `pos_*` 陣列宣告。上風點座標是 kernel 的局部變量。

#### 問題 5：memory.h 缺少 GILBM 度量項的記憶體分配

`main.cu` 宣告了變數但 `memory.h:AllocateMemory()` 沒有對應的 `cudaMallocHost/cudaMalloc`。

---

### 修正後的 main.cu 變數宣告

```cuda
//======== GILBM 度量項 ========
// 只需 2 個空間變化的度量項（ζ 方向的逆 Jacobian）
// 大小：[NYD6 * NZ6]，與 z_h 相同
double *dk_dz_h, *dk_dz_d;   // ∂ζ/∂z = 1/(∂z/∂k)
double *dk_dy_h, *dk_dy_d;   // ∂ζ/∂y = -(∂z/∂j)/(dy·∂z/∂k)

// η, ξ 方向的逆變速度是常數，不需要陣列：
// e_tilde_eta[alpha] = e[alpha][0] / dx  (在 kernel 中用 #define 或 __constant__)
// e_tilde_xi[alpha]  = e[alpha][1] / dy
```

### 需要在 memory.h 中新增的記憶體分配

```cuda
// 在 AllocateMemory() 中，z_h 分配之後加入：
nBytes = NYD6 * NZ6 * sizeof(double);
AllocateHostArray(  nBytes, 2, &dk_dz_h, &dk_dy_h);
AllocateDeviceArray(nBytes, 2, &dk_dz_d, &dk_dy_d);
```

### 記憶體比較

| 方案 | 陣列數 | 記憶體 |
|------|--------|--------|
| 用戶目前宣告 | 8 + 38 + 38 = 84 個 [NYD6*NZ6] 指標 | ~84 × 39×70×8B = 1.83 MB |
| 修正後 | 2 個 [NYD6*NZ6] 度量項 | ~2 × 39×70×8B = 43.7 KB |
| **節省** | **42× 倍** | |

---

## 六、命名修正：`discrete_jacobian.h` → `metric_terms.h`

### 問題

用戶正確指出：Imamura 2005 只需要**左側元素**（物理→計算映射的度量項）：

```
逆變速度公式：
  ẽ_α^ξ  = (∂ξ/∂y)·e_αy + (∂ξ/∂z)·e_αz     ← 左側元素
  ẽ_α^ζ  = (∂ζ/∂y)·e_αy + (∂ζ/∂z)·e_αz     ← 左側元素
```

**不需要**右側元素（計算→物理的 ∂z/∂k, ∂z/∂j）。

當前程式碼中：
- `dk_dz` = ∂ζ/∂z，`dk_dy` = ∂ζ/∂y → **存的就是左側元素，數值正確**
- 但檔名 `discrete_jacobian.h`、函數名 `ComputeDiscreteJacobian`、註解「逆 Jacobian」→ **暗示存的是右側，造成混淆**
- `dz_dk`（右側元素）只在診斷輸出中作為中間變量暫時使用，不存儲

### 修正方案

| 項目 | 現在（混淆） | 修正後（準確） |
|------|-------------|---------------|
| 檔名 | `gilbm/discrete_jacobian.h` | `gilbm/metric_terms.h` |
| 函數 | `ComputeDiscreteJacobian()` | `ComputeMetricTerms()` |
| 函數 | `DiagnoseMetricTerms()` | （不變，已經正確） |
| 標頭守衛 | `DISCRETE_JACOBIAN_FILE` | `METRIC_TERMS_FILE` |
| 檔內註解 | 「離散 Jacobian 度量項計算」 | 「座標轉換度量項計算（Imamura 2005 左側元素）」 |
| main.cu include | `"gilbm/discrete_jacobian.h"` | `"gilbm/metric_terms.h"` |
| main.cu 變數註解 | 「逆 Jacobian 矩陣」 | 「座標轉換度量項矩陣（∂計算/∂物理）」 |

### 關鍵檔案

- `gilbm/discrete_jacobian.h` → 重命名為 `gilbm/metric_terms.h`，更新檔內所有命名
- `main.cu:66-81` → 更新 GILBM 變數區塊的註解
- `gilbm/2djacobian.h` → 刪除（空白草稿，已被取代）

---

## 七、z_h vs z_global 問題修正

### 問題

`DiagnoseMetricTerms()` 使用區域 `z_h`（大小 `NYD6*NZ6`，只覆蓋 1/4 的 Y 域），
且 `if (myid != 0) return;` 只在 rank 0 執行。

**後果**：
- rank 0 的 `y_h` 覆蓋 Y ≈ [-0.21, 2.04]（全域 Y 範圍 = [0, 9.0] 的前段）
- `j_peak`（山丘峰值 H_max）可能不在 rank 0 範圍內
- `j_slope`（最陡斜面）也可能不在 rank 0 範圍內
- 6 個 Pass/Fail 判據只驗證了 1/4 的域

**釐清**：`ComputeMetricTerms()` 用區域 `z_h` 計算區域 `dk_dz/dk_dy` 本身是**正確的**
（每個 rank 未來在 evolution kernel 中只需要自己的區域度量項）。
問題出在**診斷函數的覆蓋範圍不足**。

### 修正方案（方案 A：全域重算）

在 `DiagnoseMetricTerms()` 中：
1. 在 rank 0 重新計算 `y_global[NY6]` 和 `z_global[NY6*NZ6]`（與 `GenerateMesh_Z()` 相同公式）
2. 用全域座標計算全域度量項 `dk_dz_global[NY6*NZ6]` 和 `dk_dy_global[NY6*NZ6]`
3. 所有診斷輸出和 Pass/Fail 判據改用全域資料
4. **不影響** `dk_dz_h/dk_dy_h`（區域陣列），這些仍由各 rank 在 main.cu 中獨立計算

```cuda
void DiagnoseMetricTerms(...) {
    if (myid != 0) return;

    int bfr = 3;
    double dy = LY / (double)(NY6 - 2*bfr - 1);
    double dx = LX / (double)(NX6 - 7);

    // 重建全域座標（與 GenerateMesh_Y/Z 相同公式）
    double y_global[NY6];
    double z_global[NY6 * NZ6];
    for (int j = 0; j < NY6; j++) {
        y_global[j] = dy * (j - bfr);
        double total = LZ - HillFunction(y_global[j]) - minSize;
        double a = GetNonuniParameter();
        for (int k = bfr; k < NZ6 - bfr; k++)
            z_global[j*NZ6+k] = tanhFunction(total, minSize, a, k-3, NZ6-7)
                               + HillFunction(y_global[j]);
        z_global[j*NZ6+2] = HillFunction(y_global[j]);
        z_global[j*NZ6+(NZ6-3)] = LZ;
    }

    // 用全域座標計算全域度量項
    double dk_dz_global[NY6 * NZ6];
    double dk_dy_global[NY6 * NZ6];
    ComputeMetricTerms(dk_dz_global, dk_dy_global, z_global, y_global, NY6, NZ6);

    // 以下所有診斷改用 y_global, z_global, dk_dz_global, dk_dy_global
    // j 範圍改為 bfr..NY6-bfr-1（全域物理域）
    ...
}
```

### 同時：main.cu 中的調用需要拆分

```cuda
// Phase 0 診斷（全域，只在 rank 0）
DiagnoseMetricTerms(myid);  // 不再需要傳入 y_h, z_h

// 各 rank 計算自己的區域度量項（用於未來 evolution kernel）
ComputeMetricTerms(dk_dz_h, dk_dy_h, z_h, y_h, NYD6, NZ6);
```

### 修改檔案清單

| 檔案 | 修改內容 |
|------|---------|
| `gilbm/metric_terms.h` | `DiagnoseMetricTerms()` 改為內部重建全域座標 |
| `main.cu` | 拆分：`DiagnoseMetricTerms(myid)` + `ComputeMetricTerms(...)` 兩步 |

---

## 八、C++ 語法清理：刪除 `std::` 前綴

### Context

用戶已手動將 `metric_terms.h` 從 C-style I/O (`fprintf/fopen`) 改為 C++ stream (`ofstream`)，
並在檔案頂部加入 `using namespace std;`。但目前仍有多處保留了 `std::` 前綴，需清理。

### 修改清單（`gilbm/metric_terms.h`）

1. **新增 C++ headers**（在 `#define METRIC_TERMS_FILE` 之後、`using namespace std;` 之前）：
   ```cpp
   #include <fstream>
   #include <iomanip>
   #include <cmath>
   #include <cstdlib>
   ```
   因為 `main.cu` 只有 C headers（`<math.h>`, `<stdlib.h>`），而此檔使用了 `ofstream`, `setw`, `setprecision`, `fixed`, `scientific`, `fabs`。

2. **刪除所有 `std::` 前綴**（共約 30 處）：
   - `std::ofstream` → `ofstream`（lines 85, 131, 156）
   - `std::setw(...)` → `setw(...)`（lines 95, 96, 97, 98, 99, 100, 101, 102, 103, 134, 138, 139, 169, 170, 171, 184）
   - `std::fixed` → `fixed`（lines 97, 138, 170）
   - `std::setprecision(...)` → `setprecision(...)`（lines 97, 100, 138, 139, 170）
   - `std::scientific` → `scientific`（lines 100, 139）
   - `std::fabs(...)` → `fabs(...)`（lines 120, 191, 200, 225, 238）

3. **不修改**的部分：
   - `ComputeMetricTerms()` — 純 C 風格，無 `std::`
   - `printf()` 呼叫 — 保留不動（C 函數，不受影響）
   - `malloc/free` — 保留不動

### 驗證

- 確認 `using namespace std;` 在所有 `#include` 之後
- 確認所有 `ofstream`, `setw`, `fabs` 等符號在 `using namespace std` 下無需前綴

---

## 九、判據 6 的設計缺陷與修正

### 問題分析

用戶正確指出：判據 6 的當前實作意義違和。具體問題如下：

#### 當前代碼（lines 160-161, 236-238, 312-314）

```cpp
// 初始化
int pass_slope_extra = 0;  // 用於判據 6

// 迴圈內（無任何輸出）
if (fabs(dHdy) > 0.1 && num_bc > 5) {
    pass_slope_extra = 1;
}

// 最終輸出
printf("[%s] Criteria 6: slope wall has >5 BC directions\n",
       pass_slope_extra ? "PASS" : "WARN(may be ok if slope is gentle)");
```

#### 缺陷 1：無迴圈內診斷輸出

判據 1-5 在檢測到異常時，都有**即時輸出**，例如：
- 判據 1：`FAIL: dk_dz <= 0 at j=%d, k=%d`
- 判據 5：`FAIL criteria 5: j=%d (flat, H=...), num_BC=%d (expected 5)`

但判據 6 在迴圈內**完全靜默**。用戶無法得知：
- **哪些** j 位置觸發了額外方向？
- 額外方向是 **多少個**（6？7？8？）？
- 具體是**哪些方向編號**？

#### 缺陷 2：單向鎖存邏輯，無法 FAIL

判據 5 的邏輯是「反證法」：
```
pass_flat_5dirs = 1 (預設 PASS)
→ 只要有一個反例 → pass_flat_5dirs = 0 (FAIL)
```

判據 6 的邏輯是「存在量詞」：
```
pass_slope_extra = 0 (預設「未確認」)
→ 只要找到一個斜面點 num_bc > 5 → pass_slope_extra = 1 (確認)
```

問題在於：**它永遠不可能 FAIL**。即使所有斜面點的 `num_bc` 都 ≤ 5（可能意味著度量項計算有嚴重錯誤），它只會輸出 `WARN`，而不是 `FAIL`。

更嚴重的是，**它缺少互補檢查**：如果某個點 `|dHdy| > 0.1`（明確在斜面上）但 `num_bc ≤ 5`，這應該是一個可疑信號——斜面夠陡但逆變速度沒有產生額外方向，值得輸出警告或失敗。當前代碼完全忽略了這種情況。

#### 缺陷 3：與判據 5 在同一迴圈內但邏輯不對稱

判據 5 在迴圈內有 `cout` 輸出每個失敗點的詳細資訊。判據 6 在同一迴圈內卻只做一個布林賦值。這種不對稱讓代碼閱讀者困惑——看起來像是「先佔位，忘了寫完」。

### 修正方案

將判據 6 改為與判據 5 **對稱的結構**：在迴圈內輸出診斷資訊，同時改為可 FAIL 的邏輯。

#### 修改位置：`gilbm/metric_terms.h`

**1. 初始化改為 PASS-until-fail 模式**（line 161）：
```cpp
int pass_slope_extra = 1;  // 改為預設 PASS
int found_any_slope = 0;   // 新增：是否找到任何斜面點
```

**2. 迴圈內加入對稱的診斷輸出**（lines 236-238 替換為）：
```cpp
// 判據 6：斜面應有額外方向 (num_bc > 5)
if (fabs(dHdy) > 0.1) {  // 這是一個斜面點
    found_any_slope = 1;
    if (num_bc <= 5) {
        pass_slope_extra = 0;
        cout << "  FAIL criteria 6: j=" << j
             << " (slope, |dH/dy|=" << fixed << setprecision(4) << fabs(dHdy)
             << "), num_BC=" << num_bc << " (expected >5)\n";
    }
}
```

**3. 最終輸出改為三態邏輯**（line 312-314 替換為）：
```cpp
if (found_any_slope) {
    printf("[%s] Criteria 6: slope wall has >5 BC directions\n",
           pass_slope_extra ? "PASS" : "FAIL");
} else {
    printf("[SKIP] Criteria 6: no significant slope found (|dH/dy| > 0.1)\n");
}
```

### 修正後的完整邏輯對比

| 項目 | 判據 5（平坦段） | 判據 6（斜面段，修正後） |
|------|----------------|------------------------|
| 選點條件 | `\|Hy\| < 0.01 && \|dHdy\| < 0.01` | `\|dHdy\| > 0.1` |
| 預期 | `num_bc == 5` | `num_bc > 5` |
| 初始值 | `pass = 1` (PASS) | `pass = 1` (PASS) |
| 失敗條件 | `num_bc != 5` → FAIL + 輸出 j | `num_bc <= 5` → FAIL + 輸出 j |
| 迴圈內輸出 | 有（每個失敗點） | 有（每個失敗點） |
| 無斜面時 | N/A | SKIP（不是 WARN） |

### 修改檔案

| 檔案 | 行號 | 修改內容 |
|------|------|---------|
| `gilbm/metric_terms.h` | 161 | `pass_slope_extra = 0` → `pass_slope_extra = 1; int found_any_slope = 0;` |
| `gilbm/metric_terms.h` | 236-238 | 加入 FAIL 輸出 + `found_any_slope` 追蹤 |
| `gilbm/metric_terms.h` | 312-314 | 三態輸出：PASS / FAIL / SKIP |

---

## 十、Pass/Fail 判據匯總區段：printf → cout 轉換

### Context

`DiagnoseMetricTerms()` 中 lines 255-337 的 Pass/Fail 判據匯總仍使用 C-style `printf`，
與檔案其餘部分（已改為 `cout`）風格不一致。用戶要求統一為 C++ 編碼。

### 修改範圍

`gilbm/metric_terms.h` lines 255-337（Pass/Fail 判據匯總 + free 區段）

### 轉換對照表

| C printf | C++ cout |
|----------|----------|
| `printf("...\n")` | `cout << "...\n"` |
| `printf("...%d...", j)` | `cout << "..." << j << "..."` |
| `printf("...%s...", pass ? "PASS" : "FAIL")` | `cout << "..." << (pass ? "PASS" : "FAIL") << "..."` |
| `printf("...%.6e...", val)` | `cout << "..." << scientific << setprecision(6) << val << "..."` |
| `printf("...%.2f%%...", val)` | `cout << "..." << fixed << setprecision(2) << val << "%..."` |
| `printf("...%%...")` | `cout << "...%..."` |
| `free(ptr)` | 保留不變（對應的 `malloc` 在同一函數內，`free` 在 C++ 中合法） |

### 注意事項

1. `scientific` / `fixed` 是 sticky manipulator，使用後會影響後續浮點輸出格式。
   在每次需要不同格式時必須顯式切換。
2. `setprecision` 同樣是 sticky，需要在需要不同精度時重新設定。
3. `printf` 的 `%%` 輸出一個 `%`，在 `cout` 中直接寫 `"%"` 即可。
4. 判據 2 的 FAIL 行有 `%.6e` 和 `%.2f%%` 混用，需要在同一行中切換 `scientific` → `fixed`。

---

## 十一、為什麼 Phase 0 需要 6 個判據

### Phase 0 的職責

Phase 0 的唯一任務是：**計算度量項 dk_dz、dk_dy，並在進入 Phase 1（GPU kernel）之前驗證它們的正確性。** 一旦度量項有錯，後續所有逆變速度、streaming、邊界條件都會連鎖錯誤，且在 GPU 中極難追蹤。

### 6 個判據的層次結構

```
Phase 0 判據設計
│
├── 第一層：度量項本身（dk_dz, dk_dy 數值是否正確？）
│   ├── 判據 1: dk_dz > 0 全場          ← Jacobian 不可退化（最基本）
│   ├── 判據 2: dz_dk(wall) ≈ minSize   ← 壁面處數值精度（已知答案）
│   ├── 判據 3: dk_dy ≈ 0 (平坦段)      ← 交叉項零值測試（已知答案）
│   └── 判據 4: dk_dy 符號 ∝ -H'(y)     ← 交叉項符號測試（斜面處）
│
└── 第二層：度量項的下游消費者（方向判別公式是否正確？）
    ├── 判據 5: 平坦段 → 恰好 5 方向     ← 退化為 Cartesian 的回歸測試
    └── 判據 6: 斜面段 → >5 方向          ← 座標變換確實產生效果
```

### 為什麼不能少？逐一分析

| 判據 | 驗證對象 | 若刪除的後果 | 能否被其他判據取代？ |
|------|---------|------------|-------------------|
| 1 | dk_dz 正定性 | tanh 拉伸若反向（z 非單調），整個座標變換崩潰 | **不能**。判據 2 只查壁面一點，無法覆蓋全場 |
| 2 | dk_dz 壁面精度 | z[k=2]=H(y) 的 ghost cell 值若有誤，壁面差分就錯 | **不能**。判據 1 只查符號不查量值 |
| 3 | dk_dy 零值 | 平坦段 dk_dy 若非零，意味 dz_dj 差分公式有 bug | **不能**。判據 4 查的是非零區的符號，不查零值 |
| 4 | dk_dy 符號 | 符號錯 → 逆變速度方向反轉 → 邊界完全判反 | **不能**。判據 3 在平坦段查不出符號問題 |
| 5 | 方向判別（平坦） | 公式 `ẽ_α_ζ = e[α][1]·dk_dy + e[α][2]·dk_dz` 可能打錯索引 | **不能**。判據 1-4 只驗度量項本身，不驗公式組裝 |
| 6 | 方向判別（斜面） | dk_dy 太小（量級錯）但符號對 → 判據 3,4 都過但方向數不對 | **不能**。判據 5 在平坦段 dk_dy≈0，測不出量級問題 |

### 為什麼判據 5-6 在 Phase 0 而不是 Phase 1？

判據 5-6 驗證的是「逆變速度方向判別」，這是 Phase 1.4 邊界條件的核心邏輯。但將它**提前**到 Phase 0 有兩個原因：

1. **Shift-left testing**：方向判別公式在 CPU 的診斷函數中測試，比嵌入 GPU kernel 後再 debug 容易 100 倍
2. **端到端驗證**：判據 1-4 是 unit test（驗證個別度量項），判據 5-6 是 integration test（驗證度量項→逆變速度→方向判別的完整鏈路）

### 總結

6 個判據 = **4 個 unit test + 2 個 integration test**，覆蓋：
- 2 個度量項（dk_dz, dk_dy）× 2 個面向（數值/符號）= 4 個 unit test
- 1 個公式（逆變速度方向判別）× 2 個已知場景（平坦/斜面）= 2 個 integration test

沒有一個是多餘的。

---

## 十二、判據 3 的覆蓋範圍缺陷

### 問題

用戶指出判據 3 只掃描 `j_flat`（一個 j 值）下的垂直 k 列，而非掃描整個平坦區域。

#### `j_flat` 的實際含義

```cpp
if (Hy < 0.01 && j_flat < 0) j_flat = j;  // 第一個 H(y) < 0.01 的 j
```

`j_flat` 是**第一個**滿足 H(y) < 0.01 的 j 值，不是「平坦區域最大 j 值」。

#### Periodic Hill 幾何

```
y:    0        1.93              7.07        9.0
      |--hill---|----flat (H=0)----|--hill---|
      左山丘     平坦區域           右山丘
      j=3..30   j≈31..103         j=104..131
```

- 平坦區域（H ≈ 0）佔 y ∈ [1.93, 7.07]，約 **73 個 j 值**
- `j_flat` ≈ 31，只是平坦區域的**左邊界**

#### 判據 3 vs 判據 5 的覆蓋對比

| 判據 | 掃描範圍 | 覆蓋 |
|------|---------|------|
| 判據 3（dk_dy ≈ 0） | 只有 `j_flat`（1 列） | **不完整** — 漏掉 j=32..103 |
| 判據 5（5 方向） | 所有 `\|Hy\| < 0.01 && \|dHdy\| < 0.01` 的 j | **完整** |

#### 為什麼只取一個 j 值？原因與不足

原始設計意圖是「抽樣測試」：如果度量項公式正確，那在任何平坦 j 都應該 dk_dy ≈ 0；取一個代表點即可。

但這個假設忽略了：
1. **邊界效應**：j_flat ≈ 31 恰好在山丘→平坦的過渡帶，dz_dj 的中心差分 `(z[j+1]-z[j-1])/2` 可能因為 j-1 仍在山丘邊緣而不完全為零。j = 50（平坦區域中心）才是真正的「純平坦」。
2. **數值對稱性**：右側過渡帶 j ≈ 103 也可能有邊界效應，但完全不被檢查。
3. **與判據 5 不一致**：判據 5 掃描所有平坦 j，判據 3 只掃描一個。如果判據 3 通過但判據 5 失敗，使用者會困惑為什麼「dk_dy ≈ 0 通過」但「方向數 ≠ 5 失敗」。

### 修正方案

將判據 3 改為掃描**所有平坦 j 值**，使用與判據 5 相同的選點條件：

```cpp
// 判據 3: 平坦段 dk_dy ≈ 0（掃描所有平坦 j，與判據 5 同條件）
int pass3 = 1;
for (int j = bfr; j < NY6 - bfr - 1; j++) {
    double Hy = HillFunction(y_g[j]);
    double dHdy = (HillFunction(y_g[j + 1]) - HillFunction(y_g[j - 1])) / (2.0 * dy);
    if (fabs(Hy) < 0.01 && fabs(dHdy) < 0.01) {  // 同判據 5 的平坦條件
        for (int k = bfr; k < NZ6 - bfr; k++) {
            if (fabs(dk_dy_g[j * NZ6 + k]) > 0.1) {
                pass3 = 0;
                cout << "  FAIL: flat region j=" << j << " k=" << k
                     << ", dk_dy=" << scientific << setprecision(6) << dk_dy_g[j * NZ6 + k]
                     << " (expected ~0)\n";
            }
        }
    }
}
```

#### 修改要點

1. 移除 `if (j_flat >= 0)` 的單一 j 檢查
2. 改為迴圈掃描所有 j，用 `|Hy| < 0.01 && |dHdy| < 0.01` 篩選平坦點
3. 選點條件與判據 5 **完全一致**，確保兩個判據定義的「平坦」是同一組點
4. `j_flat` 變數仍保留（用於輸出 2 的剖面選點和 console 輸出），但不再用於判據 3

### 修改檔案

| 檔案 | 行號 | 修改內容 |
|------|------|---------|
| `gilbm/metric_terms.h` | 284-296 | 判據 3 改為全平坦區掃描 |

---

## 十三、驗證計劃

1. **Poiseuille 流解析解**
   - 平行平板間，解析解 u(z) = U_max·[1 - (2z/H - 1)²]
   - 壁面剪應力 τ = μ·∂u/∂z|wall → 解析值已知
   - 分別用 NEE 和 C-E BC，對比精度

2. **方向判別驗證**
   - 在山丘最高點（H'(y)=0）和最陡處（H'(y) 最大），列印需要 BC 的方向集合
   - 最高點處應退化為標準 Cartesian 結果（f5, f11, f12, f15, f16）
   - 斜面處應有額外方向需要 BC

3. **Periodic Hill Re=200 壁面剪應力**
   - 與 ISLBM 基準對比
   - 與 Mellen (2000) DNS 數據對比

---

## 十三、判據 6 的 `pass_slope_extra` 是否會被覆蓋導致最終彙整遺漏錯誤？

### Context

用戶提問：判據 6 雖然在迴圈內有即時 FAIL 輸出，但 `pass_slope_extra` 是否可能被「覆蓋」，導致最後彙整輸出時漏掉錯誤？

### 分析：目前邏輯完全安全，不存在覆蓋問題

追蹤 `pass_slope_extra` 的完整資料流：

```
初始化:   pass_slope_extra = 1   (line 161, 預設 PASS)
                     │
          ┌──── for j 迴圈 ────┐
          │                     │
          │  if |dHdy| > 0.1:   │   ← 斜面點
          │    found_any_slope=1│
          │    if num_bc <= 5:  │   ← 斜面卻方向不足
          │      pass_slope_extra = 0  ← 唯一的賦值：只會 1→0
          │      cout FAIL ...  │   ← 即時輸出
          │                     │
          └─────────────────────┘
                     │
最終彙整:  if found_any_slope == 0 → SKIP
           elif pass_slope_extra == 1 → PASS
           else → FAIL
```

**關鍵觀察**：迴圈內對 `pass_slope_extra` 的唯一操作是 `= 0`。這是**單向門閂（one-way latch）**設計：

| 情境 | 操作 | 結果 |
|------|------|------|
| 第一個斜面點 FAIL | `1 → 0` | 正確記錄 |
| 後續斜面點也 FAIL | `0 → 0` | 冪等，無影響 |
| 後續斜面點 PASS | 不觸發 `= 0` | `pass_slope_extra` 維持 0 |

**結論：不可能被覆蓋回 1**。只要有任何一個斜面點 FAIL，最終彙整一定輸出 `[FAIL]`。

### 但存在一個資訊呈現的不對稱

用戶擔心的可能不是邏輯錯誤，而是**最終彙整的資訊量不足**：

- **即時輸出**（迴圈內 line 243-245）：每個 FAIL 點都逐一印出 `j`, `|dH/dy|`, `num_BC`
- **最終彙整**（line 338）：只印一行 `[FAIL] Criteria 6: slope wall has >5 BC directions`

對比判據 5 也一樣 — 即時輸出了逐點 FAIL，但最終彙整只有一行 `[PASS/FAIL]`。

**這其實是所有 6 個判據的統一設計**：
- 逐點 FAIL 資訊已經在迴圈中即時印出
- 最終彙整區段是 6 行「總覽」，方便一眼確認全部通過/失敗

### 可選增強：在最終彙整追加失敗計數

若要讓最終彙整更有資訊量，可以追加一個 `fail_count_slope` 計數器，在最終輸出 `[FAIL]` 時同時顯示有多少個斜面點失敗：

**修改檔案**: `gilbm/metric_terms.h`

**修改 1 — 初始化** (line 162 附近)：
```cpp
int found_any_slope = 0;
int fail_count_slope = 0;  // 新增：統計斜面 FAIL 點數
```

**修改 2 — 迴圈內** (line 242 附近)：
```cpp
if (num_bc <= 5) {
    pass_slope_extra = 0;
    fail_count_slope++;  // 新增：累計
    cout << "  FAIL criteria 6: j=" << j ...
```

**修改 3 — 最終彙整** (line 337-338)：
```cpp
} else {
    cout << "[FAIL] Criteria 6: slope wall has >5 BC directions ("
         << fail_count_slope << " slope points failed)\n";
}
```

同理，判據 5 也可加上 `fail_count_flat` 做相同增強。

### 驗證方式

- 編譯執行後，觀察最終彙整輸出：`[FAIL] Criteria 6: ... (N slope points failed)`
- N 值應等於迴圈中即時印出的 FAIL 行數

---

## 十四、伺服器編譯錯誤修正（3 個編譯錯誤 + 1 個潛在 Runtime 錯誤）

### Context

用戶將程式碼推到伺服器（CUDA 10.2, nvcc, sm_35）編譯，出現 3 個編譯錯誤：

```
gilbm/metric_terms.h(139): error: identifier "cout" is undefined
main.cu(170): error: argument of type "double *" is incompatible with parameter of type "int"
main.cu(170): error: too many arguments in function call
```

### 根因分析

#### 錯誤 1：`cout` is undefined（metric_terms.h:139）

**原因**：`metric_terms.h` 的 `#include` 有 `<fstream>`, `<iomanip>`, `<cmath>`, `<cstdlib>`，但**缺少 `<iostream>`**。

`<fstream>` 在某些編譯器（如 GCC）中會間接引入 `<iostream>`，所以本地可能不報錯。但 nvcc (CUDA 10.2) 的標準庫實作**不保證**這種間接引入。

**修正**：在 `metric_terms.h` line 4-7 的 `#include` 區加入 `#include <iostream>`。

**修改檔案**：`gilbm/metric_terms.h` line 4

```cpp
// 修正前：
#include <fstream>
#include <iomanip>
#include <cmath>
#include <cstdlib>

// 修正後：
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
```

---

#### 錯誤 2 & 3：函數簽名不匹配

**原因**：`DiagnoseMetricTerms()` 定義為：
```cpp
void DiagnoseMetricTerms(int myid)  // metric_terms.h:56 — 只接受 1 個參數
```

但 `main.cu:170` 呼叫時傳了 6 個參數：
```cpp
DiagnoseMetricTerms(y_h, z_h, xi_h, dk_dz_h, dk_dy_h, myid);
```

nvcc 報告第一個參數 `y_h`（`double*`）與預期的 `int` 不匹配，且參數數量過多。

**設計決策**：`DiagnoseMetricTerms()` 內部已自行重建全域座標（`y_g`, `z_g`），**不需要外部傳入的區域陣列**。因此正確的修正方向是：**修改 `main.cu` 的呼叫端**，與定義端保持一致。

**修正**：將 `main.cu:170` 改為：
```cpp
// 修正前：
DiagnoseMetricTerms(y_h, z_h, xi_h, dk_dz_h, dk_dy_h, myid);

// 修正後：
DiagnoseMetricTerms(myid);
```

**修改檔案**：`main.cu` line 170

---

#### 潛在 Runtime 錯誤：j_flat/j_peak/j_slope = -1 未防護

**位置**：`metric_terms.h` lines 140-142

```cpp
cout << "j_flat  = " << ... << y_g[j_flat] << ...;   // j_flat 可能 = -1
cout << "j_peak  = " << ... << y_g[j_peak] << ...;   // j_peak 可能 = -1
cout << "j_slope = " << ... << y_g[j_slope] << ...;   // j_slope 可能 = -1
```

**風險**：如果 Periodic Hill 幾何中沒有找到平坦段/峰值/斜面（理論上不可能，但防禦性編程需要防護），`y_g[-1]` 會造成**陣列越界**，可能導致 segfault 或輸出垃圾值。

**修正**：在每個 `cout` 前加入 `if (j_xxx >= 0)` 防護。

**修改檔案**：`gilbm/metric_terms.h` lines 139-142

```cpp
// 修正後：
cout << "\n===== Phase 0: Metric Terms Diagnostics (Global) =====\n";
if (j_flat >= 0)
    cout << "j_flat  = " << setw(4) << j_flat << "  (y=" << fixed << setprecision(4)
         << setw(8) << y_g[j_flat] << ", H=" << setw(8) << HillFunction(y_g[j_flat]) << ")\n";
else
    cout << "j_flat  = NOT FOUND\n";

if (j_peak >= 0)
    cout << "j_peak  = " << setw(4) << j_peak << "  (y=" << fixed << setprecision(4)
         << setw(8) << y_g[j_peak] << ", H=" << setw(8) << HillFunction(y_g[j_peak]) << ")\n";
else
    cout << "j_peak  = NOT FOUND\n";

if (j_slope >= 0)
    cout << "j_slope = " << setw(4) << j_slope << "  (y=" << fixed << setprecision(4)
         << setw(8) << y_g[j_slope] << ", H=" << setw(8) << HillFunction(y_g[j_slope])
         << ", |H'|=" << setw(8) << dH_max << ")\n";
else
    cout << "j_slope = NOT FOUND\n";
```

---

### 修改檔案清單

| 檔案 | 行號 | 修改內容 | 嚴重度 |
|------|------|---------|--------|
| `gilbm/metric_terms.h` | 4 | 新增 `#include <iostream>` | **編譯錯誤** |
| `main.cu` | 170 | `DiagnoseMetricTerms(y_h, z_h, xi_h, dk_dz_h, dk_dy_h, myid)` → `DiagnoseMetricTerms(myid)` | **編譯錯誤** |
| `gilbm/metric_terms.h` | 139-142 | 加入 `if (j_xxx >= 0)` 防護 | **Runtime 防護** |

### 驗證方式

伺服器上重新編譯：
```bash
nvcc main.cu -arch=sm_35 -I/home/chenpengchung/openmpi-3.0.3/include \
     -L/home/chenpengchung/openmpi-3.0.3/lib -lmpi -o a.out
```

預期結果：零錯誤、零警告，成功生成 `a.out`。

---

## 十五、伺服器執行結果分析：判據 2/3/5 失敗的根因與修正

### Context

程式碼成功編譯並在 4 GPU 上執行，Phase 0 診斷結果：

| 判據 | 結果 | 失敗數 |
|------|------|--------|
| 1: dk_dz > 0 全場 | **PASS** | 0 |
| 2: dz_dk(wall) ≈ minSize | **FAIL** | 32 點（j=3-18, 116-131）|
| 3: dk_dy ≈ 0 平坦段 | **FAIL** | 12 點（j=31, j=103 各 6 個 k）|
| 4: dk_dy 符號正確 | **PASS** | 0 |
| 5: 平坦段 5 方向 | **FAIL** | 2 點（j=31, j=103）|
| 6: 斜面段 >5 方向 | **PASS** | 0 |

程式在約 step 1001 後收到 SIGTERM（signal 15），此為 job scheduler 或 mpirun 超時所致，與 metric terms 無關。

---

### 根因 A：判據 2 — `minSize` 不是所有 j 的正確期望值

#### 數學推導

壁面處 (k=3) 的中心差分：
```
dz_dk|_(k=3) = (z[k=4] - z[k=2]) / 2
```

代入座標值：
```
z[k=2] = H(y)                          ← ghost cell
z[k=4] = tanhFunction(total_j, minSize, a, 1, NZ6-7) + H(y)
```

其中 `tanhFunction(..., 0, ...)` 恆等於 `minSize/2`（與 total 無關），所以：
```
z[k=4] = minSize/2 + dx(total_j) + H(y)
```

`dx(total_j)` = 第一個完整格距 = 由 GetNonuniParameter 校準，但**線性隨 total 縮放**：
```
dx(total_j) = minSize × total_j / total_0
```

其中 `total_0 = LZ - HillFunction(0.0) - minSize = 3.036 - 1.0 - 0.0191 = 2.0169`
      `total_j = LZ - HillFunction(y_g[j]) - minSize`

因此：
```
dz_dk|_(k=3) = (minSize/2 + minSize × total_j/total_0) / 2
```

#### 數值驗證

| 位置 | H(y) | total_j | total_j/total_0 | dz_dk | 對 minSize 的偏差 |
|------|------|---------|-----------------|-------|-----------------|
| 山丘頂 j=3 | 1.000 | 2.017 | 1.000 | 0.75×minSize = **0.01432** | **-25.0%** |
| 平坦段 j=50 | 0.000 | 3.017 | 1.496 | 0.998×minSize ≈ **0.01905** | **-0.2%** |

- 山丘頂偏差 25% → 完全吻合輸出 `rel_err=25.00%` ✓
- 平坦段偏差 0.2% → 通過 10% 門檻 ✓

#### 結論

**判據 2 的比較基準 `minSize` 是常數，但 dz_dk(wall) 隨 j 而變 — 這是 tanh 拉伸在不同 total 下的正常行為。判據 2 的設計有誤，不是度量項計算有錯。**

#### 修正方案

改為 **j-dependent 期望值**：直接從 tanh 公式計算每個 j 的理論 dz_dk，再與中心差分結果比較。

**修改檔案**：`gilbm/metric_terms.h` 判據 2 區段（lines 292-305）

```cpp
// 判據 2: 壁面 dz_dk 與 tanh 解析值一致（10% 容差）
int pass2 = 1;
double a = GetNonuniParameter();
double total_0 = LZ - HillFunction(0.0) - minSize;
for (int j = bfr; j < NY6 - bfr; j++) {
    double total_j = LZ - HillFunction(y_g[j]) - minSize;
    double dx_j = minSize * total_j / total_0;            // 第一個完整格距
    double expected_dz_dk = (minSize / 2.0 + dx_j) / 2.0; // 中心差分期望值
    double dz_dk_wall = 1.0 / dk_dz_g[j * NZ6 + bfr];
    double rel_err = fabs(dz_dk_wall - expected_dz_dk) / expected_dz_dk;
    if (rel_err > 0.1) {
        pass2 = 0;
        cout << "  FAIL: j=" << j
             << ", dz_dk[wall]=" << scientific << setprecision(6) << dz_dk_wall
             << ", expected=" << expected_dz_dk
             << ", rel_err=" << fixed << setprecision(2) << rel_err * 100 << "%\n";
    }
}
cout << "[" << (pass2 ? "PASS" : "FAIL")
     << "] Criteria 2: dz_dk(wall) matches tanh analytical value (within 10%)\n";
```

---

### 根因 B：判據 3 & 5 — 山丘-平坦過渡帶的中心差分穿越效應

#### 失敗點分析

j=31（y≈1.97）和 j=103（y≈7.03）恰好位於**山丘→平坦的過渡帶**。

```
y:    0        ~1.93    j=31    ~7.07   j=103   9.0
      |--hill---|........|----flat----|........|--hill---|
                 過渡帶                  過渡帶
```

這兩個 j 的 `H(y) ≈ 0`、`|dH/dy| ≈ 0`，通過了「平坦條件」。但 dk_dy 的計算用中心差分：
```
dz_dj = (z[j+1, k] - z[j-1, k]) / 2
```

j=31 的鄰居 j=30 仍在山丘邊緣（H(y_30) > 0），所以 `z[j-1]` 帶有山丘影響 → `dz_dj ≠ 0` → `dk_dy ≠ 0`。

**這是物理上正確的**：過渡帶的座標系確實有非零交叉項。問題出在**平坦條件太寬鬆**，誤將過渡帶歸入平坦區域。

#### 修正方案

收緊平坦條件：**要求鄰居 j-1、j+1 也滿足平坦條件**。這確保中心差分 stencil 不會跨越山丘-平坦邊界。

**判據 3 修改**（lines 310-313）：
```cpp
if (fabs(Hy_c3) < 0.01 && fabs(dHdy_c3) < 0.01
    && fabs(HillFunction(y_g[j - 1])) < 0.01
    && fabs(HillFunction(y_g[j + 1])) < 0.01) {
```

**判據 5 修改**（line 233，掃描迴圈內）：
```cpp
if (fabs(Hy) < 0.01 && fabs(dHdy) < 0.01
    && fabs(HillFunction(y_g[j - 1])) < 0.01
    && fabs(HillFunction(y_g[j + 1])) < 0.01) {
```

修正後 j=31 和 j=103 會被排除（因為 j=30 和 j=104 的 H(y) > 0.01），判據 3 和 5 應全部 PASS。

---

### 根因 C：Runtime 終止（SIGTERM）

```
Primary job terminated normally, but 1 process returned a non-zero exit code.
process rank 2 with PID 0 on node cfdlab-ib3 exited on signal 15 (Terminated).
```

Signal 15 = SIGTERM。可能原因：
1. nohup 下執行時 rank 2 遇到 GPU 記憶體不足或 CUDA error
2. 一個 rank 崩潰後 mpirun 發 SIGTERM 終止其他 rank
3. Job scheduler 超時

**此問題與 metric terms 無關**，屬於既有 ISLBM 程式碼的行為。程式已成功執行 1001 步並輸出 VTK 檔案，Phase 0 診斷也完整輸出。

---

### 修改檔案清單

| 檔案 | 行號 | 修改內容 | 對應判據 |
|------|------|---------|---------|
| `gilbm/metric_terms.h` | 292-305 | 判據 2 改為 j-dependent 期望值 | 判據 2 |
| `gilbm/metric_terms.h` | 313 | 判據 3 加入鄰居 H(y) 檢查 | 判據 3 |
| `gilbm/metric_terms.h` | ~233 | 判據 5 加入鄰居 H(y) 檢查 | 判據 5 |

### 預期修正後結果

```
[PASS] Criteria 1: dk_dz > 0 everywhere
[PASS] Criteria 2: dz_dk(wall) matches tanh analytical value (within 10%)
[PASS] Criteria 3: dk_dy ≈ 0 at flat region
[PASS] Criteria 4: dk_dy sign consistent with -H'(y)
[PASS] Criteria 5: flat wall has exactly 5 BC directions
[PASS] Criteria 6: slope wall has >5 BC directions
```

### 驗證方式

重新編譯、執行，確認 6 個判據全部 PASS。

＊＊每一次對話回覆皆需要上傳github且生成commit 摘要

---

## 十六、邊界條件施加位置分析：k=2 (wall node) vs k=3 (first interior)

### Context

GILBM 使用貼體座標系，壁面（山丘表面）精確落在 k=2 格線上。但目前 k=2 屬於 buffer layer（不計算），邊界條件施加在 k=3（距牆面 0.5·minSize）。用戶希望評估是否應將 k=2 納入計算域作為壁面節點，以便施加 Inamuro / Chapman-Enskog 邊界條件。

### 16.1 現狀回顧：網格結構與各層角色

```
k 索引    角色              z 座標                     是否計算    是否初始化
──────────────────────────────────────────────────────────────────────────
k=0      ghost layer       未定義                      否          否
k=1      ghost layer       未定義                      否          否
k=2      buffer/壁面       H(y) (山丘表面)              否          是 ← z_h[j*NZ6+2] = HillFunction(y_h[j])
k=3      第一計算點        H(y) + minSize/2             是          是 ← tanhFunction(..., j=0, ...)
k=4      第二計算點        H(y) + minSize/2 + minSize   是          是 ← tanhFunction(..., j=1, ...)
 ...      內部計算點        tanh 拉伸                    是          是
k=66     最後計算點        ≈ LZ - minSize/2             是          是
k=67     buffer/頂壁       LZ = 3.036                   否          是 ← z_h[j*NZ6+67] = LZ
k=68     ghost layer       未定義                      否          否
k=69     ghost layer       未定義                      否          否
```

**Kernel guard** (`evolution.h:80`): `if( k <= 2 || k >= NZ6-3 ) return;`
→ k=2 完全不進入計算循環

**當前壁面 BC** (`evolution.h:147-153`):
```cuda
if( k == 3 ){
    F5_in  = f6_old[index];   // simple bounce-back
    F11_in = f14_old[index];  // 5 個 z+ 方向反彈
    F12_in = f13_old[index];
    F15_in = f18_old[index];
    F16_in = f17_old[index];
}
```
這是 **halfway bounce-back**：有效壁面位置在 k=2.5（計算坐標），物理空間約 H(y) + minSize/4。

**BFL (Bouzidi-Firdaouss-Lallemand)** (`evolution.h:169-223`):
在 k=3,4 處，針對 y 方向穿越山丘表面的速度方向（F3, F4, F15, F16 及對角方向），用 Q 參數插值處理曲面邊界。

### 16.2 方案 A 詳細分析：維持 BC at k=3

#### 優點
1. **無需修改 kernel guard**：現有 `k <= 2` 邏輯不動
2. **度量項全用中心差分**：k=3 處 `dz_dk = (z[4]-z[2])/2 = 3·minSize/4`，二階精度
3. **7 階插值 stencil 安全**：cell_z 已有 clamp 機制，k=3 → cell_z=3 → stencil k=3..9
4. **已有 BFL 基礎設施**：y 方向曲面處理已實作完畢

#### 缺點
1. **壁面位置偏差**：bounce-back 有效壁面在 k=2.5，不在 H(y)
   - 物理空間偏差 ≈ minSize/4 ≈ 0.0048（相對山丘高度 1.0 為 0.48%）
   - 這是一階壁面位置誤差，不是二階
2. **與貼體座標哲學矛盾**：body-fitted 的核心意義就是消除壁面幾何近似
3. **Inamuro / C-E 施加困難**：
   - 這兩種方法設計時假設 BC 點 **就在牆面上**（u_wall = 0 在該點精確成立）
   - 若在 k=3 施加：u_physical(k=3) ≠ 0（k=3 是流體內點，有非零速度）
   - 需額外修正 / 外推 → 增加複雜度且引入額外誤差源

#### 方案 A 的補償方法（若堅持 k=3）

| 方法 | 做法 | 精度 | 複雜度 |
|------|------|------|--------|
| BFL 插值 (q=0.5) | 已知壁面在 k=2，用 q=0.5 做插值反彈 | O(Δx²) | 低：已有 BFL 基礎設施 |
| Non-Eq. Extrapolation | f_wall = f^eq(ρ_w, 0) + (1-ω)·f^neq\|_{k=3} | O(Δx) | 低 |
| 速度外推 | 從 k=3,4,5 外推 u 到 k=2，再設 u_wall=0 | O(Δx²) | 中 |
| 修正 Inamuro | 在 k=3 施加 Inamuro 但修正壁面距離 | O(Δx²) | 高：需改 Inamuro 公式 |

**最務實的 A 方案補償**：BFL 插值 (q=0.5)，精度二階且利用現有程式碼。但這本質上是在 k=3 做近似，不如 Option B 直接在 k=2 精確施加。

### 16.3 方案 B 詳細分析：將 k=2 納入計算域

#### 核心概念
```
方案 B 的 k 角色重新定義：

k=0,1  → ghost layer（不變，目前也未使用）
k=2    → 壁面計算點 (wall node) ← 新角色：施加 Inamuro / C-E BC
k=3    → 第一內部計算點 (first interior node)
 ...   → 正常內部計算點
k=66   → 倒數第二內部計算點
k=67   → 頂壁計算點 (top wall node) ← 同步修改
k=68,69 → ghost layer
```

#### 優點

1. **壁面精確落在格點上**：z[k=2] = H(y)，幾何零誤差
2. **Inamuro / C-E 自然施加**：u_wall = 0 在 k=2 精確成立
3. **與 GILBM 文獻一致**：Imamura (2005) 的推導假設壁面是座標線
4. **消除 BFL 的必要性**：body-fitted 座標系已將山丘映射為直線 (k=2)
   → y 方向的 BFL 處理 **理論上不再需要**（大幅簡化 kernel）
5. **dk_dy 在壁面精度更高**（詳見 16.4）

#### 代價

1. **dk_dz 需 one-sided difference**：k=2 無下方鄰居 (k=1 未初始化)
2. **修改 kernel guard**：`k <= 2` → `k <= 1`（或 `k < 2`）
3. **插值 stencil 需調整**：k=2 的 cell_z = k-3 = -1，需特殊處理
4. **需計算 k=2 處的分佈函數**：streaming + collision + BC

#### 代價的詳細評估

**代價 1 (one-sided difference)**：見 16.4 小節 — 影響極小

**代價 2 (kernel guard)**：一行改動
```cuda
// 修改前：
if( k <= 2 || k >= NZ6-3 ) return;
// 修改後：
if( k <= 1 || k >= NZ6-2 ) return;  // 允許 k=2 和 k=67 進入計算
```

**代價 3 (插值 stencil)**：
- k=2 的 cell_z = k-3 = -1 → 被 clamp 到 cell_z=3 → stencil k=3..9
- 這意味著 k=2 處「從內部來的」分佈函數（ẽ_α_ζ < 0 的方向），其上風點在 k≥3，插值 stencil 完全在有效域內
- 「從壁外來的」分佈函數（ẽ_α_ζ > 0）由 BC 提供，不需要插值
- **結論：現有 clamp 機制已足夠，無需大改**

**代價 4 (k=2 計算流程)**：
```
k=2 壁面節點的計算流程：
1. 對 ẽ_α_ζ < 0 的方向：正常 streaming（插值從 k≥3 內部取值）
2. 對 ẽ_α_ζ > 0 的方向：施加 Inamuro / C-E BC
3. 執行碰撞（MRT collision with Jacobian correction）
4. 將結果寫入 f_new[k=2]
```
這與 k=3 的流程幾乎相同，差別僅在 BC 的施加方式。

### 16.4 度量項精度分析：one-sided difference 的實際影響

#### dk_dy 在 k=2：**中心差分 + 解析已知 → 無精度損失**

```
dk_dy = -dz_dj / (dy · dz_dk)
```

其中 dz_dj 在 k=2 處用 **j 方向中心差分**（NOT k 方向！）：
```
dz_dj|_{k=2} = (z[(j+1)*NZ6+2] - z[(j-1)*NZ6+2]) / 2
             = (H(y[j+1]) - H(y[j-1])) / 2
             = dH/dy  (中心差分近似)
```

**關鍵洞察**：
- dz_dj 在 k=2 只涉及 j±1 的鄰居（同一個 k=2 層），**不需要 k 方向的鄰居**
- z[j*NZ6+2] = H(y[j])，所以 dz_dj = dH/dy（山丘導數），這是已知的分段三次多項式
- 甚至可以用 **解析微分**（分段二次多項式）取代離散差分，精度更高！
- **結論：dk_dy 在 k=2 的精度 ≥ 內部點，不是瓶頸**

#### dk_dz 在 k=2：one-sided forward difference

```
dk_dz = 1 / dz_dk
```

dz_dk 在 k=2 需 forward difference（k=1 未定義）：

**一階 forward**：`dz_dk = z[3] - z[2] = minSize/2`
**二階 forward**：`dz_dk = (-3·z[2] + 4·z[3] - z[4]) / 2`

數值驗證（平坦段 H=0）：
```
z[2] = 0,  z[3] = minSize/2,  z[4] = 3·minSize/2

一階: dz_dk = minSize/2 ≈ 0.009544
二階: dz_dk = (-3×0 + 4×minSize/2 - 3·minSize/2)/2 = minSize/4 ≈ 0.004772
中心 at k=3: dz_dk = (z[4]-z[2])/2 = 3·minSize/4 ≈ 0.01432
```

**哪個值正確？**

網格在 k=2 有一個「折點」：z[2] = H(y) 是獨立定義的壁面點，不在 tanh 曲線上。
- k=2→k=3 間距 = minSize/2（小）
- k=3→k=4 間距 = minSize（大，由 bisection 保證）

一階 forward (minSize/2) 反映的是 **壁面到第一計算點的實際物理間距**，這是決定逆變速度和 BC 行為的正確尺度。

二階 forward (minSize/4) 試圖捕捉 tanh 曲線的曲率，但 k=2 不在 tanh 曲線上，外推不合理。

**建議：使用一階 forward difference**
```
dk_dz|_{k=2} = 1 / (z[k=3] - z[k=2]) = 1 / (minSize/2) = 2/minSize
```

這給出的 dk_dz 約為 k=3 處值的 2 倍（k=3 處 dk_dz = 4/(3·minSize)），反映壁面附近網格更密集。

#### 對逆變速度判別的影響

在 k=2 壁面，dk_dz ≈ 2/minSize 而非 k=3 的 4/(3·minSize)。dk_dy 則幾乎相同。

逆變速度 ẽ_α_ζ = e_αy · dk_dy + e_αz · dk_dz

dk_dz 較大 → e_αz 的貢獻被放大 → 壁面的方向判別可能與 k=3 略有不同，但方向分類的定性結論不會改變（只影響邊界方向，dk_dz > 0 恆成立）。

### 16.5 四個問題的具體回答

#### Q1: 方案 B 是否值得引入 one-sided difference 的額外複雜性？

**值得。** 原因：
1. **dk_dy 不需要 one-sided**：它用 j 方向中心差分，精度不降
2. **dk_dz 的 one-sided 只需一行公式**：`dk_dz = 1/(z[k=3] - z[k=2])`
3. 換取的是 **壁面位置零誤差** 和 **Inamuro/C-E 的自然施加**
4. one-sided 的精度降低只影響 dk_dz 在 k=2 這一層，不影響 k≥3 的所有內部點

#### Q2: Inamuro / Chapman-Enskog 方法在曲線座標下的實作注意事項

1. **平衡態用物理速度**：f^eq(ρ, u_physical)，不是計算座標速度
2. **Jacobian 修正碰撞**：碰撞算子中 Δt 需乘以局部 Jacobian (J = dx·dy·dz_dk)
3. **C-E BC 的應變率張量**需通過度量項轉換：
   ```
   ∂u/∂z = (∂u/∂ζ) · (∂ζ/∂z)     ← 需要 dk_dz
   ∂u/∂y = (∂u/∂η) · (∂η/∂y) + (∂u/∂ζ) · (∂ζ/∂y)   ← 需要 dk_dy
   ```
4. **壁面速度梯度**用單側差分：∂u/∂ζ|_{ζ=2} ≈ (u[k=3] - u[k=2]) / 1 = u[k=3]（因 u[k=2]=0）
5. **需要 BC 的方向集合隨 (i,j) 變化**：由 sign(ẽ_α_ζ) 動態判斷（判據 5-6 已驗證此邏輯）
6. **ρ_wall 的確定**：Inamuro 方法需迭代求壁面密度以滿足質量守恆；C-E BC 可從鄰近流體節點外推 ρ

#### Q3: 若維持方案 A，可用的補償方法

最實用的選項：

1. **BFL 插值 (q=0.5)**：利用已有的 BFL 基礎設施，在 k=3 做 q=0.5 的插值反彈。
   ```cuda
   // q = 0.5 時 BFL 公式退化為：
   F_alpha = f_alpha_bar[k=3]   // 即 simple bounce-back
   ```
   注意：當 q = 0.5 時 BFL **就是** halfway bounce-back！所以現狀已是此方案的特例。

2. **Guo et al. (2002) 非平衡外推**：精度較高但需從 k=3,4 外推到牆面。

3. **修正壁面距離的 Inamuro**：在 k=3 施加 Inamuro，但修正公式以考慮壁面不在 k=3 而在 k=2。文獻中此修正較複雜且不常見。

**結論：方案 A 的最佳補償仍不如方案 B 的幾何精確性。**

#### Q4: 整體建議

**建議採用方案 B（k=2 作為壁面計算點）。**

理由的權重排序：

| 排名 | 理由 | 權重 |
|------|------|------|
| 1 | 貼體座標的核心精神：BC 在 body 上 | **根本性** |
| 2 | Inamuro/C-E 方法自然需要壁面節點 | **方法論必要** |
| 3 | dk_dy 在壁面可用中心差分（甚至解析） | 消除主要顧慮 |
| 4 | dk_dz 的 one-sided 只是一行公式 | 代價極低 |
| 5 | 消除 BFL 的必要性（大幅簡化 kernel） | 額外收益 |
| 6 | 與 Imamura (2005) 文獻一致 | 學術正確性 |

### 16.6 方案 B 的實作路線圖

#### Phase 0 修改（度量項計算 + 診斷）

**檔案**：`gilbm/metric_terms.h`

1. `ComputeMetricTerms()` 擴展 k 範圍：
   ```cpp
   // 修改前：for (int k = 3; k < NZ6_local - 3; k++)
   // 修改後：for (int k = 2; k < NZ6_local - 2; k++)
   //
   // k=2 特殊處理：
   if (k == 2) {
       dz_dk = z_h[j*NZ6 + 3] - z_h[j*NZ6 + 2];  // 一階 forward
   } else if (k == NZ6_local - 3) {
       dz_dk = z_h[j*NZ6 + k] - z_h[j*NZ6 + k-1]; // 一階 backward (頂壁)
   } else {
       dz_dk = (z_h[j*NZ6 + k+1] - z_h[j*NZ6 + k-1]) / 2.0;  // 中心差分
   }
   // dz_dj 不變（j 方向中心差分，k=2 有效）
   ```

2. `DiagnoseMetricTerms()` 的診斷範圍同步擴展

#### Phase 1 修改（GILBM kernel）

**檔案**：`evolution.h`（或未來的 `evolution_gilbm.h`）

1. Kernel guard：`k <= 2` → `k <= 1`（底壁）；`k >= NZ6-3` → `k >= NZ6-2`（頂壁）
2. k=2 和 k=67 的 BC 施加邏輯（取代現有 bounce-back）
3. 刪除 BFL y 方向處理（body-fitted 已處理山丘曲率）

#### 記憶體影響

度量項陣列大小不變（仍是 `[NYD6*NZ6]`），只是填入了 k=2 和 k=67 的值。

### 16.7 關鍵檔案修改清單

| 檔案 | 修改 | Phase |
|------|------|-------|
| `gilbm/metric_terms.h` | ComputeMetricTerms() 擴展到 k=2, NZ6-3 | Phase 0 |
| `gilbm/metric_terms.h` | DiagnoseMetricTerms() 更新診斷範圍 | Phase 0 |
| `evolution.h` | kernel guard 修改 | Phase 1 |
| `evolution.h` | k=2 BC 邏輯（Inamuro/C-E） | Phase 1 |
| `evolution.h` | 刪除 BFL y 方向處理 | Phase 1 |
| `initialization.h` | GenerateMesh_Z() — 確認 k=2 已正確初始化（已是） | 確認 |
| `variables.h` | 無修改 | — |

### 16.8 驗證計劃

1. **Phase 0 診斷**：擴展後的度量項在 k=2 層的數值應合理
   - dk_dz[k=2] ≈ 2/minSize（平坦段）
   - dk_dy[k=2] ≈ 0（平坦段）
   - dk_dy[k=2] 符號與 -dH/dy 一致（斜面段）
2. **方向判別**：k=2 處的 BC 方向集合應與 k=3 的定性一致
3. **Poiseuille 流**：壁面速度 < 1e-6，壁面剪應力與解析解匹配
4. **Periodic Hill Re=200**：壁面剪應力與 Mellen (2000) DNS 對比

---

## 十七、方案 B 完整實作計劃：k=2 成為壁面計算點

### Context

本節是十六節「建議採用方案 B」的具體實作。核心變更：**將 buffer layer 從 3 層縮減為 2 層，使 k=2 成為底壁計算點（wet-node），k=NZ6-3 成為頂壁計算點。** k=0,1 保持為 ghost layer（periodic BC 填充），k=NZ6-2,NZ6-1 同理。

### 關鍵設計決策

**NZ6 定義不變！** `#define NZ6 (NZ+6)` 保持為 70。理由：
- NZ6 同時決定了 3D 陣列大小 `[NX6 × NYD6 × NZ6]`、MPI 通訊 buffer、xi_h 陣列大小
- 改 NZ6 會牽動**整個記憶體佈局**和所有 `sizeof(double)*NX6*NYD6*NZ6` 的分配
- 相反，只改 `bfr` 和相應的迴圈範圍/偏移量，影響範圍可控

**結果：**
- 舊：k=0,1,2 ghost/buffer | k=3..66 計算 | k=67,68,69 ghost/buffer → 64 個計算點
- 新：k=0,1 ghost | k=2..67 計算（含壁面） | k=68,69 ghost → 66 個計算點
- tanh 分割數 N = NZ6-7 = 63（不變）
- 物理計算點增加 2（底壁 k=2 + 頂壁 k=67），但 tanh 內部點數不變

### 數學映射

```
舊設計 (bfr=3):
  k=2:  z = H(y)               ← 壁面坐標（不計算）
  k=3:  z = tanh(j=0) + H(y)   = minSize/2 + H(y)   ← 第一計算點
  k=66: z = tanh(j=63) + H(y)  = LZ - minSize/2      ← 最後計算點
  k=67: z = LZ                  ← 頂壁坐標（不計算）
  tanh 映射：j = k-3, N = NZ6-7 = 63

新設計 (bfr=2):
  k=2:  z = H(y)               ← 壁面計算點（u=0 BC）
  k=3:  z = tanh(j=0) + H(y)   = minSize/2 + H(y)   ← 第一內部點（不變！）
  k=66: z = tanh(j=63) + H(y)  = LZ - minSize/2      ← 最後內部點（不變！）
  k=67: z = LZ                  ← 頂壁計算點（u=0 BC）
  tanh 映射：j = k-3, N = NZ6-7 = 63                 ← 不變！
```

**核心洞察**：tanh 的 j 和 N 參數**完全不變**。k=3..66 的所有 z 座標保持原樣。唯一的改變是 k=2 和 k=67 從「不計算」變為「壁面計算點」。

---

### 十七.1 逐檔案修改清單

---

#### 檔案 1: `variables.h`

| 行 | 現在 | 修改後 | 說明 |
|---|---|---|---|
| 19 | `#define NZ6 (NZ+6)` | **不變** | NZ6 定義不變 |
| 23 | `#define minSize ((LZ-1.0)/(NZ6-6)*CFL)` | **不變** | minSize = 壁面附近最小格距，由 tanh 拉伸控制，與 bfr 無關 |

**結論：`variables.h` 不需修改。**

---

#### 檔案 2: `initialization.h`

##### 2a. `GenerateMesh_X()` (lines 51-75)

```c
// line 53: int bfr = 3;
// line 56: dx = LX / (double)(NX6-2*bfr-1);
// line 58: x_h[i] = dx*((double)(i-bfr));
```

**不修改。** X 方向 buffer 仍為 3（NX6=NX+7 的定義就是為 bfr=3 設計的）。本次只改 Z 方向。

##### 2b. `GenerateMesh_Y()` (lines 77-108)

**不修改。** Y 方向 buffer 仍為 3。

##### 2c. `GenerateMesh_Z()` (lines 110-169) — 需要修改

**修改前** (lines 111-127):
```c
int bfr = 3;
// ...
for( int k = bfr; k < NZ6-bfr; k++ ){                    // k=3..66
    z_h[j*NZ6+k] = tanhFunction( total, minSize, a, (k-3), (NZ6-7) )
                   + HillFunction( y_h[j] );
}
z_h[j*NZ6+2] = HillFunction( y_h[j] );                   // k=2
z_h[j*NZ6+(NZ6-3)] = (double)LZ;                          // k=67
```

**修改後**:
```c
int bfr = 3;  // ← X/Y 方向 buffer 仍為 3
int bfr_z = 2;  // ← Z 方向 buffer 改為 2
// ...
// tanh 內部點：k=3..66，映射 j=k-3, N=NZ6-7（完全不變）
for( int k = 3; k < NZ6-3; k++ ){
    z_h[j*NZ6+k] = tanhFunction( total, minSize, a, (k-3), (NZ6-7) )
                   + HillFunction( y_h[j] );
}
// 壁面計算點（新：這些現在是計算域的一部分）
z_h[j*NZ6+2] = HillFunction( y_h[j] );                    // k=2: 底壁
z_h[j*NZ6+(NZ6-3)] = (double)LZ;                           // k=67: 頂壁
```

**關鍵：tanh 迴圈的 k 範圍和映射公式完全不變！** 只是語義改變：k=2 和 k=67 從「buffer 座標」升級為「壁面計算座標」。

**xi_h 映射** (lines 130-132):
```c
// 現在：k=3..66，xi_h[k] 只在這個範圍有值
for( int k = bfr; k < NZ6-bfr; k++ ){
    xi_h[k] = tanhFunction( LXi, minSize, a, (k-3), (NZ6-7) ) - minSize/2.0;
}
```

**修改後**：需要為 k=2 和 k=67 也賦予 xi_h 值（因為它們現在是計算點）：
```c
// tanh 內部點不變
for( int k = 3; k < NZ6-3; k++ ){
    xi_h[k] = tanhFunction( LXi, minSize, a, (k-3), (NZ6-7) ) - minSize/2.0;
}
// 壁面計算點的 xi 值（外推）
xi_h[2] = 2.0 * xi_h[3] - xi_h[4];           // k=2: 線性外推
xi_h[NZ6-3] = 2.0 * xi_h[NZ6-4] - xi_h[NZ6-5]; // k=67: 線性外推
```

為什麼用線性外推？因為 k=2 在 tanh 曲線之外（tanh 只定義 j=0..63 → k=3..66），壁面的 xi 值需要合理定義以供插值 stencil 使用。線性外推保證 xi_h 在 k=2,3,4 之間的順序正確。

**z_global** (lines 135-147): 同理修改——tanh 範圍不變，k=2 和 k=67 的賦值不變，只需確保 xi_h 的外推也應用到 z_global 的對應邏輯中（z_global 本身不需要 xi 外推，它的 k=2 和 k=67 賦值已經正確）。

##### 2d. `GetXiParameter()` (lines 171-187) — 需要修改

**修改前** (lines 180-186):
```c
if( k >= 3 && k <= 6 ){
    GetParameter_6th( XiPara_h, pos_xi, Pos_xi, IdxToStore, 3 );
} else if ( k >= NZ6-7 && k <= NZ6-4 ) {
    GetParameter_6th( XiPara_h, pos_xi, Pos_xi, IdxToStore, NZ6-10 );
} else {
    GetParameter_6th( XiPara_h, pos_xi, Pos_xi, IdxToStore, k-3 );
}
```

**修改後**: 擴展壁面附近的 clamping 範圍以包含 k=2 和 k=67：
```c
if( k >= 2 && k <= 6 ){                                    // 加入 k=2
    GetParameter_6th( XiPara_h, pos_xi, Pos_xi, IdxToStore, 3 );
} else if ( k >= NZ6-7 && k <= NZ6-3 ) {                   // 加入 k=NZ6-3
    GetParameter_6th( XiPara_h, pos_xi, Pos_xi, IdxToStore, NZ6-10 );
} else {
    GetParameter_6th( XiPara_h, pos_xi, Pos_xi, IdxToStore, k-3 );
}
```

注意：stencil 起始位置仍為 `3`（即 xi_h[3]），因為 xi_h 的有效 tanh 值從 k=3 開始。k=2 的插值仍使用 stencil k=3..9 的 xi 值。

##### 2e. `GetIntrplParameter_Xi()` (lines 217-244) — 需要修改

**修改前** (line 220):
```c
for( int k = 3; k < NZ6-3; k++ )
```

**修改後**:
```c
for( int k = 2; k < NZ6-2; k++ )   // 包含壁面計算點 k=2 和 k=67
```

但注意：壁面點 k=2 和 k=67 的插值參數會觸發 `GetXiParameter` 的 clamping 邏輯（§2d），stencil 被固定在 3 或 NZ6-10，這是正確的。

##### 2f. `BFLInitialization()` (lines 246-346) — **暫不修改，Phase 1 再處理**

BFL 處理 y 方向穿越山丘表面的速度方向。在方案 B 的最終形態下，body-fitted 座標已將山丘映射為直線 k=2，**理論上 BFL 不再需要**。但在過渡期間：

- Phase 0（本次）：保持 BFL 不動，讓 ISLBM 部分繼續正常運行
- Phase 1（GILBM kernel）：新的 GILBM kernel 不使用 BFL，改用逆變速度方向判別
- 最終：BFL 相關代碼可以在 GILBM 完全驗證後移除

**如果需要讓 ISLBM 在方案 B 下繼續工作**（過渡期），BFL 的 `k+3` 偏移需要改為 `k+2`。但由於 GILBM 不使用 BFL，暫時跳過。

---

#### 檔案 3: `initializationTool.h`

##### 3a. `tanhFunction` macro (lines 4-7)

**不修改。** tanhFunction 本身不含任何 bfr 依賴。它接受 (L, LatticeSize, a, j, N) 參數，完全由調用端控制。

##### 3b. `GetNonuniParameter()` (lines 9-33)

**不修改。** 它使用 `NZ6-7` 作為 N 參數，這等於 tanh 的分割數 63。在方案 B 中 tanh 分割數不變，所以 `a` 參數的計算結果不變。

##### 3c. `GetBFLXiParameter()` (lines 123-135)

**修改前** (line 132):
```c
if( k >= 3 && k <= 4 ){
    GetParameter_6th( XiPara_h, pos_xi, Pos_xi, IdxToStore, 3 );
```

**修改後** (若需要讓 BFL 在過渡期工作):
```c
if( k >= 2 && k <= 4 ){                                    // 加入 k=2
    GetParameter_6th( XiPara_h, pos_xi, Pos_xi, IdxToStore, 3 );
```

**暫時跳過**（與 §2f BFL 相同理由）。

---

#### 檔案 4: `evolution.h` — 核心 CUDA kernel

##### 4a. Kernel guard — `stream_collide_Buffer` (line 80)

**修改前**:
```cuda
if( i <= 2 || i >= NX6-3 || k <= 2 || k >= NZ6-3 ) return;
```

**修改後**:
```cuda
if( i <= 2 || i >= NX6-3 || k <= 1 || k >= NZ6-2 ) return;
```

k 的有效範圍從 [3, NZ6-4] 擴展到 [2, NZ6-3]。i 的 guard 不變。

##### 4b. Kernel guard — `stream_collide` (line 369)

**修改前**:
```cuda
if( i <= 2 || i >= NX6-3 || j <= 6 || j >= NYD6-7 || k <= 2 || k >= NZ6-3 ) return;
```

**修改後**:
```cuda
if( i <= 2 || i >= NX6-3 || j <= 6 || j >= NYD6-7 || k <= 1 || k >= NZ6-2 ) return;
```

##### 4c. cell_z 計算和 clamping (lines 96-98, 387-389) — 兩個 kernel 各一處

**修改前**:
```cuda
int cell_z = k-3;
if( k <= 6 ) cell_z = 3;
if( k >= NZ6-7 ) cell_z = NZ6-10;
```

**修改後**:
```cuda
int cell_z = k-3;                        // 映射不變：k=3 → cell_z=0
if( k <= 6 ) cell_z = 3;                 // 包含 k=2 → cell_z=3（壁面用 stencil k=3..9）
if( k >= NZ6-7 ) cell_z = NZ6-10;       // 不變：k=67 → cell_z=NZ6-10（stencil k=60..66）
```

**結論：clamping 邏輯不需要修改！** 因為 k=2 的 cell_z = k-3 = -1 → 被 `k <= 6` 條件攔截 → cell_z = 3。這意味著 k=2 的插值 stencil 從 k=3 開始，完全在有效域內。

##### 4d. 底壁 bounce-back (lines 147-153, 438-444) — 兩個 kernel 各一處

**修改前**:
```cuda
if( k == 3 ){
    F5_in  = f6_old[index];
    F11_in = f14_old[index];
    F12_in = f13_old[index];
    F15_in = f18_old[index];
    F16_in = f17_old[index];
}
```

**修改後**: 將 BC 施加點從 k=3 移到 k=2
```cuda
if( k == 2 ){                            // 壁面在 k=2
    F5_in  = f6_old[index];
    F11_in = f14_old[index];
    F12_in = f13_old[index];
    F15_in = f18_old[index];
    F16_in = f17_old[index];
}
```

注意：對於 Phase 1 ISLBM（非 GILBM），bounce-back 在壁面 k=2 施加是正確的——這是 on-grid (wet-node) bounce-back，壁面精確在格點上。

##### 4e. 頂壁 bounce-back (lines 154-160, 445-451) — 兩個 kernel 各一處

**修改前**:
```cuda
if( k == NZ6-4 ){     // k=66
    F6_in  = f5_old[index];
    ...
}
```

**修改後**:
```cuda
if( k == NZ6-3 ){     // k=67，壁面在 k=67
    F6_in  = f5_old[index];
    F13_in = f12_old[index];
    F14_in = f11_old[index];
    F17_in = f16_old[index];
    F18_in = f15_old[index];
}
```

##### 4f. BFL 處理 (lines 169-223, 459-513)

**修改前**:
```cuda
if( k == 3 || k == 4 ) {
    idx_xi = (k-3)*NYD6+j;
    ...
}
```

**過渡期修改**：BFL 仍在 k=3,4 運行（因為山丘表面仍在 y 方向穿越這些格點附近）。
```cuda
if( k == 3 || k == 4 ) {                 // BFL 仍作用在第一、二內部點
    idx_xi = (k-3)*NYD6+j;               // 索引不變
    ...
}
```

**或者**若要讓 BFL 也覆蓋 k=2（壁面點，某些 y 方向的分佈函數可能需要 BFL 修正）：
```cuda
if( k == 2 || k == 3 || k == 4 ) {
    idx_xi = (k-2)*NYD6+j;               // 偏移改為 k-2，三層
    ...
}
```
→ 需要同步修改 BFL 記憶體分配為 `3*NYD6`（見 §8 memory.h）。

**建議**：Phase 0 暫時不改 BFL。k=2 的壁面 BC 由 bounce-back (§4d) 完全處理。BFL 繼續在 k=3,4 運行。

##### 4g. `periodicNML` — Z 方向 periodic BC (lines 717-759)

**修改前** (line 727):
```cuda
int buffer = 3;
```

**修改後**: Z 方向 buffer 改為 2
```cuda
int buffer = 2;    // Z 方向 ghost layer 從 3 縮減為 2
```

這影響 periodic Z 邊界的複製邏輯：
```
舊：z ghost k=0,1,2 ← 從 k=64,65,66 複製  (offset = NZ6-2*3-1 = NZ6-7)
新：z ghost k=0,1   ← 從 k=65,66 複製      (offset = NZ6-2*2-1 = NZ6-5)
```

**但是**：在 Periodic Hill 中，Z 方向**不是**周期性的（壁面邊界）！`periodicNML` 被調用但 Z 方向的 periodic BC 實際上被壁面 BC 覆蓋。需要確認 Z-periodic 是否真的被使用：

- 如果 Z 方向有 periodic kernel call → 需要改 buffer=2
- 如果 Z 方向的 k=0,1 ghost 從未被讀取（因為 kernel guard 已排除）→ 可以不改

**確認**：kernel guard `k <= 1` 排除了 k=0,1。但 k=2 的 bounce-back BC 需要讀 `f6_old[index]` 等，而 `f6_old` 在 k=2 本身的記憶體位置，**不需要** k=0,1 的數據。**結論：periodic Z 暫不修改。**

##### 4h. `periodicUD`, `periodicSW` — Y/X 方向 periodic BC

**不修改。** Y 和 X 方向的 buffer 仍為 3。

##### 4i. `AccumulateUbulk` (line 770)

**修改前**:
```cuda
if( i <= 2 || i >= NX6-3 || k <= 2 || k >= NZ6-3 ) return;
```

**修改後**:
```cuda
if( i <= 2 || i >= NX6-3 || k <= 1 || k >= NZ6-2 ) return;
```

注意：AccumulateUbulk 計算壁面附近的體積流量。k=2（壁面，u=0）也應被包含在積分中（壁面 u=0 對積分沒有貢獻，但 z 座標差 `z[k+1]-z[k-1]` 需要 k=1 的 z 值有效）。

**問題**：AccumulateUbulk 的 dz 計算（line 773）：
```cuda
double dz = z[j*NZ6+k+1] - z[j*NZ6+k-1];
```
在 k=2 時需要 `z[k-1] = z[1]`，但 k=1 的 z_h **目前未被初始化**！

**修正**：在 GenerateMesh_Z() 中為 k=1 和 k=NZ6-2 賦予 ghost z 值：
```c
// 在 GenerateMesh_Z() 中新增：
z_h[j*NZ6+1] = 2.0 * z_h[j*NZ6+2] - z_h[j*NZ6+3];     // k=1: 線性外推
z_h[j*NZ6+(NZ6-2)] = 2.0 * z_h[j*NZ6+(NZ6-3)] - z_h[j*NZ6+(NZ6-4)]; // k=68: 線性外推
```

這些 ghost z 值只用於差分 stencil，不影響物理計算。

##### 4j. `Launch_CollisionStreaming` — kernel launch (lines 778-857)

**修改前** (lines 779, 786):
```cuda
int buffer = 3;
dim3 blockdimNML(NT, 1, buffer);
```

**Z 方向 periodic launch 暫不修改**（見 §4g 分析）。

**修改前** (line 821):
```cuda
stream_collide_Buffer<<<griddimBuf, blockdimBuf, 0, stream0>>>(..., 3, rho_modify_d, ...);
```
最後的 `3` 是 `start` 參數（j 的起始值），與 k 無關。**不修改。**

##### 4k. `Launch_ModifyForcingTerm` (line 928)

**修改前**:
```cuda
for( int k = 3; k < NZ6-3; k++ ){
```

**修改後**: 包含壁面計算點（雖然壁面 u=0，forcing 在壁面可能不需要，但統一範圍更安全）：
```cuda
for( int k = 2; k < NZ6-2; k++ ){
```

---

#### 檔案 5: `gilbm/metric_terms.h`

##### 5a. `ComputeMetricTerms()` (lines 22-51) — 需要修改

**修改前** (lines 33-34):
```cpp
for (int j = 3; j < NYD6_local - 3; j++) {
    for (int k = 3; k < NZ6_local - 3; k++) {
```

**修改後**: 擴展 k 範圍至壁面
```cpp
for (int j = 3; j < NYD6_local - 3; j++) {
    for (int k = 2; k < NZ6_local - 2; k++) {     // 包含 k=2 和 k=NZ6-3
```

**新增壁面差分邏輯** (在 k-loop 內部):
```cpp
double dz_dk, dz_dj;

// dz_dk：k 方向差分
if (k == 2) {
    // 壁面：一階 forward difference
    dz_dk = z_h[j * NZ6_local + 3] - z_h[j * NZ6_local + 2];
} else if (k == NZ6_local - 3) {
    // 頂壁：一階 backward difference
    dz_dk = z_h[j * NZ6_local + k] - z_h[j * NZ6_local + (k - 1)];
} else {
    // 內部：中心差分（不變）
    dz_dk = (z_h[j * NZ6_local + (k + 1)] - z_h[j * NZ6_local + (k - 1)]) / 2.0;
}

// dz_dj：j 方向中心差分（所有 k 都一樣，包含壁面）
dz_dj = (z_h[(j + 1) * NZ6_local + k] - z_h[(j - 1) * NZ6_local + k]) / 2.0;

// 度量項計算（不變）
int idx = j * NZ6_local + k;
dk_dz_h[idx] = 1.0 / dz_dk;
dk_dy_h[idx] = -dz_dj / (dy * dz_dk);
```

**壁面 dz_dk 的數值**：
- k=2 (底壁): `dz_dk = z[3] - z[2] = minSize/2 ≈ 0.00955`
- k=67 (頂壁): `dz_dk = z[67] - z[66] = LZ - (LZ-minSize/2) = minSize/2 ≈ 0.00955`

**壁面 dz_dj 的數值**：
- k=2: `dz_dj = (H(y[j+1]) - H(y[j-1])) / 2 = dH/dy`（山丘導數，可解析計算）
- k=67: `dz_dj = 0`（頂壁 z=LZ 不隨 j 變化）

##### 5b. `DiagnoseMetricTerms()` — 需要修改

**全域 z 座標生成** (line 80): 不變（tanh 範圍 k=3..66 不變，k=2 和 k=67 的壁面賦值不變）。

**全域度量項計算** (在 DiagnoseMetricTerms 內部): 需要將 ComputeMetricTerms 調用的 k 範圍從 `bfr..NZ6-bfr` 擴展到 `2..NZ6-2`。

**k_wall 定義** (line 183):
```cpp
// 修改前：int k_wall = bfr;  // = 3
// 修改後：
int k_wall = 2;  // 壁面現在在 k=2
```

**判據 1** (dk_dz > 0): k 範圍從 `bfr..NZ6-bfr` → `2..NZ6-2`
**判據 2** (壁面 dz_dk): 壁面 k 從 `bfr` → `2`
**判據 3** (dk_dy ≈ 0): k 範圍包含 k=2
**判據 5-6** (方向判別): k_wall 改為 2

---

#### 檔案 6: `main.cu`

##### 6a. MPI Buffer (lines 114-119)

**修改前**:
```c
int Buffer     = 3;
int icount_sw  = Buffer * NX6 * NZ6;
int iToLeft    = (Buffer+1) * NX6 * NZ6;
int iFromLeft  = 0;
int iToRight   = NX6 * NYD6 * NZ6 - (Buffer*2+1) * NX6 * NZ6;
int iFromRight = iToRight + (Buffer+1) * NX6 * NZ6;
```

**不修改。** 這是 Y 方向的 MPI 通訊 buffer（3 層 ghost/overlap）。Y 方向 buffer 不變。

##### 6b. Rho 求和迴圈 (lines 247-252, 267-272)

**修改前**:
```c
for( int k = 3 ; k < NZ6-3; k++){
```

**修改後**: 包含壁面 k=2 和 k=67
```c
for( int k = 2 ; k < NZ6-2; k++){
```

##### 6c. Rho 正規化 (lines 255-256)

**修改前**:
```c
rho_global = 1.0*(NX6-7)*(NY6-7)*(NZ6-6);
rho_modify_h[0] = ( rho_global - rho_GlobalSum ) / ((NX6-7)*(NY6-7)*(NZ6-6));
```

**修改後**: 計算點數從 NZ6-6=64 增加到 NZ6-4=66
```c
rho_global = 1.0*(NX6-7)*(NY6-7)*(NZ6-4);      // 66 個 k 計算點
rho_modify_h[0] = ( rho_global - rho_GlobalSum ) / ((NX6-7)*(NY6-7)*(NZ6-4));
```

##### 6d. Local rho average (line 274)

**修改前**:
```c
rho_LocalAvg = rho_LocalSum / ((NX6-7)*(NYD6-7)*(NZ6-6));
```

**修改後**:
```c
rho_LocalAvg = rho_LocalSum / ((NX6-7)*(NYD6-7)*(NZ6-4));
```

---

#### 檔案 7: `statistics.h`

##### 7a. `KineticEnergy_dissipation` kernel guard (line 21)

**修改前**:
```cuda
if( i <= 2 || i >= NX6-3 || j <= 2 || j >= NYD6-3 || k <= 2 || k >= NZ6-3 ) return;
```

**修改後**:
```cuda
if( i <= 2 || i >= NX6-3 || j <= 2 || j >= NYD6-3 || k <= 1 || k >= NZ6-2 ) return;
```

注意：此 kernel 使用 4 階差分 stencil（`index±2*NX6` 即 k±2），在 k=2 時需要 k=0 的數據。**但 k=0 是 ghost layer**，其速度未初始化。

**解決方案**：將此 kernel 的 k guard 保持為 `k <= 2 || k >= NZ6-3`（壁面點不參與 KE/dissipation 統計），或確保 k=0,1 的 u,v,w 被 periodic Z BC 填充。

**建議**：statistics kernel 暫不改 k guard，壁面 u=0 對統計量貢獻為零。

##### 7b. `Reynolds_stress_avg` 和 `gradient_velocity` (lines 138, 192)

同上分析，**暫不修改**。

##### 7c. Summation loop (line 89)

**修改前**:
```cuda
for( int k = 3; k < NZ6-4;  k=k+2 ){
```

**修改後** (若需要包含壁面):
```cuda
for( int k = 2; k < NZ6-3;  k=k+2 ){
```

**建議**：統計量暫不改，等 GILBM 穩定後再擴展。

---

#### 檔案 8: `memory.h`

**不修改。** 所有陣列大小基於 NZ6（不變），BFL 陣列 `2*NYD6`（暫不改）。

---

#### 檔案 9: `communication.h`

##### 9a. `SendBdryToCPU_Sideways` (line 202)

**修改前**:
```c
const size_t nBytes = 3 * NX6 * NZ6 * sizeof(double);
```

**不修改。** 這是 Y 方向 MPI 通訊的 buffer 大小（3 層 Y-ghost × NX6 × NZ6）。Y 方向 buffer 仍為 3。

---

#### 檔案 10: `fileIO.h`

##### 10a. VTK output dimensions (line 96)

**修改前**:
```c
out << "DIMENSIONS " << NX6-6 << " " << NYD6-6 << " " << NZ6-6 << "\n";
```

**修改後**: Z 方向計算點數從 NZ6-6=64 變為 NZ6-4=66
```c
out << "DIMENSIONS " << NX6-6 << " " << NYD6-6 << " " << NZ6-4 << "\n";
```

##### 10b. nPoints (line 99)

**修改後**:
```c
int nPoints = (NX6-6) * (NYD6-6) * (NZ6-4);
```

##### 10c. VTK output loops (lines 102, 112)

**修改前**:
```c
for( int k = 3; k < NZ6-3; k++ ){
```

**修改後**:
```c
for( int k = 2; k < NZ6-2; k++ ){
```

##### 10d. statistics write/read (lines 197, 217)

同理：k 範圍從 `3..NZ6-3` → `2..NZ6-2`

##### 10e. merged VTK (lines 324, 343, 356)

```c
// line 324:
const int nzLocal = NZ6 - 6;  →  NZ6 - 4;

// line 343, 356:
for( int k = 3; k < NZ6-3; k++ ){  →  for( int k = 2; k < NZ6-2; k++ ){
```

---

#### 檔案 11: `combinepltv2.cpp`

所有 `NZ6-6` → `NZ6-4`，所有 `k=3; k<NZ6-3` → `k=2; k<NZ6-2`。

具體行號：85, 92, 170, 198, 206, 208, 212, 222, 230, 250, 253, 270, 273, 320。

---

### 十七.2 GenerateMesh_Z() 完整修改後代碼

```c
void GenerateMesh_Z() {
    int bfr = 3;     // X/Y 方向 buffer 仍為 3

    if( Uniform_In_Zdir ){
        printf("Error: Periodic Hill requires non-uniform Z mesh.\n");
        exit(1);
    }

    double a = GetNonuniParameter();

    // === 區域 z_h 座標 ===
    for( int j = 0; j < NYD6; j++ ){
        double total = LZ - HillFunction( y_h[j] ) - minSize;

        // tanh 內部點：k=3..66（映射 j=k-3, N=NZ6-7=63）
        for( int k = 3; k < NZ6-3; k++ ){
            z_h[j*NZ6+k] = tanhFunction( total, minSize, a, (k-3), (NZ6-7) )
                           + HillFunction( y_h[j] );
        }

        // 壁面計算點
        z_h[j*NZ6+2]       = HillFunction( y_h[j] );    // k=2: 底壁
        z_h[j*NZ6+(NZ6-3)] = (double)LZ;                 // k=67: 頂壁

        // Ghost z 值（線性外推，供差分 stencil 使用）
        z_h[j*NZ6+1]       = 2.0 * z_h[j*NZ6+2] - z_h[j*NZ6+3];         // k=1
        z_h[j*NZ6+(NZ6-2)] = 2.0 * z_h[j*NZ6+(NZ6-3)] - z_h[j*NZ6+(NZ6-4)]; // k=68
    }

    // === xi_h 映射座標 ===
    // tanh 內部點（不變）
    for( int k = 3; k < NZ6-3; k++ ){
        xi_h[k] = tanhFunction( LXi, minSize, a, (k-3), (NZ6-7) ) - minSize/2.0;
    }
    // 壁面計算點的 xi 值（線性外推）
    xi_h[2]       = 2.0 * xi_h[3] - xi_h[4];
    xi_h[NZ6-3]   = 2.0 * xi_h[NZ6-4] - xi_h[NZ6-5];

    // === z_global（全域，用於檔案輸出）===
    if( myid == 0 ){
        double dy = LY / (double)(NY6-2*bfr-1);
        double y_global[NY6];
        for( int j = 0; j < NY6; j++ )
            y_global[j] = dy * ((double)(j-bfr));

        for( int j = 0; j < NY6; j++ ){
            double total = LZ - HillFunction( y_global[j] ) - minSize;
            for( int k = 3; k < NZ6-3; k++ ){
                z_global[j*NZ6+k] = tanhFunction( total, minSize, a, (k-3), (NZ6-7) )
                                   + HillFunction( y_global[j] );
            }
            z_global[j*NZ6+2]       = HillFunction( y_global[j] );
            z_global[j*NZ6+(NZ6-3)] = (double)LZ;
            // Ghost z 值
            z_global[j*NZ6+1]       = 2.0 * z_global[j*NZ6+2] - z_global[j*NZ6+3];
            z_global[j*NZ6+(NZ6-2)] = 2.0 * z_global[j*NZ6+(NZ6-3)] - z_global[j*NZ6+(NZ6-4)];
        }
    }
}
```

---

### 十七.3 ComputeMetricTerms() 完整修改後代碼

```cpp
void ComputeMetricTerms(
    double *dk_dz_h, double *dk_dy_h,
    const double *z_h, const double *y_h,
    int NYD6_local, int NZ6_local
) {
    double dy = y_h[4] - y_h[3];

    for (int j = 3; j < NYD6_local - 3; j++) {
        for (int k = 2; k < NZ6_local - 2; k++) {     // 擴展至壁面
            double dz_dk, dz_dj;
            int idx = j * NZ6_local + k;

            // dz_dk：k 方向差分
            if (k == 2) {
                // 底壁：一階 forward difference
                dz_dk = z_h[j * NZ6_local + 3] - z_h[j * NZ6_local + 2];
            } else if (k == NZ6_local - 3) {
                // 頂壁：一階 backward difference
                dz_dk = z_h[j * NZ6_local + k] - z_h[j * NZ6_local + (k - 1)];
            } else {
                // 內部：中心差分
                dz_dk = (z_h[j * NZ6_local + (k + 1)] -
                         z_h[j * NZ6_local + (k - 1)]) / 2.0;
            }

            // dz_dj：j 方向中心差分（所有 k 通用）
            dz_dj = (z_h[(j + 1) * NZ6_local + k] -
                     z_h[(j - 1) * NZ6_local + k]) / 2.0;

            dk_dz_h[idx] = 1.0 / dz_dk;
            dk_dy_h[idx] = -dz_dj / (dy * dz_dk);
        }
    }
}
```

---

### 十七.4 壁面度量項數值驗證

| 位置 | dz_dk | dk_dz | dz_dj | dk_dy |
|------|-------|-------|-------|-------|
| k=2, 平坦段 (H=0) | minSize/2 ≈ 0.00955 | ≈ 104.7 | 0 | 0 |
| k=2, 山丘頂 (H=1, H'=0) | minSize/2 ≈ 0.00955 | ≈ 104.7 | 0 | 0 |
| k=2, 斜面 (H>0, H'≠0) | minSize/2 ≈ 0.00955 | ≈ 104.7 | dH/dy | -dH/dy / (dy·minSize/2) |
| k=3, 內部（中心差分） | (z[4]-z[2])/2 | ... | (z[j+1]-z[j-1])/2 | ... |

k=2 的 dk_dz 約為 k=3 的 1.5 倍（因為 forward vs 中心差分），但兩者的 dk_dy **符號一致**（同受 dH/dy 控制）。

---

### 十七.5 修改優先序與分段實作

**第一批（Phase 0 立即執行）— 度量項擴展**:
1. `initialization.h`: GenerateMesh_Z() — 加入 ghost z 值 (k=1, k=68) 和 xi_h 外推 (k=2, k=67)
2. `gilbm/metric_terms.h`: ComputeMetricTerms() — 擴展 k 範圍 + 壁面差分
3. `gilbm/metric_terms.h`: DiagnoseMetricTerms() — 更新 k_wall=2，擴展診斷範圍

**第二批（Phase 0 診斷通過後）— kernel 範圍修改**:
4. `evolution.h`: kernel guard 修改 (k<=1, k>=NZ6-2)
5. `evolution.h`: bounce-back 移到 k=2 和 k=NZ6-3
6. `evolution.h`: cell_z clamping — 確認不需修改（自動覆蓋 k=2）
7. `main.cu`: rho 迴圈和正規化

**第三批（驗證通過後）— 輸出修改**:
8. `fileIO.h`: VTK 輸出範圍和維度
9. `combinepltv2.cpp`: 後處理
10. `statistics.h`: 統計 kernel 的 k guard（可選）

**第四批（GILBM Phase 1）— BFL 和 interpolation**:
11. `initialization.h`: GetXiParameter() k=2 覆蓋
12. `initialization.h`: GetIntrplParameter_Xi() 範圍擴展
13. `evolution.h`: BFL 邏輯調整或移除

---

### 十七.6 驗證計劃

1. **編譯測試**：修改後在本地和伺服器（nvcc sm_35）成功編譯，零錯誤零警告
2. **Phase 0 診斷（6 判據）**：
   - 判據 1: dk_dz > 0 全場（含 k=2, k=67）→ PASS
   - 判據 2: dz_dk(wall) — 壁面現在在 k=2，expected = minSize/2 → PASS
   - 判據 3-6: 同前但用 k_wall=2
3. **Poiseuille 流回歸測試**：
   - 壁面速度 u[k=2] < 1e-6
   - 壁面剪應力與解析解比較
4. **Periodic Hill Re=200**：
   - 壁面剪應力曲線
   - 分離/再附著點位置

---

### 十七.7 風險評估

| 風險 | 嚴重度 | 緩解措施 |
|------|--------|---------|
| k=1 的 ghost z 值外推精度不足 | 低 | 只影響 AccumulateUbulk 的壁面 dz，可驗證 |
| xi_h[2] 外推值影響插值 | 中 | k=2 的 cell_z clamping 已將 stencil 固定在 k=3..9，xi_h[2] 實際不被 7 階插值讀取 |
| BFL 在 k=3,4 的行為因壁面移動而改變 | 低 | BFL 仍在 k=3,4 運行，壁面幾何不變 |
| rho 正規化的 NZ6-6 → NZ6-4 導致質量不守恆 | 高 | 必須同時修改 rho 求和迴圈範圍和正規化分母 |
| 二進制檔案格式不相容（舊 result 無法讀取） | 中 | INIT=0 重新初始化，不讀舊檔案 |

---

## 十八、Option B 剩餘項目：combinepltv2.cpp 後處理修改

### Context

Option B 主體已完成並提交（commit `251bc09`）。Batch 1-3 中的 `initialization.h`、`metric_terms.h`、`evolution.h`、`main.cu`、`fileIO.h` 均已修改。**唯一未完成的 Batch 3 項目是 `combinepltv2.cpp`（後處理程式）**。

`statistics.h` 按計劃延後（壁面 k=2 速度為零，對統計量無貢獻；MeanDerivatives 的 Z-stencil 在 guard `k≤2` 下安全——k=3 的 5 點 stencil 透過 n=3 clamping 存取 k=3..7，無越界風險）。

### 修改清單

**檔案**: `/Users/yetianzhong/Desktop/4.GitHub/D3Q27_PeriodicHill/combinepltv2.cpp`

**維度修改** (`NZ6-6` → `NZ6-4`，共 7 處):

| 行號 | 函式 | 當前代碼 | 修改後 |
|------|------|---------|--------|
| 85 | Output3Dvelocity | `(NZ6 - 6)` 在檔名 | `(NZ6 - 4)` |
| 92 | Output3Dvelocity | `int KMax = NZ6 - 6` | `NZ6 - 4` |
| 198 | Output3Dvelocity_VTK | `(NZ6 - 6)` 在 VTK 檔名 | `(NZ6 - 4)` |
| 206 | Output3Dvelocity_VTK | `(NZ6 - 6)` 在 DIMENSIONS | `(NZ6 - 4)` |
| 208 | Output3Dvelocity_VTK | `(NZ6 - 6)` 在 nPoints | `(NZ6 - 4)` |
| 250 | Outputstreamwise | `(NZ6 - 6)` 在 K= | `(NZ6 - 4)` |
| 270 | OutputMiddlePlane | `(NZ6 - 6)` 在 k= | `(NZ6 - 4)` |

**迴圈範圍修改** (`k = 3; k < NZ6 - 3` → `k = 2; k < NZ6 - 2`，共 7 處):

| 行號 | 函式 | 用途 |
|------|------|------|
| 170 | Output3Dvelocity | Tecplot 3D 速度寫入 |
| 212 | Output3Dvelocity_VTK | VTK 座標寫入 |
| 222 | Output3Dvelocity_VTK | VTK rho 寫入 |
| 230 | Output3Dvelocity_VTK | VTK 速度向量寫入 |
| 253 | Outputstreamwise | 流向剖面 |
| 273 | OutputMiddlePlane | 中間平面 |
| 320 | main() | 資料合併迴圈 |

**不修改的函式**:
- `printutau()` (lines 61-72): 壁面剪應力公式 `(9*v[k=3] - v[k=4]) / (3*minSize)` 仍然正確。推導：wall 在 k=2（v=0），k=3 距壁 d1=minSize/2，k=4 距壁 d2=3*minSize/2。二點 Lagrange 導數：`f'(0) = [f(d1)*d2 - f(d2)*d1] / [d1*(d2-d1)] = [9*f(k=3) - f(k=4)] / (3*minSize)`。與 Option B 前的半格彈射幾何**距離完全相同**。
- `Mesh_scan()`: 讀取整個 NZ6 網格，無需修改。
- `ReadData()`: 讀取 NX6*NZ6*NYD6 二進制資料，無需修改。

### 驗證

1. 編譯：`g++ -std=c++17 -O3 combinepltv2.cpp -o combineplt`（本地）
2. 確認 VTK 輸出 66×129×66 格點（原 64×129×64）
3. Tecplot 輸出 KMax=66

### 延後項目確認

以下項目維持原計劃延後：
- `statistics.h`: 暫不改（guard `k<=2` 已安全，壁面 u=0 對統計無貢獻）
- `initialization.h` GetXiParameter/GetIntrplParameter_Xi: Phase 1
- `evolution.h` BFL: Phase 1
- `memory.h`: 不修改（NZ6 不變）

---

## 十九、Option B 完成後：Z 方向節點佈局總覽與變更對照

### 19.1 節點佈局圖（Before vs After）

```
                 ┌─── Option A (舊) ───┐         ┌─── Option B (新) ───┐
  k 索引   NZ6=70│  角色       是否計算  │         │  角色       是否計算  │
  ──────────────┼──────────────────────┤         ├──────────────────────┤
  k=0           │  ghost        ✗      │         │  ghost        ✗      │
  k=1           │  ghost        ✗      │         │  ghost(外推z) ✗      │ ← NEW: z_h 線性外推
  k=2           │  buffer       ✗      │  =====> │  壁面(底)     ✓      │ ← 核心變更: 升級為計算點
  k=3           │  壁面(底) BB  ✓ 第1點 │         │  第1流體      ✓      │
  k=4           │  第1流體      ✓      │         │  第2流體      ✓      │
  k=5..65       │  內部流體     ✓      │         │  內部流體     ✓      │
  k=66(=NZ6-4)  │  最後計算     ✓ 末點  │         │  倒數第2      ✓      │
  k=67(=NZ6-3)  │  buffer       ✗      │  =====> │  壁面(頂)     ✓      │ ← 核心變更: 升級為計算點
  k=68(=NZ6-2)  │  ghost        ✗      │         │  ghost(外推z) ✗      │ ← NEW: z_h 線性外推
  k=69(=NZ6-1)  │  ghost        ✗      │         │  ghost        ✗      │
  ──────────────┴──────────────────────┘         └──────────────────────┘
  計算點數       │  64 (k=3..66, NZ6-6) │         │  66 (k=2..67, NZ6-4) │
  壁面位置       │  k=2.5 (半格彈射)    │         │  k=2 (精確在格點上)   │
  壁面BC方式     │  halfway bounce-back  │         │  wet-node bounce-back │
```

### 19.2 逐項變更對照表

| 項目 | Option A (舊) | Option B (新) | 所在檔案 |
|------|--------------|--------------|---------|
| **記憶體佈局** |
| NZ6 定義 | NZ+6 = 70 | NZ+6 = 70 (**不變**) | variables.h:19 |
| minSize 定義 | (LZ-1)/(NZ6-6)*CFL | (**不變**，tanh 參數) | variables.h:23 |
| 3D 陣列大小 | NX6×NYD6×NZ6 | (**不變**) | memory.h |
| **網格生成** |
| tanh 迴圈 | k=3..66, j=k-3, N=63 | (**不變**) | initialization.h:122 |
| 底壁 z 座標 | z[k=2] = H(y) (buffer) | z[k=2] = H(y) (**計算點**) | initialization.h:126 |
| 頂壁 z 座標 | z[k=67] = LZ (buffer) | z[k=67] = LZ (**計算點**) | initialization.h:127 |
| ghost z 值 | 無 | **k=1, k=68 線性外推** | initialization.h:130-131 |
| xi_h 壁面 | 無 | **k=2, k=67 線性外推** | initialization.h:138-139 |
| **Kernel guard** |
| stream_collide_Buffer | k≤2 ∥ k≥NZ6-3 | **k≤1 ∥ k≥NZ6-2** | evolution.h:80 |
| stream_collide | k≤2 ∥ k≥NZ6-3 | **k≤1 ∥ k≥NZ6-2** | evolution.h:369 |
| AccumulateUbulk | k≤2 ∥ k≥NZ6-3 | **k≤1 ∥ k≥NZ6-2** | evolution.h:770 |
| statistics kernels | k≤2 ∥ k≥NZ6-3 | **不變** (延後) | statistics.h:138,192 |
| **壁面邊界條件** |
| 底壁 bounce-back | k==3 (半格 BB) | **k==2** (wet-node BB) | evolution.h:147,438 |
| 頂壁 bounce-back | k==NZ6-4 (=66) | **k==NZ6-3** (=67) | evolution.h:154,445 |
| bounce-back 方向數 | 5 方向 | 5 方向 (**不變**) | evolution.h |
| BFL 位置 | k==3 ∥ k==4 | **不變** (延後至 Phase 1) | evolution.h:169,459 |
| **Forcing / rho** |
| ModifyForcingTerm | k=3..NZ6-4 | **k=2..NZ6-3** | evolution.h:928 |
| rho 求和 k 迴圈 | k=3..NZ6-4 | **k=2..NZ6-3** | main.cu:248,268 |
| rho 正規化分母 | NZ6-6 (=64) | **NZ6-4** (=66) | main.cu:255-256,274 |
| **度量項** |
| ComputeMetricTerms k迴圈 | k=3..NZ6-4 | **k=2..NZ6-3** | metric_terms.h:34 |
| 壁面差分 | 中心差分 (全場) | **k=2 forward, k=67 backward** | metric_terms.h:39-47 |
| DiagnoseMetricTerms k_wall | 3 | **2** | metric_terms.h:193 |
| 判據 2 期望值 | j-dependent | **minSize/2 (常數)** | metric_terms.h:309 |
| **檔案輸出** |
| VTK Z 維度 | NZ6-6 (=64) | **NZ6-4** (=66) | fileIO.h:96, combinepltv2.cpp:206 |
| 輸出 k 迴圈 | k=3..NZ6-4 | **k=2..NZ6-3** | fileIO.h, combinepltv2.cpp (共13處) |
| nzLocal (merged VTK) | NZ6-6 | **NZ6-4** | fileIO.h:324 |
| **不變的項目** |
| cell_z clamping | k≤6→3, k≥NZ6-7→NZ6-10 | **不變** (自動處理 k=2) | evolution.h:96-98 |
| periodicNML buffer | 3 | **不變** (Z 方向是壁面非周期) | evolution.h:727 |
| MPI Y 通訊 | 3×NX6×NZ6 | **不變** | communication.h:202 |
| GetXiParameter clamp | k≥3→stencil=3 | **不變** (Phase 1) | initialization.h:191 |
| GetIntrplParameter_Xi | k=3..NZ6-4 | **不變** (Phase 1) | initialization.h:231 |
| printutau() | k=3 (第一流體點) | **不變** (距離相同) | combinepltv2.cpp:66 |

### 19.3 物理意義的關鍵差異

| 物理量 | Option A | Option B |
|--------|----------|----------|
| 壁面幾何精度 | 壁面在 k=2.5 (半格偏移)，誤差 ≈ minSize/4 ≈ 0.48% | **壁面精確在格點 k=2**，零幾何誤差 |
| 壁面 dz_dk | 中心差分 (z[4]-z[2])/2 = 3·minSize/4 | **forward 差分** z[3]-z[2] = minSize/2 |
| 壁面 dk_dz | ≈ 4/(3·minSize) ≈ 69.8 | **≈ 2/minSize ≈ 104.7** (壁面處度量放大) |
| 計算點數 | 64 (k=3..66) | **66** (k=2..67，含兩壁面) |
| Inamuro/C-E BC 可行性 | 不可直接施加 (壁面不在格點) | **可直接施加** (Phase 1 目標) |
| BFL 必要性 | 需要 (壁面不在格點) | **理論上不需要** (Phase 1 移除) |

### 19.4 「不變」項目的安全性確認

以下項目雖然保留舊數值，但在 Option B 下仍然正確：

1. **`minSize = (LZ-1)/(NZ6-6)*CFL`**：這是 tanh 拉伸的 base 解析度參數，由 j=0..63 的映射決定，與 buffer 層數無關。NZ6-6=64 是 tanh 分割數，不是計算點數。

2. **`statistics.h` guard `k≤2 ∥ k≥NZ6-3`**：排除壁面 k=2 和 k=67。壁面速度 u=0，不參與紊流統計。MeanDerivatives 的 Z-stencil (5 點) 在 k=3 處透過 n=3 clamping 存取 k=3..7，**無越界風險**。

3. **`GetXiParameter()` 的 k≥3 clamping**：壁面 k=2 的 cell_z = -1 → 被 evolution.h 的 `if(k≤6) cell_z=3` 攔截 → stencil 起始 k=3。GetXiParameter 不會被以 k=2 呼叫（Phase 1 待擴展）。

4. **`GetIntrplParameter_Xi()` 的 k=3..66 迴圈**：壁面 k=2 和 k=67 不需要預計算插值參數（壁面 BC 由 bounce-back 直接處理，不走 7 階插值路徑）。Phase 1 的 GILBM kernel 若需要在壁面做 streaming，才需擴展。

5. **BFL 在 k=3,4**：BFL 處理 y 方向穿越山丘表面的離散速度方向。在 Option B 中，k=3 和 k=4 仍然是壁面上方第 1、2 層流體，BFL 幾何 (q 參數) 不受影響。Phase 1 的 GILBM kernel 將透過逆變速度方向判別取代 BFL。

### 19.5 修改統計

| 檔案 | 修改行數 (插入/刪除) | 修改類型 |
|------|---------------------|---------|
| initialization.h | +12 / -0 | ghost z, xi_h 外推 |
| gilbm/metric_terms.h | +30 / -20 | k 迴圈擴展, 壁面差分, 診斷更新 |
| evolution.h | +12 / -12 | guard, BB, forcing 範圍 |
| main.cu | +6 / -6 | rho 迴圈, 正規化 |
| fileIO.h | +12 / -12 | VTK 維度, 輸出迴圈 |
| combinepltv2.cpp | +14 / -14 | 維度, 輸出迴圈 |
| **合計** | **+86 / -64** | 6 檔案, 0 新增檔案 |

### 19.6 提交記錄

| Commit | 訊息 | 檔案 |
|--------|------|------|
| `251bc09` | Implement Option B: k=2 as wall computation point (wet-node BC) | initialization.h, metric_terms.h, evolution.h, main.cu, fileIO.h |
| `b3e2b5e` | Update combinepltv2.cpp post-processor for Option B (k=2 wall) | combinepltv2.cpp |

---

## 二十、Option B 後判據 3/5 再次失敗：根因分析與修正

### Context

Option B 實作完成後（k=2 為壁面計算點），重新在伺服器執行，判據 3 和 5 再次失敗：

```
FAIL criteria 3: flat region j=31 k=2, dk_dy=3.932939e-01 (expected ~0)
FAIL criteria 3: flat region j=31 k=3, dk_dy=...
... (k=2..8 逐漸衰減到 ~0.10)
FAIL criteria 3: flat region j=103 k=2, dk_dy=... (對稱)
FAIL criteria 5: j=31 (flat, H=0.0000), num_BC=8 (expected 5)
FAIL criteria 5: j=103 (flat, H=0.0000), num_BC=8 (expected 5)
```

### 20.1 根因分析：山丘-平坦過渡帶 H 殘值放大效應

#### 精確數值推導

j=31 的三個鄰居 y 座標（dy = 9.0/128 = 0.0703125）：
```
y[30] = 0.0703125 × 27 = 1.8984375   ← 在山丘最後一段 (Yb ≤ 54/28 = 1.92857)
y[31] = 0.0703125 × 28 = 1.9687500   ← 在山丘外部 (H = 0)
y[32] = 0.0703125 × 29 = 2.0390625   ← 在山丘外部 (H = 0)
```

HillFunction 在 y[30] = 1.8984375 的值：
```
Yb*28 = 53.15625
最後一段多項式 max(0, 56.390... - 2.0105*53.156 + 0.01645*53.156² + 0.0000267*53.156³) / 28
= max(0, 0.0148) / 28
= 5.278 × 10⁻⁴
```

所以 **H(y[30]) ≈ 5.28 × 10⁻⁴**，極小但非零。

#### dk_dy 放大機制

```
dz_dj|_{k=2} = (H(y[32]) - H(y[30])) / 2 = (0 - 5.28e-4) / 2 = -2.64e-4

dz_dk|_{k=2} = z[k=3] - z[k=2] = minSize/2 = 0.009544   ← Option B 的 forward difference

dk_dy = -dz_dj / (dy × dz_dk) = 2.64e-4 / (0.0703 × 0.009544) = 0.393
```

**關鍵放大因子**：`1/(dy × dz_dk) ≈ 1490`。即使 H 差值僅 5.28 × 10⁻⁴，乘以 1490 後 dk_dy ≈ 0.39。

#### 為什麼舊的鄰居檢查無法排除 j=31？

當前條件（lines 242-244, 329-331）：
```cpp
fabs(Hy) < 0.01           ← H(y[31]) = 0        ✓
fabs(dHdy) < 0.01         ← |dHdy| = 0.00375     ✓
fabs(H(y[j-1])) < 0.01   ← H(y[30]) = 5.28e-4   ✓  ← 通過！
fabs(H(y[j+1])) < 0.01   ← H(y[32]) = 0          ✓
```

**四個條件全部通過**，但 H 的**差值** 5.28e-4 足以產生 dk_dy = 0.39。
問題核心：閾值 0.01 允許的 H 差值最大為 0.02，對應 dk_dy 最大 ≈ 14.9（遠超 0.1 門檻）。

#### 為什麼 Option B 使問題惡化？

| | Option A (k=3) | Option B (k=2) |
|---|---|---|
| dz_dk | 中心差分 (z[4]-z[2])/2 ≈ 0.01432 | forward 差分 z[3]-z[2] = 0.009544 |
| 放大因子 | 1/(dy×0.01432) ≈ 994 | 1/(dy×0.009544) ≈ 1490 |
| dk_dy at j=31 | ≈ 0.26 | ≈ 0.39 |

Option A 時 dk_dy ≈ 0.26 > 0.1，**也應該失敗**。但 Option A 的 k 迴圈從 k=3 開始（`for (k=bfr; ...)`），k=2 不在計算範圍內。如果之前判據 3 在 Option A 下通過了，可能是因為迴圈從 k=3 開始且 dk_dy 在 k=3 略低於 0.1 門檻。

#### dk_dy 隨 k 衰減的原因

在 k>2 時，dz_dj 包含兩項：
```
dz_dj|_k = ΔH/2 + Δ(tanhFunction)/2
```

第二項 `Δ(tanhFunction)` 由 `total_j = LZ - H(y_j) - minSize` 決定。
當 H(y_{j-1}) > H(y_{j+1})（j=31 的情況），total_{j-1} < total_{j+1}，
tanh 拉伸的差值**部分補償** ΔH 項。
隨著 k 增大（離壁面越遠），tanh 補償越完全 → dk_dy 衰減。

### 20.2 修正方案

#### 方案選擇：添加 H 差值檢查（直接命中根因）

在判據 3 和判據 5 的平坦區域條件中，添加第五個條件：

```cpp
double H_stencil_diff = fabs(HillFunction(y_g[j+1]) - HillFunction(y_g[j-1]));
if (fabs(Hy) < 0.01 && fabs(dHdy) < 0.01
    && fabs(HillFunction(y_g[j-1])) < 0.01
    && fabs(HillFunction(y_g[j+1])) < 0.01
    && H_stencil_diff < 1e-6) {              // ← 新增：差值必須接近機器精度
```

#### 為什麼閾值選 1e-6？

| H_stencil_diff | 對應 dk_dy at k=2 | 含義 |
|---|---|---|
| 5.28e-4 (j=31) | 0.393 | 過渡帶，應排除 |
| 1e-6 | 7.5e-4 | 安全門檻（dk_dy << 0.1） |
| 0 (j=32..102) | 0 | 純平坦段，正確包含 |

在真正平坦的區域（j=32..102），H(y[j±1]) = 0 exactly（因為 y 值遠超出山丘範圍），
所以 H_stencil_diff = 0，新條件自動通過。

### 20.3 修改清單

**檔案**：`gilbm/metric_terms.h`

**修改 1 — 判據 5（line 242-244）**：
```cpp
// 修改前：
if (fabs(Hy) < 0.01 && fabs(dHdy) < 0.01
    && fabs(HillFunction(y_g[j - 1])) < 0.01
    && fabs(HillFunction(y_g[j + 1])) < 0.01) {

// 修改後：
double H_stencil_diff_c5 = fabs(HillFunction(y_g[j + 1]) - HillFunction(y_g[j - 1]));
if (fabs(Hy) < 0.01 && fabs(dHdy) < 0.01
    && fabs(HillFunction(y_g[j - 1])) < 0.01
    && fabs(HillFunction(y_g[j + 1])) < 0.01
    && H_stencil_diff_c5 < 1e-6) {
```

**修改 2 — 判據 3（line 329-331）**：
```cpp
// 修改前：
if (fabs(Hy_c3) < 0.01 && fabs(dHdy_c3) < 0.01
    && fabs(HillFunction(y_g[j - 1])) < 0.01
    && fabs(HillFunction(y_g[j + 1])) < 0.01) {

// 修改後：
double H_stencil_diff_c3 = fabs(HillFunction(y_g[j + 1]) - HillFunction(y_g[j - 1]));
if (fabs(Hy_c3) < 0.01 && fabs(dHdy_c3) < 0.01
    && fabs(HillFunction(y_g[j - 1])) < 0.01
    && fabs(HillFunction(y_g[j + 1])) < 0.01
    && H_stencil_diff_c3 < 1e-6) {
```

### 20.4 驗證

修正後預期結果：
```
[PASS] Criteria 1: dk_dz > 0 everywhere
[PASS] Criteria 2: dz_dk(wall) = minSize/2 (within 10%)
[PASS] Criteria 3: dk_dy ≈ 0 at flat region
[PASS] Criteria 4: dk_dy sign consistent with -H'(y)
[PASS] Criteria 5: flat wall has exactly 5 BC directions
[PASS] Criteria 6: slope wall has >5 BC directions
```

j=31 和 j=103 被排除後：
- 判據 3 的掃描範圍：j=32..102（純平坦段），dk_dy = 0 exactly → 全部 PASS
- 判據 5 的掃描範圍：j=32..102，num_bc = 5 → 全部 PASS

### 20.5 重要澄清

**度量項計算本身完全正確**。dk_dy ≈ 0.39 at j=31, k=2 是物理上正確的值——
山丘-平坦過渡帶確實存在微小的座標扭曲。問題僅在於**診斷判據的平坦區域定義太寬鬆**，
誤將過渡帶包含在內。修正只影響診斷邏輯，不改變任何度量項計算。

### 20.6 提交計劃

修改後需上傳 GitHub：
```bash
git add gilbm/metric_terms.h
git commit -m "Fix criteria 3/5 flat-region detection: add H stencil-diff check"
git push origin Edit3_GILBM
```

---

## 二十一、Phase 1: GILBM Streaming + Chapman-Enskog BC（跳過 NEE，直接實作 C-E BC）

### Context

Phase 0 完成（6 判據全 PASS，commit `ec7f002`）。現在進入 Phase 1：**將現有 ISLBM streaming（7 階 Lagrange Xi 插值 + BFL + bounce-back）替換為 GILBM streaming（逆變速度 RK2 + 二階上風插值 + Chapman-Enskog BC）**。

**決策**：直接跳過 NEE（Non-Equilibrium Extrapolation），因為：
1. C-E BC 與 GILBM 理論框架**自洽**（Imamura 2005 §3.2、Appendix A 明確推導）
2. NEE 的 f^neq 外推假設在扭曲晶格下有理論缺陷（§一.1.2 問題 B）
3. C-E BC 實作複雜度增加有限（壁面 u=0 大幅簡化公式）

### 21.1 Imamura 2005 核心公式（2D→3D 推廣）

#### A. 逆變速度（Eq. 13 推廣到 3D）

我們的座標系統：物理 (x,y,z) → 計算 (i,j,k)

```
ẽ_α^i = e_{α,x} / dx                          ← 常數
ẽ_α^j = e_{α,y} / dy                          ← 常數
ẽ_α^k = e_{α,y}·dk_dy[j,k] + e_{α,z}·dk_dz[j,k]  ← 隨空間變化（唯一非平凡分量）
```

其中 dk_dy = ∂ζ/∂y、dk_dz = ∂ζ/∂z 已在 Phase 0 計算完成。

#### B. RK2 上風點積分（Eq. 19-20 推廣）

i、j 方向為均勻網格，Δi = dt·ẽ_α^i、Δj = dt·ẽ_α^j 為常數，不需 RK2。

k 方向**必須**使用 RK2（壁面附近網格劇烈拉伸，Euler 積分會導致 ~50% 誤差，見 Fig. 7）：
```
Δk^(1) = ½·dt·ẽ_α^k(j, k)                     ← 第一步
k_half = k - Δk^(1)
ẽ_α^k_half = interpolate dk_dy, dk_dz at k_half, then compute ẽ^k
Δk = dt·ẽ_α^k_half                              ← 第二步（O(Δt³) 精度）
```

**關鍵優化（Imamura p.650）**：Δk 只在初始化時計算一次（度量項不隨時間變化），存儲後每步直接讀取，**節省約 50% kernel 時間**。

#### C. 二階上風二次插值（Eq. 23-24 推廣到 3D）

上風點 (up_i, up_j, up_k) = (i - Δi, j - Δj, k - Δk) 落在整數格點之間。

1D 插值係數（3 點 stencil，base = floor(upwind_position)）：
```
t = upwind_position - base    ∈ [0, 1)
a_0(t) = ½(t - 1)(t - 2)     ← 基點
a_1(t) = -t(t - 2)            ← 鄰點
a_2(t) = ½·t·(t - 1)          ← 遠點
```

3D 張量積：g(up) = Σ_{l,m,n} a_l^i · a_m^j · a_n^k · g[base_i+l, base_j+m, base_k+n]

最多 3×3×3 = 27 點，但對角速度為零的維度退化為精確格點（a_0=1, a_1=a_2=0），
實際每個方向只需 1~27 點插值（遠少於現有 ISLBM 的 7³=343 點最大 stencil）。

#### D. Chapman-Enskog BC（Eq. 26/A.9 at no-slip wall）

完整公式：
```
f_α|_{wall} = f_α^eq · [1 - ω·dt · Σ_i Σ_j (3·e_{α,i}·e_{α,j}/c² - δ_{ij}) · ∂u_i/∂x_j]
```

**在 no-slip wall (u=0) 的簡化**：

由於壁面上 u = 0 在所有 j 位置都成立，∂u/∂j|_{wall} = 0，∂u/∂i|_{wall} = 0。
速度梯度只有 ∂u/∂k 分量：
```
∂u_i/∂x = 0                                    (x 方向不耦合)
∂u_i/∂y = dk_dy · ∂u_i/∂k                      (y 通過 k 耦合)
∂u_i/∂z = dk_dz · ∂u_i/∂k                      (z 直接耦合)
```

壁面 ∂u/∂k 由二階單側有限差分計算（u[k=2]=0）：
```
∂u_α/∂k|_{k=2} = (4·u_α[k=3] - u_α[k=4]) / 2
```

壁面密度：ρ_wall ≈ ρ[k=3]（壓力法向梯度為零，Imamura §3.2）

代入後的 C-E 修正項 C_α：
```
C_α = -ω·dt · Σ_i ∂u_i/∂k · {
    [9·e_{α,i}·e_{α,y} - δ_{i,y}]·dk_dy +
    [9·e_{α,i}·e_{α,z} - δ_{i,z}]·dk_dz
}
```
其中 9 = 3/c²（D3Q19 的 c² = 1/3）。

最終：`f_α|_{wall} = w_α · ρ_wall · (1 + C_α)`

#### E. 方向判別：哪些方向需要 BC？

底壁 k=2：`ẽ_α^k > 0` → 上風點在壁外 → 需要 C-E BC
頂壁 k=NZ6-3：`ẽ_α^k < 0` → 上風點在壁外 → 需要 C-E BC

平坦段：恰好 5 方向 {5, 11, 12, 15, 16}（已由 Phase 0 判據 5 驗證）
斜面段：可能 >5 方向（已由 Phase 0 判據 6 驗證）

---

### 21.2 架構設計

#### 核心決策

| 決策 | 選擇 | 理由 |
|------|------|------|
| Kernel 方式 | 新建 `stream_collide_GILBM()` | 保留 ISLBM 作為 fallback，用 `#define USE_GILBM` 切換 |
| 逆變速度 | 預計算存儲 | Imamura 建議，節省 ~50% kernel 時間 |
| 插值階數 | 二階（3 點 stencil） | Imamura Eq. 23-24，足以壓制數值耗散 |
| 壁面 BC | C-E BC（跳過 NEE） | 與 GILBM 理論自洽，no-slip 簡化公式 |
| BFL | 移除 | GILBM 逆變速度自動處理山丘曲率 |
| 碰撞模型 | MRT 不變 | 重用現有 MRT_Matrix.h 和 MRT_Process.h |
| 速度模型 | D3Q19 不變 | 19 速度，c² = 1/3 |

#### 記憶體新增

| 陣列 | 大小 | 用途 |
|------|------|------|
| `delta_k_h/d[19 * NYD6 * NZ6]` | 19 × 39 × 70 × 8B ≈ 408 KB | 預計算 RK2 上風位移 |
| **合計新增** | **≈ 816 KB** (host+device) | 極小（對比 19 個分佈函數 ≈ 15.6 MB） |

不需要新增的陣列：
- dk_dz_d, dk_dy_d：已在 Phase 0 分配（memory.h:101-103）
- e_tilde_i[19], e_tilde_j[19]：常數，用 `__constant__` 記憶體或 `#define`

---

### 21.3 逐檔案修改清單

---

#### 新建檔案 1: `gilbm/precompute.h` — 預計算逆變速度 RK2

```cpp
#ifndef GILBM_PRECOMPUTE_H
#define GILBM_PRECOMPUTE_H

// 預計算 RK2 上風位移 delta_k[alpha][j*NZ6+k]
// 在 main.cu 初始化階段調用一次，結果拷貝到 GPU
void PrecomputeGILBM_DeltaK(
    double *delta_k_h,       // output: [19 * NYD6 * NZ6]
    const double *dk_dz_h,   // input: metric term ∂ζ/∂z
    const double *dk_dy_h,   // input: metric term ∂ζ/∂y
    int NYD6_local,
    int NZ6_local
) {
    double dx = LX / (double)(NX6 - 7);
    double dy_val = LY / (double)(NY6 - 7);

    for (int alpha = 0; alpha < 19; alpha++) {
        for (int j = 0; j < NYD6_local; j++) {
            for (int k = 2; k < NZ6_local - 2; k++) {
                int idx_jk = j * NZ6_local + k;

                if (alpha == 0) {
                    // 靜止方向，無位移
                    delta_k_h[alpha * NYD6_local * NZ6_local + idx_jk] = 0.0;
                    continue;
                }

                // Step 1: 當前點的逆變速度
                double e_tilde_k0 = e[alpha][1] * dk_dy_h[idx_jk]
                                  + e[alpha][2] * dk_dz_h[idx_jk];

                // Step 2: RK2 半步
                double dk_half = 0.5 * dt * e_tilde_k0;
                double k_half = (double)k - dk_half;

                // 在半步位置線性插值度量項
                int k_lo = (int)floor(k_half);
                k_lo = max(2, min(k_lo, NZ6_local - 4));
                double frac = k_half - (double)k_lo;
                frac = max(0.0, min(frac, 1.0));

                int idx_lo = j * NZ6_local + k_lo;
                int idx_hi = j * NZ6_local + k_lo + 1;

                double dk_dy_half = (1.0 - frac) * dk_dy_h[idx_lo]
                                  + frac * dk_dy_h[idx_hi];
                double dk_dz_half = (1.0 - frac) * dk_dz_h[idx_lo]
                                  + frac * dk_dz_h[idx_hi];

                // Step 3: 半步位置的逆變速度
                double e_tilde_k_half = e[alpha][1] * dk_dy_half
                                      + e[alpha][2] * dk_dz_half;

                // Step 4: 完整 RK2 位移
                delta_k_h[alpha * NYD6_local * NZ6_local + idx_jk] = dt * e_tilde_k_half;
            }
        }
    }
}
#endif
```

---

#### 新建檔案 2: `gilbm/interpolation_gilbm.h` — 二階上風插值

```cuda
#ifndef GILBM_INTERPOLATION_H
#define GILBM_INTERPOLATION_H

// Imamura Eq. 24: 二次 Lagrange 插值係數
__device__ __forceinline__ void quadratic_coeffs(
    double t,           // fractional position ∈ [0, 1)
    double &a0, double &a1, double &a2
) {
    a0 = 0.5 * (t - 1.0) * (t - 2.0);
    a1 = -t * (t - 2.0);
    a2 = 0.5 * t * (t - 1.0);
}

// 3D 二次上風插值（張量積）
// up_i, up_j, up_k: 上風點在計算空間的座標
// f_alpha: 該方向的分佈函數陣列 [NYD6 × NZ6 × NX6]
__device__ double interpolate_quadratic_3d(
    double up_i, double up_j, double up_k,
    const double *f_alpha,
    int NX6_val, int NZ6_val
) {
    // 各維度基點和小數部分
    int bi = (int)floor(up_i);
    int bj = (int)floor(up_j);
    int bk = (int)floor(up_k);
    double ti = up_i - (double)bi;
    double tj = up_j - (double)bj;
    double tk = up_k - (double)bk;

    // 各維度插值係數
    double ai[3], aj[3], ak[3];
    quadratic_coeffs(ti, ai[0], ai[1], ai[2]);
    quadratic_coeffs(tj, aj[0], aj[1], aj[2]);
    quadratic_coeffs(tk, ak[0], ak[1], ak[2]);

    // 張量積求和
    double result = 0.0;
    for (int n = 0; n < 3; n++) {         // k 方向
        for (int m = 0; m < 3; m++) {     // j 方向
            for (int l = 0; l < 3; l++) { // i 方向
                int idx = (bj + m) * NZ6_val * NX6_val
                        + (bk + n) * NX6_val
                        + (bi + l);
                result += ai[l] * aj[m] * ak[n] * f_alpha[idx];
            }
        }
    }
    return result;
}
#endif
```

---

#### 新建檔案 3: `gilbm/boundary_conditions.h` — C-E BC

```cuda
#ifndef GILBM_BOUNDARY_CONDITIONS_H
#define GILBM_BOUNDARY_CONDITIONS_H

// 方向判別：該方向在壁面是否需要 BC？
__device__ __forceinline__ bool NeedsBoundaryCondition(
    int alpha,
    double dk_dy_val, double dk_dz_val,
    bool is_bottom_wall
) {
    double e_tilde_k = e[alpha][1] * dk_dy_val + e[alpha][2] * dk_dz_val;
    return is_bottom_wall ? (e_tilde_k > 0.0) : (e_tilde_k < 0.0);
}

// Chapman-Enskog BC (Imamura Eq. A.9, no-slip wall u=0)
// 返回壁面分佈函數 f_alpha|_{wall}
__device__ double ChapmanEnskogBC(
    int alpha,
    double rho_wall,
    double du_x_dk, double du_y_dk, double du_z_dk,  // ∂u/∂k at wall
    double dk_dy_val, double dk_dz_val,
    double omega_val, double dt_val
) {
    // 在 no-slip wall, U_{α,i} = e_{α,i}（因為 u = 0）
    double ex = e[alpha][0], ey = e[alpha][1], ez = e[alpha][2];

    // C-E 修正項：C = -ω·dt · Σ_i ∂u_i/∂k · [(9·e_i·e_j - δ_ij)·∂k/∂x_j]
    // 其中 ∂k/∂x = 0, ∂k/∂y = dk_dy, ∂k/∂z = dk_dz, 9 = 3/c² (c²=1/3)
    double C_alpha = 0.0;

    // i = x 分量
    C_alpha += du_x_dk * (
        (9.0 * ex * ey) * dk_dy_val +
        (9.0 * ex * ez) * dk_dz_val
    );

    // i = y 分量
    C_alpha += du_y_dk * (
        (9.0 * ey * ey - 1.0) * dk_dy_val +
        (9.0 * ey * ez) * dk_dz_val
    );

    // i = z 分量
    C_alpha += du_z_dk * (
        (9.0 * ez * ey) * dk_dy_val +
        (9.0 * ez * ez - 1.0) * dk_dz_val
    );

    C_alpha *= -omega_val * dt_val;

    // f_α = w_α · ρ_wall · (1 + C_α)
    return W[alpha] * rho_wall * (1.0 + C_alpha);
}
#endif
```

---

#### 新建檔案 4: `gilbm/evolution_gilbm.h` — GILBM 流場演化 kernel

核心 kernel 結構（概要）：

```cuda
__global__ void stream_collide_GILBM_Buffer(
    /* 19 個 f_old 和 f_new 分佈函數指標 */
    double *dk_dz_d, double *dk_dy_d,
    double *delta_k_d,  // [19 * NYD6 * NZ6]
    double *Force, double *rho_modify
) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    int k = blockIdx.z*blockDim.z + threadIdx.z;

    if (i <= 2 || i >= NX6-3 || k <= 1 || k >= NZ6-2) return;

    int index = j*NX6*NZ6 + k*NX6 + i;
    int idx_jk = j*NZ6 + k;

    // 常數
    double dx = LX / (double)(NX6 - 7);
    double dy_val = LY / (double)(NY6 - 7);

    double F_in[19];
    F_in[0] = f0_old[index];  // 靜止方向

    // ===== GILBM Streaming =====
    for (int alpha = 1; alpha < 19; alpha++) {
        double delta_i = dt * e[alpha][0] / dx;
        double delta_j = dt * e[alpha][1] / dy_val;
        double delta_k = delta_k_d[alpha * NYD6 * NZ6 + idx_jk];

        // --- 壁面 C-E BC ---
        bool is_bottom = (k == 2);
        bool is_top = (k == NZ6 - 3);

        if (is_bottom && NeedsBoundaryCondition(alpha, dk_dy_d[idx_jk], dk_dz_d[idx_jk], true)) {
            // 從 k=3 和 k=4 讀取舊分佈，計算宏觀速度
            double rho3 = 0, ux3 = 0, uy3 = 0, uz3 = 0;
            double rho4 = 0, ux4 = 0, uy4 = 0, uz4 = 0;
            // ... (sum f_old at k=3 and k=4 for macroscopic variables)

            // 壁面速度梯度 (二階單側)
            double du_x_dk = (4.0*ux3 - ux4) / 2.0;
            double du_y_dk = (4.0*uy3 - uy4) / 2.0;
            double du_z_dk = (4.0*uz3 - uz4) / 2.0;

            double rho_wall = rho3;  // 壓力外推

            F_in[alpha] = ChapmanEnskogBC(alpha, rho_wall,
                du_x_dk, du_y_dk, du_z_dk,
                dk_dy_d[idx_jk], dk_dz_d[idx_jk],
                omega, dt);

        } else if (is_top && NeedsBoundaryCondition(alpha, dk_dy_d[idx_jk], dk_dz_d[idx_jk], false)) {
            // 頂壁同理（用 k=NZ6-4 和 k=NZ6-5）
            // ...

        } else {
            // --- GILBM 插值 Streaming ---
            double up_i = (double)i - delta_i;
            double up_j = (double)j - delta_j;
            double up_k = (double)k - delta_k;

            F_in[alpha] = interpolate_quadratic_3d(
                up_i, up_j, up_k,
                f_alpha_old[alpha],  // 該方向的分佈函數陣列
                NX6, NZ6);
        }
    }

    // ===== 宏觀量計算 (不變) =====
    double rho = F_in[0] + F_in[1] + ... + F_in[18];
    double ux = (F_in[1] - F_in[2] + F_in[7] - ... ) / rho;
    // ...

    // ===== Force 施加 (不變) =====
    // ...

    // ===== MRT Collision (完全重用現有邏輯) =====
    m_matrix;    // f → moment space
    meq;         // equilibrium moments
    collision;   // apply relaxation

    // ===== 寫入新分佈 =====
    f0_new[index] = F_in[0];
    // ... f18_new[index] = F_in[18];
}
```

---

#### 修改檔案 5: `memory.h` — 新增記憶體分配

在 `AllocateMemory()` 中 dk_dz/dk_dy 之後加入：
```cpp
// GILBM 預計算逆變速度 RK2 位移
nBytes = 19 * NYD6 * NZ6 * sizeof(double);
AllocateHostArray(  nBytes, 1, &delta_k_h);
AllocateDeviceArray(nBytes, 1, &delta_k_d);
```

在 `FreeSource()` 中同步加入釋放。

---

#### 修改檔案 6: `main.cu` — GILBM 初始化與 kernel 切換

```cpp
// === 在 DiagnoseMetricTerms(myid) 之後 ===

// Phase 1: 計算各 rank 的區域度量項
ComputeMetricTerms(dk_dz_h, dk_dy_h, z_h, y_h, NYD6, NZ6);

// Phase 1: 預計算 GILBM RK2 上風位移
PrecomputeGILBM_DeltaK(delta_k_h, dk_dz_h, dk_dy_h, NYD6, NZ6);

// 拷貝到 GPU
cudaMemcpy(dk_dz_d, dk_dz_h, NYD6*NZ6*sizeof(double), cudaMemcpyHostToDevice);
cudaMemcpy(dk_dy_d, dk_dy_h, NYD6*NZ6*sizeof(double), cudaMemcpyHostToDevice);
cudaMemcpy(delta_k_d, delta_k_h, 19*NYD6*NZ6*sizeof(double), cudaMemcpyHostToDevice);

// === 主迴圈中 ===
#ifdef USE_GILBM
    stream_collide_GILBM_Buffer<<<grid, block>>>(..., dk_dz_d, dk_dy_d, delta_k_d, ...);
#else
    stream_collide_Buffer<<<grid, block>>>(...);
#endif
```

---

#### 修改檔案 7: `variables.h` — 新增 GILBM 開關

```cpp
// GILBM 開關：啟用逆變速度 streaming + C-E BC
#define USE_GILBM
```

---

### 21.4 ISLBM 元件處置表

| 現有元件 | Phase 1 處置 | 理由 |
|---------|-------------|------|
| `interpolationHillISLBM.h` (7 階 Lagrange 插值) | `#ifndef USE_GILBM` 保留 | ISLBM fallback |
| bounce-back at k=2/NZ6-3 | 被 C-E BC 取代 | GILBM 不用 bounce-back |
| BFL at k=3/k=4 | `#ifndef USE_GILBM` 保留 | GILBM 不需要 BFL |
| GetXiParameter / GetIntrplParameter_Xi | `#ifndef USE_GILBM` 保留 | GILBM 不用 Xi 座標 |
| MRT_Matrix.h / MRT_Process.h | **不變** | 碰撞模型完全重用 |
| metric_terms.h | **不變** | Phase 0 已完成 |
| force / rho 正規化 | **不變** | 與 streaming 方法無關 |

---

### 21.5 實作順序（7 步）

| 步驟 | 檔案 | 動作 | 依賴 |
|------|------|------|------|
| 1 | `gilbm/precompute.h` | 新建：PrecomputeGILBM_DeltaK() | Phase 0 度量項 |
| 2 | `gilbm/interpolation_gilbm.h` | 新建：interpolate_quadratic_3d() | — |
| 3 | `gilbm/boundary_conditions.h` | 新建：NeedsBoundaryCondition(), ChapmanEnskogBC() | — |
| 4 | `gilbm/evolution_gilbm.h` | 新建：stream_collide_GILBM_Buffer() | 步驟 1-3 |
| 5 | `memory.h` | 修改：delta_k 分配/釋放 | — |
| 6 | `main.cu` | 修改：初始化 + kernel 切換 | 步驟 1-5 |
| 7 | `variables.h` | 修改：`#define USE_GILBM` | — |

---

### 21.6 驗證計劃

#### 階段 A：編譯通過
- `nvcc -arch=sm_35` 零錯誤零警告
- `#undef USE_GILBM` 切回 ISLBM 仍可編譯運行

#### 階段 B：Phase 0 不受影響
- 6 判據仍全部 PASS（度量項未修改）

#### 階段 C：Poiseuille 流驗證
- 設 HillFunction ≡ 0（平坦壁面），Re=200
- 壁面速度 |u[k=2]| < 1e-6
- 壁面剪應力 τ_wall 與解析解 `6·ρ·ν·U_max/H²` 比較，誤差 < 5%
- 速度剖面 u(z) 與拋物線解析解比較

#### 階段 D：Periodic Hill Re=200
- 收斂監測：residual 單調遞減
- 壁面剪應力 C_f(y) 與 Mellen (2000) DNS 比較
- 分離/再附著點位置

---

### 21.7 風險評估

| 風險 | 嚴重度 | 緩解措施 |
|------|--------|---------|
| C-E BC 在高 dk_dz (~105) 壁面的數值穩定性 | 高 | 監控壁面 f 值，必要時 clamp |
| 二階插值在壁面附近精度不足 | 中 | 與 ISLBM 結果交叉驗證 |
| RK2 半步超出 k 有效範圍 | 低 | PrecomputeGILBM 中已 clamp |
| MRT 的 ω 在 GILBM 下含義是否改變 | 低 | Global time step 下 ω 不變（Imamura Eq. 10） |
| 壁面 ρ 外推精度 | 低 | 低 Ma 流，ρ 變化 O(Ma²) ≈ O(10⁻⁴) |

---

### 21.8 關鍵檔案路徑

| 檔案 | 路徑 | 動作 |
|------|------|------|
| `gilbm/precompute.h` | 新建 | 預計算 RK2 |
| `gilbm/interpolation_gilbm.h` | 新建 | 二階插值 |
| `gilbm/boundary_conditions.h` | 新建 | C-E BC |
| `gilbm/evolution_gilbm.h` | 新建 | GILBM kernel |
| `memory.h` | 修改 (~101 行附近) | delta_k 分配 |
| `main.cu` | 修改 (~170 行附近) | 初始化 + kernel 切換 |
| `variables.h` | 修改 (末尾) | `#define USE_GILBM` |
| `evolution.h` | 修改 (加 `#ifndef USE_GILBM` guard) | ISLBM fallback |
**每一次對話結束後的修改軍需傳github並生成commit摘要修改部分

---

## 二十二、minSize 與 dt 在位移公式中的角色：ISLBM vs GILBM 對比分析（2026-02-20）

### 22.1 問題背景

User 觀察到 ISLBM 的 `GetIntrplParameter_Xi()`（`initialization.h:228-255`）使用 `minSize` 作為物理偏移量：

```c
// initialization.h:232-239
GetXiParameter(XiParaF3_h,  z_h[k],         y_h[j]-minSize, xi_h, ...);  // F3: y-minSize
GetXiParameter(XiParaF5_h,  z_h[k]-minSize, y_h[j],         xi_h, ...);  // F5: z-minSize
GetXiParameter(XiParaF15_h, z_h[k]-minSize, y_h[j]-minSize, xi_h, ...);  // F15: 雙方向
```

同樣，x 方向和 y 方向也用 `minSize`：
```c
// GetIntrplParameter_X (initialization.h:203)
GetParameter_6th(XPara0_h, x_h[i]-minSize, x_h, i, i-3);

// GetIntrplParameter_Y (initialization.h:217)
GetParameter_6th(YPara0_h, y_h[i]-minSize, y_h, i, i-3);
```

User 主張：「y方向與z方向遷移前偏移點都是 minSize×1，那是座標變換前的樣態，座標變換後仍然應該多乘上 minSize。」

User 提議的 GILBM 公式：
```
δη = dt × e_x / dx × minSize    ← 多乘 minSize
δξ = dt × e_y / dy × minSize    ← 多乘 minSize
```

### 22.2 ISLBM 代碼追蹤

ISLBM 的 departure point 計算分兩步：

**Step 1: 物理空間計算 departure point**

以 F5（e_z = +1）為例，arrival point 在 (j, k)：
```
z_depart = z_h[k] - minSize     ← 物理空間後退 minSize
y_depart = y_h[j]               ← F5 只沿 z 移動
```

**Step 2: 座標變換到計算空間**

`GetXiParameter()` 將物理座標 (y, z) 變換為計算座標 ξ：
```c
// initialization.h:186-187
double L = LZ - HillFunction(pos_y) - minSize;
double pos_xi = LXi * (pos_z - (HillFunction(pos_y)+minSize/2.0)) / L;
```

然後 `GetParameter_6th()` 在計算空間 pos_xi 處計算 6 階 Lagrange 插值權重。

### 22.3 關鍵發現：dt = minSize

Pre-Phase 3 的 `variables.h`（commit d273169）：
```c
#define dt (minSize)    // variables.h:71
```

**`minSize` 和 `dt` 是同一個數值。** ISLBM 的 departure point 公式等價於：
```c
z_h[k] - minSize  ≡  z_h[k] - dt    // 完全等價
y_h[j] - minSize  ≡  y_h[j] - dt    // 完全等價
```

物理位移 = c × dt × e_α = 1 × dt × e_α = dt × e_α（c=1 約定）。

### 22.4 數值驗證：三方向對比

設定：`NZ6 = 70, minSize = 0.0191, dx = 0.0857, dy = 0.05`

#### x 方向 (η)：F1 (e_x = +1)

| 方法 | 計算 | 結果 |
|------|------|------|
| ISLBM | `x_h[i] + minSize` → grid: minSize/dx | 0.0191/0.0857 = **0.2229** |
| GILBM | δη = dt × e_x / dx = minSize / dx | 0.0191/0.0857 = **0.2229** |
| User 提議 | δη = dt × e_x / dx × minSize = minSize²/dx | 0.0191²/0.0857 = **0.00426** ❌ |

#### y 方向 (ξ)：F3 (e_y = -1)

| 方法 | 計算 | 結果 |
|------|------|------|
| ISLBM | `y_h[j] - minSize` → ξ 偏移 | minSize/dy = **0.382** |
| GILBM | δξ = dt × e_y / dy | minSize/dy = **0.382** |
| User 提議 | δξ = dt × e_y / dy × minSize | minSize²/dy = **0.00730** ❌ |

#### z 方向 (ζ) at wall (k=3)：F5 (e_z = +1)

| 方法 | 計算 | 結果 |
|------|------|------|
| ISLBM | `z_h[k] - minSize` → ξ 偏移 | minSize × dk_dz = **4/3** |
| GILBM | δζ = dt × dk_dz | minSize × 4/(3minSize) = **4/3** |
| User 提議 | δζ = dt × dk_dz × minSize | minSize × 4/3 = **0.0255** ❌ |

**ISLBM 和 GILBM 完全一致。User 提議的公式偏差 ~50 倍。**

### 22.5 兩者的範式差異

| 特性 | ISLBM | GILBM |
|------|-------|-------|
| Departure point 計算 | 物理空間 → 座標變換 → 插值 | 直接在計算空間計算 |
| 物理偏移量 | `minSize × e_α` = `dt × e_α` | `dt × e_α`（同值）|
| 座標變換 | `GetXiParameter` 非線性變換 | `dt × c̃` 線性化（含 RK2 修正）|
| 插值方法 | 6 階 Lagrange | 二階 Quadratic |
| 精度差異 | Euler（在 k 點取 metric）| RK2（在 k_half 取 metric，更精確）|
| dt = minSize 時 | — | 兩者數值一致（差 O(dt²)）|

### 22.6 結論

1. **ISLBM 的 `minSize` 就是 `dt × e_α`**：因為 `#define dt (minSize)`，二者永遠相等
2. **GILBM 的 `dt × c̃` 已包含物理位移和座標變換**：不需要額外乘 minSize
3. **額外乘 minSize 會導致**：
   - 量綱不一致（δ 應為無因次，多一個 [length]）
   - 數值錯誤（比 ISLBM 小 ~50 倍）
   - Imamura CFL 公式失效（改變 dt 不改變位移 → CFL 不可控）
4. **minSize 隱含在度量項中**：dk_dz(wall) = 2/minSize，物理效果已正確反映