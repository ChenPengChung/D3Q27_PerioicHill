# GILBM 開發文件：理論、演算法與實作

> 基於 Imamura 2005 (J. Comput. Phys. 202, 645-663) 的 Generalized Interpolation-Supplemented LBM

---

## 一、座標系統與基本設定

### 1.1 座標變換

物理空間 (x, y, z) → 計算空間 (η, ξ, ζ)：
- η (i 方向)：均勻 x 網格，dx = LX/(NX6-7)
- ξ (j 方向)：均勻 y 網格，dy = LY/(NY6-7)
- ζ (k 方向)：tanh 拉伸 z 網格（body-fitted），壁面處最密

壁面（山丘表面）精確落在 k=3 格線上：z[j\*NZ6+3] = H(y[j])

### 1.2 Lattice Speed Convention (c = 1)

**核心結論：c 是固定的模型參數（streaming speed），c = 1。**

驗證來源：
- Imamura Eq. 2: `c_i = c × e_i` — 速度明確包含 c
- Imamura Eq. 13: `c̃ = c × e × ∂ξ/∂x` — 逆變速度包含 c
- Imamura Section 5.1: "U/c = 0.1" — c 由 Mach 數設定，獨立於網格/dt
- **c=1 時所有 c 因子消失**，位移 = `dt × e × ∂ξ/∂x`
- 黏度確認 c=1：`ν = (τ-0.5)/3 × dt`（variables.h:100）

### 1.3 速度模型

D3Q19，c² = 1/3，19 個離散速度方向（含 1 個靜止方向）。
權重：w₀ = 1/3，w₁₋₆ = 1/18，w₇₋₁₈ = 1/36

### 1.4 網格結構（Buffer=3，k=3 為壁面計算點）

```
k 索引    角色               z 座標                  是否計算
─────────────────────────────────────────────────────────
k=0,1    ghost layer        線性外插                   否
k=2      buffer layer       外插: 2*z[3]-z[4]          否
k=3      壁面計算點          H(y)（山丘表面）          是 ← C-E BC
k=4      第一內部計算點      H(y) + Δz                是
...      內部計算點          tanh 拉伸                是
k=NZ6-5  倒數第二內部點                               是
k=NZ6-4  頂壁計算點          LZ                       是 ← C-E BC
k=NZ6-3  buffer layer       外插: 2*z[NZ6-4]-z[NZ6-5]  否
k=NZ6-2, NZ6-1  ghost layer                          否
```

Kernel guard: `if (k <= 2 || k >= NZ6-3) return;`

---

## 二、逆變速度與位移預計算

### 2.1 三方向位移公式

```
δη[α]     = dt · e_x[α] / dx              → 常數 [19]（均勻 x）
δξ[α]     = dt · e_y[α] / dy              → 常數 [19]（均勻 y）
δζ[α,j,k] = dt · ẽ^ζ_α(k_half)           → RK2 中點值 [19×NYD6×NZ6]
```

**不需要額外乘 minSize**：因為 `#define dt (minSize)`（pre-Phase 3），dt 本身就是物理位移量。

### 2.2 ζ 方向 RK2 預計算（Imamura Eq. 19-20）

k 方向度量項隨空間劇烈變化（壁面附近 tanh 拉伸），必須用 RK2：

```
Step 1: ẽ^ζ_α(k) = e_y[α]·dk_dy(j,k) + e_z[α]·dk_dz(j,k)
Step 2: k_half = k - 0.5·dt·ẽ^ζ_α(k)
Step 3: 線性插值 dk_dy, dk_dz at k_half
Step 4: δζ = dt · ẽ^ζ_α(k_half)
```

Euler 積分在壁面附近有 ~50% 誤差（Imamura Fig. 7），RK2 將精度提升到 O(Δt³)。

### 2.3 位移存儲

| 陣列 | 大小 | GPU 記憶體 | 說明 |
|------|------|-----------|------|
| `GILBM_delta_eta[19]` | 19 | `__constant__` | 均勻 x |
| `GILBM_delta_xi[19]` | 19 | `__constant__` | 均勻 y |
| `delta_zeta_d[19*NYD6*NZ6]` | ~52K | device global | 非均勻 z |

### 2.4 ISLBM vs GILBM 位移對比

| 特性 | ISLBM | GILBM |
|------|-------|-------|
| Departure point | 物理空間 → 座標變換 → 插值 | 直接在計算空間計算 |
| 物理偏移量 | `minSize × e_α` = `dt × e_α` | `dt × e_α`（同值）|
| 座標變換 | `GetXiParameter` 非線性 | `dt × c̃` 線性化 + RK2 |
| 精度 | Euler（在 k 點取 metric） | RK2（在 k_half 取 metric）|

---

## 三、Imamura 4-Step Evolution Algorithm

### 3.1 核心思想

Imamura Eq. 14：`f_α(x, t+Δt) = f̃_α(x_D, t)`

其中 f̃ 是**碰撞後**分佈函數，x_D 是出發點（departure point）。
GILBM 的本質是：先在所有 stencil 節點執行碰撞，再對碰撞後的值做插值。

### 3.2 當前實作的 4 步流程

對每個到達點 A = (i,j,k)，每個速度方向 α：

**Step 1+2（合併）：讀取 + 插值 → f_streamed**
```
1. 讀取 f_old[α] 到 7×7×7 stencil buffer f_re[7][7][7]
2. 計算出發點座標：(up_i, up_j, up_k) = (i-δη, j-δξ, k-δζ)
3. 計算 7 點 Lagrange 權重 Lxi[7], Leta[7], Lzeta[7]
4. Tensor-product 插值 f_re → f_streamed（純量值）
5. 累加 f_streamed 到宏觀量（ρ, momentum）
```

**Step 3：Re-estimation（以各節點 τ_B 碰撞）**
```
對 343 個 stencil 節點 B：
  feq_B = compute_feq_alpha(α, ρ_B, ux_B, uy_B, uz_B)
  f_re[B] += (1/τ_B) × (feq_B - f_re[B])
```
→ f_re 從 pre-collision 變為 post-collision（Imamura Eq. 15）

**Step 4：Collision（以到達點 τ_A 碰撞）**
```
feq_A = compute_feq_alpha(α, ρ_A, ux_A, uy_A, uz_A)
對 343 個 stencil entries：
  f_re[B] += (1/τ_A) × (feq_A - f_re[B])
```

**輸出**：
- `f_new[α] = f_re[ci][cj][ck]`（stencil 中心位置，即 A 點，經雙階段碰撞後的值）
- 宏觀量從 f_streamed 累加計算（post-streaming 值）

### 3.3 邊界條件分支

對壁面節點（k=3 底壁, k=NZ6-4 頂壁），某些方向需 Chapman-Enskog BC：
```
判據：ẽ^ζ_α = e_y[α]·dk_dy + e_z[α]·dk_dz
底壁：ẽ^ζ_α > 0 → 出發點在壁外 → C-E BC
頂壁：ẽ^ζ_α < 0 → 出發點在壁外 → C-E BC
```

平坦底壁 BC 方向：α = {5,11,12,15,16}（5 個，皆 e_z > 0）
斜面底壁（slope < 45°）：額外包含 e_y 分量方向，共 ~8 個

---

## 四、7-Point Lagrange 插值權重組裝

### 4.1 共享 Stencil 架構

**核心優化**：每個到達點 A 使用 **一個** 7×7×7 stencil，由所有 18 個速度方向共享。

宏觀量 (ρ, ux, uy, uz) 在 343 個 stencil 節點預計算一次：
```cpp
double rho_s[7][7][7], ux_s[7][7][7], uy_s[7][7][7], uz_s[7][7][7];
double inv_tau_s[7][7];  // τ 只隨 (j,k) 變化，不隨 i 變化
```

不共享時：每個 α 獨立讀取 343×19 = 6517 個 f 值 → 18 方向共 117,306 次讀取
共享後：343×19 = 6517 次讀取（一次），節省 ~18 倍 global memory traffic

### 4.2 Stencil Base 與邊界自適應夾緊

Stencil 基底居中於到達點：
```
bi = i - 3,  bj = j - 3,  bk = k - 3
```

邊界夾緊確保 stencil 不超出有效域：
```
bi: clamp to [0, NX6-7]        （x 週期邊界）
bj: clamp to [0, NYD6-7]       （y MPI 邊界）
bk: clamp to [3, NZ6-10]       （z 壁面，k=3 到 bk+6=NZ6-4）
```

到達點 A 在 stencil 中的位置：
```
ci = i - bi  （內部恆為 3）
cj = j - bj  （內部恆為 3）
ck = k - bk  （靠近壁面時從 0 到 6 變化）← 這是最關鍵的變化
```

### 4.3 Lagrange 權重計算

**出發點座標**（計算空間）：
```
up_i = i - a_local × δη[α]
up_j = j - a_local × δξ[α]       （a_local = dt_local/dt_global，LTS 加速因子）
up_k = k - δζ[α, j, k]
```

**小數位置相對於 stencil base**：
```
t_i = up_i - bi  ∈ [0, 6]（居中時約 3±δ）
t_j = up_j - bj  ∈ [0, 6]
t_k = up_k - bk  ∈ [0, 6]（壁面附近因夾緊而偏移）
```

**7 點 Lagrange 基函數**（節點在 0,1,2,3,4,5,6）：
```
L_m(t) = Π_{n≠m} (t - n) / (m - n),  m = 0,...,6
```

精度：6 階多項式插值，O(Δx⁷) 截斷誤差

### 4.4 Tensor-Product 三維組裝

三層嵌套化簡 7³ = 343 點為三次 7 點 1D 插值：

```
// 最內層：沿 i 方向插值 → val_xi[sm]
for each (sm, sn):
    val_xi[sm] = Σ_{sl=0}^{6} Lxi[sl] × f_re[sl][sm][sn]

// 中間層：沿 j 方向插值 → val_eta[sn]
for each sn:
    val_eta[sn] = Σ_{sm=0}^{6} Leta[sm] × val_xi[sm]

// 最外層：沿 k 方向插值 → f_streamed
f_streamed = Σ_{sn=0}^{6} Lzeta[sn] × val_eta[sn]
```

計算量：7×7×7 + 7×7 + 7 = 343 + 49 + 7 = 399 次 multiply-add
（而非直接 7³ = 343 次三重乘積 = 1029 次 multiply）

---

## 五、Chapman-Enskog 壁面邊界條件

### 5.1 理論基礎（Imamura 2005 Eq. A.9）

選擇 C-E BC 而非 NEE（Non-Equilibrium Extrapolation）的理由：
1. **自洽性**：GILBM 本身基於 C-E 展開推導，BC 應在同一框架下
2. **度量項處理**：C-E 通過顯式應變率張量自然包含座標變換效應
3. **文獻一致性**：Imamura 文獻明確使用 C-E 分析推導的壁面條件

### 5.2 No-slip 壁面公式

完整 C-E 修正項：
```
C_α = -ω·dt · Σ_i (∂u_i/∂k) · [(9·e_{α,i}·e_{α,j} - δ_{ij}) · (∂k/∂x_j)]
```

其中 9 = 3/c²（D3Q19 的 c² = 1/3），∂k/∂x = 0, ∂k/∂y = dk_dy, ∂k/∂z = dk_dz

壁面速度梯度（一階差分，u[wall] = 0，二階版本已註解保留）：
```
∂u_i/∂k|_{k=3} = u_i[k=4]               // 一階差分 (u[wall]=0)
// 二階版本 (註解保留): (4·u_i[k=4] - u_i[k=5]) / 2
```

壁面密度：ρ_wall ≈ ρ[k=4]（Imamura §3.2：壁面法向壓力梯度為零）

最終：`f_α|_{wall} = w_α · ρ_wall · (1 + C_α)`

---

## 六、Local Time Step（LTS，Imamura 2005 §2.3）

### 6.1 全域時間步（Phase 3, Eq. 22/25）

```
Δt_g = λ / max_{α,j,k} |c̃_α(j,k)|
```

λ 為 CFL 安全因子（< 1），max 掃描所有方向和空間點。
壁面處 dk_dz 最大 → CFL 最嚴格 → 決定 Δt_g。

### 6.2 局部時間步（Phase 4, Eq. 28）

```
dt_local(j,k) = λ / max_α |c̃_α(j,k)|
τ_local(j,k) = 0.5 + 3ν / dt_local(j,k)
a(j,k) = dt_local / dt_global  （加速因子，≥ 1）
```

壁面附近：a ≈ 1（CFL 最嚴格，dt_local ≈ dt_global）
通道中心：a >> 1（CFL 寬鬆，dt_local >> dt_global → 加速收斂）

### 6.3 LTS 對位移的影響

η, ξ 方向位移用 a_local 縮放：
```
δη_local = a_local × δη[α]
δξ_local = a_local × δξ[α]
```

ζ 方向使用 `PrecomputeGILBM_DeltaZeta_Local`，以 dt_local 重做完整 RK2。

---

## 七、關鍵檔案結構

| 檔案 | 職責 |
|------|------|
| `gilbm/evolution_gilbm.h` | 主 kernel：4-step 演算法（Buffer + Full 兩版本）|
| `gilbm/interpolation_gilbm.h` | Lagrange 權重、feq 計算、插值函數 |
| `gilbm/precompute.h` | δη/δξ/δζ 預計算、CFL 時間步 |
| `gilbm/boundary_conditions.h` | Chapman-Enskog BC、方向判別 |
| `gilbm/metric_terms.h` | 度量項 dk_dy, dk_dz 計算 |

### 7.1 __constant__ 記憶體佈局

```cpp
__constant__ double GILBM_e[19][3];       // 離散速度
__constant__ double GILBM_W[19];          // 權重
__constant__ double GILBM_dt;             // 全域 dt
__constant__ double GILBM_tau;            // 全域 τ
__constant__ double GILBM_delta_eta[19];  // η 位移
__constant__ double GILBM_delta_xi[19];   // ξ 位移
```

---

## 八、度量項與壁面精度

### 8.1 dk_dy 在壁面（k=3）

```
dk_dy = -dz_dj / (dy · dz_dk)
```

dz_dj 在 k=3 只涉及 j±1 鄰居（同一 k=3 層）：
```
dz_dj|_{k=3} = (H(y[j+1]) - H(y[j-1])) / 2 = dH/dy
```
**不需要 k 方向鄰居**，精度 ≥ 內部點。

### 8.2 dk_dz 在壁面（k=3）

使用二階前差分（k=2 為 buffer）：
```
dz_dk = (-3·z[3] + 4·z[4] - z[5]) / 2
```

---

## 九、NEE vs Chapman-Enskog BC 理論對比

### 9.1 NEE 在 GILBM 中的三個問題

**問題 A**：需要 BC 的方向不再固定
- Cartesian LBM：底壁固定 5 方向
- GILBM：ẽ^ζ_α 隨空間變化 → BC 方向集合是 (j) 的函數

**問題 B**：f^neq 張量結構被座標變換改變（核心問題）
- NEE 假設 f^neq|wall ≈ f^neq|fluid_neighbor
- 在 GILBM 中，streaming 沿計算空間特徵線 → f^neq 混入度量項效應
- 壁面-流體間外推忽略度量項在 ζ 方向的變化

**問題 C**：LTS 下鬆弛參數含義改變
- 不同格點有效 ω 不同
- Re-estimation 機制修正 f^neq 傳遞
- NEE 的 (1-ω) 因子需考慮 local dt

### 9.2 C-E BC 優勢

1. 顯式使用速度梯度 → 通過度量項正確轉換
2. 不依賴 f^neq 外推假設
3. 與 GILBM 的 C-E 理論框架自洽
4. 高 Re 時不會因壁面 f^neq 梯度增大而降級

---

## 十、minSize 與 dt 的關係

### 10.1 關鍵發現

Pre-Phase 3 的 variables.h:
```c
#define dt (minSize)    // dt 和 minSize 是同一個數值
```

ISLBM 的 departure point：
```c
z_h[k] - minSize  ≡  z_h[k] - dt    // 完全等價
```

### 10.2 數值驗證

設定：NZ6=70, minSize=0.0191, dx=0.0857, dy=0.05

| 方向 | ISLBM | GILBM | 額外乘 minSize（錯誤）|
|------|-------|-------|---------------------|
| x (F1) | 0.2229 | 0.2229 | 0.00426 |
| y (F3) | 0.382 | 0.382 | 0.00730 |
| z (F5, wall) | 4/3 | 4/3 | 0.0255 |

**ISLBM 和 GILBM 完全一致。額外乘 minSize 偏差 ~50 倍。**

### 10.3 結論

1. ISLBM 的 minSize 就是 dt×e_α（dt=minSize 恆等式）
2. GILBM 的 dt×c̃ 已含物理位移 + 座標變換，不需額外乘 minSize
3. minSize 隱含在度量項中：dk_dz(wall) = 2/minSize
