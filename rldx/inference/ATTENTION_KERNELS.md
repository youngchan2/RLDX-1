# RLDX-1 Attention 커널 Configuration 정리

`rldx/inference/` 하위 5개 모듈의 `engine/ops` + `engine/kernels`를 분석하여, 각 attention
커널이 **어떤 fusion 구성·마스킹·dtype·head 설정으로 실행되는지**와 **각 텐서의 size**를 정리한 문서.

> **표기**
> - `M` = 토큰(시퀀스) 길이, `D` = head_dim, `H` = num_heads
> - bf16 = bfloat16, fp32 = float32
> - "기본값"은 benchmark/registry 기준 default (변경 가능, 입력 의존)

---

## 0. 전체 요약

| 모듈 | op 이름 | Attention 구현 | 마스킹 | H / D | GQA | QK-Norm |
|------|---------|----------------|--------|-------|-----|---------|
| backbone/llm | `rldx_backbone::fused_attention` | **Triton flash (fully fused)** + Split-KV | Causal | 32 / 128 | ✅ (8 KV, G=4) | ✅ RMSNorm |
| backbone/vision_encoder | `rldx_backbone::vision_attention` | **Triton flash (fully fused)** + Split-KV | Non-causal (per-image) | config | ❌ | ❌ |
| memory | `mem::fused_attention` | **Triton flash (fully fused)** | **Block-causal** | 16 / 256 | ❌ | ❌ |
| action/double_stream | `ds::fused_attention_2way` / `_3way` | Triton(RMSNorm+RoPE) → **`F.sdpa`** | Non-causal (joint) | 24 / 64 | ❌ | ✅ RMSNorm |
| action/single_stream | `ss::fused_attention_2way` / `_3way` | Triton(RMSNorm+RoPE) → **`F.sdpa`** | Non-causal (joint) | 24 / 64 | ❌ | ✅ RMSNorm |

**핵심 차이**
- **backbone(llm/vision/memory)**: RoPE(+RMSNorm)부터 online-softmax attention까지 **하나의 Triton 커널로 완전 융합**.
- **action_model(double/single stream)**: Triton 커널은 **RMSNorm + RoPE까지만** 수행하고, attention 자체는 PyTorch `F.scaled_dot_product_attention`에 위임.

공통 dtype 정책: **bf16 저장 / fp32 compute (RMSNorm·softmax 누적)**, `QK^T`·`P@V`는 bf16 입력 + fp32 accumulator (tensor core).

---

## 1. backbone/llm — `rldx_backbone::fused_attention`

**파일**: `backbone/llm/engine/ops/op_fused_llm_attention.py`, `kernels/fused_llm_attention.py`

융합 구성: **Fused QKV(caller) → Q/K RMSNorm + weight → RoPE → Causal online-softmax attention**.
GQA 그룹핑(KV head 1개를 G개 Q head가 공유) + Split-KV(Flash-Decoding) 디스패치.

### Head 설정 (Qwen3-8B 기준, `forward` 기본값)
| 항목 | 값 |
|------|-----|
| `num_heads` (Q) | 32 |
| `num_kv_heads` | 8 |
| `head_dim` D | 128 |
| `GROUP_SIZE` | 32 / 8 = **4** |
| `Q_DIM` | 32 × 128 = **4096** |
| `K_DIM` = `V_DIM` | 8 × 128 = **1024** |
| `QKV_DIM` | 4096 + 2×1024 = **6144** |
| scale | 1/√128 |

> 실제 chain(`libra_backbone_chain.py`)에서는 `head_dim = self_attn.head_dim`, `num_heads/num_kv_heads`를 가중치 shape에서 동적으로 읽음.

### 텐서 size
| 텐서 | shape | dtype |
|------|-------|-------|
| `qkv` (입력) | (M, 6144) | bf16 |
| `q_norm_w` / `q_norm_w_rot` / `k_norm_w` / `k_norm_w_rot` | (128,) | bf16 (per-layer) |
| `cos` / `signed_sin` | (M, 128) | bf16 (chain 공유) |
| **출력** | (M, 4096) = (M, Q_DIM) | bf16 |
| `O_partial` (split 경로) | (MAX_SPLITS=8, M, 32, 128) | bf16 |
| `m_partial` / `l_partial` | (8, M, 32) | fp32 (m은 -inf prefill) |

`M` = LLM 입력 토큰 수 (vision merged 토큰 + text + cog 토큰). `select_layer=18` → 18개 decoder layer 통과, layer당 1회 호출.

### 디스패치 / 커널 config
- **`SPLIT_M_THRESHOLD = 128`** (build 시 `autotune_split_m_threshold`로 64/128/256/512 중 선택)
  - `M ≥ threshold` → **direct kernel** (partial 미할당, reduce 미호출). Grid: `(cdiv(M, BLOCK_S), num_kv_heads=8)`
  - `M < threshold` → **split kernel + reduce kernel**. Grid: `(cdiv(M, BLOCK_S), 8, NUM_SPLITS)`
- `MAX_SPLITS = 8`, reduce는 `BLOCK_S_R=32`
- Q row flatten: `BS_EFF = BLOCK_S × GROUP_SIZE`
- **autotune key = `["M"]`**
  - direct: `BLOCK_S ∈ {16,32}`, `BLOCK_P ∈ {16,32,64,128}`, `num_warps ∈ {2,4,8}`, `num_stages ∈ {2,3}`
  - split: `BLOCK_S ∈ {16,32}`, `BLOCK_P ∈ {16,32,64}`, `NUM_SPLITS ∈ {2,4,8}`
- Causal 최적화: early-exit(`kv_end = min(M, s_start+BLOCK_S)`), prefix/boundary 분할(prefix 구간은 causal mask 생략), group-shared cos/sin 로드.

---

## 2. backbone/vision_encoder — `rldx_backbone::vision_attention`

**파일**: `backbone/vision_encoder/engine/ops/op_fused_vision_attention.py`, `kernels/fused_vision_attention.py`

융합 구성: **Fused QKV에서 직접 Q/K/V 로드 → RoPE(rotate_half 부호를 `rope_sin`에 pre-bake) → Non-causal online-softmax attention** + Split-KV. **RMSNorm 없음, GQA 없음**.

### Head 설정
Qwen3-VL vision config에서 동적으로 읽음 (`vis_attn.num_heads`, `vis_attn.head_dim`, `vis_attn.scaling`).
| 항목 | 값 |
|------|-----|
| `num_heads` H | config 의존 |
| `head_dim` D | config 의존 |
| `Q_DIM` | H × D |
| `QKV_DIM` | **3 × Q_DIM** (Q\|K\|V 동일 차원, GQA 없음) |
| scale | `vis_attn.scaling` (= D^-0.5) |
| 마스킹 | **Non-causal** (이미지/seq 내부 full attention) |

### 텐서 size
| 텐서 | shape | dtype |
|------|-------|-------|
| `qkv` (입력) | (M, 3·H·D) | bf16 |
| `rope_cos` / `rope_sin` | (M, D) | fp32 (sin엔 rotate_half 부호 baked) |
| `cu_seqlens` | (num_seqs+1,) | int32 |
| **출력** | (M, H·D) | bf16 |
| `O_partial` | (8, M, H, D) | bf16 |
| `m_partial` / `l_partial` | (8, M, H) | fp32 |

- `M` = 전체 패치 수 = (뷰 V × 프레임 T) × (이미지당 패치 수). 기본 입력: **V=2, T=4, 224×224**.
- `num_seqs = len(cu_seqlens) - 1`, `seq_len = M / num_seqs` (균일 grid 가정 — RLDX 고정 grid_thw).
- **CTA는 cu_seqlens 경계를 넘지 않음** (이미지 단위 attention). Blackwell(sm_120) 대응 위해 masked lane을 마지막 유효 인덱스로 safe-clamp.

### 디스패치 / 커널 config
- **`SPLIT_M_THRESHOLD = 128`**, `MAX_SPLITS = 8`, **autotune key = `["SEQ_LEN"]`**
- direct grid: `(cdiv(seq_len, BLOCK_S), H, num_seqs)`
- split grid: `(cdiv(seq_len, BLOCK_S), H, NUM_SPLITS × num_seqs)` — `program_id(2)`에 (split_idx, seq_idx) packing
- `BLOCK_D = next_power_of_2(head_dim)`
- autotune: direct `BLOCK_S ∈ {32,64,128}` / `BLOCK_P ∈ {32,64,128}`; split `NUM_SPLITS ∈ {1,2,4,8}`

---

## 3. memory — `mem::fused_attention`

**파일**: `memory/engine/ops/op_fused_memory_attention.py`, `kernels/fused_memory_attention.py`

융합 구성: **RoPE → Block-Causal online-softmax attention**. **QK RMSNorm 없음, GQA 없음** (num_heads == num_kv_heads).

### Head 설정 (TransformerMemory)
| 항목 | 값 |
|------|-----|
| `num_heads` H | **16** |
| `head_dim` D | **256** |
| `block_attn_size` | **16** |
| `Q_DIM` = `K_DIM` = `V_DIM` | 16 × 256 = **4096** |
| `QKV_DIM` | 3 × 4096 = **12288** |
| scale | 1/√256 |
| 마스킹 | **Block-causal**: i가 j를 보려면 `i // 16 ≥ j // 16` |

### 텐서 size
| 텐서 | shape | dtype |
|------|-------|-------|
| `qkv` (입력) | (M, 12288) | bf16 |
| `cos` / `signed_sin` | (M, 256) | bf16 |
| **출력** | (M, 4096) | bf16 |

- `M = seq_length = memory_length(K) × memory_n_cog_tokens` (체크포인트 `mem_cfg`에서 결정).
  - `n_cog_tokens = 64` (registry). `memory_length`, `memory_n_cog_tokens`는 모델별.
- RoPE는 전부 bf16 연산 (eager `apply_rotary_pos_emb`와 일치).

### 커널 config
- 단일 커널 (split 없음). Grid: `(cdiv(M, BLOCK_S), num_heads=16)`
- `BLOCK_D = head_dim = 256`
- **autotune key = `["M"]`**: `BLOCK_S ∈ {16,32,64}`, `BLOCK_P ∈ {16,32,64}`, `num_stages ∈ {2,3}`, `num_warps ∈ {4,8}` (전조합)

---

## 4. action_model/double_stream

**파일**: `action_model/double_stream/engine/ops/op_fused_attention.py` (2way), `op_fused_attention_3way.py` (3way),
`kernels/rmsnorm_rope_ds.py`, `rmsnorm_rope_ds_3way.py`

> Triton 커널은 **RMSNorm + RoPE까지만** 수행 → `q_out/k_out/v_out` 생성 → **`F.scaled_dot_product_attention`** (mask 없음 = **non-causal full joint attention**) → permute/reshape.

### 공통 head 설정
| 항목 | 값 |
|------|-----|
| `inner_dim` N | **1536** |
| `H` | **24** |
| `D` | **64** (24 × 64 = 1536) |
| QK-Norm | ✅ stream별 RMSNorm weight (D,) |
| 마스킹 | **Non-causal** (`F.sdpa`에 mask/causal 미지정) |
| rmsnorm_rope grid | `(cdiv(TOTAL, BLOCK_S=128), H=24)`, `BLOCK_N = D = 64` |

### 4-1. `ds::fused_attention_2way` (DoubleStreamBlock, VL\|SA)
스트림 2개(SA, VL)를 따로 받아 joint attention.

| 텐서 | shape | dtype | 기본값(예) |
|------|-------|-------|-----------|
| `sa_qkv` | (n_sa, 3·1536=4608) | bf16 | n_sa=**18** |
| `vl_qkv` | (n_vl, 4608) | bf16 | n_vl=**64** |
| `q/k_norm_{sa,vl}_weight` | (64,) | bf16 | |
| `rope_cos` / `rope_sin` | (TOTAL, …) | — | |
| `q_out`/`k_out`/`v_out` (중간) | (H=24, TOTAL, D=64) | bf16 | TOTAL=82 |
| **출력 (attn)** | (1, TOTAL, 1536) — **[VL\|SA] 순서** | bf16 | (1, 82, 1536) |

- `TOTAL = n_vl + n_sa`
- `n_sa = n_sa_pure + num_temb_tokens = 17 + 1 = 18` (state 1 + action_horizon 16 + time token 1)
- 출력은 `attn[:, :n_vl]` = VL, `attn[:, n_vl:]` = SA로 분리.

### 4-2. `ds::fused_attention_3way` (ExpandedDoubleStreamBlock, VL\|SA\|P)
2way에 **physics(P) 스트림** 추가.

| 텐서 | shape | dtype |
|------|-------|-------|
| `sa_qkv` | (n_sa, 4608) | bf16 |
| `vl_qkv` | (n_vl, 4608) | bf16 |
| `p_qkv` | (n_p, 4608) | bf16 |
| `q/k_norm_{sa,vl,p}_weight` | (64,) | bf16 |
| `sa_rope_cos/sin` | (n_sa, 32) | fp32 (D//2) |
| `p_rope_cos/sin` | (n_p, 32) | fp32 |
| `q/k/v_out` (중간) | (24, TOTAL, 64) | bf16 |
| **출력 (attn)** | (1, TOTAL, 1536) — **[VL\|SA\|P] 순서** | bf16 |

- `TOTAL = n_vl + n_sa + n_p`, `n_p = physics_hist_len + physics_fut_len`
- physics 사용 시 n_vl 기본값은 **80** (`benchmark_action_model`).

---

## 5. action_model/single_stream

**파일**: `action_model/single_stream/engine/ops/op_fused_attention.py` (2way), `op_fused_attention_3way.py` (3way),
`kernels/rmsnorm_rope_ss.py`, `rmsnorm_rope_ss_3way.py`

> double_stream과 동일하게 **Triton(RMSNorm+RoPE) → `F.sdpa`(non-causal)** 구조. 단, VL/SA가 **하나의 스트림으로 concat**됨.

### 공통 head 설정: N=**1536**, H=**24**, D=**64** (double_stream과 동일). rmsnorm_rope grid `(cdiv(M, 128), 24)`.

토큰 레이아웃: `x = [VL(n_vl) | time_token(num_temb=1) | SA(n_sa_pure)]` → `M = n_vl + 1 + n_sa_pure`.
- linear1 출력에서 **앞 3·1536=4608**이 QKV, 나머지가 MLP(SwiGLU)용 (`qkv_mlp[:, :, 4608:]`).

### 5-1. `ss::fused_attention_2way` (SingleStreamBlock)
| 텐서 | shape | dtype | 기본값(예) |
|------|-------|-------|-----------|
| `qkv` (입력, QKV portion) | (M, 4608) | bf16 | M=**82** (64+1+17) |
| `q_norm_weight` / `k_norm_weight` | (64,) | bf16 | |
| `rope_cos` / `rope_sin` | (n_sa, …) | — | |
| `q/k/v_out` (중간) | (H=24, M, D=64) | bf16 | (24, 82, 64) |
| **출력** | (1, M, 1536) | bf16 | (1, 82, 1536) |

- `n_sa`(RoPE 적용 구간) = time_token + SA 부분 = `num_temb + n_sa_pure = 18`.

### 5-2. `ss::fused_attention_3way` (ExpandedSingleStreamBlock, [VL+SA] \| P)
| 텐서 | shape | dtype |
|------|-------|-------|
| `x_qkv` (VL+SA 스트림) | (n_x, 4608) | bf16 |
| `p_qkv` (physics 스트림) | (n_p, 4608) | bf16 |
| `x_q/k_norm_weight` | (64,) | bf16 |
| `p_q/k_norm_weight` | (64,) | bf16 |
| `sa_rope_cos/sin` | (n_sa, 32) | fp32 |
| `p_rope_cos/sin` | (n_p, 32) | fp32 |
| `q/k/v_out` (중간) | (24, TOTAL, 64) | bf16 |
| **출력** | (1, TOTAL, 1536) — **[VL+SA \| P] 순서** | bf16 |

- `n_x = n_vl + num_temb + n_sa_pure` (= 2way의 M), `TOTAL = n_x + n_p`
- `n_sa` = VL+SA 중 마지막 SA 부분(RoPE 대상), `n_p` = physics 토큰 수.

---

## 6. 기본 입력 파라미터 (benchmark/registry default)

| 파라미터 | 기본값 | 출처 |
|----------|--------|------|
| num_images (뷰 V) | 2 | benchmark_vla |
| num_frames (T) | 4 | benchmark_vla |
| image_size | 224×224 | benchmark_vla |
| n_state | 1 | benchmark |
| action_horizon | 16 | benchmark |
| num_temb_tokens | 1 | MSAT |
| n_sa_pure (= n_state + action_horizon) | 17 | benchmark |
| n_sa (DS, +time token) | 18 | graph_safe_msat |
| n_cog_tokens (→ n_vl_raw) | 64 | registry |
| n_vl (no physics) | 64 | benchmark_action_model |
| n_vl (physics) | 80 | benchmark_action_model |
| select_layer (LLM) | 18 | registry |
| denoising_steps | 4 | benchmark |

---

## 부록: 커널 dtype 파이프라인 (공통)
```
QKV(GEMM)        : bf16
RoPE cos/sin     : bf16 (memory/llm) · fp32(load)→연산 (vision)
RMSNorm 계산     : fp32 (llm/action) — memory/vision은 없음
Q·cos + rot(Q)·sin: bf16 × bf16 → bf16
QK^T             : bf16 × bf16, fp32 accumulator (tensor core)
Softmax          : fp32 (online softmax)
P @ V            : bf16 × bf16, fp32 accumulator
출력             : bf16
```
