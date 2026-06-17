# FragTile로 RLDX Attention 커널 튜닝하기 — 적용 방법 정리

목표: `Libra/FragTile/`를 **on-chip placement 튜닝 라이브러리**로 사용해서, RLDX-1의 5개
attention([ATTENTION_KERNELS.md](ATTENTION_KERNELS.md))을 작성할 때 **tile size / QK·PV·P RegFrags /
ring buffer 등을 sweep**하여 input shape별 최적 SMEM↔Register 배치를 찾는다.

> 핵심 프레이밍 (이전 버전 수정): vanilla decode 커널을 **복제**하는 게 아니다.
> **FragTile이 제공하는 placement 튜닝 기능**(tile 크기·register fragment 분배 sweep)을, RLDX attention의
> compute(QK / softmax / mask / RoPE / RMSNorm / PV)는 직접 작성하면서 **데이터 배치 부분에만** 끼워 넣는 것.
> vanilla 커널은 "FragTile을 이렇게 쓴다"는 **사용 예시**일 뿐이다.

---

## 1. FragTile은 무엇을 튜닝해 주는가 (멘탈 모델)

FragTile은 **버퍼별 on-chip 배치만** 소유한다(`FragTile/README.md`). attention compute는 전부 user 코드.

```
[user 작성]   QKV column-slice, RMSNorm, RoPE, QK^T 스케일, mask, online-softmax, P@V, epilogue
[FragTile]    K/V/Q/P/O 각 버퍼를 SMEM에 둘지 register에 둘지 + ldmatrix/cp.async/mma.sync 배관
```

왜 튜닝이 의미 있나 (`CLAUDE.md` Core Concept): GPU 커널은 한쪽 자원(보통 register)이 먼저 소진되어
CTAs/SM이 묶인다. FragTile은 K/V/Q/P tile의 일부를 **register↔SMEM 사이로 옮기는 비율**을 template knob으로
노출하고, 이를 sweep해서 occupancy(CTAs/SM)를 최대화하는 배치를 input shape별로 자동 선택한다.

---

## 2. 튜닝 표면 — sweep 가능한 knob (이게 user가 쓰려는 "기능")

출처: `FragTile/placement/policy.cuh` + `kernels/libra_common/fragtile_sweep.py`(`AXIS_TO_POLICY`) +
`kernels/attention/libra/libra_vanilla_attn_wrapper.py`(`generate_vanilla_attn_configs`).

| sweep 축 (Python) | FragTile 필드 | 의미 | 자원 trade | 대표 범위 |
|---|---|---|---|---|
| **TileRows** | `TileShape::Rows` (=BLOCK_S) | K/V tile의 seq 길이 | ↑ → SMEM·reg 둘 다 ↑ | {16,32,64,128} |
| **TileCols** | `TileShape::Cols` (=D) | head_dim | RLDX는 고정(64/128/256) | — |
| **Stages** | `RingPolicy::Stages` (=RING_BUFS) | ring buffer 슬롯 수(prefetch 깊이) | ↑ → SMEM ↑, latency hide ↑ | {1,2,3,4} |
| **NumSmem** | `RingPolicy::NumSmem` | 순수 SMEM 슬롯 수 (`NumHybrid = Stages−NumSmem`) | ↑ → SMEM ↑ / reg ↓ | [0, Stages] |
| **KRegFrags** | K `HybridSpec<Cols, RegFrags>` (=QK_REG_FRAGS) | hybrid 슬롯에서 **K/Q의 D축** 중 register에 둘 16-frag 수 | ↑ → reg ↑ / SMEM ↓ | [0, D/16] |
| **VRegFrags** | V `HybridSpec<Rows, RegFrags>` (=PV_REG_FRAGS) | hybrid 슬롯에서 **V의 BS축** 중 register에 둘 frag 수 | ↑ → reg ↑ / SMEM ↓ | [0, BLOCK_S/16] |
| **PRegFrags** | P `FragCachePolicy<CacheFrags>` (=P_REG_FRAGS) | softmax 후 P의 A-fragment를 register에 캐시할 수 | ↑ → reg ↑, SMEM reload ↓ | [0, BLOCK_S/16] |
| **NumWarps** | `NUM_WARPS` (커널 템플릿) | CTA당 warp 수 | — | {1,2,4,8} |
| **WarpSpec** | `WARP_SPEC` (커널 템플릿) | QK warp군 / PV warp군 분리 | live-set 분리 → reg ↓ | {0,1} |
| (커널) **p_splits** | — | seq축 split(flash-decoding 식) | 작은 M에 parallelism ↑ | {1..8} |
| (커널) **acc_buffers** | ScoreTile `Stages` | score ping-pong 버퍼 | — | {2} |

`KRegFrags`·`PRegFrags`가 user가 언급한 **"QKV Reg Frag"** 에 해당. 핵심 아이디어:
- `NumSmem=Stages`, `KRegFrags=VRegFrags=0` → **순수 full-SMEM** (register 최소, occupancy를 SMEM이 제한).
- `NumSmem` 줄이고 `KRegFrags/VRegFrags` 키움 → **hybrid**(register로 분산, SMEM 절약, 단 reg pressure ↑).
- vanilla의 top-1(B=16,S=4K)은 `BS=64 RB=3 NS=3 QKRF=5 PVRF=0 PRF=4` = **순수 full-SMEM**인데, 이유는 그
  커널이 register에 먼저 묶이기 때문(161 regs → 1 CTA/SM). **RLDX는 shape가 다르니 sweep 결과도 다를 것** — 그게 sweep의 목적.

---

## 3. 커널 작성자 API — FragTile을 코드에 끼우는 법

vanilla 커널(`libra_vanilla_attn_unified.cuh`)에서 추출한 사용 패턴. 새 RLDX 커널도 동일하게 쓴다.

**(a) tile 타입 선언** — knob을 템플릿 인자로 (`libra_vanilla_attn_common.cuh`의 alias 그대로 차용):
```cpp
namespace ft = libra::fragtile;
// K: D축(Cols) hybrid split, V: BS축(Rows) hybrid split
template <int BS,int RB,int NS,int QKRF> using KTile = ft::FragTile<
    ft::TileShape<BS, D>, ft::RingPolicy<NS,0,RB-NS, ft::HybridSpec<ft::HybridAxis::Cols, QKRF>>,
    ft::FragCachePolicy<0>, ft::Source::Input, ft::MmaOperand::B>;
template <int BS,int RB,int NS,int PVRF> using VTile = ft::FragTile<
    ft::TileShape<BS, D>, ft::RingPolicy<NS,0,RB-NS, ft::HybridSpec<ft::HybridAxis::Rows, PVRF>>,
    ft::FragCachePolicy<0>, ft::Source::Input, ft::MmaOperand::B>;
template <int BS,int PRF,int WS> using PTile = ft::FragTile<
    ft::TileShape<16, BS>, ft::RingPolicy<(WS?2:1),0,0>,
    ft::FragCachePolicy<PRF>, ft::Source::Intermediate, ft::MmaOperand::A>;
// Q/O는 커널 내부에서 KTile 기준 파생 (Q cache tier == K hybrid tier 강제)
```

**(b) SMEM 할당** — cursor ctor가 알아서 누적:
```cpp
extern __shared__ __align__(16) unsigned char smem_raw[];
int cur = 0;
KTile k_tile(smem_base, cur);  VTile v_tile(smem_base, cur);  PTile p_tile(smem_base, cur);
// k_tile.smem_pure() / .smem_hybrid() 로 포인터 획득. smem_bytes()로 총량 컴파일타임 계산.
```

**(c) prefetch / fragment load / mma** — FragTile 자유함수:
```cpp
ft::prefetch<NUM_WARPS, ...>(k_tile, seq, K, K_stride0, ...);  // cp.async 스테이징
cp_async_commit_group(); cp_async_wait_group<N>();             // cadence는 user 결정
ft::fill(score, 0.f);  ft::init(stat);                          // acc/softmax state
// TilePipe로 ldmatrix→mma_atom: K_STEPS(D/16) 루프, hybrid/pure 슬롯 자동 분기
```

**(d) user compute (FragTile 밖)** — RLDX 고유 부분을 여기에:
- **RMSNorm**(llm·action): SMEM Q/K tile에 `load_scalar/store_scalar` (**Case A**, 쉬움. README 예시가 그대로 RLDX RMSNorm).
- **RoPE**: `mma_step`의 `BFragTransform` hook (**Case B**). `kernels/fused_decode/libra/`(Convention B)의 `apply_rope_to_b_pair`가 레퍼런스.
- **mask**: score fragment(register, Case B)에 softmax 직전 `-inf` 주입 (causal / block-causal / none).
- **softmax / scale / epilogue**: 전부 user.

---

## 4. RLDX 텐서 shape → FragTile 인스턴스 (커널별)

[ATTENTION_KERNELS.md](ATTENTION_KERNELS.md)의 size를 `TileShape`에 직접 대입. `K_STEPS=D/16`.

| 커널 | D | TileCols | KRegFrags 범위 | VRegFrags/PRegFrags 범위 | mask | 추가 user compute |
|------|---|----------|----------------|--------------------------|------|--------------------|
| backbone/llm | 128 | 128 | 0..8 | 0..BS/16 | causal | RMSNorm+RoPE, GQA flatten |
| memory | 256 | 256 | 0..16 | 0..BS/16 | block-causal(16) | RoPE |
| action ds/ss | 64 | 64 | 0..4 | 0..BS/16 | none(joint) | (RMSNorm+RoPE는 기존 Triton 재활용 가능) |
| backbone/vision | config | =D | 0..D/16 | 0..BS/16 | none(이미지별) | RoPE, cu_seqlens 경계 |

- **TileRows(BLOCK_S)·BLOCK_M(쿼리 tile)** 은 자유 sweep 대상. M축에는 **쿼리 seq 위치**를 싣는다(prefill;
  vanilla의 "group head pack" decode와 다른 점 — compute 쪽 변경이고 FragTile 배치는 동일).
- D가 작을수록(action D=64) `KRegFrags`의 sweep 폭이 좁다(0..4). D=256(memory)은 SMEM 예산이 2배라 BLOCK_S를 작게.

---

## 5. sweep 배관 — autotuner / config 생성 / dispatch (그대로 복제)

vanilla의 3-레이어를 새 커널용으로 복제:

1. **config 생성** (`generate_vanilla_attn_configs` 복제): knob 카르테시안을 만들고
   - **SMEM 예산**으로 prune (`k_full_bytes = NS*BS*D*2` 등, **D를 RLDX 값으로**) — `_SMEM_LIMIT = get_sm_resources()`.
   - **FragTile 제약**으로 prune: `fragtile_sweep.fragtile_constraints_ok(point, tile_cols=D)` 호출.
     ⚠️ `tile_cols` 기본이 **128**이라 RLDX는 **반드시 `tile_cols=D`(64/256)로 넘겨야** KRegFrags 상한이 맞다.
2. **autotuner** (`LibraVanillaAttnAutotuner` 복제): `select_config`이 `(p_splits, seq_bucket, B, H_q, H_kv, group_size)`로
   캐싱, miss 시 grid 전체를 CUDA event로 타이밍해 min 선택. RLDX는 seq가 작고 고정적이라(예: action TOTAL≈82)
   `seq_bucket` 캐시가 잘 들어맞아 1회 autotune 후 재사용.
3. **dispatch shard** (`generate_cuda_dispatch.py` 복제): knob 튜플 → 템플릿 인스턴스 라우팅. `-mcmodel=large` +
   `--split-compile` 그대로.

> RLDX 통합 지점: build-time autotune은 RLDX의 기존 패턴(`autotune_split_m_threshold`, `libra_backbone_chain.py`)과
> 동일 철학 — 고정 shape에서 한 번 튜닝 후 캐시.

---

## 6. 권장 작업 순서

1. **D 일반화 확인** — `TileShape<·,D>`로 D∈{64,128,256} 컴파일 + `tests/fragtile/test_smem_bytes.cu` 패턴으로
   `smem_bytes()` 검증. (vanilla 커널의 `static_assert(kHeadDim==128)`은 vanilla 전용 — 새 커널엔 안 들어감.)
2. **최소 PoC (non-causal, attention-only)** — action ss/ds(D=64) 타깃. FragTile로 K/V/Q/P/O 선언 + QK/softmax/PV만.
   기존 `F.sdpa` 출력과 수치 비교(`_smoke_test.py` 패턴).
3. **sweep 배관 연결** — §5의 config 생성+autotuner를 붙여 `TileRows/Stages/NumSmem/KRegFrags/VRegFrags/PRegFrags/NumWarps/WarpSpec`를 실제로 sweep, top-1 확인.
4. **mask 추가** — block-causal(memory), causal(llm).
5. **RMSNorm/RoPE 융합** — Case A + Case B hook.
6. **GQA(llm)/cu_seqlens(vision)** — compute 쪽 확장.

단계 게이트(Libra roadmap): build OK / top-10 config ±2% latency / top-1 reg ±2 / 수치 정확도 보존.

---

## 7. 우선순위

| 우선 | 커널 | 이유 |
|------|------|------|
| 1 | **action ss/ds (D=64)** | mask 없음, 입력이 `(H,TOTAL,D)`로 정돈 + 현재 `F.sdpa`라 교체 효과 큼. sweep 기능 검증에 가장 단순 |
| 2 | **memory (D=256)** | block-causal만 추가, MHA(GQA 없음). D=256이라 SMEM↔reg trade가 커서 sweep 효과 클 가능성 |
| 3 | **backbone/llm (D=128)** | vanilla와 D 동일해 swizzle 호환 ↑. 단 causal+GQA+RMSNorm+RoPE 다 필요 |
| 4 | **backbone/vision** | 가변 seq + cu_seqlens. head_dim 확인 선행 |

---

## 8. 참고 파일 / 검증 필요

| 파일 | 용도 |
|------|------|
| `FragTile/placement/policy.cuh` | TileShape/RingPolicy/HybridSpec/FragCachePolicy/FragAccess 정의 (**튜닝 knob 원본**) |
| `FragTile/README.md` | placement API + Case A/B (user compute 경계) + 버퍼별 인스턴스 예시 |
| `FragTile/mma/ops.cuh`, `mma/tile_mma.cuh`, `placement/io.cuh` | `ft::fill/init/reduce`, `prefetch`, `load_fragment`, `mma_step` |
| `kernels/libra_common/fragtile_sweep.py` | **sweep 축 ↔ policy 매핑 + `fragtile_constraints_ok`** (그대로 import) |
| `kernels/attention/libra/libra_vanilla_attn_wrapper.py` | `generate_vanilla_attn_configs` + autotuner (복제 베이스) |
| `kernels/attention/libra/libra_vanilla_attn_common.cuh` | FT_KTile/VTile/PTile alias (knob→FragTile 바인딩 패턴) |
| `kernels/attention/libra/libra_vanilla_attn_unified.cuh` | FragTile 호출 시퀀스(prefetch/TilePipe/fill) 사용 예 |
| `kernels/fused_decode/libra/` | RMSNorm+RoPE in-kernel 융합(Convention B, RoPE hook) 레퍼런스 |

> **검증 필요**: ① vision `head_dim`(Qwen3-VL config) ② D=64/256에서 CUTLASS swizzle·ldmatrix 동작 ③ bf16 mma 정확도
> ④ `fragtile_constraints_ok(..., tile_cols=D)` 호출 시 RegFrags 상한이 D별로 맞는지.
