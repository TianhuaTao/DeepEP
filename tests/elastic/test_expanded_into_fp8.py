"""Focused multi-rank check for FP8 ``dispatch_expanded_into``.

Run directly with, for example::

    PYTHONPATH=$PWD EP_REUSE_NCCL_COMM=0 \
        torchrun --standalone --nproc-per-node=8 \
        tests/elastic/test_expanded_into_fp8.py

The test intentionally uses caller-owned, capacity-sized q/scale outputs.  It
canonicalizes expanded rows through ``recv_src_metadata`` so that atomic row
placement does not need to be identical between the regular and ``*_into``
dispatch launches.
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist

import deep_ep


def _make_inputs(
    *,
    rank: int,
    tokens: int,
    hidden: int,
    top_k: int,
    world_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    element_ids = torch.arange(tokens * hidden, device="cuda", dtype=torch.float32)
    qdata = (
        (element_ids.reshape(tokens, hidden) + rank * 13).remainder(29) - 14
    ).to(torch.float8_e4m3fn)

    # DeepEP treats each scale element as an opaque four-byte sf_pack_t.  Fill
    # all four UE8M0 bytes with nontrivial values so row/column stride mistakes
    # are visible, then expose the payload as packed int32.
    scale_bytes = (
        torch.arange(tokens * (hidden // 32), device="cuda", dtype=torch.int64)
        + rank * 17
    ).remainder(31).add(111).to(torch.uint8).reshape(tokens, hidden // 32)
    packed_scales = scale_bytes.contiguous().view(torch.int32)

    token_ids = torch.arange(tokens, device="cuda", dtype=torch.long).view(-1, 1)
    lane_ids = torch.arange(top_k, device="cuda", dtype=torch.long).view(1, -1)
    # One expert per rank; lanes for a token always target distinct ranks.
    topk_idx = ((token_ids + rank + lane_ids) % world_size).to(deep_ep.topk_idx_t)

    if top_k != 2:
        topk_weights = torch.full(
            (tokens, top_k),
            1.0 / top_k,
            device="cuda",
            dtype=torch.float32,
        )
    else:
        first = 0.2 + token_ids.to(torch.float32).remainder(7) * 0.03
        topk_weights = torch.cat((first, 1.0 - first), dim=1).contiguous()
    return qdata, packed_scales, topk_idx.contiguous(), topk_weights


def _canonicalize_expanded(
    recv_x: tuple[torch.Tensor, torch.Tensor],
    recv_weights: torch.Tensor,
    handle: object,
    *,
    world_size: int,
    tokens: int,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    recv_q, recv_sf = recv_x
    actual_recv_tokens = int(handle.psum_num_recv_tokens_per_scaleup_rank[-1].item())
    metadata = handle.recv_src_metadata[:actual_recv_tokens]
    src_global = metadata[:, 0].to(torch.long)
    expanded_slots = metadata[:, 2 : 2 + top_k].to(torch.long)

    canonical_q = torch.zeros(
        (world_size * tokens, top_k, recv_q.shape[1]),
        device=recv_q.device,
        dtype=recv_q.dtype,
    )
    canonical_sf = torch.zeros(
        (world_size * tokens, top_k, recv_sf.shape[1]),
        device=recv_sf.device,
        dtype=recv_sf.dtype,
    )
    canonical_weights = torch.zeros(
        (world_size * tokens, top_k),
        device=recv_weights.device,
        dtype=recv_weights.dtype,
    )
    canonical_valid = torch.zeros(
        (world_size * tokens, top_k),
        device=recv_q.device,
        dtype=torch.bool,
    )

    for lane in range(top_k):
        slots = expanded_slots[:, lane]
        valid = slots >= 0
        src = src_global[valid]
        slots = slots[valid]
        canonical_q[src, lane] = recv_q.index_select(0, slots)
        canonical_sf[src, lane] = recv_sf.index_select(0, slots)
        canonical_weights[src, lane] = recv_weights.index_select(0, slots)
        canonical_valid[src, lane] = True

    return canonical_q, canonical_sf, canonical_weights, canonical_valid


def _all_gather_cat(x: torch.Tensor) -> torch.Tensor:
    gathered = [torch.empty_like(x) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, x)
    return torch.cat(gathered, dim=0)


def _assert_static_matches_source(
    canonical: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    local_q: torch.Tensor,
    local_sf: torch.Tensor,
    local_topk_idx: torch.Tensor,
    local_weights: torch.Tensor,
    rank: int,
) -> None:
    canonical_q, canonical_sf, canonical_weights, canonical_valid = canonical
    source_q = _all_gather_cat(local_q)
    source_sf = _all_gather_cat(local_sf)
    source_topk_idx = _all_gather_cat(local_topk_idx)
    source_weights = _all_gather_cat(local_weights)
    expected_valid = source_topk_idx == rank

    if not torch.equal(canonical_valid, expected_valid):
        raise AssertionError("expanded route validity differs from source top-k routing")
    for lane in range(source_topk_idx.shape[1]):
        valid = expected_valid[:, lane]
        if not torch.equal(canonical_q[valid, lane], source_q[valid]):
            raise AssertionError(f"expanded FP8 q payload mismatch in lane {lane}")
        if not torch.equal(canonical_sf[valid, lane], source_sf[valid]):
            raise AssertionError(f"expanded packed scale payload mismatch in lane {lane}")
        if not torch.equal(canonical_weights[valid, lane], source_weights[valid, lane]):
            raise AssertionError(f"expanded top-k weight mismatch in lane {lane}")


def _reference_rows(src_global: torch.Tensor, hidden: int) -> torch.Tensor:
    cols = torch.arange(hidden, device=src_global.device, dtype=torch.float32).view(1, -1)
    values = (src_global.to(torch.float32).view(-1, 1) * 0.03125) + (
        cols.remainder(23) * 0.0078125
    )
    return values.to(torch.bfloat16)


def _check_bf16_cached_dispatch(
    buffer: deep_ep.ElasticBuffer,
    handle: object,
    *,
    rank: int,
    tokens: int,
    hidden: int,
    top_k: int,
    rank_capacity: int,
    num_sms: int,
    num_qps: int,
) -> None:
    """Exercise the BF16 backward replay on an FP8-configured buffer."""
    local_src = rank * tokens + torch.arange(tokens, device="cuda", dtype=torch.long)
    source_grad = _reference_rows(local_src, hidden)
    recv_grad_out = torch.full(
        (rank_capacity, hidden),
        float("nan"),
        device="cuda",
        dtype=torch.bfloat16,
    )
    recv_grad, recv_idx, recv_weights, _cached_handle, _event = (
        buffer.dispatch_cached_expanded_into(
            source_grad,
            handle=handle,
            recv_x_out=recv_grad_out,
            num_sms=num_sms,
            num_qps=num_qps,
            async_with_compute_stream=False,
        )
    )
    if recv_grad.data_ptr() != recv_grad_out.data_ptr():
        raise AssertionError("cached dispatch did not preserve caller-owned BF16 storage")
    if recv_idx is not None or recv_weights is not None:
        raise AssertionError("cached expanded dispatch unexpectedly returned routing payloads")

    actual_recv_tokens = int(handle.psum_num_recv_tokens_per_scaleup_rank[-1].item())
    metadata = handle.recv_src_metadata[:actual_recv_tokens]
    src_global = metadata[:, 0].to(torch.long)
    expanded_slots = metadata[:, 2 : 2 + top_k].to(torch.long)
    expected_rows = _reference_rows(src_global, hidden)
    for lane in range(top_k):
        slots = expanded_slots[:, lane]
        valid = slots >= 0
        if not torch.equal(recv_grad.index_select(0, slots[valid]), expected_rows[valid]):
            raise AssertionError(f"cached BF16 dispatch payload mismatch in lane {lane}")


def _check_bf16_combine(
    buffer: deep_ep.ElasticBuffer,
    recv_weights: torch.Tensor,
    handle: object,
    *,
    rank: int,
    tokens: int,
    hidden: int,
    top_k: int,
    rank_capacity: int,
    num_sms: int,
    num_qps: int,
) -> None:
    expert_out = torch.zeros(
        (rank_capacity, hidden),
        device="cuda",
        dtype=torch.bfloat16,
    )
    actual_recv_tokens = int(handle.psum_num_recv_tokens_per_scaleup_rank[-1].item())
    metadata = handle.recv_src_metadata[:actual_recv_tokens]
    src_global = metadata[:, 0].to(torch.long)
    expanded_slots = metadata[:, 2 : 2 + top_k].to(torch.long)
    source_rows = _reference_rows(src_global, hidden)
    for lane in range(top_k):
        slots = expanded_slots[:, lane]
        valid = slots >= 0
        slots = slots[valid]
        expert_out[slots] = (
            source_rows[valid] * recv_weights.index_select(0, slots).view(-1, 1)
        ).to(torch.bfloat16)

    combined, combined_weights, _event = buffer.combine(
        expert_out,
        handle=handle,
        num_sms=num_sms,
        num_qps=num_qps,
        async_with_compute_stream=False,
    )
    if combined_weights is not None:
        raise AssertionError("expanded combine unexpectedly returned top-k weights")

    local_src = rank * tokens + torch.arange(tokens, device="cuda", dtype=torch.long)
    base = _reference_rows(local_src, hidden)
    # The test weights sum to one. Account for BF16 rounding before reduction.
    if top_k == 2:
        token_ids = torch.arange(tokens, device="cuda", dtype=torch.long).view(-1, 1)
        first = 0.2 + token_ids.to(torch.float32).remainder(7) * 0.03
        weights = torch.cat((first, 1.0 - first), dim=1)
    else:
        weights = torch.full(
            (tokens, top_k),
            1.0 / top_k,
            device="cuda",
            dtype=torch.float32,
        )
    expected = sum(
        (base * weights[:, lane : lane + 1]).to(torch.bfloat16).to(torch.float32)
        for lane in range(top_k)
    ).to(torch.bfloat16)
    torch.testing.assert_close(combined, expected, rtol=2e-2, atol=2e-2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=2)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 8:
        raise RuntimeError(f"this focused test requires EP=8, got world_size={world_size}")
    if args.hidden % 128 != 0:
        raise ValueError(f"hidden must be divisible by 128, got {args.hidden}")
    if args.top_k > world_size:
        raise ValueError(f"top-k must be <= world size, got {args.top_k}")

    qdata, packed_scales, topk_idx, topk_weights = _make_inputs(
        rank=rank,
        tokens=args.tokens,
        hidden=args.hidden,
        top_k=args.top_k,
        world_size=world_size,
    )
    rank_capacity = args.tokens * args.top_k
    buffer = deep_ep.ElasticBuffer(
        dist.group.WORLD,
        num_max_tokens_per_rank=args.tokens,
        hidden=args.hidden,
        num_topk=args.top_k,
        use_fp8_dispatch=True,
        deterministic=False,
        allow_hybrid_mode=True,
        allow_multiple_reduction=True,
        prefer_overlap_with_compute=False,
        explicitly_destroy=True,
    )
    num_sms = int(buffer.get_theoretical_num_sms(world_size, args.top_k))
    num_qps = int(buffer.get_theoretical_num_qps(num_sms))

    try:
        standard_x, _standard_idx, standard_weights, standard_handle, _event = buffer.dispatch(
            (qdata, packed_scales),
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            num_experts=world_size,
            num_max_tokens_per_rank=args.tokens,
            expert_alignment=1,
            num_sms=num_sms,
            num_qps=num_qps,
            async_with_compute_stream=False,
            do_cpu_sync=True,
            do_expand=True,
            use_tma_aligned_col_major_sf=False,
        )
        if not isinstance(standard_x, tuple) or standard_weights is None:
            raise AssertionError("standard FP8 expanded dispatch returned an invalid payload")

        recv_q_out = torch.full(
            (rank_capacity, args.hidden),
            1,
            device="cuda",
            dtype=torch.float8_e4m3fn,
        )
        recv_sf_out = torch.full(
            (rank_capacity, args.hidden // 128),
            -1,
            device="cuda",
            dtype=torch.int32,
        )
        recv_weights_out = torch.full(
            (rank_capacity,),
            float("nan"),
            device="cuda",
            dtype=torch.float32,
        )
        static_x, static_idx, static_weights, static_handle, _event = (
            buffer.dispatch_expanded_into(
                (qdata, packed_scales),
                topk_idx=topk_idx,
                topk_weights=topk_weights,
                recv_x_out=recv_q_out,
                recv_topk_weights_out=recv_weights_out,
                recv_sf_out=recv_sf_out,
                num_experts=world_size,
                num_max_tokens_per_rank=args.tokens,
                expert_alignment=1,
                num_sms=num_sms,
                num_qps=num_qps,
                async_with_compute_stream=False,
                do_cpu_sync=False,
            )
        )
        if static_idx is not None or static_weights is None or not isinstance(static_x, tuple):
            raise AssertionError("static FP8 expanded dispatch returned an invalid payload")
        if static_x[0].data_ptr() != recv_q_out.data_ptr():
            raise AssertionError("dispatch_expanded_into did not preserve caller-owned q storage")
        if static_x[1].data_ptr() != recv_sf_out.data_ptr():
            raise AssertionError("dispatch_expanded_into did not preserve caller-owned scale storage")
        if static_weights.data_ptr() != recv_weights_out.data_ptr():
            raise AssertionError("dispatch_expanded_into did not preserve caller-owned weight storage")

        standard = _canonicalize_expanded(
            standard_x,
            standard_weights,
            standard_handle,
            world_size=world_size,
            tokens=args.tokens,
            top_k=args.top_k,
        )
        static = _canonicalize_expanded(
            static_x,
            static_weights,
            static_handle,
            world_size=world_size,
            tokens=args.tokens,
            top_k=args.top_k,
        )
        for name, standard_value, static_value in zip(
            ("qdata", "packed scales", "top-k weights", "valid routes"),
            standard,
            static,
            strict=True,
        ):
            if not torch.equal(standard_value, static_value):
                raise AssertionError(f"standard/static expanded dispatch mismatch for {name}")

        _assert_static_matches_source(
            static,
            local_q=qdata,
            local_sf=packed_scales,
            local_topk_idx=topk_idx,
            local_weights=topk_weights,
            rank=rank,
        )
        _check_bf16_combine(
            buffer,
            static_weights,
            static_handle,
            rank=rank,
            tokens=args.tokens,
            hidden=args.hidden,
            top_k=args.top_k,
            rank_capacity=rank_capacity,
            num_sms=num_sms,
            num_qps=num_qps,
        )
        _check_bf16_cached_dispatch(
            buffer,
            static_handle,
            rank=rank,
            tokens=args.tokens,
            hidden=args.hidden,
            top_k=args.top_k,
            rank_capacity=rank_capacity,
            num_sms=num_sms,
            num_qps=num_qps,
        )

        dist.barrier()
        if rank == 0:
            print(
                "PASS: EP=8 FP8 dispatch, BF16 combine, and cached BF16 dispatch",
                flush=True,
            )
    finally:
        dist.barrier()
        buffer.destroy()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
