"""Energy accounting (the SSD-offloading paper's model, as used in Q2).

    transfer     bytes moved over the link x (9.2 pJ/bit DDR5 read + 4.2 pJ/bit HBM write)
    dynamic      non-expert work per step (Q2 base pass) + per executed expert: weights read from
                 HBM (4.2 pJ/bit) + compute energy per token routed to it
    background   (70 W GPU+HBM + 57 W host DRAM) x elapsed time; the part spent stalled is reported
                 separately as stall energy (idle GPU waiting on the link)
Energy per token divides by tokens generated (steps x batch). EDP = energy per token x time per token.
"""


def energy_breakdown(cost, steps, batch, total_time, stall_time, bytes_moved, exec_experts, exec_tokens):
    transfer = bytes_moved * cost.e_link_byte
    dynamic = steps * cost.e_nonexpert_step + exec_experts * cost.e_hbm_expert \
        + exec_tokens * cost.e_comp_token_expert
    p_bg = cost.background_power(offloaded=True)
    background = p_bg * total_time
    tokens = steps * batch
    total = transfer + dynamic + background
    per_token = total / tokens
    return dict(energy_J_per_token=per_token,
                transfer_J_per_token=transfer / tokens,
                dynamic_J_per_token=dynamic / tokens,
                background_J_per_token=background / tokens,
                stall_J_per_token=p_bg * stall_time / tokens,
                edp_J_s=per_token * (total_time / steps))
