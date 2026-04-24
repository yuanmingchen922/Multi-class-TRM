"""
run_all.py -- Main Validation Entry Point (P_block Version)
Runs all V1-V7 validation modules in order, collects results, prints summary report.
Final judgment on the numerical reliability of Autonomous Multiclass TRM m3+m4 (.tex)
core research propositions. New: probabilistic blocking replaces Softplus lateral model.
"""

import os
import sys
import time
import json
import numpy as np


class _NumpyEncoder(json.JSONEncoder):
    """Handle numpy scalars and booleans in JSON output."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return super().default(obj)

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

import V1_occupancy
import V2_kinematics
import V3_fvm
import V4_probability
import V5_mass
import V6_stiffness
import V7_reactions

# ── 研究可靠性判断标准 ─────────────────────────────────────────────────────────
RESEARCH_CLAIMS = {
    'claim_1': {
        'desc': 'P_block probabilistic blocking: bounded, monotone, and well-posed (Eq. 8)',
        'checks': ['V4-a', 'V4-b', 'V4-c'],
        'module': 'V4'
    },
    'claim_2': {
        'desc': 'Singular barrier B(Omega) forms impenetrable capacity upper bound',
        'checks': ['V2-c', 'V2-f', 'V1-a'],
        'module': 'V1+V2'
    },
    'claim_3': {
        'desc': 'Lie-Trotter 3-phase splitting + Thomas algorithm resolves extreme stiffness',
        'checks': ['V6-a', 'V6-c', 'V6-d', 'V6-f'],
        'module': 'V6'
    },
    'claim_4': {
        'desc': 'Global mass conservation theorem (Theorem 1)',
        'checks': ['V5-a', 'V5-b', 'V5-c', 'V5-d'],
        'module': 'V5'
    },
    'claim_5': {
        'desc': 'FVM positivity and shockwave formation (global Godunov limiter)',
        'checks': ['V3-a', 'V3-b', 'V3-d', 'V3-e'],
        'module': 'V3'
    },
    'claim_6': {
        'desc': 'Moving bottleneck capture/release conservation (Phase 1 exact matrix exp zero-sum)',
        'checks': ['V7-a', 'V7-b', 'V7-c', 'V7-d'],
        'module': 'V7'
    },
    'claim_7': {
        'desc': 'Bs constraint: kinematic acceleration blockade (i>=i_thr) replaces lateral phase',
        'checks': ['V2-g', 'V4-f', 'V7-c'],
        'module': 'V2+V4+V7'
    },
}

SEP = '=' * 68


def run_all():
    print(f"\n{SEP}")
    print("  Autonomous Multiclass TRM m3+m4 -- Validation Framework (P_block)")
    print("  Ref: Multi-class_TRM.tex (P_block version) + Benchmark Dataset.json")
    print("  Modules: V1 Occupancy | V2 Kinematics | V3 FVM | V4 P_block")
    print("           V5 Mass Cons | V6 Stiffness  | V7 Capture/Release")
    print(SEP)

    all_results = {}
    t_start = time.perf_counter()

    modules = [
        ('V1', V1_occupancy),
        ('V2', V2_kinematics),
        ('V3', V3_fvm),
        ('V4', V4_probability),
        ('V5', V5_mass),
        ('V6', V6_stiffness),
        ('V7', V7_reactions),
    ]

    for name, mod in modules:
        try:
            t0 = time.perf_counter()
            res = mod.run()
            elapsed = time.perf_counter() - t0
            res['elapsed_s'] = round(elapsed, 2)
            all_results[name] = res
            print(f"  [{name}] 完成  ({elapsed:.1f}s)")
        except Exception as e:
            print(f"  [{name}] 异常: {e}")
            all_results[name] = {'module': name, 'checks': {}, 'summary': f'ERROR: {e}'}

    total_elapsed = time.perf_counter() - t_start

    # ── 汇总所有检验项 ─────────────────────────────────────────────────────────
    all_checks = {}
    for mod_key, res in all_results.items():
        for check_key, check_val in res.get('checks', {}).items():
            all_checks[check_key] = check_val

    n_pass  = sum(1 for v in all_checks.values() if v.get('passed', False))
    n_total = len(all_checks)

    # ── 研究命题评估 ──────────────────────────────────────────────────────────
    claim_verdicts = {}
    for ckey, claim in RESEARCH_CLAIMS.items():
        relevant = [all_checks.get(c, {}).get('passed', False)
                    for c in claim['checks'] if c in all_checks]
        verdict = all(relevant) if relevant else False
        claim_verdicts[ckey] = verdict

    # ── 打印最终报告 ──────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  验证结果汇总")
    print(SEP)

    for mod_key, res in all_results.items():
        summary = res.get('summary', 'N/A')
        elapsed = res.get('elapsed_s', '?')
        icon = '[PASS]' if 'PASSED' in summary else '[FAIL]'
        print(f"  {icon}  {mod_key}  {summary}  ({elapsed}s)")

    print(f"\n  总计: {n_pass}/{n_total} 检验项通过")

    print(f"\n{SEP}")
    print("  核心研究命题可靠性评估")
    print(SEP)
    for ckey, claim in RESEARCH_CLAIMS.items():
        verdict = claim_verdicts[ckey]
        icon = '[可靠]' if verdict else '[存疑]'
        relevant_checks = claim['checks']
        check_str = ', '.join([
            f"{c}:{'P' if all_checks.get(c, {}).get('passed', False) else 'F'}"
            for c in relevant_checks
        ])
        print(f"  {icon}  {claim['desc']}")
        print(f"          关联检验: {check_str}")

    total_passed = n_pass == n_total
    print(f"\n{SEP}")
    print(f"  总耗时: {total_elapsed:.1f}s")
    status = 'ALL PASSED' if total_passed else f'PARTIAL ({n_pass}/{n_total})'
    print(f"  最终状态: {status}")
    print(SEP)

    # ── 保存 JSON 摘要 ────────────────────────────────────────────────────────
    summary_path = os.path.join(BASE, 'validation_summary.json')
    json_out = {
        'total_checks': n_total,
        'passed_checks': n_pass,
        'total_elapsed_s': round(total_elapsed, 2),
        'modules': {},
        'claim_verdicts': claim_verdicts,
    }
    for mod_key, res in all_results.items():
        json_out['modules'][mod_key] = {
            'summary': res.get('summary', ''),
            'elapsed_s': res.get('elapsed_s', 0),
            'checks': {
                k: {'passed': v.get('passed', False), 'desc': v.get('desc', '')}
                for k, v in res.get('checks', {}).items()
            }
        }
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(json_out, f, ensure_ascii=False, indent=2, cls=_NumpyEncoder)
    print(f"\n  验证摘要已保存: {summary_path}")

    return json_out


if __name__ == '__main__':
    run_all()
