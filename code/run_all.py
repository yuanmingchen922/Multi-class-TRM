"""
run_all.py — 主验证入口  (m3+m4 升级版)
按顺序运行 V1-V7 全部验证模块，收集结果，打印汇总报告。
最终判断 Autonomous Multiclass TRM m3+m4 (.tex) 各项核心命题的数值可靠性。
"""

import os
import sys
import time
import json

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

import V1_occupancy
import V2_kinematics
import V3_fvm
import V4_lateral
import V5_mass
import V6_stiffness
import V7_reactions

# ── 研究可靠性判断标准 ─────────────────────────────────────────────────────────
RESEARCH_CLAIMS = {
    'claim_1': {
        'desc': 'Softplus 侧向变道优于 hard-max (C∞ 连续性)',
        'checks': ['V4-a', 'V4-b', 'V4-e'],
        'module': 'V4'
    },
    'claim_2': {
        'desc': '奇异屏障 B(Ω) 构成不可穿透容量上界',
        'checks': ['V2-c', 'V2-f', 'V1-a'],
        'module': 'V1+V2'
    },
    'claim_3': {
        'desc': 'Lie-Trotter 4 相分裂 + Thomas 算法解决极端刚性',
        'checks': ['V6-a', 'V6-c', 'V6-d', 'V6-f'],
        'module': 'V6'
    },
    'claim_4': {
        'desc': '全局质量守恒定理 (Theorem 1)',
        'checks': ['V5-a', 'V5-b', 'V5-c', 'V5-d'],
        'module': 'V5'
    },
    'claim_5': {
        'desc': 'FVM 正值性与激波有机形成 (全局 Godunov 限制器)',
        'checks': ['V3-a', 'V3-b', 'V3-d', 'V3-e'],
        'module': 'V3'
    },
    'claim_6': {
        'desc': '移动瓶颈捕获/释放守恒 (Phase 1 精确矩阵指数零和)',
        'checks': ['V7-a', 'V7-b', 'V7-c', 'V7-d'],
        'module': 'V7'
    },
    'claim_7': {
        'desc': 'Bs 约束系统: 加速封锁 + 绝对侧向禁止 (m3+m4 等构扩展)',
        'checks': ['V2-g', 'V4-f', 'V7-c'],
        'module': 'V2+V4+V7'
    },
}

SEP = '=' * 68


def run_all():
    print(f"\n{SEP}")
    print("  Autonomous Multiclass TRM m3+m4 — 综合验证框架")
    print("  对应: Multi-class_TRM.tex (m3+m4) + Benchmark Dataset.json")
    print("  模块: V1 占用 | V2 运动学 | V3 FVM | V4 侧向")
    print("        V5 质量守恒 | V6 刚性 | V7 捕获/释放反应")
    print(SEP)

    all_results = {}
    t_start = time.perf_counter()

    modules = [
        ('V1', V1_occupancy),
        ('V2', V2_kinematics),
        ('V3', V3_fvm),
        ('V4', V4_lateral),
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
        json.dump(json_out, f, ensure_ascii=False, indent=2)
    print(f"\n  验证摘要已保存: {summary_path}")

    return json_out


if __name__ == '__main__':
    run_all()
