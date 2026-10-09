"""
CarrotCompass - salary allocation formula v1
- 일반 파라미터(fact): Git/이슈트래커/근태/회계에서 수집되는 값
- 하이퍼파라미터(HP): 사람이 입력하는 주관값. 모두 clamp로 상·하한 적용
"""
from __future__ import annotations
from dataclasses import dataclass, field
from statistics import mean

PROJECT_NAME = "CarrotCompass"
LEVEL_TITLES = {1: "Associate", 2: "Junior", 3: "Mid", 4: "Senior", 5: "Lead"}


def level_title(level: int) -> str:
    return f"{LEVEL_TITLES.get(level, '?')} (L{level})"


# ─────────────────────────────────────────────
# 1. 설정 (하이퍼파라미터 초기값)
# ─────────────────────────────────────────────
@dataclass
class Config:
    # 기본급
    min_wage_monthly: float = 2_156_880      # 최저임금 월 환산액 (시급 10,320 × 209h)
    alpha: float = 300_000                   # 직급 1단계당 가산액

    # PR 난이도
    diff_weights: tuple = (0.3, 0.4, 0.3)    # a:b:c = 파일수 : 복잡도 : 리뷰왕복
    diff_caps: tuple = (20, 30, 8)           # 정규화 기준 (이 값 이상이면 1.0)
    d_min: float = 1.0
    d_max: float = 2.0
    u_min: float = 0.85                      # 평가자의 PR 난이도 보정 범위
    u_max: float = 1.15

    # 프로젝트 내 기여 점수 가중치 (산출, 품질, 시간, 협업)
    lambdas: tuple = (0.40, 0.25, 0.20, 0.15)
    defect_penalty: float = 2.0              # 출시 후 결함 1건당 감점(심각도 단위)

    # 몫 계산
    m_main: float = 1.5
    eps: float = 0.2                         # 평균 회귀 강도

    # 성과급 풀
    theta: float = 0.3                       # 순이익 중 직원 환원율
    smoothing: tuple = (0.5, 0.3, 0.2)       # 이번 달, 전월, 전전월

    # 사람 보정 + 상한
    e_min: float = 0.9
    e_max: float = 1.1
    kappa: float = 0.4                       # 성과급 상한 / 기본급
    min_evaluators: int = 3


# ─────────────────────────────────────────────
# 2. 데이터 모델 (fact)
# ─────────────────────────────────────────────
@dataclass
class PR:
    loc: int                  # 유효 코드량 (자동생성/설정 파일 제외)
    files: int                # 수정 파일 수
    complexity: float         # 수정 위치의 순환 복잡도 합
    review_rounds: int        # 리뷰 왕복 횟수
    u: float = 1.0            # [HP] 평가자 난이도 보정 배수


@dataclass
class Contribution:
    """한 직원의 한 프로젝트 내 활동 기록"""
    role: str                                        # "main" | "sub"
    prs: list[PR] = field(default_factory=list)
    bugs_fixed: list[int] = field(default_factory=list)   # 심각도 목록 (본인이 만든 버그 제외)
    defects: int = 0                                 # 출시 후 내가 만든 결함 수
    hours: float = 0.0
    collab: float = 0.0                              # 리뷰·페어·문서화 가중 건수
    commits: int = 0                                 # 이상 탐지용 (점수 미반영)


@dataclass
class Employee:
    name: str
    level: int
    work_ratio: float = 1.0                          # H: 출근일·근무시간 비율
    evaluator_scores: list[float] = field(default_factory=list)   # [HP] 인물 보정 점수


@dataclass
class Project:
    name: str
    revenue: float
    direct_cost: float
    contributions: dict[str, Contribution]           # 직원 이름 → 기여 기록

    @property
    def profit(self) -> float:
        return self.revenue - self.direct_cost


# ─────────────────────────────────────────────
# 3. 계산 함수
# ─────────────────────────────────────────────
def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def base_salary(emp: Employee, cfg: Config) -> float:
    return (cfg.min_wage_monthly + cfg.alpha * emp.level) * emp.work_ratio


def pr_difficulty(pr: PR, cfg: Config) -> float:
    a, b, c = cfg.diff_weights
    cf, cc, cr = cfg.diff_caps
    n_files = min(pr.files / cf, 1.0)
    n_cx = min(pr.complexity / cc, 1.0)
    n_rev = min(pr.review_rounds / cr, 1.0)
    d_auto = 1 + a * n_files + b * n_cx + c * n_rev
    u = clamp(pr.u, cfg.u_min, cfg.u_max)
    return clamp(d_auto * u, cfg.d_min, cfg.d_max)


def raw_metrics(con: Contribution, cfg: Config) -> dict[str, float]:
    output = sum(pr.loc ** 0.5 * pr_difficulty(pr, cfg) for pr in con.prs)
    quality = max(0.0, sum(con.bugs_fixed) - cfg.defect_penalty * con.defects)
    return {"output": output, "quality": quality, "time": con.hours, "collab": con.collab}


def project_shares(project: Project, cfg: Config) -> dict[str, float]:
    """프로젝트 내 몫 share' (main 배수 + 평균 회귀 반영)"""
    metrics = {n: raw_metrics(c, cfg) for n, c in project.contributions.items()}
    keys = ["output", "quality", "time", "collab"]
    totals = {k: sum(m[k] for m in metrics.values()) for k in keys}

    raw = {}
    for name, m in metrics.items():
        s = sum(
            lam * (m[k] / totals[k] if totals[k] > 0 else 0.0)
            for lam, k in zip(cfg.lambdas, keys)
        )
        mult = cfg.m_main if project.contributions[name].role == "main" else 1.0
        raw[name] = mult * s

    total = sum(raw.values())
    n = len(raw)
    if total == 0:
        return {name: 1 / n for name in raw}
    return {
        name: (1 - cfg.eps) * v / total + cfg.eps / n
        for name, v in raw.items()
    }


def person_adjustment(scores: list[float], cfg: Config) -> float:
    """평가자 점수 → 절사평균 → clamp. 평가자 부족 시 중립(1.0)"""
    if len(scores) < cfg.min_evaluators:
        return 1.0
    scores = sorted(clamp(s, cfg.e_min, cfg.e_max) for s in scores)  # 개별 입력도 clamp
    trimmed = scores[1:-1]                                             # 최고·최저 제거
    return clamp(mean(trimmed), cfg.e_min, cfg.e_max)


# ─────────────────────────────────────────────
# 4. 월 정산
# ─────────────────────────────────────────────
def settle_month(
    employees: list[Employee],
    projects: list[Project],
    fixed_cost: float,
    pool_history: list[float],       # 과거 Pool_raw, 최신이 마지막 [.., t-2, t-1]
    cfg: Config,
) -> dict:
    bases = {e.name: base_salary(e, cfg) for e in employees}
    total_base = sum(bases.values())
    total_profit = sum(p.profit for p in projects)

    # 성과급 풀 (평활화)
    pool_raw = cfg.theta * max(0.0, total_profit - fixed_cost - total_base)
    hist = (pool_history + [pool_raw])[-3:]
    hist = [hist[0]] * (3 - len(hist)) + hist            # 이력 부족 시 가장 오래된 값으로 채움
    w_now, w_prev, w_prev2 = cfg.smoothing
    pool = w_now * hist[-1] + w_prev * hist[-2] + w_prev2 * hist[-3]

    # 기여도 c_i = Σ Profit_p × share'
    contrib = {e.name: 0.0 for e in employees}
    for p in projects:
        shares = project_shares(p, cfg)
        for name, sh in shares.items():
            contrib[name] += max(0.0, p.profit) * sh
    total_c = sum(contrib.values())

    rows, reserve = [], 0.0
    for e in employees:
        bonus_raw = pool * contrib[e.name] / total_c if total_c > 0 else 0.0
        adj = person_adjustment(e.evaluator_scores, cfg)
        cap = cfg.kappa * bases[e.name]
        bonus_adj = bonus_raw * adj
        bonus = clamp(bonus_adj, 0.0, cap)
        reserve += bonus_adj - bonus if bonus_adj > cap else 0.0
        rows.append({
            "name": e.name, "title": level_title(e.level), "base": bases[e.name], "bonus_raw": bonus_raw,
            "adj": adj, "cap": cap, "bonus": bonus,
            "capped": bonus_adj > cap, "salary": bases[e.name] + bonus,
        })

    return {
        "rows": rows, "pool_raw": pool_raw, "pool": pool,
        "total_profit": total_profit, "total_base": total_base,
        "reserve": reserve,
    }


# ─────────────────────────────────────────────
# 5. 샘플 시뮬레이션
# ─────────────────────────────────────────────
def sample_data():
    emps = [
        Employee("Minsu Kim", level=4, evaluator_scores=[1.05, 1.08, 1.10, 1.02]),
        Employee("Seoyeon Lee", level=3, evaluator_scores=[1.00, 1.05, 1.02]),
        Employee("Jihoon Park", level=2, evaluator_scores=[0.98, 1.00, 1.03, 1.00]),
        Employee("Yujin Choi", level=2, evaluator_scores=[1.08, 1.10, 1.05]),
        Employee("Haneul Jung", level=1, work_ratio=0.9, evaluator_scores=[0.92, 1.00, 0.95]),
    ]
    projects = [
        Project("A Shop Backend", revenue=30_000_000, direct_cost=5_000_000, contributions={
            "Minsu Kim": Contribution("main", [PR(900, 14, 22, 6), PR(400, 8, 18, 4)],
                                 [3, 5], 0, 90, 6, 80),
            "Seoyeon Lee": Contribution("sub", [PR(600, 9, 12, 3)], [1, 1, 3], 1, 70, 8, 55),
            "Jihoon Park": Contribution("sub", [PR(500, 5, 6, 2)], [1, 1], 0, 60, 3, 120),
        }),
        Project("B Data Pipeline", revenue=20_000_000, direct_cost=3_000_000, contributions={
            "Seoyeon Lee": Contribution("main", [PR(700, 12, 25, 5)], [3], 0, 80, 5, 40),
            "Yujin Choi": Contribution("sub", [PR(1200, 6, 5, 1), PR(300, 3, 4, 1)], [1], 0, 85, 1, 90),
            "Haneul Jung": Contribution("sub", [PR(250, 4, 5, 2)], [1, 1], 0, 50, 2, 30),
        }),
        Project("C Admin Dashboard", revenue=12_000_000, direct_cost=2_000_000, contributions={
            "Jihoon Park": Contribution("main", [PR(450, 7, 8, 3)], [1, 1, 3], 0, 55, 4, 45),
            "Yujin Choi": Contribution("sub", [PR(350, 5, 6, 2)], [1], 1, 40, 2, 38),
            "Haneul Jung": Contribution("sub", [PR(300, 4, 4, 1)], [1], 0, 35, 3, 28),
        }),
    ]
    return emps, projects


def won(x: float, w: int = 11) -> str:
    return f"{x:>{w},.0f}"


def main():
    cfg = Config()
    emps, projects = sample_data()
    result = settle_month(
        emps, projects,
        fixed_cost=15_000_000,
        pool_history=[4_000_000, 5_000_000],      # Pool_raw of t-2, t-1
        cfg=cfg,
    )

    print(f"=== {PROJECT_NAME} ===")
    print(f"Total profit {result['total_profit']:,.0f} | Total base {result['total_base']:,.0f}")
    print(f"Pool_raw {result['pool_raw']:,.0f} -> smoothed Pool {result['pool']:,.0f}\n")

    header = (f"{'Name':<13}{'Rank':<14}{'Base':>11}{'Bonus(raw)':>12}"
              f"{'Adj':>7}{'Cap':>11}{'Bonus':>11}{'Salary':>12}")
    print(header)
    print("-" * len(header))
    for r in result["rows"]:
        flag = "  *capped" if r["capped"] else ""
        print(f"{r['name']:<13}{r['title']:<14}{won(r['base'])}{won(r['bonus_raw'], 12)}"
              f"{r['adj']:>7.3f}{won(r['cap'])}{won(r['bonus'])}{won(r['salary'], 12)}{flag}")
    print(f"\nReserve inflow: {result['reserve']:,.0f}")


if __name__ == "__main__":
    main()
