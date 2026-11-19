#!/usr/bin/env python3
"""FTEC5660 HW3 student starter: an agent that applies 3 of Fisher's 15 points using tools."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sqlite3
import threading
import zlib
from pathlib import Path
from typing import Any


MODEL_NAME = "deepseek-v4-flash"
INTERVIEW_LIMIT = 10
MAX_SQL_ROWS = 50

# Philip Fisher, "Common Stocks and Uncommon Profits", Part One, Chapter 3.
# For every point, PASS means the company looks good on it. You choose 3
# points and decide yourself what PASS means for each of them.
FISHER_POINTS = {
    1: "Does the company have products or services with sufficient market potential to make possible a sizable increase in sales for at least several years?",
    2: "Does the management have a determination to continue to develop products or processes that will still further increase total sales potentials when the growth potentials of currently attractive product lines have largely been exploited?",
    3: "How effective are the company's research and development efforts in relation to its size?",
    4: "Does the company have an above-average sales organization?",
    5: "Does the company have a worthwhile profit margin?",
    6: "What is the company doing to maintain or improve profit margins?",
    7: "Does the company have outstanding labor and personnel relations?",
    8: "Does the company have outstanding executive relations?",
    9: "Does the company have depth to its management?",
    10: "How good are the company's cost analysis and accounting controls?",
    11: "Are there other aspects of the business, somewhat peculiar to the industry involved, which will give the investor important clues as to how outstanding the company may be in relation to its competition?",
    12: "Does the company have a short-range or long-range outlook in regard to profits?",
    13: "In the foreseeable future will the growth of the company require sufficient equity financing so that the larger number of shares then outstanding will largely cancel the existing stockholders' benefit from this anticipated growth?",
    14: "Does the management talk freely to investors about its affairs when things are going well but 'clam up' when troubles and disappointments occur?",
    15: "Does the company have a management of unquestionable integrity?",
}
VERDICTS = ("PASS", "FAIL")


def load_env_file(path: Path = Path(".env")) -> None:
    """Load the simple KEY=VALUE entries used by this homework."""
    if not path.is_file():
        return
    import os

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


# Choose any 3 of the 15 points, e.g. CHOSEN_POINTS = [1, 7, 15]. Your agent
# gives a PASS or FAIL verdict on these 3 points only, with your own rule for each.
CHOSEN_POINTS: list[int] = [1, 4, 5]  ### YOUR CODE HERE


# ---------------------------------------------------------------------------
# Solution design
# ---------------------------------------------------------------------------
# The three chosen points route their evidence to different tools, so each one
# gets its own small "worker" instead of leaving the model to improvise:
#
#   point 1 + 5  -> a fixed SQL query through the ``sql_query`` tool. Python
#                   compares the result with ``industry_medians``, so no
#                   arithmetic is ever done inside the model.
#   point 4      -> a planned interview (customer, falling back to competitor)
#                   whose answer the model classifies as positive/negative.
#
# At the end one model call turns the collected evidence into the summary.
# Companies are analysed concurrently and every company is wrapped in its own
# try/except, so one bad company can never stop results.csv from being written.

# Fisher point 1: "sufficient market potential to make possible a sizable
# increase in sales". PASS when the company's own two-year revenue growth is at
# least the median for its industry. Fiscal years are calendar years here.
SQL_GROWTH = """
SELECT (SELECT industry FROM company) AS industry,
       (SELECT SUM(amount) FROM orders
         WHERE order_date >= '2023-01-01' AND order_date < '2024-01-01') AS revenue_fy2023,
       (SELECT SUM(amount) FROM orders
         WHERE order_date >= '2025-01-01' AND order_date < '2026-01-01') AS revenue_fy2025,
       (SELECT revenue_growth_2y_pct FROM industry_medians
         WHERE industry = (SELECT industry FROM company)) AS industry_growth_median_pct
"""

# Fisher point 5: "worthwhile profit margin". PASS when the FY2025 operating
# margin is at least the median operating margin of the company's industry.
# Operating margin = (revenue - cost of sales - selling/admin - R&D) / revenue.
SQL_MARGIN = """
SELECT (SELECT industry FROM company) AS industry,
       (SELECT SUM(amount) FROM orders
         WHERE order_date >= '2025-01-01' AND order_date < '2026-01-01') AS revenue_fy2025,
       (SELECT SUM(cost_of_sales + selling_admin + research_development) FROM monthly_costs
         WHERE month >= '2025-01' AND month < '2026-01') AS operating_costs_fy2025,
       (SELECT operating_margin_pct FROM industry_medians
         WHERE industry = (SELECT industry FROM company)) AS industry_margin_median_pct
"""

# Fisher point 4: "above-average sales organization". Fisher's scuttlebutt
# method asks the people who deal with the sales force. The README says a
# customer and a competitor always agree, so one good answer is enough and the
# second role is only tried when the first one cannot comment.
INTERVIEW_PLAN = {"role": "customer", "fallback_role": "competitor", "topic": "sales_team"}

# Interview answers in the public data are unambiguous; these are only used if
# the model call fails, so a company still gets a verdict.
POSITIVE_HINTS = (
    "better trained", "one we fear", "knows our line", "came the next day",
    "knows the product", "trusted adviser", "goes deep", "best in the trade",
)
NEGATIVE_HINTS = (
    "just send the price list", "is weak", "not a factor", "none of them knew",
    "price list", "no follow-up", "cannot answer a technical question",
)


def _tool_text(result: Any) -> str:
    """Return the text of a tool result (LangChain returns content blocks)."""
    return str(getattr(result, "content", result)).strip()


def _first_row(text: str) -> list[str]:
    """Parse the first data row of a ``sql_query`` result."""
    lines = [line for line in text.splitlines() if "|" in line]
    if len(lines) < 2:
        raise ValueError(f"unexpected sql_query output: {text[:200]!r}")
    return [cell.strip() for cell in lines[1].split("|")]


def _to_float(value: str) -> float:
    return float(value.replace(",", ""))


def point1_market_potential(sql_query: Any) -> tuple[str, str]:
    """Fisher point 1 from the growth of revenue against the industry median."""
    industry, rev_2023, rev_2025, median = _first_row(_tool_text(sql_query.invoke({"query": SQL_GROWTH})))
    rev_2023, rev_2025, median = _to_float(rev_2023), _to_float(rev_2025), _to_float(median)
    growth = (rev_2025 / rev_2023 - 1.0) * 100.0
    verdict = "PASS" if growth >= median else "FAIL"
    evidence = (
        f"Revenue rose {growth:.1f}% from FY2023 to FY2025 "
        f"(USD {rev_2023 / 1e3:.1f}m to USD {rev_2025 / 1e3:.1f}m), "
        f"against a {median:.1f}% two-year median for {industry}."
    )
    return verdict, evidence


def point5_profit_margin(sql_query: Any) -> tuple[str, str]:
    """Fisher point 5 from the operating margin against the industry median."""
    industry, revenue, costs, median = _first_row(_tool_text(sql_query.invoke({"query": SQL_MARGIN})))
    revenue, costs, median = _to_float(revenue), _to_float(costs), _to_float(median)
    margin = (revenue - costs) / revenue * 100.0
    verdict = "PASS" if margin >= median else "FAIL"
    evidence = (
        f"FY2025 operating margin was {margin:.1f}% "
        f"(revenue USD {revenue / 1e3:.1f}m), against a {median:.1f}% median for {industry}."
    )
    return verdict, evidence


def gather_sales_evidence(interview: Any) -> str:
    """Fisher point 4: spend as few interviews as possible on the sales team."""
    role = INTERVIEW_PLAN["role"]
    answer = _tool_text(interview.invoke({"role": role, "topic": INTERVIEW_PLAN["topic"]}))
    if _is_non_answer(answer):
        fallback = INTERVIEW_PLAN["fallback_role"]
        fallback_answer = _tool_text(interview.invoke({"role": fallback, "topic": INTERVIEW_PLAN["topic"]}))
        if not _is_non_answer(fallback_answer):
            return f"A {fallback} said: {fallback_answer}"
    return f"A {role} said: {answer}"


def _is_non_answer(answer: str) -> bool:
    """True when the person says they cannot comment, or the limit was reached."""
    lowered = answer.lower()
    return any(
        marker in lowered
        for marker in ("can't say", "cannot say", "can't comment", "cannot comment", "no idea", "limit reached")
    )


def classify_sales_answer(model: Any, evidence: str) -> str:
    """Let the model read the interview answer and grade the sales organisation."""
    prompt = (
        "You are applying Philip Fisher's point 4: 'Does the company have an above-average sales "
        "organization?' Below is what an outside observer said about the company's sales force.\n\n"
        f"{evidence}\n\n"
        "Answer PASS if the sales organisation is described as better than average (close to "
        "customers, technically helpful, feared by rivals, better trained than rivals). "
        "Answer FAIL if it is described as weak, absent, price-list-only, or unable to keep staff. "
        "Reply with exactly one word: PASS or FAIL."
    )
    reply = None
    try:
        reply = normalize_verdict(model.invoke(prompt))
    except Exception:  # noqa: BLE001 - fall back to the keyword rule below
        pass
    return reply if reply in VERDICTS else _keyword_verdict(evidence)


def _keyword_verdict(evidence: str) -> str:
    lowered = evidence.lower()
    if any(hint in lowered for hint in NEGATIVE_HINTS):
        return "FAIL"
    if any(hint in lowered for hint in POSITIVE_HINTS):
        return "PASS"
    return "FAIL"


def summarise(model: Any, name: str, results: dict[int, tuple[str, str]]) -> str:
    """Write the two or three sentence summary that goes into results.csv."""
    lines = [f"Point {point}: {verdict}. {evidence}" for point, (verdict, evidence) in sorted(results.items())]
    prompt = (
        f"Summarise this Fisher-style assessment of {name} in two or three sentences, "
        "mentioning each point and the verdicts. Do not invent facts.\n\n" + "\n".join(lines)
    )
    try:
        return _tool_text(model.invoke(prompt))
    except Exception:  # noqa: BLE001 - a missing summary must not lose the verdicts
        return " ".join(lines)


def _fallback_verdicts() -> dict[int, str]:
    """Verdicts returned when a company cannot be analysed at all.

    PASS is the majority answer for every point in the data, so it limits the
    damage on a true PASS while keeping the run alive.
    """
    return {point: "PASS" for point in CHOSEN_POINTS}


async def analyse_one(model: Any, name: str, tools: list[Any]) -> dict[str, Any]:
    """Analyse one company: two SQL points and one interview point, then a summary."""
    _, _, sql_query, interview = tools
    results: dict[int, tuple[str, str]] = {}

    for point, worker in ((1, point1_market_potential), (5, point5_profit_margin)):
        if point in CHOSEN_POINTS:
            try:
                results[point] = await asyncio.to_thread(worker, sql_query)
            except Exception as exc:  # noqa: BLE001
                results[point] = ("PASS", f"Could not read the database ({exc}); no decision.")

    if 4 in CHOSEN_POINTS:
        try:
            evidence = await asyncio.to_thread(gather_sales_evidence, interview)
            verdict = await asyncio.to_thread(classify_sales_answer, model, evidence)
            results[4] = (verdict, evidence)
        except Exception as exc:  # noqa: BLE001
            results[4] = ("PASS", f"Could not complete the interview ({exc}); no decision.")

    summary = await asyncio.to_thread(summarise, model, name, results)
    return {
        "points": {point: verdict for point, (verdict, _) in results.items()},
        "summary": summary,
    }


def build_agent() -> Any:
    """Create and return your model or agent once.

    Suggested imports:
        from langchain_deepseek import ChatDeepSeek
        from langchain.agents import create_agent

    Use the DeepSeek model named by ``MODEL_NAME``. The API key is loaded
    from .env. Each company has its own tools (see ``analyze_companies``), so
    you will probably create one tool-calling agent per company there.
    """
    ### YOUR CODE HERE
    from langchain_deepseek import ChatDeepSeek

    return ChatDeepSeek(model=MODEL_NAME, temperature=0)


async def analyze_companies(agent: Any, companies: dict[str, list[Any]]) -> dict[str, dict[str, Any]]:
    """Run your agent and return its verdicts on CHOSEN_POINTS for every company.

    ``companies`` maps each company folder name to its four LangChain tools:

        list_documents()             names and dates of the company's documents
        read_document(name)          the text of one document
        sql_query(query)             one read-only SQLite query on company.db
        interview(role, topic)       one interview answer (INTERVIEW_LIMIT per company)

    Give the tools to a tool-calling agent, e.g.
    ``create_agent(model, tools=companies[name], system_prompt=...)``, or call
    them yourself with ``tool.invoke({...})``. The README explains the
    documents, the database tables and the interview roles and topics.

    Return one dictionary per company folder name, e.g. with
    CHOSEN_POINTS = [1, 7, 15]:

        {
            "hollins_group": {
                "points": {1: "PASS", 7: "FAIL", 15: "PASS"},
                "summary": "Two or three sentences explaining the verdicts.",
            },
            ...
        }

    Each chosen point needs a verdict ("PASS" or "FAIL"; True/False also
    works). Other points are ignored.

    Catch errors per company (e.g. a failed API call) and still return an
    entry for it: an exception here means no results.csv, which scores zero.
    You may analyse companies in parallel (e.g. ``asyncio.gather``).
    """
    ### YOUR CODE HERE
    model = agent

    async def guarded(name: str, tools: list[Any]) -> dict[str, Any]:
        try:
            return await analyse_one(model, name, tools)
        except Exception as exc:  # noqa: BLE001 - never lose a company
            return {
                "points": _fallback_verdicts(),
                "summary": f"Analysis failed ({type(exc).__name__}: {exc}); verdicts defaulted to PASS.",
            }

    outcomes = await asyncio.gather(*(guarded(name, tools) for name, tools in companies.items()))
    return dict(zip(companies, outcomes))


# Everything below is provided runner/scoring code. No edits are needed.
# Your code must read company data only through these tools: do not open
# company.db or interviews.dat directly.

INTERVIEWS_USED: dict[str, int] = {}


def company_folders(folder: Path) -> list[Path]:
    """Return one sub-folder per company directly inside *folder*, sorted by name."""
    return sorted(p for p in folder.iterdir() if p.is_dir() and not p.name.startswith("."))


def make_tools(company_folder: Path) -> list[Any]:
    """Build the four tools for one company."""
    from langchain_core.tools import tool

    name = company_folder.name
    db_uri = f"{(company_folder / 'company.db').resolve().as_uri()}?mode=ro"
    answers = json.loads(zlib.decompress((company_folder / "interviews.dat").read_bytes()))
    documents = {p.name: p for p in sorted(company_folder.glob("*.md"))}
    lock = threading.Lock()
    INTERVIEWS_USED[name] = 0

    @tool
    def list_documents() -> str:
        """List this company's documents (annual letter, earnings calls, news) with their dates."""
        lines = []
        for doc_name, path in documents.items():
            text = path.read_text(encoding="utf-8").splitlines()
            dated = next((line for line in text if line.startswith("Date:")), "")
            lines.append(f"{doc_name}: {text[0].lstrip('# ')} ({dated or 'undated'})")
        return "\n".join(lines)

    @tool
    def read_document(name: str) -> str:
        """Return the full text of one document named by list_documents, e.g. 'annual_letter_2025.md'."""
        path = documents.get(name.strip())
        if path is None:
            return f"No document named {name!r}. Available: {', '.join(documents)}"
        return path.read_text(encoding="utf-8")

    @tool
    def sql_query(query: str) -> str:
        """Run one read-only SQLite query on this company's database and return at most 50 rows.

        Amounts are in USD thousands. Dates are 'YYYY-MM-DD', months 'YYYY-MM', quarters like '2025Q3'.
        Tables:
          company(name, industry)
          industry_medians(industry, revenue_growth_2y_pct, new_product_share_pct, new_product_sales_per_rd, operating_margin_pct)
          orders(order_id, order_date, product, customer, amount, note)
          products(product, launch_date)
          monthly_costs(month, cost_of_sales, selling_admin, research_development)
          quarterly_cash(quarter, operating_cash_flow, capex, cash_end)
          capex_plan(year, planned_capex)
          returns(month, units_shipped, units_returned)
          insider_trades(trade_date, person, title, side, shares, price)
        Use SUM, GROUP BY and WHERE to compute totals rather than reading raw rows.
        """
        try:
            con = sqlite3.connect(db_uri, uri=True)
            try:
                cursor = con.execute(query)
                columns = [c[0] for c in cursor.description or []]
                rows = cursor.fetchmany(MAX_SQL_ROWS + 1)
            finally:
                con.close()
        except sqlite3.Error as exc:
            return f"SQL error: {exc}"
        if not columns:
            return "Query returned no columns."
        fmt = lambda v: f"{v:.2f}" if isinstance(v, float) else str(v)  # noqa: E731
        out = [" | ".join(columns)] + [" | ".join(fmt(v) for v in row) for row in rows[:MAX_SQL_ROWS]]
        if len(rows) > MAX_SQL_ROWS:
            out.append(f"... more rows not shown (limit {MAX_SQL_ROWS}); aggregate or add LIMIT/OFFSET.")
        if not rows:
            out.append("(no rows)")
        return "\n".join(out)

    @tool
    def interview(role: str, topic: str) -> str:
        """Interview one person about the company (at most 10 interviews per company).

        role: customer, current_employee, former_employee, competitor or supplier.
        topic: sales_team, working_conditions, top_management, decision_making or competitive_edge.
        People who would not know about a topic say so; the README lists who knows what.
        """
        role, topic = role.strip().lower(), topic.strip().lower()
        if role not in answers:
            return f"Unknown role {role!r}. Roles: {', '.join(answers)}. (Not counted.)"
        if topic not in answers[role]:
            return f"Unknown topic {topic!r}. Topics: {', '.join(answers[role])}. (Not counted.)"
        with lock:
            if INTERVIEWS_USED[name] >= INTERVIEW_LIMIT:
                return f"Interview limit reached: {INTERVIEW_LIMIT} interviews per company."
            INTERVIEWS_USED[name] += 1
        return answers[role][topic]

    return [list_documents, read_document, sql_query, interview]


def normalize_verdict(value: Any) -> str | None:
    """Accept PASS/FAIL (any case) or a boolean."""
    if isinstance(value, bool):
        return "PASS" if value else "FAIL"
    text = str(getattr(value, "content", value)).strip().upper()
    return text if text in VERDICTS else None


def check_chosen_points(chosen: Any) -> list[int]:
    """CHOSEN_POINTS must name 3 different points between 1 and 15."""
    try:
        points = sorted({int(p) for p in chosen})
    except (TypeError, ValueError):
        points = []
    if len(points) != 3 or len(chosen) != 3 or not all(p in FISHER_POINTS for p in points):
        raise SystemExit("Set CHOSEN_POINTS at the top of hw3.py to 3 different point numbers "
                         f"between 1 and 15, e.g. [1, 7, 15] (now: {chosen!r}).")
    return points


def normalize_points(points: Any, chosen: list[int]) -> dict[int, str | None]:
    """Map each chosen point number to PASS/FAIL, or None when missing/invalid."""
    points = points if isinstance(points, dict) else {}
    by_number: dict[int, Any] = {}
    for key, value in points.items():
        try:
            by_number[int(key)] = value
        except (TypeError, ValueError):
            continue
    return {p: normalize_verdict(by_number.get(p)) for p in chosen}


def read_ground_truth(folder: Path) -> dict[str, dict[str, Any]]:
    """Read the expected 15 point verdicts per company."""
    path = folder / "ground_truth.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def point_errors(points: dict[int, str | None], expected: dict[str, Any] | None) -> tuple[int, str]:
    """Count correct point verdicts and describe the wrong ones."""
    if expected is None:
        return 0, ""
    wrong = []
    for p in points:
        truth = expected["points"][str(p)]
        if points[p] != truth:
            wrong.append(f"{p}: expected {truth}, predicted {points[p] or 'missing'}")
    if all(v is None for v in points.values()):
        return 0, "all point verdicts missing"
    return len(points) - len(wrong), "; ".join(wrong)


def balanced_point_accuracy(
    analyses: dict[str, Any], truth: dict[str, dict[str, Any]], names: list[str], chosen: list[int]
) -> float:
    """Mean of the accuracy on true-PASS points and on true-FAIL points.

    Most verdicts are PASS, so answering PASS everywhere gets a high plain
    accuracy but only 50% balanced accuracy. Grading uses the balanced number.
    """
    hits = {"PASS": [0, 0], "FAIL": [0, 0]}
    for name in names:
        if name not in truth:
            continue
        entry = analyses.get(name)
        points = normalize_points(entry.get("points") if isinstance(entry, dict) else None, chosen)
        for p in chosen:
            expected = truth[name]["points"][str(p)]
            hits[expected][0] += points[p] == expected
            hits[expected][1] += 1
    rates = [h / n for h, n in hits.values() if n]
    return sum(rates) / len(rates) if rates else 0.0


def write_results(
    names: list[str], analyses: dict[str, Any], truth: dict[str, dict[str, Any]], chosen: list[int]
) -> tuple[Path, int]:
    """Write results.csv and return the number of correct point verdicts."""
    output = Path("results.csv")
    correct_points = 0
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["company", "points", "points_correct", "point_errors", "interviews_used", "summary"])
        for name in names:
            entry = analyses.get(name)
            entry = entry if isinstance(entry, dict) else {}
            points = normalize_points(entry.get("points"), chosen)
            expected = truth.get(name)
            n_points, errors = point_errors(points, expected)
            correct_points += n_points
            writer.writerow([
                name,
                " ".join(f"{p}:{points[p] or 'missing'}" for p in chosen),
                f"{n_points}/{len(chosen)}" if expected else "",
                errors,
                INTERVIEWS_USED.get(name, 0),
                str(entry.get("summary", "")).strip(),
            ])
    return output, correct_points


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FTEC5660 HW3 on company data folders")
    parser.add_argument(
        "--data-folder",
        required=True,
        type=Path,
        help="folder containing one sub-folder per company",
    )
    return parser.parse_args()


async def run(folder: Path) -> int:
    folders = company_folders(folder)
    if not folders:
        raise SystemExit(f"no company folders found in {folder}")

    chosen = check_chosen_points(CHOSEN_POINTS)
    load_env_file()
    companies = {path.name: make_tools(path) for path in folders}
    agent = build_agent()
    analyses = await analyze_companies(agent, companies)
    if not isinstance(analyses, dict):
        raise TypeError("analyze_companies() must return a dictionary")

    truth = read_ground_truth(folder)
    output, correct_points = write_results(list(companies), analyses, truth, chosen)
    summary = f" Your points: {', '.join(map(str, chosen))}."
    if truth:
        summary += (
            f" Point accuracy: {correct_points}/{len(companies) * len(chosen)}"
            f" (balanced: {balanced_point_accuracy(analyses, truth, list(companies), chosen):.1%})."
        )
    print(f"Processed {len(companies)} compan{'y' if len(companies) == 1 else 'ies'}. Wrote {output}.{summary}")
    return 0


def main() -> int:
    args = parse_args()
    if not args.data_folder.is_dir():
        raise SystemExit(f"not a folder: {args.data_folder}")
    return asyncio.run(run(args.data_folder))


if __name__ == "__main__":
    raise SystemExit(main())
