"""
SEE — Skill Evolution Engine: Orchestrator with PUCT loop and evidence matrix.

Integrates 4 prototype components:
  - skill_evolution_sandbox.py    (safe test execution)
  - skill_evolution_routing.py    (_delegate_role with per-role model routing)
  - skill_evolution_test_cache.py  (test cache across evolutions)
  - skill_evolution_budget.py     (budget tracking with max_cost_usd)

Algorithm: PUCT (Predictor + Upper Confidence bounds applied to Trees)
  1. Selection:    traverse the tree from root choosing the child with max UCB1
  2. Expansion:    generate K candidates via delegate_patch (coefficient α ~ PUCT)
  3. Simulation:   execute the patched skill and evaluate tests → fills M[v, test_id]
  4. Backprop:     propagate the value (evidence score) from the leaf up to the root

Evidence matrix M[v, test_id] ∈ {0, 1, None}:
  - 1    = test passed (the patched skill satisfies the hypothesis)
  - 0    = test failed (the patched skill does not satisfy the hypothesis)
  - None = test not executed for this node

The best node (max evidence score) is chosen at the end as the winning patch.

Entry point: evolve_skill(skill_name, skill_content, task_context, output, parent_agent)
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from agent.skill_evolution_budget import BudgetTracker, BudgetExceededError
from agent.skill_evolution_sandbox import run_test_sandboxed, validate_test_strict
from agent.skill_evolution_test_cache import TestCase, TestCache

logger = logging.getLogger("hermes.evolution")

# ── Constants ────────────────────────────────────────────────────────
DEFAULT_BUDGET = 5            # max PUCT iterations
DEFAULT_K = 3                  # candidates per expansion
UCB_C = 1.4142                 # sqrt(2) — exploration constant
TTL_DAYS = 30


# ══════════════════════════════════════════════════════════════════════
# 1. Data structures
# ══════════════════════════════════════════════════════════════════════

@dataclass
class Hypothesis:
    """Falsifiable hypothesis about a skill defect."""
    id: str                      # "H1", "H2", ...
    description: str             # "The skill does not validate exit code"
    observable_behavior: str     # "exit_code must be 0"
    action: str = "add"          # "add" | "refine" | "refute"
    rationale: str = ""
    status: str = "pending"      # "pending" | "confirmed" | "refuted"


@dataclass
class PatchCandidate:
    """A proposed patch to the SKILL.md with ordinal ranking."""
    rank: int                    # 1 = best
    old_string: str
    new_string: str
    rationale: str = ""
    expected_improvement: str = ""


@dataclass
class EvidenceEntry:
    """An entry in the evidence matrix: result of a test at a node."""
    test_id: str
    passed: bool
    reason: str
    category: str


@dataclass
class PUCTNode:
    """PUCT search tree node.

    Each node represents a skill version (root = original skill,
    children = skill patched with each PatchCandidate).
    """
    node_id: str
    skill_content: str
    skill_hash: str
    parent: Optional["PUCTNode"] = None
    children: list["PUCTNode"] = field(default_factory=list)
    patches_used: list[PatchCandidate] = field(default_factory=list)

    # Evidence: M[v, test_id] = {"test_id": EvidenceEntry}
    evidence: dict[str, EvidenceEntry] = field(default_factory=dict)

    # PUCT stats
    visits: int = 0
    total_value: float = 0.0     # sum of evidence scores

    # metadata
    depth: int = 0
    created_at: float = field(default_factory=time.time)

    @property
    def avg_value(self) -> float:
        return self.total_value / self.visits if self.visits > 0 else 0.0

    @property
    def evidence_score(self) -> float:
        """Score = passed tests / total tests executed (excluding None)."""
        if not self.evidence:
            return 0.0
        passed = sum(1 for r in self.evidence.values() if r.passed)
        total = len(self.evidence)
        return passed / total

    def ucb1(self, exploration: float = UCB_C) -> float:
        """UCB1 = avg_value + C * sqrt(ln(N_parent) / n_self)."""
        if self.visits == 0:
            return float("inf")  # unvisited nodes are explored first
        if self.parent is None or self.parent.visits == 0:
            return self.avg_value
        exploit = self.avg_value
        explore = exploration * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploit + explore


def _hash_skill(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


# ══════════════════════════════════════════════════════════════════════
# 2. Evidence matrix
# ══════════════════════════════════════════════════════════════════════

class EvidenceMatrix:
    """Matrix M[node_id, test_id] → EvidenceEntry.

    Allows querying: which tests did each skill version pass?
    A hypothesis is confirmed if all its tests pass in at least one node.
    """

    def __init__(self):
        self._matrix: dict[str, dict[str, EvidenceEntry]] = {}

    def record(self, node: PUCTNode, test_id: str, result: EvidenceEntry) -> None:
        if node.node_id not in self._matrix:
            self._matrix[node.node_id] = {}
        self._matrix[node.node_id][test_id] = result
        node.evidence[test_id] = result

    def best_node_id(self, all_nodes: list[PUCTNode]) -> Optional[str]:
        """Returns the node_id with the highest evidence_score.
        Only considers nodes that have at least one test executed.
        Prefers higher score, then shallower depth (simpler patches).
        """
        scored = [(n.node_id, n.evidence_score, n.depth)
                  for n in all_nodes if n.evidence]
        if not scored:
            return None
        # Prefer higher score, then shallower depth (simpler patches)
        scored.sort(key=lambda x: (-x[1], x[2]))
        return scored[0][0]

    def confirmed_hypotheses(self, node: PUCTNode, hypotheses: list[Hypothesis]) -> list[str]:
        """Returns IDs of confirmed hypotheses in this node (all their tests pass)."""
        confirmed = []
        for h in hypotheses:
            # Tests linked to this hypothesis in this node
            h_tests = [
                r for tid, r in node.evidence.items()
                if tid.startswith(h.id + "_")
            ]
            if h_tests and all(r.passed for r in h_tests):
                confirmed.append(h.id)
        return confirmed

    def to_dict(self) -> dict:
        return {
            nid: {tid: {"passed": r.passed, "reason": r.reason, "category": r.category}
                  for tid, r in tests.items()}
            for nid, tests in self._matrix.items()
        }


# ══════════════════════════════════════════════════════════════════════
# 3. Bucle PUCT
# ══════════════════════════════════════════════════════════════════════

class PUCTSearch:
    """PUCT search over the skill's patch tree."""

    def __init__(
        self,
        skill_name: str,
        skill_content: str,
        task_context: str,
        output_stdout: str,
        failure_signal: str,
        parent_agent,
        budget_iterations: int = DEFAULT_BUDGET,
        max_children: int = DEFAULT_K,
        budget_tracker: Optional[BudgetTracker] = None,
        test_cache: Optional[TestCache] = None,
        # Injectables for testing (mock delegates)
        delegate_hypothesis_fn=None,
        delegate_test_fn=None,
        delegate_patch_fn=None,
        delegate_execute_fn=None,
    ):
        self.skill_name = skill_name
        self.skill_content = skill_content
        self.skill_hash = _hash_skill(skill_content)
        self.task_context = task_context
        self.output_stdout = output_stdout
        self.failure_signal = failure_signal
        self.parent_agent = parent_agent
        self.budget_iterations = budget_iterations
        self.max_children = max_children
        self.budget_tracker = budget_tracker
        self.test_cache = test_cache

        # Delegates (injectable for testing)
        self._delegate_hypothesis = delegate_hypothesis_fn or self._default_hypothesis
        self._delegate_test = delegate_test_fn or self._default_test
        self._delegate_patch = delegate_patch_fn or self._default_patch
        self._delegate_execute = delegate_execute_fn or self._default_execute

        # State
        self.root = PUCTNode(
            node_id="root",
            skill_content=skill_content,
            skill_hash=self.skill_hash,
        )
        self.all_nodes: list[PUCTNode] = [self.root]
        self.matrix = EvidenceMatrix()
        self.hypotheses: list[Hypothesis] = []

    # ── Default delegates (use skill_evolution_routing) ──────────
    def _default_hypothesis(self, skill_content, task_context, failure_signal,
                             output_stdout, existing_hypotheses, parent_agent,
                             focus=None, round_idx=1, total_rounds=3):
        from agent.skill_evolution_routing import delegate_hypothesis
        return delegate_hypothesis(
            skill_content, task_context, failure_signal, output_stdout,
            existing_hypotheses, parent_agent, focus, round_idx, total_rounds,
            budget_tracker=self.budget_tracker,
        )

    def _default_test(self, hypothesis_description, observable_behavior,
                       output_stdout, file_list, parent_agent):
        from agent.skill_evolution_routing import delegate_test
        return delegate_test(
            hypothesis_description, observable_behavior,
            output_stdout, file_list, parent_agent,
            budget_tracker=self.budget_tracker,
        )

    def _default_patch(self, skill_content, evidence_summary,
                        task_context, max_children, parent_agent):
        from agent.skill_evolution_routing import delegate_patch
        return delegate_patch(
            skill_content, evidence_summary, task_context,
            max_children, parent_agent,
            budget_tracker=self.budget_tracker,
        )

    def _default_execute(self, skill_content, task_context, parent_agent):
        from agent.skill_evolution_routing import delegate_execute
        return delegate_execute(
            skill_content, task_context, parent_agent,
            budget_tracker=self.budget_tracker,
        )

    # ── Fase 1: Hypothesis generation ─────────────────────────────────
    def _generate_hypotheses(self, round_idx: int) -> list[Hypothesis]:
        """Generate 3-5 falsifiable hypotheses about the skill defect."""
        existing = [
            {"id": h.id, "description": h.description, "status": h.status}
            for h in self.hypotheses
        ]
        raw = self._delegate_hypothesis(
            skill_content=self.skill_content,
            task_context=self.task_context,
            failure_signal=self.failure_signal,
            output_stdout=self.output_stdout,
            existing_hypotheses=existing,
            parent_agent=self.parent_agent,
            round_idx=round_idx,
            total_rounds=self.budget_iterations,
        )
        if not isinstance(raw, list):
            return []
        result = []
        for i, item in enumerate(raw):
            if not isinstance(item, dict):
                continue
            result.append(Hypothesis(
                id=item.get("id", f"H{len(self.hypotheses) + i + 1}"),
                description=item.get("description", ""),
                observable_behavior=item.get("observable_behavior", ""),
                action=item.get("action", "add"),
                rationale=item.get("rationale", ""),
            ))
        return result

    # ── Phase 2: Test generation (with cache) ───────────────────────────
    def _generate_test_for_hypothesis(self, h: Hypothesis) -> Optional[str]:
        """Generate (or recover from cache) an executable test for the hypothesis."""

        # 1. Cache lookup
        if self.test_cache:
            try:
                cached = self.test_cache.get_cached_test(
                    hypothesis_desc=h.description,
                    skill_hash=self.skill_hash,
                    observable_behavior=h.observable_behavior,
                )
                if cached is not None:
                    logger.info("[evolve] cache HIT for %s", h.id)
                    return cached.code
            except Exception:
                pass  # best-effort

        # 2. Generate via expensive LLM model
        logger.info("[evolve] cache MISS for %s — generating test", h.id)
        code = self._delegate_test(
            hypothesis_description=h.description,
            observable_behavior=h.observable_behavior,
            output_stdout=self.output_stdout,
            file_list=[],
            parent_agent=self.parent_agent,
        )
        if not code:
            return None

        # 3. Validate before accepting
        ok, reason = validate_test_strict(code)
        if not ok:
            logger.warning("[evolve] test rejected for %s: %s", h.id, reason)
            return None

        # 4. Store in cache
        if self.test_cache:
            try:
                test_case = TestCase(
                    test_id=f"{h.id}_{_hash_skill(code)[:8]}",
                    hypothesis_id=h.id,
                    hypothesis_description=h.description,
                    observable_behavior=h.observable_behavior,
                    code=code,
                    category="hard",
                    validated=True,
                )
                self.test_cache.store_test(test_case, skill_hash=self.skill_hash)
            except Exception:
                pass  # best-effort

        return code

    # ── Fase 3: Selection (UCB1 traversal) ────────────────────────────
    def _select(self) -> PUCTNode:
        """Traverse the tree from root choosing max UCB1 until reaching a leaf."""
        node = self.root
        while node.children:
            node = max(node.children, key=lambda c: c.ucb1())
        return node

    # ── Phase 4: Expansion (generate K patches) ──────────────────────────
    def _expand(self, node: PUCTNode) -> list[PUCTNode]:
        """Generate K candidate patches and create K child nodes."""
        evidence_summary = json.dumps({
            "tests": {tid: {"passed": r.passed, "reason": r.reason}
                      for tid, r in node.evidence.items()},
            "hypotheses": [{"id": h.id, "description": h.description,
                            "status": h.status} for h in self.hypotheses],
        }, indent=2)

        raw_patches = self._delegate_patch(
            skill_content=node.skill_content,
            evidence_summary=evidence_summary,
            task_context=self.task_context,
            max_children=self.max_children,
            parent_agent=self.parent_agent,
        )

        if not isinstance(raw_patches, list) or not raw_patches:
            return []

        children = []
        for i, p in enumerate(raw_patches[: self.max_children]):
            if not isinstance(p, dict):
                continue
            old = p.get("old_string", "")
            new = p.get("new_string", "")
            if not old:
                continue
            # Apply patch
            patched = node.skill_content.replace(old, new, 1)
            if patched == node.skill_content:
                continue  # patch not applicable
            child = PUCTNode(
                node_id=f"n{len(self.all_nodes)}",
                skill_content=patched,
                skill_hash=_hash_skill(patched),
                parent=node,
                depth=node.depth + 1,
                patches_used=node.patches_used + [PatchCandidate(
                    rank=i + 1,
                    old_string=old,
                    new_string=new,
                    rationale=p.get("rationale", ""),
                    expected_improvement=p.get("expected_improvement", ""),
                )],
            )
            node.children.append(child)
            self.all_nodes.append(child)
            children.append(child)

        return children

    # ── Phase 5: Simulation (execute skill + tests) ────────────────────
    def _simulate(self, node: PUCTNode) -> float:
        """Execute the patched skill, run tests, return evidence score."""
        # 1. Execute the patched skill against the task
        output = self._delegate_execute(
            skill_content=node.skill_content,
            task_context=self.task_context,
            parent_agent=self.parent_agent,
        )

        # 2. Run each test against the output
        for h in self.hypotheses:
            test_code = h._test_code if hasattr(h, "_test_code") else None
            if test_code is None:
                test_code = self._generate_test_for_hypothesis(h)
                if test_code is None:
                    continue
                h._test_code = test_code  # in-memory cache per hypothesis

            test_id = f"{h.id}_test"
            result = run_test_sandboxed(test_code, output or "", {}, timeout_s=5.0)
            tr = EvidenceEntry(
                test_id=test_id,
                passed=result.passed,
                reason=result.reason,
                category=result.category,
            )
            self.matrix.record(node, test_id, tr)

            # Update hypothesis status
            if tr.passed:
                h.status = "confirmed"
            elif h.status != "confirmed":
                h.status = "refuted"

        # 3. Track budget
        if self.budget_tracker:
            # Real tokens would come from the child agent; here we use 0 as placeholder
            # Real tracking happens in _delegate_role
            pass

        return node.evidence_score

    # ── Phase 6: Backpropagation ───────────────────────────────────────
    def _backprop(self, node: PUCTNode, value: float) -> None:
        """Propagate the value from the leaf up to the root."""
        while node is not None:
            node.visits += 1
            node.total_value += value
            node = node.parent

    # ── Main PUCT loop ────────────────────────────────────────────────
    def run(self) -> dict[str, Any]:
        """Run the full PUCT loop. Returns the evolution result."""
        t0 = time.time()
        budget_exceeded = False

        # 1. Generate initial hypotheses
        try:
            self.hypotheses = self._generate_hypotheses(round_idx=1)
            logger.info("[evolve] generated %d hypotheses", len(self.hypotheses))
        except BudgetExceededError as e:
            return self._result(budget_exceeded=True, error=str(e))

        if not self.hypotheses:
            return self._result(error="no hypotheses generated")

        # 2. PUCT loop
        for iteration in range(self.budget_iterations):
            logger.info("[evolve] PUCT iteration %d/%d", iteration + 1, self.budget_iterations)

            try:
                if self.budget_tracker:
                    self.budget_tracker.check_budget("test", "?")
            except BudgetExceededError as e:
                logger.warning("[evolve] budget exceeded: %s", e)
                budget_exceeded = True
                break

            node = self._select()
            children = self._expand(node)

            if not children:
                # Could not generate children — simulate the current node
                value = self._simulate(node)
                self._backprop(node, value)
                continue

            # Simulate the best child (rank 1) or all if budget allows
            for child in children:
                try:
                    value = self._simulate(child)
                    self._backprop(child, value)
                except BudgetExceededError as e:
                    budget_exceeded = True
                    logger.warning("[evolve] budget exceeded during simulation: %s", e)
                    break

            if budget_exceeded:
                break

            # Refine hypotheses: if all confirmed, generate new ones with focus
            confirmed = self.matrix.confirmed_hypotheses(node, self.hypotheses)
            if len(confirmed) == len(self.hypotheses) and iteration < self.budget_iterations - 1:
                new_hyps = self._generate_hypotheses(round_idx=iteration + 2)
                self.hypotheses.extend(new_hyps)

        # 3. Select best node
        best_id = self.matrix.best_node_id(self.all_nodes)
        best_node = next((n for n in self.all_nodes if n.node_id == best_id), None)

        return self._result(best_node=best_node, budget_exceeded=budget_exceeded)

    def _result(
        self,
        best_node: Optional[PUCTNode] = None,
        budget_exceeded: bool = False,
        error: Optional[str] = None,
    ) -> dict[str, Any]:
        """Build the final evolution result."""
        result = {
            "skill_name": self.skill_name,
            "iterations": self.root.visits,
            "nodes_explored": len(self.all_nodes),
            "hypotheses": [{"id": h.id, "description": h.description, "status": h.status}
                           for h in self.hypotheses],
            "evidence_matrix": self.matrix.to_dict(),
            "budget_exceeded": budget_exceeded,
            "elapsed_s": time.time() - getattr(self, "_t0", time.time()),
        }

        if best_node and best_node != self.root:
            result["best_patch"] = {
                "node_id": best_node.node_id,
                "evidence_score": best_node.evidence_score,
                "depth": best_node.depth,
                "patches": [
                    {"rank": p.rank, "old_string": p.old_string[:200],
                     "new_string": p.new_string[:200]}
                    for p in best_node.patches_used
                ],
                "patched_skill_preview": best_node.skill_content[:500],
            }
        else:
            result["best_patch"] = None

        if error:
            result["error"] = error

        if self.budget_tracker:
            result["cost"] = self.budget_tracker.summary()

        return result


# ══════════════════════════════════════════════════════════════════════
# 4. Public entry point
# ══════════════════════════════════════════════════════════════════════

def evolve_skill(
    skill_name: str,
    skill_content: str,
    task_context: str,
    output_stdout: str,
    parent_agent,
    failure_signal: str = "",
    config: Optional[dict] = None,
    category: str = "uncategorized",
    # For testing
    mock_delegates: Optional[dict] = None,
) -> dict[str, Any]:
    """Orchestrate a full skill evolution.

    Args:
        skill_name: name of the skill to evolve.
        skill_content: current SKILL.md content.
        task_context: context of the task that failed.
        output_stdout: output produced by the agent (that failed).
        parent_agent: parent AIAgent (for delegates).
        failure_signal: description of the failure signal.
        config: evolution config (max_cost_usd, models, etc.).
        category: skill category (for cache).
        mock_delegates: dict with injectable delegates for testing.

    Returns:
        dict with: iterations, nodes_explored, hypotheses, evidence_matrix,
        best_patch, budget_exceeded, cost (if tracker).
    """
    config = config or {}
    evolution_cfg = config.get("evolution", {})

    # 1. Budget tracker
    budget_tracker = BudgetTracker.from_config(evolution_cfg)

    # 2. Test cache
    cache_cfg = evolution_cfg.get("cache", {})
    test_cache = TestCache(category, skill_name)
    # Prune at start
    test_cache.prune_expired_tests(ttl_days=cache_cfg.get("test_ttl_days", TTL_DAYS))

    # 3. PUCT search
    delegates = mock_delegates or {}
    search = PUCTSearch(
        skill_name=skill_name,
        skill_content=skill_content,
        task_context=task_context,
        output_stdout=output_stdout,
        failure_signal=failure_signal,
        parent_agent=parent_agent,
        budget_iterations=evolution_cfg.get("budget", DEFAULT_BUDGET),
        max_children=evolution_cfg.get("max_children", DEFAULT_K),
        budget_tracker=budget_tracker,
        test_cache=test_cache,
        delegate_hypothesis_fn=delegates.get("hypothesis"),
        delegate_test_fn=delegates.get("test"),
        delegate_patch_fn=delegates.get("patch"),
        delegate_execute_fn=delegates.get("execute"),
    )

    search._t0 = time.time()
    return search.run()
