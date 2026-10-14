"""Validate a generated set before export."""
from __future__ import annotations

from dataclasses import dataclass, field

from eu4unitgen.core.generator import GeneratedSet
from eu4unitgen.core.nation import Nation
from eu4unitgen.core.project import Project
from eu4unitgen.library.scanner import GameFiles, Index
from eu4unitgen.validate.graph import Graph, World, build_graph
from eu4unitgen.validate.rules import ERROR, INFO, RULES, WARNING, Context, Issue


@dataclass
class Report:
    issues: list[Issue] = field(default_factory=list)
    graph: Graph | None = None

    def by(self, severity: str) -> list[Issue]:
        return [i for i in self.issues if i.severity == severity]

    @property
    def errors(self) -> list[Issue]:
        return self.by(ERROR)

    @property
    def warnings(self) -> list[Issue]:
        return self.by(WARNING)

    @property
    def ok(self) -> bool:
        return not self.errors

    def summary(self) -> str:
        return f"{len(self.errors)} errors, {len(self.warnings)} warnings, {len(self.by(INFO))} notes"


def validate(project: Project, nation: Nation, generated: GeneratedSet, game: GameFiles, game_idx: Index,
             mod: GameFiles | None = None, rules: list[str] | None = None) -> Report:
    world = World(game, game_idx, mod, generated, project.owned_dlc)
    graph = build_graph(world, generated)
    ctx = Context(project, nation, generated, world, graph)
    report = Report(graph=graph)
    for name in rules or list(RULES):
        report.issues += RULES[name](ctx)
    order = {ERROR: 0, WARNING: 1, INFO: 2}
    report.issues.sort(key=lambda i: order[i.severity])
    return report
