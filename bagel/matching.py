"""
Evaluation-time identifier matching. Benchmarks hold their source's identifiers; a matcher decides whether an
identifier BAGEL produced means the same thing as a benchmark identifier.
"""
import httpx

from bagel.cache import Cache, cache_key


class IdentityMatcher:
    """Two identifiers match only if they are identical."""

    def describe(self) -> dict:
        return {"matcher": "identity"}

    def prefetch(self, curies) -> None:
        pass

    def known(self, curie: str) -> bool:
        return True

    def same(self, a: str, b: str) -> bool:
        return a == b


class BabelMatcher:
    """
    Two identifiers match if they are identical or NodeNorm puts them in the same clique, under the given conflation
    settings. The Babel release is recorded, since matches can change between releases.
    """

    def __init__(self, nodenorm_url: str = "https://nodenormalization-sri.renci.org/get_normalized_nodes",
                 conflate: bool = True, drug_chemical_conflate: bool = True, cache: Cache | None = None,
                 timeout: float = 120):
        self.url, self.conflate, self.drug_chemical_conflate = nodenorm_url, conflate, drug_chemical_conflate
        self.client = httpx.Client(timeout=timeout, transport=httpx.HTTPTransport(retries=3))
        status = self.client.get(nodenorm_url.replace("get_normalized_nodes", "status"))
        status.raise_for_status()
        self.babel_version = status.json()["babel_version"]
        self.cache = cache
        self._preferred: dict[str, str | None] = {}

    def describe(self) -> dict:
        return {"matcher": "babel", "babel_version": self.babel_version, "conflate": self.conflate,
                "drug_chemical_conflate": self.drug_chemical_conflate, "nodenorm_url": self.url}

    def _key(self, curie: str) -> str:
        return cache_key("preferred", self.babel_version, self.conflate, self.drug_chemical_conflate, curie)

    def prefetch(self, curies) -> None:
        """Looks up all given identifiers in batches (and through the cache, keyed by Babel release)."""
        todo = []
        for curie in {c for c in curies if c and c not in self._preferred}:
            cached = self.cache.get(self._key(curie)) if self.cache else None
            if cached is not None:
                self._preferred[curie] = cached["id"]
            else:
                todo.append(curie)
        todo.sort()
        for start in range(0, len(todo), 500):
            batch = todo[start:start + 500]
            response = self.client.post(self.url, json={"curies": batch, "conflate": self.conflate,
                                                        "drug_chemical_conflate": self.drug_chemical_conflate})
            response.raise_for_status()
            for curie, node in response.json().items():
                preferred = node["id"]["identifier"] if node else None
                self._preferred[curie] = preferred
                if self.cache:
                    self.cache.set(self._key(curie), {"id": preferred})

    def preferred(self, curie: str) -> str | None:
        if curie not in self._preferred:
            self.prefetch([curie])
        return self._preferred[curie]

    def known(self, curie: str) -> bool:
        return self.preferred(curie) is not None

    def same(self, a: str, b: str) -> bool:
        if a == b:
            return True
        pa = self.preferred(a)
        return pa is not None and pa == self.preferred(b)


def judged_verdict(curie: str, judgments: list[dict], matcher, closed_world: bool) -> str:
    """
    The verdict a benchmark gives an identifier: correct | incorrect | conflicting (it matches judgments with both
    verdicts) | unjudged (matches none; in a closed-world benchmark that is incorrect).
    """
    verdicts = {j["verdict"] for j in judgments if matcher.same(curie, j["curie"])}
    if verdicts == {"correct"}:
        return "correct"
    if verdicts == {"incorrect"}:
        return "incorrect"
    if verdicts:
        return "conflicting"
    return "incorrect" if closed_world else "unjudged"
