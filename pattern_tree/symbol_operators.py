import re
import numpy as np
from typing import List, Tuple, Optional, Union
from collections import Counter
from itertools import combinations


class SymbolOperator:
    """
    Treats regex patterns as mathematical operators on strings.
    Enables inner/outer products and pattern algebra.
    """

    def __init__(self, pattern: Union[str, re.Pattern], name: str = None):
        if isinstance(pattern, str):
            self.pattern = re.compile(pattern)
        else:
            self.pattern = pattern
        self.name = name or pattern.pattern

    def inner_product(self, text: str) -> float:
        """
        Compute similarity metric between pattern and text.
        Returns normalized match density.
        """
        if not text:
            return 0.0

        matches = list(self.pattern.finditer(text))
        if not matches:
            return 0.0

        # Coverage: how much of the text is covered by matches
        covered_chars = sum(m.end() - m.start() for m in matches)
        coverage = covered_chars / len(text)

        # Frequency: how many matches relative to text length
        frequency = len(matches) / (len(text) / 10)  # normalized per 10 chars

        # Regularity: how evenly distributed are the matches
        if len(matches) > 1:
            positions = [m.start() for m in matches]
            intervals = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
            mean_interval = np.mean(intervals)
            std_interval = np.std(intervals)
            regularity = 1 / (1 + std_interval / mean_interval) if mean_interval > 0 else 0
        else:
            regularity = 1.0

        return coverage * 0.4 + min(frequency, 1.0) * 0.3 + regularity * 0.3

    def outer_product(self, text: str) -> Tuple[List[str], str]:
        """
        Extract pattern matches and transform text into symbol basis.
        Returns (extracted_symbols, transformed_text).
        """
        matches = list(self.pattern.finditer(text))
        if not matches:
            return [], text

        extracted = [text[m.start():m.end()] for m in matches]

        # Replace matches with placeholders
        transformed = text
        for i, match in enumerate(reversed(matches)):
            placeholder = f"<{self.name}_{len(matches)-1-i}>"
            transformed = transformed[:match.start()] + placeholder + transformed[match.end():]

        return extracted, transformed

    def compose(self, other: 'SymbolOperator') -> 'SymbolOperator':
        """
        Create a composite pattern that matches either pattern.
        """
        combined_pattern = f"(?:{self.pattern.pattern})|(?:{other.pattern.pattern})"
        return SymbolOperator(combined_pattern, f"{self.name}+{other.name}")

    def difference(self, other: 'SymbolOperator', text: str) -> List[str]:
        """
        Find matches of self that don't overlap with other's matches.
        """
        self_matches = list(self.pattern.finditer(text))
        other_matches = list(other.pattern.finditer(text))

        non_overlapping = []
        for sm in self_matches:
            overlaps = False
            for om in other_matches:
                if not (sm.end() <= om.start() or sm.start() >= om.end()):
                    overlaps = True
                    break
            if not overlaps:
                non_overlapping.append(text[sm.start():sm.end()])

        return non_overlapping


class PatternLearner:
    """
    Learns regex patterns from data samples through generalization.
    """

    @staticmethod
    def learn_from_sample(text: str) -> str:
        """
        Generate initial regex pattern from a single text sample.
        Tokenizes and creates character class patterns.
        """
        # Identify structural components
        tokens = []
        i = 0
        while i < len(text):
            if text[i].isdigit():
                # Digit sequence
                j = i
                while j < len(text) and text[j].isdigit():
                    j += 1
                tokens.append(r'\d+')
                i = j
            elif text[i].isalpha():
                # Letter sequence
                j = i
                while j < len(text) and text[j].isalpha():
                    j += 1
                if text[i:j].isupper():
                    tokens.append(r'[A-Z]+')
                elif text[i:j].islower():
                    tokens.append(r'[a-z]+')
                else:
                    tokens.append(r'[A-Za-z]+')
                i = j
            elif text[i] in '.-_/:':
                # Common separators - keep literal
                tokens.append(re.escape(text[i]))
                i += 1
            elif text[i].isspace():
                # Whitespace
                j = i
                while j < len(text) and text[j].isspace():
                    j += 1
                tokens.append(r'\s+')
                i = j
            else:
                # Special character - escape it
                tokens.append(re.escape(text[i]))
                i += 1

        return ''.join(tokens)

    @staticmethod
    def generalize_patterns(patterns: List[str]) -> str:
        """
        Generalize multiple regex patterns into one.
        Finds commonalities and creates flexible pattern.
        """
        if not patterns:
            return ""
        if len(patterns) == 1:
            return patterns[0]

        # Parse patterns to find common structure
        parsed = []
        for p in patterns:
            # Split pattern into tokens - simplified tokenization
            tokens = []
            i = 0
            while i < len(p):
                if i < len(p) - 1 and p[i] == '\\':
                    # Escaped character or pattern
                    if p[i+1] in 'dws':
                        # Look for quantifier
                        if i + 2 < len(p) and p[i+2] in '+*?':
                            tokens.append(p[i:i+3])
                            i += 3
                        else:
                            tokens.append(p[i:i+2])
                            i += 2
                    else:
                        tokens.append(p[i:i+2])
                        i += 2
                elif p[i] == '[':
                    # Character class
                    j = i + 1
                    while j < len(p) and p[j] != ']':
                        j += 1
                    if j < len(p):
                        j += 1  # Include the ]
                        if j < len(p) and p[j] in '+*?':
                            j += 1  # Include quantifier
                        tokens.append(p[i:j])
                        i = j
                    else:
                        tokens.append(p[i])
                        i += 1
                else:
                    tokens.append(p[i])
                    i += 1
            parsed.append(tokens)

        # Find longest common subsequence structure
        result = []
        min_len = min(len(p) for p in parsed)

        for i in range(min_len):
            tokens_at_i = [p[i] for p in parsed]

            if len(set(tokens_at_i)) == 1:
                # All same - keep as is
                result.append(tokens_at_i[0])
            elif all(t in [r'\d+', r'\d*', r'\d'] for t in tokens_at_i):
                # All digits with different quantifiers
                result.append(r'\d+')
            elif all(t in [r'[A-Za-z]+', r'[A-Z]+', r'[a-z]+', r'\w+'] for t in tokens_at_i):
                # All word-like
                result.append(r'\w+')
            elif all(t.startswith('[') and t.endswith(']') for t in tokens_at_i if t):
                # All character classes - keep first one for simplicity
                result.append(tokens_at_i[0])
            else:
                # Make optional group with alternatives
                unique = list(set(tokens_at_i))
                if len(unique) <= 3:
                    result.append(f"(?:{'|'.join(unique)})")
                else:
                    result.append('.*?')  # Too varied - use non-greedy any

        return ''.join(result)

    @staticmethod
    def learn_structure_from_samples(samples: List[str]) -> List[Tuple[str, str]]:
        """
        Learn multiple pattern types from a dataset.
        Returns list of (pattern_name, regex) tuples.
        """
        patterns = []

        # Learn individual patterns
        sample_patterns = [PatternLearner.learn_from_sample(s) for s in samples]

        # Cluster similar patterns
        clusters = PatternLearner._cluster_patterns(sample_patterns)

        for i, cluster in enumerate(clusters):
            if len(cluster) > 1:
                # Generalize cluster
                general_pattern = PatternLearner.generalize_patterns(cluster)
                patterns.append((f"PATTERN_{i}", general_pattern))
            else:
                patterns.append((f"UNIQUE_{i}", cluster[0]))

        return patterns

    @staticmethod
    def _cluster_patterns(patterns: List[str]) -> List[List[str]]:
        """
        Group similar patterns together for generalization.
        """
        if not patterns:
            return []

        # Simple clustering based on pattern similarity
        clusters = []
        used = set()

        for i, p1 in enumerate(patterns):
            if i in used:
                continue

            cluster = [p1]
            used.add(i)

            for j, p2 in enumerate(patterns[i+1:], i+1):
                if j in used:
                    continue

                # Check similarity (simple heuristic)
                if PatternLearner._patterns_similar(p1, p2):
                    cluster.append(p2)
                    used.add(j)

            clusters.append(cluster)

        return clusters

    @staticmethod
    def _patterns_similar(p1: str, p2: str) -> bool:
        """
        Check if two patterns are similar enough to cluster.
        """
        # Extract pattern components
        tokens1 = set(re.findall(r'\\[dws]|\\[A-Z]|\\[[^\]]+\]', p1))
        tokens2 = set(re.findall(r'\\[dws]|\\[A-Z]|\\[[^\]]+\]', p2))

        # Check overlap
        if not tokens1 or not tokens2:
            return False

        overlap = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)

        return overlap / union > 0.5 if union > 0 else False


class AdaptiveSymbolGenerator:
    """
    Evolves regex patterns based on data feedback.
    """

    def __init__(self, initial_pattern: str, name: str):
        self.pattern = re.compile(initial_pattern)
        self.name = name
        self.match_history = []
        self.false_positives = []
        self.false_negatives = []

    def test_and_adapt(self, text: str, expected_matches: List[str] = None):
        """
        Test pattern on text and adapt based on expected matches.
        """
        actual_matches = [m.group() for m in self.pattern.finditer(text)]

        if expected_matches is not None:
            # Calculate errors
            actual_set = set(actual_matches)
            expected_set = set(expected_matches)

            fps = actual_set - expected_set
            fns = expected_set - actual_set

            if fps:
                self.false_positives.extend(fps)
            if fns:
                self.false_negatives.extend(fns)

            # Adapt pattern if errors accumulate
            if len(self.false_positives) + len(self.false_negatives) > 5:
                self._adapt_pattern()

        self.match_history.extend(actual_matches)
        return actual_matches

    def _adapt_pattern(self):
        """
        Modify pattern based on error history.
        """
        # Analyze false positives to tighten pattern
        if self.false_positives:
            # Find common elements to exclude
            fp_patterns = [PatternLearner.learn_from_sample(fp) for fp in self.false_positives]
            # Could implement negative lookahead based on these

        # Analyze false negatives to loosen pattern
        if self.false_negatives:
            fn_patterns = [PatternLearner.learn_from_sample(fn) for fn in self.false_negatives]
            generalized = PatternLearner.generalize_patterns([self.pattern.pattern] + fn_patterns)
            try:
                self.pattern = re.compile(generalized)
            except re.error:
                pass  # Keep original if generalization fails

        # Clear history after adaptation
        self.false_positives = []
        self.false_negatives = []