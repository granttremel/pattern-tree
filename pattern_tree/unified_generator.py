"""
Unified Symbol system where symbols are both patterns and operators.
This integrates the SymbolGenerator with the mathematical operator approach.
"""

import re
from typing import Union, List, Tuple, Optional, Callable
from .symbol import Symbol
from .symbol_operators import SymbolOperator, PatternLearner


class UnifiedSymbol(SymbolOperator):
    """
    A Symbol that acts as both a pattern matcher and a mathematical operator.
    Can generate child symbols, perform transformations, and evolve patterns.
    """

    def __init__(self,
                 pattern: Union[str, re.Pattern],
                 name: str,
                 formatter: Callable = None,
                 at_branch: bool = False,
                 divisible: bool = False,
                 priority: int = 0):
        super().__init__(pattern, name)
        self.formatter = formatter or (lambda x: f"<{self.name}:{x}>")
        self.at_branch = at_branch
        self.divisible = divisible
        self.priority = priority  # Higher priority patterns extract first

    def detect_and_extract(self, text: str) -> List[Symbol]:
        """
        Find patterns in text and create Symbol objects.
        """
        matches = list(self.pattern.finditer(text))
        symbols = []
        for match in matches:
            sym = Symbol(
                match.group(),
                self.name,
                has_data=True,
                divisible=self.divisible
            )
            # Store position for reconstruction
            sym.start_pos = match.start()
            sym.end_pos = match.end()
            symbols.append(sym)
        return symbols

    def transform(self, text: str) -> Tuple[List[Symbol], str]:
        """
        Extract symbols and transform text simultaneously.
        """
        symbols = self.detect_and_extract(text)
        if not symbols:
            return [], text

        # Sort by position for proper replacement
        symbols.sort(key=lambda s: s.start_pos)

        # Replace from end to start to maintain indices
        transformed = text
        for i, sym in enumerate(reversed(symbols)):
            placeholder = self.formatter(len(symbols) - 1 - i)
            transformed = (
                transformed[:sym.start_pos] +
                placeholder +
                transformed[sym.end_pos:]
            )

        return symbols, transformed

    def learn_from_examples(self, examples: List[str]) -> 'UnifiedSymbol':
        """
        Create a new evolved symbol by learning from examples.
        """
        patterns = [PatternLearner.learn_from_sample(ex) for ex in examples]
        generalized = PatternLearner.generalize_patterns(patterns)

        return UnifiedSymbol(
            pattern=generalized,
            name=f"{self.name}_evolved",
            formatter=self.formatter,
            at_branch=self.at_branch,
            divisible=self.divisible,
            priority=self.priority
        )

    def mutate(self, mutation_rate: float = 0.1) -> 'UnifiedSymbol':
        """
        Create a mutated version of this symbol's pattern.
        Useful for evolutionary approaches.
        """
        import random

        pattern_str = self.pattern.pattern
        mutations = []

        for char in pattern_str:
            if random.random() < mutation_rate:
                if char == '+':
                    mutations.append('*')  # Make greedy
                elif char == '*':
                    mutations.append('+')  # Make non-greedy
                elif char == '\\d':
                    mutations.append('\\w')  # Broaden to word chars
                elif char == '\\w':
                    mutations.append('\\d')  # Narrow to digits
                else:
                    mutations.append(char)
            else:
                mutations.append(char)

        mutated_pattern = ''.join(mutations)

        try:
            return UnifiedSymbol(
                pattern=mutated_pattern,
                name=f"{self.name}_mut",
                formatter=self.formatter,
                at_branch=self.at_branch,
                divisible=self.divisible,
                priority=self.priority
            )
        except re.error:
            # If mutation creates invalid regex, return original
            return self


class HierarchicalSymbolExtractor:
    """
    Manages a hierarchy of UnifiedSymbols for progressive text analysis.
    """

    def __init__(self):
        self.symbols = []
        self.extraction_history = []

    def add_symbol(self, symbol: UnifiedSymbol):
        """Add a symbol to the hierarchy."""
        self.symbols.append(symbol)
        # Sort by priority
        self.symbols.sort(key=lambda s: s.priority, reverse=True)

    def extract_hierarchically(self, text: str) -> dict:
        """
        Extract symbols level by level, building a parse tree.
        """
        remaining = text
        extractions = {}
        tree_levels = []

        for symbol in self.symbols:
            extracted, transformed = symbol.transform(remaining)
            if extracted:
                extractions[symbol.name] = extracted
                tree_levels.append({
                    'symbol': symbol.name,
                    'extracted': [str(e) for e in extracted],
                    'transformed': transformed
                })
                remaining = transformed

        result = {
            'original': text,
            'final': remaining,
            'extractions': extractions,
            'tree': tree_levels
        }

        self.extraction_history.append(result)
        return result

    def learn_new_patterns(self, texts: List[str]) -> List[UnifiedSymbol]:
        """
        Discover new patterns from unmatched text portions.
        """
        unmatched_portions = []

        for text in texts:
            result = self.extract_hierarchically(text)
            # Collect non-placeholder portions
            final = result['final']
            # Simple heuristic: find text between placeholders
            parts = re.split(r'<[^>]+>', final)
            unmatched_portions.extend([p.strip() for p in parts if p.strip()])

        if not unmatched_portions:
            return []

        # Learn patterns from unmatched portions
        learned_patterns = PatternLearner.learn_structure_from_samples(unmatched_portions)

        new_symbols = []
        for name, pattern in learned_patterns:
            try:
                sym = UnifiedSymbol(
                    pattern=pattern,
                    name=name,
                    priority=-1  # Lower priority for learned patterns
                )
                new_symbols.append(sym)
            except re.error:
                continue

        return new_symbols


class SymbolAlgebra:
    """
    Implements algebraic operations on symbols.
    """

    @staticmethod
    def union(sym1: UnifiedSymbol, sym2: UnifiedSymbol) -> UnifiedSymbol:
        """Create a symbol that matches either pattern."""
        pattern = f"(?:{sym1.pattern.pattern})|(?:{sym2.pattern.pattern})"
        return UnifiedSymbol(
            pattern=pattern,
            name=f"{sym1.name}∪{sym2.name}",
            priority=max(sym1.priority, sym2.priority)
        )

    @staticmethod
    def intersection(sym1: UnifiedSymbol, sym2: UnifiedSymbol, test_texts: List[str]) -> UnifiedSymbol:
        """
        Create a symbol that matches only where both patterns overlap.
        This is approximate since regex intersection is complex.
        """
        # Find examples that match both
        common_matches = []
        for text in test_texts:
            matches1 = sym1.detect_and_extract(text)
            matches2 = sym2.detect_and_extract(text)

            for m1 in matches1:
                for m2 in matches2:
                    if m1.data == m2.data:  # Exact match
                        common_matches.append(m1.data)

        if common_matches:
            # Learn pattern from common matches
            patterns = [PatternLearner.learn_from_sample(m) for m in common_matches]
            intersect_pattern = PatternLearner.generalize_patterns(patterns)

            return UnifiedSymbol(
                pattern=intersect_pattern,
                name=f"{sym1.name}∩{sym2.name}",
                priority=min(sym1.priority, sym2.priority)
            )
        else:
            # No intersection found
            return UnifiedSymbol(
                pattern=r"(?!.*)",  # Never matches
                name=f"{sym1.name}∩{sym2.name}",
                priority=0
            )

    @staticmethod
    def sequence(sym1: UnifiedSymbol, sym2: UnifiedSymbol, separator: str = r'\s*') -> UnifiedSymbol:
        """Create a symbol that matches sym1 followed by sym2."""
        pattern = f"(?:{sym1.pattern.pattern}){separator}(?:{sym2.pattern.pattern})"
        return UnifiedSymbol(
            pattern=pattern,
            name=f"{sym1.name}→{sym2.name}",
            priority=(sym1.priority + sym2.priority) // 2
        )

    @staticmethod
    def kleene_star(sym: UnifiedSymbol, separator: str = r'\s*') -> UnifiedSymbol:
        """Create a symbol that matches zero or more repetitions."""
        pattern = f"(?:(?:{sym.pattern.pattern}){separator})*(?:{sym.pattern.pattern})?"
        return UnifiedSymbol(
            pattern=pattern,
            name=f"{sym.name}*",
            priority=sym.priority
        )


# Example usage patterns
def create_scientific_log_symbols():
    """Create symbols for parsing scientific instrument logs."""

    # Base symbols with priorities
    timestamp = UnifiedSymbol(
        r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?',
        'TIMESTAMP',
        priority=10
    )

    log_level = UnifiedSymbol(
        r'\b(?:DEBUG|INFO|WARN|ERROR|FATAL)\b',
        'LEVEL',
        priority=9
    )

    measurement = UnifiedSymbol(
        r'\b\w+:\s*-?\d+(?:\.\d+)?(?:\s*[A-Za-z°]+)?',
        'MEASUREMENT',
        priority=5
    )

    sensor_id = UnifiedSymbol(
        r'\[[\w-]+\]',
        'SENSOR',
        priority=8
    )

    # Composite symbols using algebra
    algebra = SymbolAlgebra()

    # Log entry = timestamp followed by level
    log_header = algebra.sequence(timestamp, log_level)

    # Measurement series = repeated measurements
    measurement_series = algebra.kleene_star(measurement, separator=r',\s*')

    return {
        'timestamp': timestamp,
        'log_level': log_level,
        'measurement': measurement,
        'sensor_id': sensor_id,
        'log_header': log_header,
        'measurement_series': measurement_series
    }


# Integration with existing SymbolGenerator
def upgrade_symbol_generator(generator) -> UnifiedSymbol:
    """
    Convert existing SymbolGenerator to UnifiedSymbol.
    """
    return UnifiedSymbol(
        pattern=generator.detector,
        name=generator.name,
        formatter=lambda x: generator.formatter(x),
        at_branch=generator.at_branch,
        divisible=generator.divisible,
        priority=0
    )