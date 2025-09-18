
from typing import Optional, Dict, List, Any, Set, Tuple, TYPE_CHECKING

import re


# if TYPE_CHECKING:
from .symbol import Symbol

class SymbolGenerator:
    """
    a kind of symbol that allows generating a symbol from other symbols
    """
    
    def __init__(self, name, detector, formatter, at_branch=False, divisible=False):
        
        self.name=name
        
        if not isinstance(detector, re.Pattern):
            detector = re.compile(detector)
        self.detector = detector
        self.formatter = formatter
        self.at_branch=at_branch
        self.divisible=divisible
    
    def detect(self, input_str):
        res = re.findall(self.detector, input_str)
        return res
    
    def extract(self, symbol:'Symbol'):
        if symbol.name == self.name:
            return []
        
        if symbol.is_branch:
            # do we still need the set logic
            agg=list()
            for child in symbol.children:
                if not child.divisible:
                    continue
                res = self.extract(child)
                for _res in res:
                    agg.append(_res)
            return agg
        else:
            reres = re.findall(self.detector, symbol)
            syms = [self.to_symbol(r) for r in reres]
            syms = list(set(syms))
            return syms
    
    def substitute(self, input_str):
        return re.sub(self.detector, self.name, input_str)
    
    def generate(self, *symbols):
        return self.formatter(symbols)
    
    def to_symbol(self, input_str):
        return Symbol(input_str, self.name, has_data=True, divisible=self.divisible)

    def __str__(self):
        return self.name

# Atomic patterns - fundamental units that can't be decomposed further
uuid_generator = SymbolGenerator(
    "UUID",
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",
    "{}"
)

# Numeric patterns with non-capturing groups and word boundaries
integer_generator = SymbolGenerator(
    "INT",
    r"\b(?<!\.)(\d+)(?!\.\d)\b",  # Integer not part of decimal
    "{}",
    at_branch=True
)

decimal_generator = SymbolGenerator(
    "DEC",
    r"\b(\d+\.\d+)\b",  # Decimal with word boundaries
    "{}",
    at_branch=True
)

# Structural patterns - capture content AND delimiters
enclosed_generator = SymbolGenerator(
    "ENC",
    r"(?:\[[^\]]*\]|\([^)]*\)|\{[^}]*\}|<[^>]*>)",  # Non-greedy, captures full structure
    "{}"
)

quote_generator = SymbolGenerator(
    "QUO",
    r'(?:"[^"]*"|\'[^\']*\')',  # Captures full quoted strings
    "{}"
)

# Pattern for identifiers (variable names, etc)
identifier_generator = SymbolGenerator(
    "ID",
    r"\b[a-zA-Z_][a-zA-Z0-9_]*\b",
    "{}"
)

# Pattern for timestamps (ISO-like)
timestamp_generator = SymbolGenerator(
    "TS",
    r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?",
    "{}"
)

# Pattern for IPv4 addresses
ipv4_generator = SymbolGenerator(
    "IP4",
    r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b",
    "{}"
)

# Pattern for hex values
hex_generator = SymbolGenerator(
    "HEX",
    r"0x[0-9a-fA-F]+\b",
    "{}"
)

generators = {
    uuid_generator.name: uuid_generator,
    integer_generator.name: integer_generator,
    decimal_generator.name: decimal_generator,
    enclosed_generator.name: enclosed_generator,
    quote_generator.name: quote_generator,
    identifier_generator.name: identifier_generator,
    timestamp_generator.name: timestamp_generator,
    ipv4_generator.name: ipv4_generator,
    hex_generator.name: hex_generator,
}
