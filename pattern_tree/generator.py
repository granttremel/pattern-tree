
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

uuid_generator = SymbolGenerator("UUID",r"\w{8}-\w{4}-\w{4}-\w{4}-\w{12}","{}")
integer_generator = SymbolGenerator("INT",r"([\d]+)", "{}", at_branch=True)
decimal_generator = SymbolGenerator("DEC",r"([\d]+\.[\d]+)","{}", at_branch=True)
# enclosed_generator = SymbolGenerator("ENC",r"[\{|\(|\[]([\w]+)[\})\]]","{}")
enclosed_generator = SymbolGenerator("ENC",r"\[\w+\]|\(\w+\)|\(\w+\)|<\w+>","{}")
quote_generator = SymbolGenerator("QUO",r"\"\w+\"|\'\w+\'","{}")

generators = {
    uuid_generator.name:uuid_generator,
    integer_generator.name:integer_generator,
    decimal_generator.name:decimal_generator,
    enclosed_generator.name:enclosed_generator,
}
