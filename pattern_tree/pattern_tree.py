
from typing import Optional, Dict, List, Any, Set, Tuple
import re
import tiktoken
from itertools import chain, zip_longest
import string

from .symbol import Symbol
from .generator import SymbolGenerator, generators
from .zipper import SymbolZipper


class PatternTree:
    
    def __init__(self):
        self.frequencies: Dict[Symbol, int] = {}
        self.generators:Tuple[SymbolGenerator]=tuple()
        self.root = Symbol("","A",is_root=True)

    def set_generators(self, *generators):
        self.generators = tuple(generators)

    def preprocess(self, new_string):
        
        sym = Symbol(new_string)
        
        for gen in self.generators:
            subsyms = gen.extract(sym)
            for subsym in subsyms:
                sym.divide(subsym, at_branch = gen.at_branch)
        
        # zp = SymbolZipper.create(sym)
        sym.tokenize()
                
        return sym

    def combine_symbols(self):
        
        def op_func(sym:'Symbol',zp:'SymbolZipper'):
            
            reg = zp.state.register
            if reg.name == sym.name:
                new_sym = reg.merge(sym)
                zp.parent.symbols
            
        zp = SymbolZipper.leaf_walker(self.root,op_func=op_func)
        
        pass

    def append(self, new_string):
        """
        check for generators
        substitute symbols
        etc
        """
        
        sym = self.preprocess(new_string)
        self.root = self.root + sym