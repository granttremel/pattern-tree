
from typing import Optional, Dict, List, Any, Set, Tuple

import re
import tiktoken
from itertools import chain, zip_longest
import string

tokenizer =  tiktoken.get_encoding("cl100k_base")

def encode_string(input_str:str):
    return tokenizer.encode(input_str)
    
def decode_string(tokens:List[int]):
    return tokenizer.decode(tokens)

class Symbol(str):
    """
    needs to be a class but also an instance.. hmm
    its a symbol but also.. a function from lists of symbols to a symbol?
    
    """
    
    _name_base = string.ascii_uppercase
    # when we lose a reference, i.e. symbol is fully replaced with sub-symbols, we should remove
    _symbols = {}
    
    def __new__(cls, value: str, 
            name:str,
            divisible=True,
            children: Optional[List['Symbol']] = None,
            metadata: Optional[Dict[str, Any]] = None,
            position: Optional[int] = None):
        """Create a new AnnotatedToken that behaves as a string"""
        instance = str.__new__(cls, value)
        return instance
    
    def __init__(self, value: str,
                name:str,
                is_root=True,
                divisible=True,
                children: Optional[List['Symbol']] = None,
                metadata: Optional[Dict[str, Any]] = None,
                position: Optional[int] = None):
        """Initialize the annotation metadata"""
        self.name=name
        self.is_root = is_root # root symbol 
        self.children:List['Symbol'] = children or [] #maybe a dict for treelike
        self.divisible = divisible
        self.metadata = metadata or {}
        self.position = position
    
    @property
    def is_leaf(self):
        return len(self.children)==0
    
    @property
    def is_branch(self):
        return len(self.children)>0
    
    def divide(self, symbol:'Symbol', at_root=False):
        """
        Symbol(ABABCABAB).split(Sym(C)) -> Symbol(children=[Sym(ABAB), Sym(C), Sym(ABAB)])
        Symbol(children=[Sym(ABAB), Sym(C), Sym(ABAB)]).split(Sym(ABAB)) -> Symbol(children=[Sym(AB),Sym(AB), Sym(C), Sym(AB),Sym(AB)])
        maybe add a count parameter to summarize repetitive sequences
        """
        if at_root:
            self.__class__._divide_at_root(self, symbol)
        else:
            self.__class__._divide(self, symbol)
    
    def collect_leaves(self):
        
        if not self.children:
            return [self]
        else:
            all_leaves = []
            for sym in self.children:
                leaves = sym.collect_leaves()
                all_leaves.extend(leaves)
            return all_leaves
    
    # def walk(self):
    #     pass
    
    # def __iter__(self):
    #     return iter(self.children)
    
    # def __next__(self):
    #     pass
    
    def split_string(self, input_string:str):
        
        if input_string == str(self):
            return []
        parts = input_string.split(str(self))
        
        # print(f"splitting {input_string} with {str(self)} to produce {parts}")
        
        # If no splits occurred, return the original string as a symbol
        if len(parts) == 1:
            return [self.get_symbol(input_string)]
            # return []
        
        # Interleave parts with symbol using zip_longest
        # zip_longest pairs each part with symbol, except the last part gets None
        paired = zip_longest(parts, [self] * (len(parts) - 1), fillvalue=None)
        
        # Flatten, keeping empty strings but filtering None values
        result = []
        for pair in paired:
            for item in pair:
                if item is not None:
                    if item == self:
                        result.append(item)
                    elif item == '':
                        # Skip empty strings for now
                        continue
                    else:
                        result.append(self.get_symbol(item))
        
        # print(f"split produced symbols {result}")
        return result
        
    def __repr__(self)->str:
        if self.children:
            innerstr = ','.join(repr(child) for child in self.children)
        else:
            innerstr=str(self)
        return f"Symbol({self.name},{innerstr})"
    
    def to_string(self)->str:
        # or just self.value, str(self)
        return ''.join([sym.to_string() for sym in self.children])
    
    def __add__(self, other):
        self.children.append(other)
        # or we could group
        # self.__class__._group(self,other)
    
    def __div__(self, other:'Symbol'):
        return self.__class__._divide(self, other)
    
    def __divmod__(self, other:'Symbol'):
        return self.__class__._divide_at_root(self, other)
    
    def __eq__(self, other:'Symbol'):
        return str(self) == str(other)
        # return self.to_string() == other.to_string()
    
    def __hash__(self):
        return hash(str(self))
    
    @classmethod
    def get_symbol(cls, symbolstring, name:str="", children = [], divisible = True)->'Symbol':
        
        if not symbolstring:
            return
        
        for _name, sym in cls._symbols.items():
            # can add more exotic behavior for is ..?
            if _name==name:
                return sym # hmm needs to generate symbols with varying data
                
            elif symbolstring is sym or symbolstring==str(sym):
                return sym
        
        if not name or name in cls._symbols:
            name = cls.get_next_name()
        new_symbol = cls(symbolstring, name, children=children, divisible=divisible)
        cls._symbols[name] = new_symbol
        return new_symbol
    
    @classmethod
    def _group(cls, *symbols)->'Symbol':
        symstr = cls.make_string(*symbols)
        # return Symbol(symstr, "get_next_name", True, children = list(symbols))
        return cls.get_symbol(symstr, children = list(symbols))

    @classmethod
    def _divide(cls, num:'Symbol', den:'Symbol'):
        if not num or not den:
            return 
        elif not num.children:
            # print(f"calling split on {num} with {den}")
            syms = den.split_string(str(num))
            num.children = syms
        else:
            newchildren = []
            for child in num.children:
                if child.children:
                    # print(f"calling divide on {child} with {den}")
                    cls._divide(child, den)
                    newchildren.append(child)
                else:
                    # print(f"calling split on {child} with {den}")
                    child_syms = den.split_string(str(child))
                    child.children = child_syms
                    newchildren.append(child)
            num.children = newchildren
    
    @classmethod
    def _divide_at_root(cls, num:'Symbol', den:'Symbol'):
        
        if not num or not den:
            return
        elif not num.children:
            # print(f"calling split on {num} with {den}")
            syms = den.split_string(str(num))
            num.children = syms
        else:
            newchildren = []
            for child in num.children:
                # print(f"calling divide on {child} with {den}")
                syms = den.split_string(str(child))
                newchildren.extend(syms)
            num.children=newchildren
        
    @classmethod
    def make_string(cls, *symbols)->str:
        return ''.join([sym.to_string() for sym in symbols])
    
    @classmethod
    def get_next_name(cls)->str:
        nsyms = len(cls._symbols)
        nchars = len(cls._name_base)
        
        if nsyms==0:
            return cls._name_base[0]
        
        new_name = []
        
        n = nsyms
        while n>0:
            modres = n % nchars
            new_name.append(cls._name_base[modres])
            n = n//nchars
        
        return ''.join(reversed(new_name))
    
    @classmethod
    def _clear(cls):
        cls._symbols = {}
    

class SymbolGenerator:
    """
    a kind of symbol that allows generating a symbol from other symbols
    """
    
    def __init__(self, name, detector, formatter):
        
        self.name=name
        
        if not isinstance(detector, re.Pattern):
            detector = re.compile(detector)
        self.detector = detector
        
        self.formatter = formatter
    
    def detect(self, input_str):
        res = re.findall(self.detector, input_str)
        return res
    
    def extract(self, symbol:'Symbol'):
        if symbol.name == self.name:
            return []
        
        if symbol.children:
            agg=list()
            for child in symbol.children:
                res = self.extract(child)
                for _res in res:
                    agg.append(_res)
            return agg
        else:
            reres = re.findall(self.detector, symbol)
            syms = [self.to_symbol(r) for r in reres]
            return list(set(syms))
    
    def split(self, input_str:str, symbol:Symbol):
        # Split the string by the symbol
        parts = input_str.split(str(symbol))
        
        # If no splits occurred, return the original string
        if len(parts) == 1:
            return [input_str]
        
        # Interleave parts with symbol using zip_longest
        # zip_longest pairs each part with symbol, except the last part gets None
        paired = zip_longest(parts, [symbol] * (len(parts) - 1), fillvalue=None)
        
        # Flatten and filter out None values
        result = [item for pair in paired for item in pair if item is not None]
        
        return result
    
    def substitute(self, input_str):
        return re.sub(self.detector, self.name, input_str)
    
    def generate(self, *symbols):
        return self.formatter(symbols)
    
    def to_symbol(self, input_str):
        return Symbol.get_symbol(input_str, self.name, divisible=False)
        # return Symbol.get_symbol(input_str, self.name, [], {}, 0)

    def __str__(self):
        return self.name

integer_generator = SymbolGenerator("INT",r"([\d]+)", "{}")
decimal_generator = SymbolGenerator("DEC",r"([\d]+\.[\d]+)","{}")

class PatternTree:
    
    def __init__(self, initial_symbols, initial_generators):
        
        self.frequencies: Dict[Symbol, int] = {}
        self.symbols:Dict[str,Symbol]={sym.name:sym for sym in initial_symbols}
        self.generators:Tuple[SymbolGenerator]=tuple(initial_generators)

    def extract(self, input_str):
        
        parent_symbol = Symbol(input_str)
        syms = []
        
        for gen in self.generators:
            vsym = gen.to_symbol(input_str)
            
        
        pass

