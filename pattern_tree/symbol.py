
from typing import Optional, Dict, List, Any, Set, Tuple

import numpy as np

import re
import tiktoken
from itertools import chain, zip_longest
import string
import weakref

from .zipper import SymbolZipper

tokenizer =  tiktoken.get_encoding("cl100k_base")

def encode_string(input_str:str):
    return tokenizer.encode(input_str)
    
def decode_string(tokens:List[int]):
    return tokenizer.decode(tokens)

class Symbol(str):

    _name_base = string.ascii_uppercase
    _new_token_ind=tokenizer.n_vocab
    _symbols = weakref.WeakValueDictionary()
    _freed_names = set()  # Track freed names for reuse
    
    def __new__(cls, value: str, 
            name:str="",
            is_root=False,
            has_data=False,
            **kwargs
        ):
        
        existing = cls._get_from_registry(
            value, name, 
            is_root=is_root)
        
        if existing and not has_data:
            return existing
        
        instance = super().__new__(cls, value)
        return instance
    
    def __init__(self, value: str,
                name:str="",
                is_root=False,
                has_data=False,
                divisible=True,
                repeats:Optional[int] = 1,
                children: Optional[List['Symbol']] = None,
                metadata: Optional[Dict[str, Any]] = None,
                position: Optional[int] = None
            ):
        if hasattr(self, '_initialized'):
            return
        self._initialized=True
        
        ind=-1
        if not name:
            name = self.get_next_name()
        
        if not name in self._symbols:
            self._symbols[name] = self
            ind = self.__class__._new_token_ind
            self.__class__._new_token_ind += 1
        else:
            ind = self._symbols[name]._token_index
            
        self.name=name
        self.is_root = is_root # root symbol
        self.has_data=has_data
        self.children:List['Symbol'] = children or [] #maybe a dict for treelike
        self.divisible = divisible
        if len(value) < 2:
            self.divisible=False
            
        self.metadata = metadata or {}
        self.repeats=repeats
        self.position = position
        self._token_index = ind
        self._tokens=[]

        # Set up callback to track when this symbol is garbage collected
        def cleanup_callback(ref):
            Symbol._freed_names.add(name)

        self._weakref = weakref.ref(self, cleanup_callback)
    
    @classmethod
    def _get_from_registry(cls, value, name="", is_root=False):
        value = value.strip() # does this discard information .. ?
        if not value and not is_root:
            return None
        
        for _name, sym in cls._symbols.items():
            # can add more exotic behavior for is ..?
            if (name and _name==name) or value==str(sym):
                return sym
        
        return None
    
    @property
    def num_branches(self):
        return len(self.children)
    
    @property
    def num_nodes(self):
        res = 1
        for ch in self.children:
            res += len(ch)
        return res
    
    @property
    def is_leaf(self):
        return self.num_branches==0
    
    @property
    def is_branch(self):
        return self.num_branches>0
    
    def dot(self, other):
        return self.__class__._dot(self, other)
    
    def get_tokens(self, expand_branches=True):
        
        if self.is_branch and expand_branches:
            tks =[]
            for child in self.children:
                tks.extend(child.get_tokens(expand_branches=expand_branches))
            return tks
        elif self.is_branch:
            return [self._new_token_ind]
        elif self.is_leaf:
            return self._tokens
        else:
            return []
        
    def tokenize(self):
        def op(s,z):
            if s.is_leaf and s.divisible:
                s._tokens = tokenizer.encode(str(s))
        
        zp = SymbolZipper.leaf_walker(self, op_func = op)
        res = zp.run()
        return res
    
    def merge(self, symbol:'Symbol'):
        return self.__class__._merge(self, symbol)
    
    def divide(self, symbol:'Symbol', at_branch=False):
        """
        Symbol(ABABCABAB).split(Sym(C)) -> Symbol(children=[Sym(ABAB), Sym(C), Sym(ABAB)])
        Symbol(children=[Sym(ABAB), Sym(C), Sym(ABAB)]).split(Sym(ABAB)) -> Symbol(children=[Sym(AB),Sym(AB), Sym(C), Sym(AB),Sym(AB)])
        maybe add a count parameter to summarize repetitive sequences
        """
        if at_branch:
            self.__class__._divide_at_branch(self, symbol)
        else:
            self.__class__._divide(self, symbol)
    
    def collect_leaves(self):
        zp = SymbolZipper.leaf_collector(self)
        res = zp.run()
        return res.fetched
    
    def split_symbol(self, symbol:'Symbol')->List['Symbol']:
        """
        B.split_string(ABABABA) -> [A,B,A,B,A,B,A]
        
        """
        if symbol == str(self):
            return []
        parts = symbol.split(str(self))

        if len(parts) == 1:
            return [symbol]
        
        paired = zip_longest(parts, [self] * (len(parts) - 1), fillvalue=None)
        result = []
        for pair in paired:
            for item in pair:
                if item:
                    if item == self:
                        result.append(item)
                    # elif item == '':
                    #     continue
                    else:
                        result.append(self.__class__(item))
        
        return result
    
    def split_symbols(self, symbol_list:List['Symbol']):
        outlist = []
        for sym in symbol_list:
            splits = self.split_symbol(sym)
            outlist.extend(splits)
        
        return outlist
    
    def split_children(self, symbol:'Symbol'):
        if symbol.is_branch:
            return self.split_symbols(symbol.children)
        else:
            return []
    
    def get_zipper(self, **kwargs):
        return SymbolZipper.create(self,**kwargs)

    
    def print_tree(self, nlevel=0, tokens=False, info=False):
        
        _frm = "{name}: \"{value}\""
        _frm_root = "{name}(root): \"{value}\""
        frm = _frm_root if self.is_root else _frm
    
        tab = '  '
        
        if tokens and self.tokens:
            value=self.tokens
        elif self.is_leaf:
            value = str(self)
        else:
            value = repr(self)
        
        if self.is_leaf:
            printstr = tab*nlevel + frm.format(name=self.name,value=value)
        else:
            printstr=tab*nlevel + frm.format(name=self.name,value=value)
        
        if info:
            if self.is_leaf:
                printstr += f" (leaf)"
            else:
                printstr += f" (branchx{self.num_branches})"
        
        print(printstr)
        
        for child in self.children:
            child.print_tree(nlevel+1, tokens=tokens, info=info)
    
    def repr_compact(self, tokens=False):
        
        innerstrs = []
        if self.is_branch:
            for child in self.children:
                innerstrs.append(child.repr_compact(tokens=tokens))
        else:
            if tokens:
                return str(self._token_index)
        
        innerstr = ' '.join(innerstrs)
        namestr = str(self._token_index) if tokens else self.name
        if innerstr:
            return f"{namestr}[{innerstr}]"
        else:
            return f"{namestr}"
            
    def __repr__(self)->str:
        if self.children:
            innerstr = ','.join(repr(child) for child in self.children)
        else:
            innerstr=str(self)
        return f"Symbol({self.name},{innerstr})"
    
    def to_string(self)->str:
        return ''.join([sym.to_string() for sym in self.children])
    
    def to_dict(self):
        subdicts = []
        for child in self.children:
            subdicts.append(child.to_dict())
        
        outdict= {
            "name":self.name,
            "value":str(self),
            "is_root":self.is_root,
            "divisible":self.divisible,
            "repeats":self.repeats,
            "metadata":self.metadata,
            "token_index":self._token_index,
            "tokens":','.join(map(str,self.get_tokens())),
            "children":subdicts
        }
        
        if not outdict.get("metadata"):
            del outdict["metadata"]
        if not outdict.get("tokens"):
            del outdict["tokens"]
        if not outdict.get("children"):
            del outdict["children"]
            
        return outdict
    
    def __add__(self, other:'Symbol'):
        self.children.append(other)
        return self
    
    def __div__(self, other:'Symbol'):
        return self.__class__._divide(self, other)
    
    def __divmod__(self, other:'Symbol'):
        return self.__class__._divide_at_branch(self, other)
    
    def __mul__(self, other:'Symbol'):
        """
        A . B
        hhmmmmm
        """
        pass
    
    def __matmul__(self, other:'Symbol'):
        """
        A @ B
        do something like "compare patterns" or "project A onto B"
        """
        pass
    
    def __eq__(self, other:'Symbol'):
        return str(self) == str(other)
        # return self.to_string() == other.to_string()
    
    def __hash__(self):
        return hash(str(self))

    @classmethod
    def get_symbol(cls, symbol_string:str="", name:str=""):
        if not symbol_string and not name:
            return
        for _name, sym in cls._symbols.items():
            # can add more exotic behavior for is ..?
            if (name and _name==name) or symbol_string==str(sym):
                return sym
    
    @classmethod
    def _linear_dot(cls, sym1, sym2):
        freqs = {}
        score=0
        for c1,c2 in zip(sym1.children, sym2.children):
            if c1.name==c2.name:
                score += 1
                
            if not c1.name in freqs:
                freqs[c1.name] = [0,0]
            freqs[c1.name][0]+=1
            
            if not c2.name in freqs:
                freqs[c2.name] = [0,0]
            freqs[c2.name][1]+=1
            
        sums = [sum([freqs[name][i] for name in freqs]) for i in range(2)]
        for name in freqs:
            score += min(freqs[name][0] / sums[0], freqs[name][1] / sums[1])
        
        return score
    
    @classmethod
    def _tree_dot(cls):
        
        pass
    
    @classmethod
    def _pool(cls, *symbols)->'Symbol':
        sym1 = symbols[0]
        for othersym in symbols[1:]:
            sym1=cls._merge(sym1,othersym)
        return sym1
        
    @classmethod
    def _merge(cls, symbol1, symbol2)->'Symbol':
        if symbol1.name==symbol2.name:
            newval = ' '.join((str(symbol1),str(symbol2)))
            newsym = Symbol(
                newval,
                name=symbol1.name,
                repeats = symbol1.repeats + symbol2.repeats,
                children = symbol1.children + symbol2.children,
                metadata={
                    "symbol1":repr(symbol1),
                    "symbol2":repr(symbol2)
                },
                position = symbol1.position
            )
            return newsym

    @classmethod
    def _group(cls, *symbols)->'Symbol':
        symstr = cls.make_string(*symbols)
        name = cls.get_next_name()
        return cls(symstr, name, children = list(symbols), is_root=True)

    @classmethod
    def _divide(cls, num:'Symbol', den:'Symbol'):
        if not num.divisible:
            return
        elif num.is_leaf:
            syms = den.split_symbol(num)
            num.children = syms
        else:
            for child in num.children:
                cls._divide(child, den)
    
    @classmethod
    def _divide_at_branch(cls, num:'Symbol', den:'Symbol'):
        if not num.divisible:
            return
        elif num.is_leaf:
            syms = den.split_symbol(num)
            num.children = syms
        else:
            newchildren = []
            newchildren = den.split_children(num)
            num.children=newchildren
        
    @classmethod
    def make_string(cls, *symbols)->str:
        return ''.join([sym.to_string() for sym in symbols])
    
    @classmethod
    def get_next_name(cls, iter=0)->str:
        # Try to reuse a freed name first
        if cls._freed_names:
            return cls._freed_names.pop()

        # Otherwise generate a new name
        nsyms = len(cls._symbols) + iter
        nchars = len(cls._name_base)

        if nsyms==0:
            return cls._name_base[0]

        new_name = []

        n = nsyms
        while n>0:
            n -= 1  # Convert to 0-based for proper base conversion
            modres = n % nchars
            new_name.append(cls._name_base[modres])
            n = n//nchars
        
        new_name_str = ''.join(reversed(new_name))
        
        if new_name_str in cls._symbols:
            return cls.get_next_name(iter+1)
        else:
            return new_name_str 
    
    def __len__(self):
        
        ct = 0
        sym=self
        zp = SymbolZipper.create(sym)
        while zp:
            if sym.is_leaf:
                ct+=1
            sym = zp.step()
    
    @classmethod
    def __len__(cls):
        return len(cls._symbols)
    
    @classmethod
    def _clear(cls):
        cls._symbols = {}
    
