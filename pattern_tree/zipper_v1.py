
from typing import Optional, Callable, Dict, List, Any, Set, Tuple, TYPE_CHECKING

from math import factorial

if TYPE_CHECKING:
    from .symbol import Symbol
    
def random_leader_func(symbol, zipper):
    if zipper.count > 20:
        return 0b0
    
    if symbol.name == "INT":
        return 0b10011100
    else:
        return 0b11100100    

def value_retrieve_func(symbol, zipper):
    return str(symbol)

class SymbolZipper:
    
    _depth_inst = 0b100100
    _continue_inst = 0b1001
    _exit_inst = 0b10
    _map = ["down","right","up","left"]
    # _heritable=["mode","register","leader_func","retrieve_func","op_func"]
    
    def __init__(self, 
            symbol:'Symbol', 
            parent:'SymbolZipper'=None,
            previous:'SymbolZipper'=None,
            index=0, 
            mode='depth',
            leader_func:Optional[Callable]=None,
            retrieve_func:Optional[Callable]=None,
            op_func:Optional[Callable]=None
            ):
        self.symbol=symbol
        self.parent=parent
        self.index=index
        self.count = previous.count + 1 if previous else 0 

        self.mode = previous.mode if previous else mode
        self.register = previous.register if previous else None
        self.leader_func = previous.leader_func if previous else leader_func
        self.retrieve_func = previous.retrieve_func if previous else retrieve_func
        self.op_func = previous.op_func if previous else op_func
        
        self.retrieved=previous.retrieved if previous else []
        # self.path = previous.path if previous else ["start"]
        self.path = (previous.path if previous else []) + [id(symbol)]
        
        self.run_retrieve()
        self.run_op_instruction()
    
    def confer(self, other:'SymbolZipper'):
        other.count = self.count + 1
        other.retrieved = self.retrieved
        other.path = self.path
        
        # these are optional.. but may change over the course of execution? 
        other.mode = self.mode
        other.register = self.register
        other.leader_func = self.leader_func
        other.retrieve_func = self.retrieve_func
        other.op_func = self.op_func
    
    def get_symbol(self, inst):
        dir = self._map[inst]
        if dir=="down":
            if self.symbol.is_branch:
                return self.symbol.children[0], 0
        elif dir=="right":
            if self.parent and self.index < len(self.parent.symbol.num_branches):
                return self.parent.symbol.children[self.index+1], self.index+1
        elif dir=="up":
            if self.parent and self.parent.parent:
                return self.parent.get_symbol("right")
                # return self.parent.parent.symbol.children[self.parent.index+1], self.parent.index+1
        elif dir=="left":
            if self.index > 0:
                return self.parent.symbol.children[self.index-1], self.index-1
        else:
            pass
        return None, 0
    
    def down(self):
        """
        0b00
        """
        print("zipper down")
        if not self.symbol.is_leaf:
            return SymbolZipper(self.symbol.children[0], self, self, 0)
        return None
    
    def up(self):
        """
        0b10
        or we could just make a new one?
        up means up and to the right.. but what if you need to keep going up?
        """
        print("zipper up")
        self.confer(self.parent)
        return self.parent
        # return self.parent.right()
    
    def right(self):
        """
        0b01
        """
        print("zipper right")
        if self.parent and self.index+1 < self.parent.symbol.num_branches:
            return SymbolZipper(
                self.parent.symbol.children[self.index+1],
                self.parent,
                self,
                self.index+1
            )
    
    def left(self):
        """
        0b11
        or we could just make a new one?
        """
        print("zipper left")
        if self.parent and self.index-1 > 0:
            return SymbolZipper(
                self.parent.symbol.children[self.index-1],
                self.parent,
                self,
                self.index-1
            )
    
    def run_move_instruction(self,inst:int):
        """
        depth: 0100 (never go up or left)
        """
        if not inst > 0 and inst < 2**len(self._map):
            return None
        
        binst = inst
        next=None
        while binst > 0:
            i1 = binst & 3
            
            dir = self._map[int(i1)]
            next = getattr(self, dir)()
            if next:
                break
            binst = binst >> 2
            
        return next
    
    def step(self):
        if self.mode=="depth":
            inst = self._depth_inst
        elif self.mode=="leader":
            inst = self._get_leader_instruction()
        else:
            # default behavior ?
            next = self.up()
        
        next = self.run_move_instruction(inst)
        
        if next:
            return next, next.symbol
        else:
            return self.halt()
            
    def halt(self):
        """
        when this happens, how to prevent zipper from disappearing..?
        """
        retrieved = self.retrieved.copy()
        return None, retrieved
    
    def _get_leader_instruction(self):
        inst = self.leader_func(self.symbol, self)
        return inst

    # retrieve
    
    def run_retrieve(self):
        if self.retrieve_func:
            res = self.retrieve_func(self.symbol, self)
            if res:
                self.retrieved.append(res)
            
    # op instruction codes

    def replace(self, new_symbol):
        if self.parent:
            self.parent.symbol.children[self.index] = new_symbol
        self.symbol=new_symbol

    def do_nothing(self):
        pass

    def overwrite_register(self):
        self.register = self.symbol

    def write_register(self):        
        if self.parent:
            self.parent.symbol.children[self.index] = self.register
        self.symbol=self.register
    
    def swap_register(self):
        """
        shuffling, destroys info i think
        """
        sym = self.symbol
        self.replace(self.register)
        self.register = sym
        
    def divide_sym_reg(self):        
        """
        div is in-place operation
        """
        new_sym = self.sym / self.reg
        
    def divide_reg_sym(self):
        new_sym = self.reg / self.sym
    
    def insert_register_right(self):
        """
        shuffling, destroys info i think
        """
        if self.parent:
            self.parent.symbol.children.insert(self.index+1, self.register)
        return self
    
    def run_op_instruction(self):
        
        ops = [self.do_nothing, self.overwrite_register, self.divide_reg_sym, self.insert_register_right]
        
        inst = self._get_op_instruction()
        
        binst = inst
        next=None
        while binst > 0:
            i1 = binst & 3
            next = ops[int(i1)]()
            if next:
                break
            binst = binst >> 2
        
        return next
        
    def _get_op_instruction(self):
        if self.op_func:
            inst = self.op_func(symbol=self.symbol, index=self.index,count=self.count)
            return inst
        else:
            return 0b0
    
    @classmethod
    def get_zipper(cls, root_symbol, parent=None, index=0, **kwargs):
        return cls(root_symbol,parent=parent,index=index, **kwargs), root_symbol
    


