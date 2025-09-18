
from typing import Optional, Callable, Dict, List, Any, Set, Tuple, TYPE_CHECKING
from enum import IntEnum
from dataclasses import dataclass, field
import random
import weakref

if TYPE_CHECKING:
    from .symbol import Symbol

class Move(IntEnum):
    """Movement instructions as bit patterns"""
    PASS =  0b000
    DOWN = 0b001
    RIGHT = 0b010
    UP = 0b011
    LEFT = 0b100
    LAST = 0b101
    
move_bits = 3
num_moves = len(Move._member_names_)

def build_instruction(*sequence):
    i=0
    inst=0b0
    for seq in sequence:
        inst = inst ^ (seq << i)
        i+=move_bits
    return inst

wait_inst = build_instruction(Move.PASS)
search_inst = build_instruction(Move.DOWN, Move.RIGHT, Move.UP)
continue_inst = build_instruction(Move.RIGHT, Move.UP)
exit_inst = build_instruction(Move.UP)

class ZipperState:
    """Immutable state that can be shared/copied between zippers"""
    def __init__(self):
        self.visited: List[int] = list()
        self.path: List[str] = []
        self.moves: List[int] = []
        self.state_index: int = 0 # just a handy index for state transitioning via callbacks
        self.fetched: List[Any] = []
        self.register: Optional['Symbol'] = None
        self.metadata: Dict[str, Any] = {}
        self.halt_reason: Optional[str] = None

    def copy(self):
        """Create a shallow copy for branching"""
        new_state = ZipperState()
        new_state.visited = self.visited.copy()
        new_state.path = self.path.copy()
        new_state.moves = self.moves.copy()
        new_state.fetched = self.fetched.copy()
        new_state.register = self.register
        new_state.metadata = self.metadata.copy()
        return new_state
    
    def __repr__(self):
        parts = []
        parts.append(f"len(visited)={len(self.visited)}")
        parts.append(f"len(path)={len(self.path)}")
        if self.fetched:
            parts.append(f"len(fetched)={len(self.fetched)}")
        if self.register:
            parts.append(f"register={repr(self.register)}")
        if self.halt_reason:
            parts.append(f"halt_reason={self.halt_reason}")
        
        partstr = ','.join(parts)
        return f"ZipperState({partstr})"
    
    def long_repr(self):
        parts = ["ZipperState data:"]
        parts.append(f"  len(visited)={len(self.visited)}")
        parts.append(f"  path={' '.join(self.path)}")
        parts.append(f"  moves={' '.join([Move(m).name for m in self.moves])}")
        if self.fetched:
            parts.append(f"  fetched={' '.join(map(str,self.fetched))}")
        if self.register:
            parts.append(f"  register={self.register}")
        if self.halt_reason:
            parts.append(f"halt_reason={self.halt_reason}")
        return '\n'.join(parts)
        

class ZipperInstruction:
    """Encapsulates behavior functions"""
    def __init__(
        self,
        move_func: Optional[Callable] = None,
        fetch_func: Optional[Callable] = None,
        op_func: Optional[Callable] = None,
        halt_func: Optional[Callable] = None
    ):
        self.move_func = move_func or (lambda s, z: Move.DOWN | (Move.RIGHT << move_bits))
        self.fetch_func = fetch_func
        self.op_func = op_func
        self.halt_func = halt_func or (lambda s, z: z.depth > 100)

class SymbolZipper:
    """
    Improved zipper that:
    - Maintains state separately from navigation
    - Tracks visited nodes to avoid cycles
    - Preserves state on halt
    - Allows resuming from halt
    """

    def __init__(
        self,
        symbol: 'Symbol',
        parent: Optional['SymbolZipper'] = None,
        index: int = 0,
        state: Optional[ZipperState] = None,
        instruction: Optional[ZipperInstruction] = None
    ):
        self.symbol = symbol
        self.parent = parent
        self.index = index
        self.depth = (parent.depth + 1) if parent else 0

        # Share or create state
        self.state = state or ZipperState()
        self.instruction = instruction or ZipperInstruction()

        # Track current position
        self.symbol_id = id(symbol)
        self.is_revisit = self.symbol_id in self.state.visited
        self.state.visited.append(self.symbol_id)
        self.state.path.append(self.symbol.name)

        # Run initial operations
        self._on_enter()

    def _on_enter(self):
        """Called when entering a node"""
        if not self.is_revisit:
            # Only fetch on first visit
            if self.instruction.fetch_func:
                result = self.instruction.fetch_func(self.symbol, self)
                if result is not None:
                    self.state.fetched.append(result)

            # Run operation
            if self.instruction.op_func:
                self.instruction.op_func(self.symbol, self)

    @property
    def count(self):
        return len(self.state.path)

    def move(self, direction: Move) -> Optional['SymbolZipper']:
        """Move in a direction, returning new zipper or None"""
        if direction == Move.PASS:
            return self
        
        elif direction == Move.DOWN:
            if self.symbol.is_branch and self.symbol.children:
                return SymbolZipper(
                    self.symbol.children[0],
                    parent=self,
                    index=0,
                    state=self.state,
                    instruction=self.instruction
                )

        elif direction == Move.RIGHT:
            if self.parent and self.index + 1 < len(self.parent.symbol.children):
                return SymbolZipper(
                    self.parent.symbol.children[self.index + 1],
                    parent=self.parent,
                    index=self.index + 1,
                    state=self.state,
                    instruction=self.instruction
                )

        elif direction == Move.UP:
            if self.parent:
                # Don't create new zipper, return existing parent
                # But mark that we've returned to it
                self.parent.state = self.state  # Update parent's state
                return self.parent

        elif direction == Move.LEFT:
            if self.parent and self.index > 0:
                return SymbolZipper(
                    self.parent.symbol.children[self.index - 1],
                    parent=self.parent,
                    index=self.index - 1,
                    state=self.state,
                    instruction=self.instruction
                )
        elif direction == Move.LAST:
            last_id = self.state.visited[-1]
            
            # lazy
            for dir in range(1,Move.LAST - 1):
                attempt= self.move(dir)
                if attempt and attempt.symbol_id == last_id:
                    print("went to last!")
                    return attempt
        
        return None

    def step(self) -> Tuple[Optional['SymbolZipper'], Any]:
        """Execute one step, returning (next_zipper, result)"""

        # Check halt condition
        if self.instruction.halt_func and self.instruction.halt_func(self.symbol, self):
            self.state.halt_reason = "halt_func triggered"
            return None, self.state

        # Skip if we've been here before (cycle detection)
        if self.is_revisit and self.state.metadata.get('skip_revisits', True):
            # Try to continue to sibling or up
            move_inst = Move.RIGHT | (Move.UP << move_bits)
        else:
            # Get movement instruction
            move_inst = self.instruction.move_func(self.symbol, self)
        
        # Try moves in order encoded in instruction
        next_zipper = self._try_moves(move_inst)

        if next_zipper:
            return next_zipper, next_zipper.symbol
        else:
            self.state.halt_reason = "no valid moves"
            return None, self.state

    def _try_moves(self, instruction: int) -> Optional['SymbolZipper']:
        """Try movement instructions in order"""
        # print(instruction)
        while instruction > 0:
            move_int = instruction & (2**move_bits - 1)
            try:
                move = Move(move_int)
                next_zipper = self.move(move)
            except Exception as e:
                print(f"Error moving zipper with instruction {bin(instruction)} and move_int {bin(move_int)}: {str(e)}")
                next_zipper = self.move(Move.UP)
            
            if next_zipper:
                return next_zipper
            instruction >>= move_bits
        return None

    def run(self, max_steps: int = 1000) -> ZipperState:
        """Run zipper to completion or max steps"""
        current = self
        steps = 0

        while current and steps < max_steps:
            current, _ = current.step()
            steps += 1

        if steps >= max_steps:
            self.state.halt_reason = f"max_steps ({max_steps}) reached"

        return self.state

    def fork(self) -> 'SymbolZipper':
        """Create a fork with copied state for parallel exploration"""
        return SymbolZipper(
            self.symbol,
            parent=self.parent,
            index=self.index,
            state=self.state.copy(),
            instruction=self.instruction
        )

    def resume(self) -> 'SymbolZipper':
        """Resume from current position with cleared halt"""
        self.state.halt_reason = None
        return self

    @property
    def path(self) -> List[str]:
        """Get path from root to current position"""
        if self.parent:
            return self.parent.path + [self.index]
        return [self.symbol.name]

    def relative_path(self, other_zipper:'SymbolZipper'):
        
        my_path = self.path
        other_path = other_zipper.path
        
        i=0
        for ind1,ind2 in zip(my_path, other_path):
            if ind1==ind2:
                i+=1
            else:
                break
        rel_path = list(reversed(my_path[i:])) + other_path[i:]
        return rel_path

    @classmethod
    def create(
        cls,
        root: 'Symbol',
        move_func: Optional[Callable] = None,
        fetch_func: Optional[Callable] = None,
        op_func: Optional[Callable] = None,
        halt_func: Optional[Callable] = None,
        **metadata
    ) -> 'SymbolZipper':
        """Factory method for creating a zipper with custom behavior"""
        instruction = ZipperInstruction(
            move_func=move_func,
            fetch_func=fetch_func,
            op_func=op_func,
            halt_func=halt_func
        )
        state = ZipperState()
        state.metadata.update(metadata)

        return cls(root, state=state, instruction=instruction)

    @classmethod
    def leaf_walker(cls, root:'Symbol', **kwargs):
        
        move_func = lambda s,z:search_inst
        _ = kwargs.pop("move_func",None)
        
        halt_func = lambda s,z: s.is_root and len(z.state.path) > 1
        _ = kwargs.pop("halt_func",None)
        
        if "fetch_func" in kwargs:
            rf = kwargs.pop("fetch_func",None)
            fetch_func = lambda s,z: rf(s,z) if s.is_leaf else None
        else:
            fetch_func=None
            
        return cls.create(root, move_func=move_func, fetch_func=fetch_func, halt_func=halt_func, **kwargs)
    
    @classmethod
    def leaf_collector(cls, root:'Symbol', **kwargs):
        fetch_func = lambda s,z: s if s.is_leaf else None
        _=kwargs.pop(fetch_func,None)
        return cls.leaf_walker(root, fetch_func=fetch_func)
    
    @classmethod
    def random_walker(cls, root:'Symbol',**kwargs):
        
        def move_func(s,z):
            moves = list(range(len(Move._member_names_)))
            random.shuffle(moves)
            return build_instruction(*moves)
        _ = kwargs.pop("move_func",None)    
        
        return cls.create(root, move_func=move_func,**kwargs)
    
    @classmethod
    def topographer(cls, root:'Symbol', **kwargs):
        
        def move_func(s,z):
            if z.state.state_index == 0:
                return Move.DOWN
            elif z.state.state_index==1:
                return search_inst
            else:
                return Move.UP
        _ = kwargs.pop("move_func",None)
            
        def fetch_func(s,z):
            return s.name
        _ = kwargs.pop("fetch_func",None)
            
        def op_func(s,z):
            if z.state.state_index == 0 and s.is_leaf:
                z.state.state_index += 1
        _ = kwargs.pop("op_func",None)
        
        # halt_func = lambda s,z: s.is_root and z.state.
        
        return cls.create(root, move_func=move_func, fetch_func = fetch_func, op_func = op_func)
        
"""
topographers finding similar sequences (both initializing from bottom):

similar=[]
dissimilar=[]

for i in range(n_iters):
    while not topo1.symbol.name == topo2.symbol.name:
        dissimilar.append(topo1.symbol.name)
        topo1.step()

    similar.append(topo1.symbol.name)
    topo2.step()

or we could alternate roles..




"""


        
# Example evolving instruction set
class EvolvableInstruction(ZipperInstruction):
    """Instruction set that can mutate and evolve"""

    def __init__(self, genome: List[int] = None):
        self.genome = genome or [0b00010010, 0b01001000, 0b10000001]  # Random starting genome
        super().__init__(
            move_func=self.genome_move,
            fetch_func=self.genome_fetch,
            halt_func=self.genome_halt
        )

    def genome_move(self, symbol, zipper):
        # Use genome and symbol properties to determine move
        gene_index = hash(symbol.name) % len(self.genome)
        return self.genome[gene_index]

    def genome_fetch(self, symbol, zipper):
        # fetch based on genome pattern matching
        if zipper.depth % 2 == 0:
            return str(symbol)
        return None

    def genome_halt(self, symbol, zipper):
        # Halt based on genome conditions
        return zipper.depth > len(self.genome) * 10

    def mutate(self, mutation_rate: float = 0.1):
        """Mutate the genome"""
        import random
        for i in range(len(self.genome)):
            if random.random() < mutation_rate:
                self.genome[i] ^= random.randint(0, 255)
        return self

    def crossover(self, other: 'EvolvableInstruction') -> 'EvolvableInstruction':
        """Create offspring from two instructions"""
        import random
        crossover_point = random.randint(1, min(len(self.genome), len(other.genome)) - 1)
        new_genome = self.genome[:crossover_point] + other.genome[crossover_point:]
        return EvolvableInstruction(new_genome)