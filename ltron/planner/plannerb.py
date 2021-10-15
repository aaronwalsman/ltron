import random

# Ok, I think what we have below is correct as pseudo-code.
# Next: test it with a simple environment and make it no-longer pseudo-code
# After that: make it into a persistent thing that I can make multiple calls to.

'''
A snag:
I need some way of figuring out whether a node has already been explored.  Currently these nodes can be anything.  Ideally they would be something hashable so I can put them in a set or a dictionary, or even just use them as edge coordinates without using the more cumbersome integer ids to stand in for them in the set of edges.

Options:
1. Follow through: make the nodes be hashable and just do this.
2. Make an additional function (which must be passed in) that returns a unique id that can be used... but this is basically just the hash function.

What I'm going to do for now:
Follow through, make everything be hashable (can go in a set, etc.) then revise if that becomes unfeasible later.
'''

'''
A few thoughts:
There are two components here:
One thing that grows this graph and another thing that tries to find a path through that graph from start to goal.  Both need similar, but slightly different structures.  The graph can be shared for all time (i.e. by multiple calls to plan for the same target configuration).  The path finder is specific to the start and goal.

The basic idea is to grow the graph until you can find a path through it from the start to the goal that is "good enough."  How do you know when to check if you have a path from start to goal?  At the moment, it seems like you can do this every time you make a connection back to the graph.  But there's a better thing here I think.  If you keep track of what nodes are reachable from the current start, then you can only check once you hit a previously unreachable node.  That's not bad.
'''

'''
Another issue (perhaps for later):
Also, how do I do the backward search that lets me avoid camera motion?  Right now this is all forward.
'''

'''
Tangles:
What is the interface that I want to provide to the gentle user of this function for any reasonable version of this?

So first off, we need an env.

My idea from earlier this morning is that each component should provide get_state and set_state (which should return an observation) methods that allow us to jump around to different states and do backtracking, etc.  The combined state is then a good thing to use as a node indentifier.

My idea from later this evening was to impose some kind of structure on this whole thing and use the fact that we are always adding or removing bricks that correspond to some target.  In that way, each state would bastically be a bit-vector for whether or not a certain target node has a corresponding node in the scene, and then another bit vector if we want to allow non-target nodes to show up when an agent makes mistakes.  This is nice because we have a nice distance metric to use here when planning (hamming distance).

The problem with the first thing is that it makes it hard to use certain nice heuristics, like assembly via disassembly.

The problem with the second thing is it's more limited and doesn't have variables for things like camera motion.  So out the window it goes maybe?

The problem with throwing it out is that this assembly via disassembly is probably pretty powerful, and without it, it will take a lot more work to make sure we don't move the camera too much.

The second problem with it though is that we now have to figure out a way to translate whatever our different state space is to this bit vector, which becomes something we either have to make as some super-generic component of the env (which doesn't really make sense, because the env can cover spaces that don't have a specific target) or make that conversion another thing the user has to specify.  What are we to do?

So conceptually, it would be really nice to be able to use whatever configuration space we want, but doing so gets rid of some powerful heuristics.

So what else must the user provide?  Honestly, it would be nice if the answer was "a start, a goal and not much else."  But then I realize that I have these functions that I've been treating as arguments: neighbor_fn and check_edge_fn.  These are two new things that must be supplied by the gentle user, and they are not trivial at all.  In the case of the reassembly planner, the neighbor_fn is:

def neighbor_fn(env, state, target):
    obs = env.set_state(state)
    current_instances = get_instances_from_obs()
    match(current_instances, target)
    if misplaced_current_instances:
        return remove_ops_for_misplaced_current_instances
    elif unplaced_target_instances:
        return add_ops_for_target_graph_neighbors
    else:
        return []

But here we find another tangle: if our state space is this convoluted thing we get directly from the environment, how do we know what the neighbors will be or what high level actions we can reason about?  I mean, yeah I guess we can just hard-code it, and make our gentle user specify it in whatever implments remove_ops_for_misplaced_current_instances.  This gets back to the fact that somebody real, either us or gentle user is going to need to do some transating from goofy complicated state space output by the environment to clean precise hashable node that can be added to a python set or dictionary and tested for equality with other nodes.

What to do what to do?

Is there a different route to take?  Lookng further afield, we could go to the bit-vector approach, and make everything else (cameras) part of the transition space between nodes.  The problem with this is that we'd then have to track this as we trace a path through high-level nodes.

What if we just tried to plan directly in the super-crazy-pants space?  Skip the high-level/low-level thing and just do all planning in the low-level space.  This seems pretty hard though.  Maybe possible with crazy heuristics?  I mean that's kind of what my last attempt was/is right?  And that has been kind of a disaster.  Why?  There's still some business to clear up there about viewpoints and other things.  What a mess.

Ok, so how can we make the high-level/low-level search work?  And make it general enough to use for multiple action spaces?  This thing really is about adding and removing nodes, so maybe we do the bit-vector thing with a... no no no.  We need the camera in there too.  So the neighbor function is going to return camera motions as well?  Yikes.  Is this where some heuristic comes in to compensate for the blow-up in graph degree?  Guess so.

We're also going to need something that can get us from point a to point b using low-level actions.  I guess we know how to do this, the last few attempts at this all do something similar, but the last few attempts have also been all this crazy specific mess.  I guess this is kind of fine if it gets me there, BUT I thought cleaning up that mess was part of what this attempt was all about, and it's kind of disappointing to throw away generality again (or require the gentle user to implement so so much).

As a thought exercise, let's pull this together from the bottom up.  At the lowest level, our action space is all the pixels in the screen plus a dozen or so extra levers.

From here we can consolidate all the pixels into a varying number of discrete regions that all do the same thing.  This takes us from ~256x256 to probably a few hundred discrete snap points.  A few hundred is still really big.  We could reduce this still if we group those few hundred into a slightly smaller functionally equivalent class, where all the snaps that result in the same next pose get grouped together.  This may help a little bit, but not as much as we need.  Is this getting us anywhere?

One thought: we can make the low-level planner be another instantiation of the high-level planner.  Fancy fancy.  In this case, the check_edge function would do nothing because all the edges are already correctly connected to each other.  Would be kinda cool if it worked.

Ok no more tangles.
'''

'''
High Level:
env: gym environment
nodes: states
edges: lazily evaluated low-level state-action-state paths
neighbor_fn: [manually specified] add-remove bricks plus camera actions
check_edge_fn: low-level planner

Low Level:
env: gym environment (shared with high-level)
nodes: states
edges: actions
neighbor_fn: 

Yo wait wait wait... what if we added a new action space that us find high-level neighbors automatically based on the target.  Yeah I like that a lot.  This is only in the training environment, so the agent can't actually use it, but it can be used for supervision.  We could even do this for the low-level environment too.  The purpose of it is just to specify the structure of the graph for planning at that particular level.  We could even make this some levels deeper by making the low-level planner operate in symbolic snap space and defer that plan to a low-low-level (basement-level) planner that picks out pixel coordinates and shit.  Eew man, that's nasty.  Maybe good nasty.  Maybe bad nasty.  Won't know until the morning.

How does all this make the planning more general and easy to use?

Oh one more thing, if we do the center-viewport-to-snap thing I've been mulling over in my head, then the number of camera actions are really big yo.  This is kind of similar to the issues I'm having with everything else though.  It seems like I need a special heuristic kind of thing to tell me which ones of these to consider.
'''

'''
There is another argument to make here: just forget all this fancy general nonsense and make one thing that works for reassembly.  Oof.  Just fuggin do it, you know?  Is this the right thing?  Don't mull it over.  Lay it out.  Let's start laying this out, enough meandering.
'''

class GraphSearch:
    def __init__(self, road_map):
        self.road_map = road_map

class RoadMap:
    def __init__(
        self,
        env,
        neighbor_fn,
        check_edge_fn,
    ):
        self.env = env
        self.nodes = set()
        self.edges = {}
        
        # TODO: make these members, and make the user subclass RoadMap?
        self.neighbor_fn = neighbor_fn
        self.check_edge_fn = check_edge_fn
    
    def plan(self, start, goal, max_cost=float('inf')):
        while True:
            path, cost = self.graph_search(start, goal, max_cost=max_cost)
            if path is not None:
                return path, cost
            
            new_nodes = self.expand_graph()
            if not new_nodes:
                raise Exception
    
    def graph_search(self, start, goal, max_cost=max_cost):
        precursors = {}
        frontier = [(0,start)]
        

def plan(
    env,
    start,
    goal,
    neighbor_fn,
    check_edge_fn,
    max_cost=float('inf'),
):
    nodes = {goal}
    edges = {}
    
    # initialize the frontier
    frontier = [(None, start)]
    reachable_nodes = {start}
    
    def pick_from_frontier():
        connected_edges = [
            (a,b) for (a,b) in frontier if b in nodes]
        if connected_edges:
            return random.choice(connected_eges)
        else:
            return random.choice(frontier)
    
    while frontier:
        source, destination = pick_from_frontier()
        nodes.add(destination)
        if source is not None:
            edges[source, destination] = float('inf'), None
        
        if destination not in reachable_nodes:
            path_steps = []
            path_cost = 0.
            
            # this should be an in-place A* update or something
            path = graph_search(start, goal, nodes, edges)
            for s, d in path:
                edge_cost, edge_steps = edges[s,d]
                if edge_steps is None:
                    cost, steps = check_edge_fn(env, s, d)
                    edge[s, d] = cost, steps
                
                path_steps.extend(steps)
                path_cost += cost
                if path_cost >= max_cost:
                    break
                
                reachable_nodes.add(d)
            
            else:
                return path_steps, path_cost
        
        neighbors = neighbor_fn(env, dest)
        for neighbor in neighbors:
            frontier.append((dest, neighbor))
    
    # if we can't find anything, return an empty sequence with infinite cost
    return [], float('inf')

def test_plan():
    '''
        d
       x \
      b---c
       \ x
        a
    '''
    
    
    nodes = set('abcd')
    start = 'a'
    goal = 'b'
    
    def neighbor_fn(env, node):
        if node == 'a':
            return 'b', 'c'
        elif node == 'b':
            return 'c', 'd'
        elif node == 'c':
            return 'b', 'd'
        elif node == 'd':
            return ()
    
    def check_fn(env, s, d):
        if s == 'a' and d == 'b':
            return 0., ['ab.1', 'ab.2']
        elif s == 'a' and d == 'c':
            return float('inf'), []
        
        elif s == 'b' and d == 'd':
            return float('inf'), []
        elif s == 'b' and d == 'c':
            return 0., ['bc.1', 'bc.2']
        
        elif s == 'c' and d == 'b':
            return 0., ['cb.1', 'cb.2']
        elif s == 'c' and d == 'd':
            return 0., ['cd.1', 'cd.2']
    
    plan(None, 'a', 'b', neighbor_fn, check_fn)

if __name__ == '__main__':
    test_plan()

