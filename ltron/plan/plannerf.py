from numpy.random import choice

'''
Thoughts:

So this is the 5th time I've been here.  I want a simpler thing that works with the expert I've been using successfully for small problems.

If we move the collision map into the expert, then we can maybe just get away
with doing this as a one-stage plan.  Then we would have a single graph of nodes (env states) and actions (integers under then new action space rules).

Actually, no, everything is a tree (we don't worry about merging graph nodes) and so a state can also be represented as a list of actions (integers).

Now, there's still something missing: visual search.  When we need to place something, but can't from the current viewpoint, we need a way to search for a way to do it.  Previously, we did an exhaustive search through camera angles, but I want to relax that because I want to support camera action spaces with translations finally.

So before what we did is have a budget for camera motions per path length or something like that.  We could do this again, and just randomly search when we can't find something.

Before I think there was some kind of "cost" associated with each path which was the number of camera moves/high level step or something like that, along with a tradeoff about preferring the longest path in order to make progress.  That tradeoff never seemed satisfactorily addressed.

Now I can punt on this, since there's no camera motion in the symbolic space, so actually even the expert is good enough there.  But I also want to get back to the visual space soon.  Soon soon.  And I kind of want to get the whole structure roughed out too.

So what does this look like?
'''

def plan(
    env,
    start_state,
    viewpoint_cost=0.5,
    progress_reward=1.0
):
    
    action_history = []
    score = 0.
    observation = env.set_state(start_state)
    leaf_actions = observation['expert']
    
    frontier = []
    
    while True:
        
        # make new leaves
        state = env.get_state()
        for action in leaf_actions:
            if env.is_finished_action(action):
                return action_history + [action]
            elif env.is_viewpoint_action(action):
                leaf_score = score - viewpoint_cost
            else:
                leaf_score = score + progress_reward
            new_leaf = (action_history, state, action, leaf_score)
            frontier.append(new_leaf)
        
        # randomly sample from the frontier according to a softmax
        p_frontier = [math.exp(score) for _, _, _, score in frontier]
        p_sum = sum(p_expand)
        p_frontier = [p/p_sum for p in p_frontier]
        leaf_index = choice(range(len(frontier)), 1, p_frontier)
        action_history, state, next_action, score = frontier.pop(leaf_index)
        
        # set the state
        env.set_state(state)
        observation, reward, terminal, info = env.step(next_action)
        
        # get expert actions
        leaf_actions = observation['expert']
