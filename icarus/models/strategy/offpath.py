"""Implementations of all off-path strategies"""
from __future__ import division

import networkx as nx

from icarus.registry import register_strategy
from icarus.util import inheritdoc, path_links

from .base import Strategy

import random

__all__ = [
       'NearestReplicaRouting'
           ]


@register_strategy('NRR')
class NearestReplicaRouting(Strategy):
    """Ideal Nearest Replica Routing (NRR) strategy.

    In this strategy, a request is forwarded to the topologically closest node
    holding a copy of the requested item. This strategy is ideal, as it is
    implemented assuming that each node knows the nearest replica of a content
    without any signaling

    On the return path, content can be caching according to a variety of
    metacaching policies. LCE and LCD are currently supported.
    """

    def __init__(self, view, controller, metacaching, implementation='ideal',
                 radius=4, **kwargs):
        """Constructor

        Parameters
        ----------
        view : NetworkView
            An instance of the network view
        controller : NetworkController
            An instance of the network controller
        metacaching : str (LCE | LCD)
            Metacaching policy used
        implementation : str, optional
            The implementation of the nearest replica discovery. Currently on
            ideal routing is implemented, in which each node has omniscient
            knowledge of the location of each content.
        radius : int, optional
            Radius used by nodes to discover the location of a content. Not
            used by ideal routing.
        """
        super(NearestReplicaRouting, self).__init__(view, controller)
        if metacaching not in ('LCE', 'LCD'):
            raise ValueError("Metacaching policy %s not supported" % metacaching)
        if implementation not in ('ideal', 'approx_1', 'approx_2'):
            raise ValueError("Implementation %s not supported" % implementation)
        self.metacaching = metacaching
        self.implementation = implementation
        self.radius = radius
        self.distance = dict(nx.all_pairs_dijkstra_path_length(self.view.topology(),
                                                               weight='delay'))

    @inheritdoc(Strategy)
    def process_event(self, time, receiver, content, log):
        # get all required data
        locations = self.view.content_locations(content)
        nearest_replica = min(locations, key=lambda x: self.distance[receiver][x])
        # Route request to nearest replica
        self.controller.start_session(time, receiver, content, log)
        if self.implementation == 'ideal':
            self.controller.forward_request_path(receiver, nearest_replica)
        elif self.implementation == 'approx_1':
            # Floods actual request packets
            paths = {loc: len(self.view.shortest_path(receiver, loc)[:self.radius])
                     for loc in locations}
            # TODO: Continue
            raise NotImplementedError("Not implemented")
        elif self.implementation == 'approx_2':
            # Floods meta-request packets
            # TODO: Continue
            raise NotImplementedError("Not implemented")
        else:
            # Should never reach this block anyway
            raise ValueError("Implementation %s not supported"
                             % str(self.implementation))
        self.controller.get_content(nearest_replica)
        # Now we need to return packet and we have options
        path = list(reversed(self.view.shortest_path(receiver, nearest_replica)))
        if self.metacaching == 'LCE':
            for u, v in path_links(path):
                self.controller.forward_content_hop(u, v)
                if self.view.has_cache(v) and not self.view.cache_lookup(v, content):
                    self.controller.put_content(v)
        elif self.metacaching == 'LCD':
            copied = False
            for u, v in path_links(path):
                self.controller.forward_content_hop(u, v)
                if not copied and v != receiver and self.view.has_cache(v):
                    self.controller.put_content(v)
                    copied = True
        else:
            raise ValueError('Metacaching policy %s not supported'
                             % self.metacaching)
        self.controller.end_session()

@register_strategy('AIE')
class AI_EMPOWERED(Strategy):
    """AI Empowered (AIE) strategy.

    In this strategy a copy of a content is replicated at the cache the
    AI model provided as a decision to put the content in.
    """

    @inheritdoc(Strategy)
    def __init__(self, view, controller, **kwargs):
        super(AI_EMPOWERED, self).__init__(view, controller)
        self.old_reward = -1e6

    @inheritdoc(Strategy)
    def process_event(self, time, receiver, content, log):
        # get all required data
        # print(self.old_reward)
        serving_node = None
        source = self.view.content_source(content)
        path = self.view.shortest_path(receiver, source)
        rsu = path[1]

        self.controller.start_session(time, receiver, content, log)

        if self.view.cache_lookup(rsu, content): 
            self.controller.get_content(rsu)
            serving_node = rsu

        # Check neighbors
        if serving_node == None:
            for node in self.controller.get_neigbbors(rsu):
                if node != source:
                    if self.view.cache_lookup(node, content):
                        self.controller.get_content(rsu)
                        self.controller.get_content(node)
                        serving_node = node
                        break
                
        # Route requests to original source and queries caches on the path
        if serving_node == None:
            for u, v in path_links(path):
                self.controller.forward_request_hop(u, v)
                if self.view.has_cache(v):
                    if self.controller.get_content(v):
                        serving_node = v
                        break
                # No cache hits, get content from source
                self.controller.get_content(v)
                serving_node = v
        # Return content
        path = list(reversed(self.view.shortest_path(receiver, serving_node)))
        for u, v in path_links(path):
            self.controller.forward_content_hop(u, v)

        self.controller.put_content(rsu)

        state = self.controller.get_state(rsu, content)
        action = self.controller.get_best_action(rsu, state)
        action_converted = self.controller.convert_action(action)
        if action_converted[content-1]:
            policy = [n+1 for n, p in enumerate(action_converted) if p and n+1 != content]
            self.controller.set_replacement_candidates(rsu, policy)
            self.controller.put_content(rsu)

            reward_after = self.controller.get_avg_reward()
                
            if reward_after < self.old_reward:
                self.controller.revert_back_cache(rsu)
            else:
                self.old_reward = reward_after
        
        self.controller.train_model(rsu, state)
        self.controller.save_observation(rsu, state, action, self.controller.get_reward(rsu))
        self.controller.end_session()

