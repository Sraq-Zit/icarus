
# TODO
- [x] Implement size-based caching in (policies.py)
- [x] Make size constraints in caching policy (Contents size mapping in Cache class (policies.py))
- [x] Caclulate popularity of contents (Content popularity mapping in workload)
- [x] Gather all necessary parameters/results in the controller
  - state
    - cache status of node (model.cache\[node]) 
    - request (engine.py -> event)
  - reward
    - popularity (workload.popularity)
    - cache status
      - neighbors (model.topology.adj\[node])
      - cache status of node (model.cache\[node])
- [ ] Integerate Qlearning..


# Notes

- (Problem: calculating cost) The cost for content delivery could be calculated from the Controller  (network.py -> NetworkController) since the path to the content is handles when executing an event (onpath?.py)

- (Problem: simulating content size) The size for contents could be handled in Cache class (policies.py). by implementing methods (get, put, ...) we can create a caching and replacement policy based on different constraints as well as cache size

- (Problem: accessing results) All needed results can be accessible through the NetworkController (network.py) which makes it the best place to store results which were not stored by default
