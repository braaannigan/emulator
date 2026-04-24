# Graph/Mesh Batch

Base setup: current 2-layer shifting-wind incumbent family with 30 epochs, periodic eval every 5 epochs, reference comparison retained, and early stopping disabled.

## G01 Grid Graph, Cross Stencil
Hypothesis: Replacing convolutional mixing with explicit 4-neighbor mesh message passing will reduce boundary artifacts by preventing learned filters from implicitly relying on exterior support.

## G02 Grid Graph, Full Stencil
Hypothesis: An 8-neighbor grid-graph stencil will better capture diagonal transport structure than the 4-neighbor mesh graph while still avoiding exterior convolution support.

## G03 Grid Graph With Boundary Features
Hypothesis: Adding explicit boundary features to the graph node updates will help the mesh operator distinguish edge dynamics from interior dynamics.

## G04 Grid Graph With Global Context
Hypothesis: Broadcasting global pooled context into each graph update will improve long-range coordination that a purely local mesh message-passing operator might miss.

## G05 Full Stencil Graph With Boundary Features And Global Context
Hypothesis: Combining richer local connectivity with explicit boundary and global context will produce the strongest graph-based baseline for this problem family.
