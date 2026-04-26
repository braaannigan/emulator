# Graph V2 Batch

Base setup: current 2-layer shifting-wind graph `g03` line, extended to richer graph message passing while keeping 30 epochs, periodic eval every 5 epochs, reference comparison retained, and early stopping disabled.

## V01 Rich Node Features
Hypothesis: Adding richer spatial node features (`x`, `y`, edge mask, and directional boundary distances) will let the graph operator distinguish boundary, corner, and interior roles more cleanly than the first-pass boundary-feature graph.

## V02 Rich Node Features With Deeper Message Passing
Hypothesis: Keeping the richer node features but increasing graph depth will improve multi-hop propagation and reduce the need for the decoder to reconstruct long-range structure from shallow graph updates.

## V03 Gated Messages With Edge-Type Embeddings
Hypothesis: Learned message gating plus explicit edge-type embeddings will let the graph operator weight north/south/east/west interactions differently and suppress harmful neighbor messages near unstable regions.

## V04 Boundary-Gated Graph Updates
Hypothesis: Applying a boundary-conditioned gate to the graph updates will let the operator treat edge-adjacent cells differently from interior cells without forcing a separate boundary model.

## V05 Deeper Boundary-Aware Graph With Global Context
Hypothesis: Combining richer node features, gated edge-aware messages, deeper propagation, and broadcast global context will give the graph model enough local and global structure to compete more directly with the incumbent convolutional backbone.
