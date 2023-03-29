There are 3 main parts of this game I coded. 
1. Pathfinding: A* algorithm that will help AI units move more efficently. 
2. Strategy: The rule base iteration system
3. PCG: Procedural map generation

The rule base system will gather information about the game state and add it to our knowledge base. 
Then if we find a unification in the knowledge base for a rule, we gather it to later arbitrate. The rule that make it arbitration are fired in the game state.

The PGC System uses kruskal algorithm to create a random MST and then creates a map with tiles that have the same connections. 
Some tiles have sets of which a random one is choosen. This can add cycles back into the MST if the players harvest the trees to create a path
