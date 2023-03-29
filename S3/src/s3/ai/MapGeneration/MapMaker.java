package s3.ai.MapGeneration;

import s3.ai.MapGeneration.Tiles.*;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class MapMaker {
    MapTile[] tiles = new MapTile[]{
            new Top(),
            new Right(),
            new Bottom(),
            new Left(),
            new LeftTop(),
            new LeftBottom(),
            new LeftRight(),
            new RightTop(),
            new RightBottom(),
            new TopBottom(),
            new TRB(),
            new TRL(),
            new TBL(),
            new RBL(),
            new TRBL(),
    };

    int n,m;
    MapEdge[] edges;
    public MapNode[] nodes;

    public MapMaker(int n, int m){
        this.n = n;
        this.m = m;
        this.createGraph(this.n, this.m);
    }

    public void createGraph(int x, int y){
        List<MapNode> graph = new ArrayList<>();
        Random random = new Random();

        // instantiate Nodes
        for (int i = 0; i < x; i++) {
            for (int j = 0; j < y; j++) {
                // create a node
                MapNode a = new MapNode(i,j);
                graph.add(a);
            }
        }

        // create edges
        List<MapEdge> map_edges = new ArrayList<>();
        for (MapNode node : graph){
            List<MapEdge> node_edges = new ArrayList<>();
            MapNode test_right = new MapNode(node.getPos_x() + 1, node.getPos_y());

            if (graph.contains(test_right)){
                MapEdge right_edge = new MapEdge(node, graph.get(graph.indexOf(test_right)), random.nextInt(4));
                node_edges.add(right_edge);
            }
            MapNode test_up = new MapNode(node.getPos_x(), node.getPos_y() + 1);
            if (graph.contains(test_up)){
                MapEdge up_edge = new MapEdge(node, graph.get(graph.indexOf(test_up)), random.nextInt(4));
                node_edges.add(up_edge);
            }
            map_edges.addAll(node_edges);
        }


        this.edges = new MapEdge[map_edges.size()];
        this.edges = map_edges.toArray(this.edges);

        this.nodes = new MapNode[graph.size()];
        this.nodes = graph.toArray(this.nodes);
    }

    class subset {
        // this holds the parent for the entire subset of nodes
        // path compression is better to detect cycle.
        // runs in almost constant time
        public MapNode parent;
        public int rank;

        public subset(MapNode parent){
            this.parent = parent;
            rank = 0;
        }

        @Override
        public String toString() {
            return "subset{" +
                    "parent=" + parent +
                    ", rank=" + rank +
                    '}';
        }
    }

    MapNode find(subset[] subsets, int id)
    {
        // follow up the chain of parents to find the root node
        if (subsets[id].parent.getId() != id)
            subsets[id].parent = find(subsets, subsets[id].parent.getId());

        return subsets[id].parent;
    }

    void Union(subset[] subsets, int x, int y)
    {
        // Here we want to update the subsets with the parent information
        // to show union between the node with id:x and the node with id:y
        // we get the root parent of the nodes we want to join
        MapNode xroot = find(subsets, x);
        MapNode yroot = find(subsets, y);

        // add lower rank to higher rank
        if (subsets[xroot.getId()].rank < subsets[yroot.getId()].rank)
            subsets[xroot.getId()].parent = yroot;
        else if (subsets[xroot.getId()].rank > subsets[yroot.getId()].rank)
            subsets[yroot.getId()].parent = xroot;

            // If ranks are same, then make one as
            // root and increment its rank by one
        else {
            subsets[yroot.getId()].parent = xroot;
            subsets[xroot.getId()].rank++;
        }
    }

    public MapEdge[] Kruskal() {

        int V = nodes.length;
        // V-1 edges in a MST
        int E = V - 1;

        MapEdge[] result = new MapEdge[E];

        for (int i = 0; i < E; ++i)
            result[i] = new MapEdge();

        // Initialize a subset for each vertex
        // it will have the rank 0 and will be its own parent
        subset[] subsets = new subset[V];
        for (int i = 0; i < V; ++i)
            subsets[i] = new subset(nodes[i]);

        // use comparator to find the smallest edge first
        Arrays.sort(edges);

        int e = 0;
        int i = 0;
        while (e < E) {

            MapEdge next_edge = edges[i];
            i++;
            // find the parent of the indexed MapNode
            MapNode x = find(subsets, next_edge.src.getId());
            MapNode y = find(subsets, next_edge.dst.getId());

            // If the parents are the same we have a cycle, and we don't want this edge
            // if not we add the edge to MST and join the x and y with Union
            if (x != y) {
                result[e] = next_edge;
                e++;
                // updates subsets with new parent info
                Union(subsets, x.getId(), y.getId());
            }
        }
        return result;
    }

    public List<String> MapString(){
        for (MapNode node: this.nodes)
        {
            for (MapTile map_tile: tiles){

                //System.out.println(Arrays.deepToString(map_tile.tile));
                if (Arrays.equals(map_tile.pattern, node.getPattern())){
                    if (node.getTile() == null)
                        node.setTile(map_tile);
                    break;
                }
            }
        }


        List<String> string_map = new ArrayList<>();
        for (int j = 0; j < m; j++) {
            String[] string_row = new String[8];
            for (int i = 0; i < n; i++){
                String[][] curr_tile = nodes[i * m + j].getTile().getTile();
                for (int row = 0; row < curr_tile.length; row++){

                    String curr_row = String.join("",curr_tile[row]);
                    if (string_row[row] != null){
                        string_row[row] = string_row[row].concat(curr_row);
                    }
                    else {
                        string_row[row] = curr_row;
                    }
                    //System.out.println(Arrays.toString(curr_tile[row]));
                }
            }
            string_map.addAll(Arrays.asList(string_row));
        }
        return string_map;
    }

    public static void main(){
        int n= 6;
        int m= 5;

        MapMaker maker = new MapMaker(n,m);
        MapEdge[] result = maker.Kruskal();
        // With these map edges we can now classify our nodes with a tile pattern
        for(MapEdge edge : result){
            // set pattern in the source and destination
            edge.src.setPattern(edge.dst.getId());
            edge.dst.setPattern(edge.src.getId());
        }

        StartTile start = new StartTile();
        //players alwasy start upper left side
        maker.nodes[0].setTile(start);
        int[] player1_start = new int[]{
                start.player_position[0],
                start.player_position[1],
                start.gold_position[0],
                start.gold_position[1],
        };

        EnemyTile enemy = new EnemyTile();
        int enemy_num = 0;
        int[] player2_start = new int[0];
        List<MapNode> enemy_zones = new ArrayList<>();
        for (MapNode node: maker.nodes){
            if(node.getPos_x() == n-1 || node.getPos_y() == m-1){
                enemy_zones.add(node);
            }
        }
        Random rand = new Random();
        MapNode randomElement = enemy_zones.get(rand.nextInt(enemy_zones.size()));
        randomElement.setTile(enemy);
        player2_start = new int[]{
            enemy.player_position[0] + 8 * randomElement.getPos_x(),
            enemy.player_position[1] + 8 * randomElement.getPos_y(),
            enemy.gold_position[0] + 8 * randomElement.getPos_x(),
            enemy.gold_position[1] + 8 * randomElement.getPos_y(),
        };

        List<String> tile_map = maker.MapString();

        XMLmap.write_mapfile(n*8, m*8, maker.nodes, tile_map, player1_start, player2_start);
    }

}
