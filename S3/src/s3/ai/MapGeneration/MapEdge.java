package s3.ai.MapGeneration;

public class MapEdge implements Comparable<MapEdge>{
    public MapNode src, dst;
    int weight;

    public MapEdge(){
    }

    public MapEdge(MapNode src, MapNode dst, int weight){
        this.src = src;
        this.dst = dst;
        this.weight = weight;
    }

    @Override
    public int compareTo(MapEdge o) {
        return this.weight - o.weight;
    }

    @Override
    public String toString() {
        return "MapEdge{" +
                "src=" + src +
                ", dst=" + dst +
                ", weight=" + weight +
                '}';
    }
}
