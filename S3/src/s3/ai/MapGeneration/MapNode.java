package s3.ai.MapGeneration;

import java.util.Objects;

public class MapNode {
    private int pos_x;
    private int pos_y;
    private static int _id = 0;
    private int id;
    private MapTile tile;


    // pattern will follow TRBL order. Set all to false will return island
    private boolean[] pattern = new boolean[] {false, false, false, false};

    public MapNode(int x, int y){
        pos_x = x;
        pos_y = y;
        id = _id;
        _id++;
    }

    public int getPos_x() {
        return pos_x;
    }

    public int getPos_y() {
        return pos_y;
    }

    public int getId() {
        return id;
    }

    public boolean[] getPattern() {
        return pattern;
    }

    public void setPattern(int other_id) {
        if (other_id == this.id - 1){
            //up
            pattern[0] = true;
        }
        if (other_id == this.id + 1){
            //down
            pattern[2] = true;
        }
        if (other_id > this.id + 1){
            // right
            pattern[1] = true;
        }
        if (other_id < this.id - 1){
            //left
            pattern[3] = true;
        }
    }

    public MapTile getTile() {
        return tile;
    }

    public void setTile(MapTile tile) {
        this.tile = tile;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        MapNode mapNode = (MapNode) o;
        return pos_x == mapNode.pos_x && pos_y == mapNode.pos_y;
    }

    @Override
    public int hashCode() {
        return Objects.hash(pos_x, pos_y);
    }

    @Override
    public String toString() {
        return "MapNode{" +
                "pos_x=" + pos_x +
                ", pos_y=" + pos_y +
                ", id=" + id +

                '}';
    }

    public int toInt(){
        return id;
    }
}
