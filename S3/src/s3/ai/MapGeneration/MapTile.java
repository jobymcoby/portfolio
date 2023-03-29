package s3.ai.MapGeneration;

import java.util.Random;

public class MapTile {
    public Random rand = new Random();
    public String[][] tile = new String[8][8];
    public boolean[] pattern = new boolean[4];

    public String w = "w";
    public String t = "t";
    public String m = ".";

    public String[][] getTile() {
        return tile;
    }

    public MapTile() {

    }

    public MapTile(MapTile another) {
        this.tile = another.tile;
        this.pattern = another.pattern;
    }

}
