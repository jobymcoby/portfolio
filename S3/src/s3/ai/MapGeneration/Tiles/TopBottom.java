package s3.ai.MapGeneration.Tiles;

import s3.ai.MapGeneration.MapTile;

import java.util.ArrayList;
import java.util.List;

public class TopBottom extends MapTile {
    String[][] tile1 = new String[][] {
        {w, w, m, m, m, m, w, w},
        {w, w, m, m, m, t, w, w},
        {w, t, m, m, m, m, w, w},
        {w, m, m, m, m, m, w, w},
        {w, m, m, m, m, m, w, t},
        {w, t, m, m, m, m, w, w},
        {w, m, m, m, m, t, m, w},
        {w, m, m, m, m, m, m, w},
    };
    String[][] tile2 = new String[][] {
        {w, w, m, m, m, m, w, w},
        {w, w, m, m, m, t, w, w},
        {w, w, w, m, m, m, w, w},
        {w, w, w, m, m, w, w, w},
        {w, m, m, m, t, w, w, t},
        {w, t, m, m, m, m, w, w},
        {w, m, m, m, m, t, m, w},
        {w, m, m, m, m, m, m, w},
    };

    List<String[][]> tiles = new ArrayList<>();

    @Override
    public String[][] getTile() {
        return tiles.get(rand.nextInt(tiles.size()));
    }

    public TopBottom(){
        tiles.add(tile1);
        tiles.add(tile2);
        this.pattern = new boolean[] {true, false, true, false};
        this.tile = tiles.get(rand.nextInt(tiles.size()));
    }
}
