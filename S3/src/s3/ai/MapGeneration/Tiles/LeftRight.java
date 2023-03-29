package s3.ai.MapGeneration.Tiles;

import s3.ai.MapGeneration.MapTile;

import java.util.ArrayList;
import java.util.List;

public class LeftRight extends MapTile {
    String[][] tile1 =  new String[][] {
            {t, w, w, w, w, w, w, w},
            {w, w, t, m, t, t, w, w},
            {m, m, m, m, m, m, m, m},
            {m, m, t, m, m, m, m, m},
            {m, m, m, m, m, m, m, m},
            {m, t, t, t, t, w, w, m},
            {w, t, t, t, t, w, w, t},
            {w, w, t, t, w, w, w, t},
    };
    String[][] tile2 =  new String[][] {
            {w, w, w, w, w, w, w, w},
            {w, w, w, m, t, t, w, w},
            {m, m, m, m, m, m, m, m},
            {m, m, t, m, m, m, m, m},
            {m, m, m, m, m, m, m, m},
            {m, t, t, t, t, w, w, m},
            {w, w, w, w, w, w, w, t},
            {w, w, w, w, w, w, w, w},
    };

    List<String[][]> tiles = new ArrayList<>();

    @Override
    public String[][] getTile() {
        return tiles.get(rand.nextInt(tiles.size()));
    }

    public LeftRight(){
        tiles.add(tile1);
        tiles.add(tile2);
        this.pattern = new boolean[] {false, true, false, true};
        this.tile = tiles.get(rand.nextInt(tiles.size()));
    }
}
