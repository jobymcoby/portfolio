package s3.ai.MapGeneration.Tiles;

import s3.ai.MapGeneration.MapTile;

import java.util.ArrayList;
import java.util.List;

public class Right extends MapTile {
    String[][] tile1 =new String[][] {
            {w, w, w, w, w, w, w, w},
            {w, w, w, m, t, t, w, w},
            {w, w, m, m, m, m, m, m},
            {w, w, t, m, m, m, m, m},
            {w, w, m, m, m, m, m, t},
            {w, w, t, t, t, w, m, m},
            {w, w, w, w, w, w, w, w},
            {w, w, w, w, w, w, w, w},
    };
    String[][] tile2 = new String[][] {
            {w, w, t, t, w, w, w, w},
            {w, t, t, m, t, t, w, w},
            {w, t, m, m, m, m, m, m},
            {t, t, t, m, m, m, m, m},
            {t, t, m, m, m, m, m, t},
            {w, t, t, t, t, w, m, m},
            {w, t, t, t, w, t, w, w},
            {w, w, w, w, w, t, t, w},
    };

    List<String[][]> tiles = new ArrayList<>();

    @Override
    public String[][] getTile() {
        return tiles.get(rand.nextInt(tiles.size()));
    }

    public Right(){

        tiles.add(tile1);
        tiles.add(tile2);
        this.pattern = new boolean[] {false, true, false, false};
        tile = tiles.get(rand.nextInt(tiles.size()));
    }
}
