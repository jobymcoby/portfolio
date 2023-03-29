package s3.ai.MapGeneration.Tiles;

import s3.ai.MapGeneration.MapTile;

public class Bottom extends MapTile {
    public Bottom(){
        this.pattern = new boolean[] {false, false, true, false};
        tile = new String[][] {
                {w, w, w, w, w, w, w, w},
                {w, w, w, t, t, w, t, w},
                {w, t, t, t, t, w, t, t},
                {w, m, m, m, m, m, t, t},
                {w, m, m, m, m, m, t, w},
                {w, t, m, m, m, m, t, w},
                {w, w, m, m, m, m, m, w},
                {w, m, m, m, m, m, m, w},
        };
    }
}
