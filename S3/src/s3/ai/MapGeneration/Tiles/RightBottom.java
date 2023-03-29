package s3.ai.MapGeneration.Tiles;

import s3.ai.MapGeneration.MapTile;

public class RightBottom extends MapTile {
    public RightBottom(){
        this.pattern = new boolean[] {false, true, true, false};
        tile = new String[][] {
            {w, w, w, w, w, m, m, m},
            {w, w, w, w, t, m, m, m},
            {w, t, w, t, m, m, m, m},
            {w, m, t, t, t, m, m, m},
            {w, t, t, m, m, m, m, m},
            {w, t, m, m, m, m, m, m},
            {w, t, m, m, m, m, m, m},
            {w, m, m, m, m, m, m, m},
        };
    }
}
