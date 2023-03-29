package s3.ai.MapGeneration.Tiles;

import s3.ai.MapGeneration.MapTile;

public class TRB extends MapTile {
    public TRB(){
        this.pattern = new boolean[] {true, true, true, false};
        tile = new String[][] {
                {w, w, m, m, m, w, w, w},
                {w, w, w, m, m, w, w, t},
                {w, t, t, m, m, w, w, m},
                {w, m, t, m, m, t, t, m},
                {w, t, t, m, m, t, m, m},
                {w, m, m, m, m, m, m, m},
                {w, m, m, m, m, t, m, w},
                {w, m, m, m, m, m, t, w},
        };
    }
}
