package s3.ai.MapGeneration.Tiles;

import s3.ai.MapGeneration.MapTile;

public class RBL extends MapTile {
    public RBL(){
        this.pattern = new boolean[] {false, true, true, true};
        tile = new String[][] {
                {w, w, w, w, w, w, w, w},
                {w, w, w, t, t, w, w, t},
                {m, t, t, t, t, w, w, m},
                {m, m, t, t, t, t, w, t},
                {m, t, t, m, m, t, t, m},
                {m, m, m, m, m, m, m, m},
                {t, m, m, m, m, t, t, w},
                {w, w, t, m, m, t, w, w},
        };
    }
}
