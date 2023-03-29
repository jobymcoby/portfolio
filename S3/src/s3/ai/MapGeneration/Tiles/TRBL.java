package s3.ai.MapGeneration.Tiles;

import s3.ai.MapGeneration.MapTile;

public class TRBL extends MapTile {
    public TRBL(){
        this.pattern = new boolean[] {true, true, true, true};
        tile = new String[][] {
                {w, m, m, m, m, t, w, w},
                {w, m, t, t, t, w, w, t},
                {m, m, t, t, t, w, w, t},
                {m, m, t, t, t, t, w, t},
                {m, t, t, m, m, t, t, m},
                {m, m, m, m, m, m, m, m},
                {t, m, m, m, m, t, t, w},
                {w, w, t, m, m, t, w, w},
        };
    }
}
