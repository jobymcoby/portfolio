package s3.ai.MapGeneration.Tiles;

import s3.ai.MapGeneration.MapTile;

public class TRL extends MapTile {
    public TRL(){
        this.pattern = new boolean[] {true, true, false, true};
        tile = new String[][] {
                {w, w, m, m, m, w, w, w},
                {w, w, w, m, m, w, w, t},
                {m, t, t, t, m, w, w, m},
                {m, m, t, t, m, t, w, t},
                {m, t, t, m, m, t, t, m},
                {m, m, m, m, t, m, m, m},
                {t, m, m, m, m, t, t, w},
                {w, w, w, w, w, w, w, w},
        };
    }
}
