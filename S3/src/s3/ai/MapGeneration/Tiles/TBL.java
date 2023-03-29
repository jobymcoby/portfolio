package s3.ai.MapGeneration.Tiles;

import s3.ai.MapGeneration.MapTile;

public class TBL extends MapTile {
    public TBL(){
        this.pattern = new boolean[] {true, false, true, true};
        tile = new String[][] {
                {w, w, m, m, m, w, w, w},
                {w, w, w, m, m, w, w, w},
                {w, t, t, m, m, w, w, w},
                {w, m, t, t, m, m, w, t},
                {m, t, t, m, m, t, t, w},
                {m, m, m, m, m, m, t, w},
                {m, m, m, m, m, m, m, w},
                {w, m, m, m, m, m, m, w},
        };
    }
}
