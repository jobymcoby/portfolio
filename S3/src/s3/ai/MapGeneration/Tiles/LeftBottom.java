package s3.ai.MapGeneration.Tiles;

import s3.ai.MapGeneration.MapTile;

public class LeftBottom extends MapTile {
    public LeftBottom(){
        this.pattern = new boolean[] {false, false, true, true};
        tile = new String[][] {
                {w, w, w, w, w, w, w, w},
                {w, w, w, w, w, w, w, w},
                {w, t, t, t, t, w, w, w},
                {w, m, t, t, t, t, w, t},
                {m, t, t, m, m, t, t, w},
                {m, m, m, m, m, m, t, w},
                {m, m, m, m, m, m, m, w},
                {m, m, m, m, m, m, m, w},
        };
    }
}
