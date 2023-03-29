package s3.ai.MapGeneration.Tiles;

import s3.ai.MapGeneration.MapTile;

public class RightTop extends MapTile {
    public RightTop(){
        this.pattern = new boolean[] {true, true, false, false};
        tile = new String[][] {
                {w, m, m, m, m, m, m, m},
                {w, t, m, m, m, m, m, m},
                {w, t, m, m, m, m, m, m},
                {w, t, t, m, m, t, m, m},
                {w, m, t, t, t, m, m, m},
                {w, t, w, t, t, m, m, m},
                {w, w, w, w, w, w, m, m},
                {w, w, w, w, w, w, w, w},
        };
    }
}

