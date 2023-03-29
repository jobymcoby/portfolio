package s3.ai.MapGeneration.Tiles;

import s3.ai.MapGeneration.MapTile;

public class Top extends MapTile {
    public Top(){
        this.pattern = new boolean[] {true, false, false, false};
        tile = new String[][] {
                {w, w, m, m, m, m, w, w},
                {w, w, m, m, m, t, w, w},
                {w, t, m, m, m, m, w, w},
                {w, m, m, m, m, m, w, w},
                {w, m, m, m, m, m, w, t},
                {w, t, t, t, t, w, w, w},
                {w, w, w, w, w, w, w, w},
                {w, w, w, w, w, w, w, w},
        };
    }
}
