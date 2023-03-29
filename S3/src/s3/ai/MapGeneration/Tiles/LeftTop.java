package s3.ai.MapGeneration.Tiles;

import s3.ai.MapGeneration.MapTile;
public class LeftTop extends MapTile {
    public LeftTop(){
        this.pattern = new boolean[] {true, false, false, true};
        tile = new String[][] {
                {m, m, m, m, m, t, t, w},
                {m, m, m, m, m, m, t, w},
                {m, m, m, m, m, m, t, w},
                {m, t, t, m, m, t, t, w},
                {m, m, m, m, t, t, w, t},
                {w, m, t, t, t, w, w, w},
                {m, m, w, w, w, w, w, w},
                {w, w, w, w, w, w, w, w},
        };
    }
}
