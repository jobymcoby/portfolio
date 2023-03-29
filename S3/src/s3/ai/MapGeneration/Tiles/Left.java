package s3.ai.MapGeneration.Tiles;

import s3.ai.MapGeneration.MapTile;

public class Left extends MapTile {
    public Left(){
        this.pattern = new boolean[] {false, false, false, true};
        this.tile = new String[][] {
                {w, w, w, w, w, w, w, w},
                {m, m, m, m, t, t, w, w},
                {m, m, m, m, m, m, w, w},
                {m, m, t, m, m, m, w, w},
                {m, m, m, m, m, m, w, t},
                {m, m, t, t, t, w, w, w},
                {m, m, m, w, w, w, w, w},
                {w, w, w, w, w, w, w, w},
        };
    }
}
