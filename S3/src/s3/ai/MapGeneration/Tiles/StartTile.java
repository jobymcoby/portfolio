package s3.ai.MapGeneration.Tiles;

public class StartTile extends RightBottom {
    public int[] player_position = new int[]{1,2};
    public int[] gold_position = new int[]{0,-1};
    public StartTile(){
        super();
        tile = new String[][] {
                {m, m, m, m, m, m, m, m},
                {m, m, m, m, m, m, m, m},
                {m, m, m, m, m, m, m, m},
                {m, m, m, m, m, m, m, m},
                {m, m, m, m, m, m, m, m},
                {m, m, m, m, m, m, m, m},
                {m, m, m, m, m, m, m, m},
                {m, m, m, m, m, m, m, m},
        };
    }
}
