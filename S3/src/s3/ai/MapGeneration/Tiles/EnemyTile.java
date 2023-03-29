package s3.ai.MapGeneration.Tiles;

public class EnemyTile extends RightBottom {
    public int[] player_position = new int[]{4,2};
    public int[] gold_position = new int[]{5,-2};
    public EnemyTile(){
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
