package s3.ai.strategy;

import s3.base.S3;
import s3.base.S3Action;
import s3.entities.*;

import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

public class Perception {
    static int[] player_start_location = null;
    static float ratio = 3/4f;

    public static void obtain_knowledge(S3 game, KnowledgeBase kb, String m_playerID){
        // On each cycle we would like to update the knowledge base with the new info about the game state
        // this is a way to represent the game state more efficiently.
        //    # auxiliary predicates:
        //    ownBase(X) :- type(X,WTownhall),own(X).
        //    ownWorker(X) :- type(X,WPeasant),own(X).
        //    ownBarrack(X) :- type(X,WBarracks),own(X).
        //    idleWorker(X) :- type(X,WPeasant),own(X),idle(X).
        //    idleLight(X) :- type(X,WFootman),own(X),idle(X).
        //
        //    # inferred from auxiliary predicates (no longer used)
        //    workerNeeded() :- ~ownWorker(X).
        //    bestMine(X) :- closest Mine to the worker.

        HashMap<String, Integer> count_player1 = new HashMap<String, Integer>();
        for (S3Entity unit : game.getAllUnits()) {
            // unit knowledge
            //Type(x,y)
            String unit_type = unit.getClass().getSimpleName();
            kb.addTerm(new Term("type", Integer.toString(unit.entityID), unit_type));

            if(Objects.equals(unit.getOwner(), m_playerID)){
                if (count_player1.containsKey(unit_type)) {
                    count_player1.put(unit_type, count_player1.get(unit_type) + 1);
                } else {
                    count_player1.put(unit_type, 1);
                }
            }


            //goldAvailable(x)
            //woodAvailable(y)
            if (unit instanceof WPlayer unit1){
                if(Objects.equals(unit1.getOwner(), m_playerID)){
                    kb.addTerm(new Term("goldAvailable", Integer.toString(unit1.getGold())));
                    kb.addTerm(new Term("woodAvailable", Integer.toString(unit1.getWood())));
                }
            }
            else if(unit instanceof WUnit unit2) {
                if (Objects.equals(unit2.getOwner(), m_playerID)) {
                    //own(x)
                    kb.addTerm(new Term("own", Integer.toString(unit.entityID)));
                    if (player_start_location == null) {
                        player_start_location = new int[2];
                        player_start_location[0] = unit2.getX();
                        player_start_location[1] = unit2.getY();
                    }
                    //idle(x)
                    if (unit2.getStatus() == null) {
                        kb.addTerm(new Term("idle", Integer.toString(unit.entityID)));
                        if (unit2.getClass().getSimpleName().equals("WPeasant")){
                            kb.addTerm(new Term("idleWorker", Integer.toString(unit.entityID)));
                        }
                        if (unit2.getClass().getSimpleName().equals("WFootman")){
                            kb.addTerm(new Term("idleLight", Integer.toString(unit.entityID)));
                        }
                    }

                    if (unit2.getClass().getSimpleName().equals("WTownhall")){
                        kb.addTerm(new Term("ownBase", Integer.toString(unit.entityID)));
                    }
                    if (unit2.getClass().getSimpleName().equals("WPeasant")){
                        kb.addTerm(new Term("ownWorker", Integer.toString(unit.entityID)));
                    }
                    if (unit2.getClass().getSimpleName().equals("WBarracks")){
                        kb.addTerm(new Term("ownBarrack", Integer.toString(unit.entityID)));
                    }
                }
                else {
                    //enemy(x)
                    if (!(unit2 instanceof WGoldMine)) {
                        kb.addTerm(new Term("enemy", Integer.toString(unit.entityID)));
                    }
                }
            }
        }

        // try this
        // game.locateNearestMapEntity();
        int closest_mine_id = 0;
        double min_distance = Double.POSITIVE_INFINITY;
        for (S3Entity unit : game.getAllUnits()){
            if (unit instanceof WGoldMine unit1){
                double ac = Math.abs(player_start_location[0] - unit1.getX());
                double cb = Math.abs(player_start_location[1] - unit1.getY());
                double distance = Math.hypot(ac, cb);
                if (distance < min_distance && unit1.remaining_gold > 0){
                    min_distance = distance;
                    closest_mine_id = unit1.entityID;
                }
            }
        }
        kb.addTerm(new Term("ownMine", Integer.toString(closest_mine_id)));

        int miners = 0;
        int loggers = 0;
        int max_id = 0;
        for(S3Entity e:game.getAllUnits()) {
            if (e instanceof WPeasant && e.getOwner().equals(m_playerID)) {
                WPeasant peasant = (WPeasant)e;
                if (peasant.entityID > max_id){
                    max_id = peasant.entityID;
                }
                if (peasant.getStatus()!=null && peasant.getStatus().m_action== S3Action.ACTION_HARVEST) {
                    if (peasant.getStatus().m_parameters.size()==1) miners++;
                    else loggers++;
                }
            }
        }
        kb.addTerm(new Term("builder", Integer.toString(max_id)));

        if (miners > 2){
            if ((float) miners / (miners+ loggers) > ratio) {
                kb.addTerm(new Term("needLoggers", "1"));
            }
            else{
                kb.addTerm(new Term("needMiners", "1"));
            }
        }
        else {
            kb.addTerm(new Term("needMiners", "1"));
        }



        // we want to make worker log when there are over 2/3 mining
        kb.addTerm(
                new Term("total",Integer.toString(miners),"Miners")
        );

        kb.addTerm(
                new Term("total", Integer.toString(loggers), "Loggers")
        );

        for (Map.Entry<String, Integer> type_count : count_player1.entrySet()) {
            kb.addTerm(
                    new Term("total", Integer.toString(type_count.getValue()), type_count.getKey())
            );
        }
    }
}
