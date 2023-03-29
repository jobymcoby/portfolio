package s3.ai.strategy;

import s3.ai.*;
import s3.base.S3;
import s3.base.S3Action;
import s3.entities.*;
import s3.util.Pair;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.*;

public class Rule_AI implements AI {

    String m_playerID;
    KnowledgeBase unit_cost_base = new KnowledgeBase();
    KnowledgeBase perception_base = new KnowledgeBase();
    Rule[] rules;
    String ran_once ="1";

    public Rule_AI(String playerID) {
        m_playerID = playerID;
    }

    public Rule[] load_rules(String file_name) {
        List<Rule> rule_list = new ArrayList<Rule>();
        try {
            String file_location = System.getProperty("user.dir");
            file_location = file_location.concat(file_name);
            File ruleFile = new File(file_location);

            Scanner scan = new Scanner(ruleFile);

            while (scan.hasNextLine()) {
                String rule_line = scan.nextLine();
                if (rule_line.length() > 1) {
                    char firstChar = rule_line.charAt(0);
                    char pound = "#".charAt(0);
                    if (firstChar == pound) {
                        continue;
                    }

                    Rule rule = parse_rule(rule_line);
                    rule_list.add(rule);
                }
            }
            scan.close();
        }
        catch (FileNotFoundException e) {
            System.out.println("An error occurred.");
            e.printStackTrace();
        }
        return rule_list.toArray(new Rule[0]);
    }

    public Rule parse_rule(String rule_line) {
        String[] rule_parts = rule_line.split(" :- ");
        String predicate_string = rule_parts[0];
        Term predicate = new Term(predicate_string, false);

        // regex to find upper level commas that make term pattern for rule match
        String[] antecedents = rule_parts[1].split(",(?![^(]*\\))");
        Term[] antecedent_pattern = parse_antecedent_terms(antecedents);

        return new Rule(antecedent_pattern, new Term[]{predicate});
    }

    public Term[] parse_antecedent_terms(String[] antecedents){
        Term[] antecedent_pattern = new Term[antecedents.length];
        for (int i = 0; i < antecedents.length; i++) {
            String antecedent  = antecedents[i];
            boolean not_term = false;
            Term term;

            // handle comparator terms
            if (antecedent.contains("<")){
                String[] antecedent_parts = antecedent.split("<");
                term = new Term("less", antecedent_parts[0], antecedent_parts[1]);
            }
            else if(antecedent.contains(">")){
                String[] antecedent_parts = antecedent.split(">");
                term = new Term("great", antecedent_parts[0], antecedent_parts[1]);
            }
            else {
                // not terms
                if(antecedent.contains("~")){
                    String[] antecedent_parts = antecedent.split("~");
                    antecedent = antecedent_parts[1];
                    not_term = true;
                }
                term = new Term(antecedent, not_term);
            }

            antecedent_pattern[i] = term;
        }
        return antecedent_pattern;
    }

    public void get_unit_cost(KnowledgeBase kb){
        WPeasant worker = new WPeasant();
        kb.addTerm(new Term("goldNeededFor", "WPeasant", Integer.toString(worker.getCost_gold())));
        kb.addTerm(new Term("woodNeededFor", "WPeasant", Integer.toString(worker.getCost_wood())));

        WFootman light = new WFootman();
        kb.addTerm(new Term("goldNeededFor", "WFootman", Integer.toString(light.getCost_gold())));
        kb.addTerm(new Term("woodNeededFor", "WFootman", Integer.toString(light.getCost_wood())));

        WBarracks barracks = new WBarracks();
        kb.addTerm(new Term("goldNeededFor", "WBarracks", Integer.toString(barracks.getCost_gold())));
        kb.addTerm(new Term("woodNeededFor", "WBarracks", Integer.toString(barracks.getCost_wood())));

        WTownhall town = new WTownhall();
        kb.addTerm(new Term("goldNeededFor", "WTownhall", Integer.toString(town.getCost_gold())));
        kb.addTerm(new Term("woodNeededFor", "WTownhall", Integer.toString(town.getCost_wood())));
    }

    @Override
    public void gameStarts() throws Exception {
        rules = load_rules("\\RuleList");
        get_unit_cost(unit_cost_base);
    }


    @Override
    public void game_cycle(S3 game, WPlayer player, List<S3Action> actions) throws ClassNotFoundException, IOException {

        Perception.obtain_knowledge(game, perception_base, m_playerID);

        KnowledgeBaseCollection all_knowledge = new KnowledgeBaseCollection(new KnowledgeBase[]{unit_cost_base, perception_base});

        List<Rule> matched_rules = InferenceEngine.RuleBasedSystemIteration(rules, all_knowledge);

        //for (Rule r: matched_rules) {
        //    System.out.println(r);
        //}
        //if (matched_rules.size() > 0){
        //    all_knowledge.print_list();
        //    System.out.println();
        //}

        WPlayer player1 = game.getPlayer(m_playerID);
        arbitrate_rules(player1.getGold(), player1.getWood(), matched_rules);

        fire_rules(game, actions, all_knowledge, matched_rules);

        perception_base.clear();
    }

    public static void arbitrate_rules(int current_gold, int current_wood, List<Rule> firedRules) {
        // arbitrate rules
        List<Rule> toDelete = new ArrayList<Rule>();
        for(Rule r1: firedRules) {
            for(Rule r2: firedRules) {
                if (r1!=r2 && !toDelete.contains(r1)) {
                    String unit_id1 =  r1.getEffect()[0].parameters[0];
                    String unit_id2 =  r2.getEffect()[0].parameters[0];

                    if (Objects.equals(unit_id1, unit_id2)) {
                        if (!toDelete.contains(r2)) {
                            toDelete.add(r2);
                        }
                    }
                }
            }
        }
        // make sure rules dont overspend resources.
        firedRules.removeAll(toDelete);
        for (Rule rule: firedRules){
            if (rule.getEffectType() == 1 || rule.getEffectType() == 3){
                if(rule.getEffect()[0].functor.contains("Barracks")){

                }
                if(rule.getEffect()[0].functor.contains("Base")){

                }
                if(rule.getEffect()[0].functor.contains("Worker")){

                }
                if(rule.getEffect()[0].functor.contains("Light")){

                }

            }
        }
    }

    private void fire_rules(S3 game, List<S3Action> actions, KnowledgeBaseCollection all_knowledge, List<Rule> firedRules) {
        for (Rule r : firedRules) {
            if (r.getEffectType() == 0){
                // add term to kb
                all_knowledge.addTerm(r.getEffect()[0]);
            }
            else if(r.getEffectType() == 1){
                //build
                Term effect_action  = r.getEffect()[0];
                int worker_id = Integer.parseInt(effect_action.parameters[0]);
                WPeasant peasant =  (WPeasant) game.getUnit(worker_id);


                // First try one locating with space to walk around it:
                Pair<Integer, Integer> loc = game.findFreeSpace(peasant.getX(), peasant.getY(), 5);
                if (null == loc) {
                    loc = game.findFreeSpace(peasant.getX(), peasant.getY(), 3);
                    if (loc==null) break;
                }
                S3Action build_act = null;

                if(r.getEffect()[0].functor.contains("Barracks")){
                    build_act = new S3Action(worker_id, S3Action.ACTION_BUILD, WBarracks.class.getSimpleName(), loc.m_a, loc.m_b);
                    actions.add(build_act);
                } else {
                    build_act = new S3Action(worker_id, S3Action.ACTION_BUILD, WTownhall.class.getSimpleName(), loc.m_a, loc.m_b);
                    actions.add(build_act);
                }

            }
            else if(r.getEffectType() == 2){
                //Harvest

                Term effect_action  = r.getEffect()[0];

                int worker_id = Integer.parseInt(effect_action.parameters[0]);

                if (effect_action.parameters.length == 3) {
                    int gold_id = Integer.parseInt(effect_action.parameters[1]);

                    //
                    WPeasant peasant = (WPeasant) game.getUnit(worker_id);
                    WGoldMine mine = (WGoldMine) game.getUnit(gold_id);
                    S3Action harvest_act = new S3Action(peasant.entityID, S3Action.ACTION_HARVEST, mine.entityID);

                    actions.add(harvest_act);
                }
                else {
                    WPeasant peasant = (WPeasant) game.getUnit(worker_id);
                    List<WOTree> trees = new LinkedList<WOTree>();
                    for(int i = 0; i< game.getMap().getWidth(); i++) {
                        for(int j = 0; j< game.getMap().getHeight(); j++) {
                            S3PhysicalEntity e = game.getMap().getEntity(i, j);
                            if (e instanceof WOTree) trees.add((WOTree)e);
                        }
                    }

                    WOTree tree = null;

                    int leastDist = 9999;
                    for (WOTree unit : trees) {
                        int dist = Math.abs(unit.getX() - peasant.getX())
                                + Math.abs(unit.getY() - peasant.getY());
                        if (dist < leastDist) {
                            leastDist = dist;
                            tree = unit;
                        }
                    }

                    if (tree != null) {
                        actions.add(new S3Action(peasant.entityID, S3Action.ACTION_HARVEST, tree.getX(), tree.getY()));
                    }
                }
            }
            else if(r.getEffectType() == 3){
                //Train
                Term effect_action  = r.getEffect()[0];
                int worker_id = Integer.parseInt(effect_action.parameters[0]);

                S3Action train_act;

                if(r.getEffect()[0].functor.contains("Worker")){
                    WTownhall th = (WTownhall) game.getUnit(worker_id);
                    train_act = new S3Action(th.entityID,S3Action.ACTION_TRAIN, "WPeasant");
                    actions.add(train_act);
                } else {
                    WBarracks bk = (WBarracks) game.getUnit(worker_id);
                    train_act = new S3Action(bk.entityID,S3Action.ACTION_TRAIN, "WFootman");
                    actions.add(train_act);
                }
            }
            else if(r.getEffectType() == 4){
                Term effect_action  = r.getEffect()[0];
                int worker_id = Integer.parseInt(effect_action.parameters[0]);
                int enemy_id = Integer.parseInt(effect_action.parameters[1]);

                S3Action attack_act = new S3Action(worker_id,S3Action.ACTION_ATTACK, enemy_id);
                actions.add(attack_act);
            }

        }
    }

    @Override
    public void gameEnd() { }

    public String getPlayerId() {
        return m_playerID;
    }
}
