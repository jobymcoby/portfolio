package s3.ai.strategy;

import java.util.*;

public class Testing {

    public static void main(String[] args) {
        KnowledgeBase kb1 = new KnowledgeBase();
        Rule_AI test = new Rule_AI("1");
//        Rule[] rule_list = test.load_rules("\\RuleList");
//        for(Rule r: rule_list){
//            //System.out.println(r);
//        }
        kb1.addTerm(new Term("type", "P", "1"));
        kb1.addTerm(new Term("own", "1"));
        kb1.addTerm(new Term("idle","1"));
        kb1.addTerm(new Term("type", "P", "2"));
        kb1.addTerm(new Term("own", "2"));
        kb1.addTerm(new Term("idle","2"));
        //kb1.addTerm(new Term("enemy", "3"));

        Term a  = new Term("type", "P", "X");
        Term b  = new Term("idle","X");
        Term c  = new Term("own", "X");
        Term d  = new Term("great", "X", "Y");
        Term f  = new Term("ownP", "X");

        Rule rule1 = new Rule(new Term[]{a,c,b}, new Term[]{f});
        Rule rule2 = new Rule(new Term[]{b}, new Term[]{f});

        Rule[] rule_arr = new Rule[]{rule1, rule2};
        List<Rule> fired_rules = InferenceEngine.RuleBasedSystemIteration(rule_arr, kb1);
        for (Rule r: fired_rules){
            System.out.println(r);
        }

        Rule_AI.arbitrate_rules(500, 299, fired_rules);
        System.out.println();
        for (Rule r: fired_rules){
            System.out.println(r);
        }

    }
}
