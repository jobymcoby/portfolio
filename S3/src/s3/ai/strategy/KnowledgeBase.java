package s3.ai.strategy;

import java.util.ArrayList;
import java.util.List;

public class KnowledgeBase {
    List<Term> facts = new ArrayList<Term>();

    public void addTerm(Term t){
        if (!facts.contains(t)) {
            facts.add(t);
        }
    }

    public void clear(){
        facts.clear();
    }

    public void print_list() {
        for (Term fact: facts) {
            System.out.println(fact);
        }
    }
}
