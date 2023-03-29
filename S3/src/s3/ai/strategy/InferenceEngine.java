package s3.ai.strategy;

import java.util.*;

public class InferenceEngine {


    public static List<Rule> RuleBasedSystemIteration(Rule[] rules, KnowledgeBase KB) {

        List<Rule> FiredRules = new ArrayList<Rule>();
        for (Rule r : rules) {

            HashMap<String, String> bindings = new HashMap<String, String>();
            HashMap<String, String> next_term_bindings = new HashMap<String, String>();
            if (r.pattern.length <= 1) {
                bindings = unification_kb(r.pattern[0], KB);
            } else {
                // unify rule pattern with kb
                Term one_prime = r.pattern[0];
                Term two_prime = r.pattern[1];

                for (int i = 1; i < r.pattern.length; i++) {
                    //System.out.println(one_prime);
                    //System.out.println(two_prime);

                    next_term_bindings = unification_and(one_prime, two_prime, KB, next_term_bindings);

                    if (next_term_bindings == null) {
                        // no unification in kb, don't use rule
                        //System.out.println("no union");
                        bindings = null;
                        break;
                    }

                    if (bindings != null) {
                        bindings.putAll(next_term_bindings);
                    }

                    one_prime = Rule.apply_binding(r.pattern[i], bindings);
                    if (i + 1 < r.pattern.length) {
                        two_prime = r.pattern[i + 1];
                    }


                }
            }
            // Rule has been unified with first result not in used bindings
            if (bindings != null) {
                FiredRules.add(new Rule(r, bindings));
            }
        }

        return FiredRules;
    }


    public static HashMap<String, String> unification_terms(Term one, Term two){
        HashMap<String, String> bindings = new HashMap<String, String>();
        if (one.f_id != two.f_id) {return null;}
        // only functor match

        if (one.parameters == null || two.parameters == null){
            if (one.parameters == null && two.parameters == null){
                return bindings;
            }
            else {
                return null;
            }
        }


        if (one.parameters.length != two.parameters.length) {return null;}
        for (int i = 0; i < one.parameters.length ; i++) {
            //unbound variable

            if (one.parameters[i].matches("[ABCJXYZ]")){
                bindings.put(one.parameters[i], two.parameters[i]);
            }
            else if(!one.parameters[i].equals(two.parameters[i])){
                return null;
            }

        }
        return bindings;
    }

    public static HashMap<String, String> unification_kb(Term term, KnowledgeBase knowledgeBase){
        HashMap<String, String> bindings;
        for (Term fact: knowledgeBase.facts) {
            bindings = unification_terms(term, fact);
            if(bindings!=null){
                if(term.not){return null;}
                else {return bindings;}
            }
        }
        if(term.not){return new HashMap<String, String>();}
        else {return null;}
    }

    public static HashMap<String, String> unification_and(Term one, Term two, KnowledgeBase kb, HashMap<String, String> current_bindings) {
        HashMap<String, String> bindings = null;

        for (Term fact1 : kb.facts) {

            bindings = unification_terms(one, fact1);

            add_comparisons_knowledge_base(one, kb);


            if (bindings != null) {
                // we have found a unification
                // now we check for a not operator to adjust the return.
                if (one.not) {
                    return null;
                }
                current_bindings.putAll(bindings);
                HashMap<String, String> bindings2 = unification_and_term2(two,kb,current_bindings);
                if (bindings2 != null){
                    return bindings2;
                }
            }
        }

        if (one.not){
            HashMap<String, String> bindings2 = unification_and_term2(two,kb,current_bindings);
            if (bindings2 != null){
                return bindings2;
            }
        }
        // if there is no match
        return null;
    }

    public static HashMap<String, String> unification_and_term2(Term two, KnowledgeBase kb, HashMap<String, String> current_bindings){
        HashMap<String, String> bindings2 = null;
        Term two_prime = Rule.apply_binding(two, current_bindings);

        add_comparisons_knowledge_base(two_prime, kb);


        for (Term fact2 : kb.facts) {

            bindings2 = unification_terms(two_prime, fact2);

            if (bindings2 != null) {
                if (two.not) {
                    return null;
                }
                current_bindings.putAll(bindings2);
                return current_bindings;
            }
        }
        if (two.not){
            return current_bindings;
        }
        return null;
    }

    public static void add_comparisons_knowledge_base(Term term_to_unify, KnowledgeBase kb){
        if (Objects.equals(term_to_unify.functor, "great")) {
            if (Integer.parseInt(term_to_unify.parameters[0]) > Integer.parseInt(term_to_unify.parameters[1])) {
                kb.addTerm(new Term("great", term_to_unify.parameters[0], term_to_unify.parameters[1]));
            }
        }
        if (Objects.equals(term_to_unify.functor, "less")) {
            if (Integer.parseInt(term_to_unify.parameters[0]) < Integer.parseInt(term_to_unify.parameters[1])) {
                kb.addTerm(new Term("less", term_to_unify.parameters[0], term_to_unify.parameters[1]));
            }
        }
    }




}
