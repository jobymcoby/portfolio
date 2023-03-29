package s3.ai.strategy;

import java.util.*;

public class Rule {
    Term[] pattern;
    Term[] effect;
    int effectType;

    public int getEffectType() {
        return effectType;
    }

    public Term[] getEffect() {
        return effect;
    }

    public Term[] getPattern() {
        return pattern;
    }

    public Rule(Term[] pattern, Term[] effect)
    {
        this.pattern = pattern;
        this.effect = effect;

        Term predicate = effect[0];
        int effect_int = 0;
        if(predicate.functor.contains("do")){
            if(predicate.functor.contains("Build")){
                effect_int = 1;
            }
            if(predicate.functor.contains("Harvest")){
                effect_int = 2;
            }
            if(predicate.functor.contains("Train")){
                effect_int = 3;
            }
            if(predicate.functor.contains("Attack")){
                effect_int = 4;
            }
        }
        this.effectType = effect_int;
    }

    public Rule(Rule r, HashMap<String, String> bindings) {
        // Instantiate a new rule with the bindings given
        List<Term> bound_pattern = new ArrayList<Term>();
        // first apply binds to the pattern
        for (Term term: r.pattern){
            Term bound_term = apply_binding(term, bindings);
            bound_pattern.add(bound_term);
        }
        List<Term> bound_effect = new ArrayList<Term>();
        for(Term term: r.effect){
            Term bound_term  = apply_binding(term, bindings);
            bound_effect.add(bound_term);
        }

        Term[] bound_pattern_a = new Term[bound_pattern.size()];
        bound_pattern_a = bound_pattern.toArray(bound_pattern_a);
        this.pattern = bound_pattern_a;

        Term[] bound_effect_a = new Term[bound_effect.size()];
        bound_effect_a = bound_effect.toArray(bound_effect_a);
        this.effect = bound_effect_a;

        this.effectType = r.effectType;
    }

    public static Term apply_binding(Term term, HashMap<String, String> bindings){
        List<String> bound_parameters = new ArrayList<String>();
        for (String parameter: term.parameters){
            // use this to pass already bound variable to the new term
            String bound_variable = parameter;
            // look for the bindings
            if(bindings.containsKey(parameter)){
                bound_variable = bindings.get(parameter);
            }
            // replace unbound variable with bound variable
            bound_parameters.add(bound_variable);
        }
        String[] bound_parameters_a = new String[bound_parameters.size()];
        bound_parameters_a = bound_parameters.toArray(bound_parameters_a);
        return new Term(term.functor, term.not, bound_parameters_a);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Rule rule = (Rule) o;
        return effectType == rule.effectType && Arrays.equals(pattern, rule.pattern) && Arrays.equals(effect, rule.effect);
    }

    @Override
    public int hashCode() {
        int result = Objects.hash(effectType);
        result = 31 * result + Arrays.hashCode(pattern);
        result = 31 * result + Arrays.hashCode(effect);
        return result;
    }

    @Override
    public String toString() {
        return "Rule{" +
                "effect=" + Arrays.toString(effect) +
                ",\n pattern=" + Arrays.toString(pattern) +
                ",\n effectType=" + effectType +
                "}";
    }
}
