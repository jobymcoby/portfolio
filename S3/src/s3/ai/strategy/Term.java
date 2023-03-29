package s3.ai.strategy;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Objects;

public class Term {

    public static HashMap<String, Integer> functors = new HashMap<>();
    public String functor;
    // used because it is faster than string comparisons
    public int f_id;
    public String[] parameters;
    public boolean not = false;

    public Term (String term_string, boolean not_term){
        // split functor from parameters
        String[] term_parts = term_string.split("\\(");
        String[] parameters = term_parts[1].split(",");
        for (int i = 0; i < parameters.length; i++) {
            parameters[i] = parameters[i].replace(")", "");
            parameters[i] = parameters[i].replace(".", "");
        }

        this.functor = term_parts[0];
        setF_id(this.functor);
        this.parameters = parameters;
        this.not = not_term;
    }

    public Term(String f, String... parameters){
        this.functor = f;
        setF_id(this.functor);
        for (int i = 0; i < parameters.length; i++) {
            parameters[i] = parameters[i].replace(")", "");
            parameters[i] = parameters[i].replace(".", "");
        }
        this.parameters = parameters;
    }

    public Term(String f){
        this.functor = f;
        setF_id(this.functor);
        parameters = new String[0];

    }

    public Term(String f, boolean not, String... parameters){
        this.functor = f;
        setF_id(this.functor);
        this.parameters = parameters;
        this.not = not;
    }



    private void setF_id(String funct){
        // if functor is in dictionary, set f_id
        int f_id;
        if (functors.containsKey(funct)){
            f_id = functors.get(funct);
        }
        else{
            f_id = functors.size();
            functors.put(funct, f_id);
        }
        this.f_id = f_id;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Term term = (Term) o;
        return not == term.not && functor.equals(term.functor) && Arrays.equals(parameters, term.parameters);
    }

    @Override
    public int hashCode() {
        int result = Objects.hash(functor, not);
        result = 31 * result + Arrays.hashCode(parameters);
        return result;
    }

    @Override
    public String toString() {
        return "Term{" +
                "function='" + functor + '\'' +
                ", f_id='" + f_id + '\'' +
                ", parameters=" + Arrays.toString(parameters) +
                ", not=" + not +
                '}';
    }
}
