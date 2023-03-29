package s3.ai.strategy;

public class KnowledgeBaseCollection extends KnowledgeBase{

    public KnowledgeBaseCollection(KnowledgeBase[] bases){
        for (KnowledgeBase kb: bases) {
            this.facts.addAll(kb.facts);
        }
    }
}
