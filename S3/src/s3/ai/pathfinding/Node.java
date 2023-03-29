package s3.ai.pathfinding;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

public class Node implements Comparable<Node>{
    int x,y,g = 0, max_x, max_y;
    double h = 0, f;
    Node parent, goal;

    public int getX() {
        return x;
    }

    public int getY() {
        return y;
    }

    public int getG() {
        return g;
    }

    public double getH() {
        return h;
    }

    public double getF() {
        return f;
    }

    public Node(int x, int y, Node parent, Node _goal, int max_x, int max_y){
        this.x = x;
        this.y = y;
        this.max_x = max_x;
        this.max_y = max_y;

        this.parent = parent;
        if (parent != null){
            this.goal = parent.goal;
            set_G();
            set_H(this.goal);
            calc_F();
        }
        else if (_goal != null){
            this.goal = _goal;
        }
    }

    private void set_G(){
        //Distance from start, uses the parents g

        if (parent == null){
            return;
        }

        // if 1 moves away cost += 10, else +=14
        if (parent.getX() == this.getX() || parent.getY() == this.getY()){
            g = parent.g + 10;
        }
        else{
            g = parent.g + 14;
        }
    }

    public void set_H(Node goal){
        //Distance from end use the euclidean heuristic
        h = Math.sqrt(Math.pow(x - goal.getX(), 2) + Math.pow(y - goal.getY(),2)) * 10;
        calc_F();
    }
    private void calc_F(){
        f = g + h;
    }

    public List<Node> getNeighbors() {


        Node right = new Node(this.x+1, this.y, this, null, max_x, max_y);
        Node down = new Node(this.x, this.y+1, this, null, max_x, max_y);
        Node rdown = new Node(this.x+1, this.y+1, this,null, max_x, max_y);
        Node lup = new Node(this.x-1, this.y-1, this,null, max_x, max_y);
        Node left = new Node(this.x-1, this.y, this, null, max_x, max_y);
        Node ldown = new Node(this.x-1, this.y+1, this,null, max_x, max_y);
        Node up = new Node(this.x, this.y-1, this, null, max_x, max_y);
        Node rup = new Node(this.x+1, this.y-1, this,null, max_x, max_y);

        List<Node> neighbors = new ArrayList<Node>(
                List.of(right, down, rdown, lup, left, ldown, up, rup)
        );

        if(this.x - 1 < 0 && this.y - 1 < 0){
            neighbors.remove(lup);
            neighbors.remove(left);
            neighbors.remove(ldown);
            neighbors.remove(up);
            neighbors.remove(rup);
        }
        else if(this.x - 1 < 0){
            neighbors.remove(lup);
            neighbors.remove(left);
            neighbors.remove(ldown);
        }
        else if(this.y - 1 < 0){
            neighbors.remove(lup);
            neighbors.remove(up);
            neighbors.remove(rup);
        }

        if(this.x + 1 > max_x && this.y + 1 > max_y){
            neighbors.remove(rup);
            neighbors.remove(right);
            neighbors.remove(rdown);
            neighbors.remove(down);
            neighbors.remove(ldown);
        }
        else if(this.y + 1 > max_y){
            neighbors.remove(rdown);
            neighbors.remove(down);
            neighbors.remove(ldown);
        }
        else if(this.x + 1 > max_x){
            neighbors.remove(rup);
            neighbors.remove(right);
            neighbors.remove(rdown);
        }

        return neighbors;
    }

    @Override
    public String toString() {
        return "Node{" +
                "x=" + x +
                ", y=" + y +
                ", g=" + g +
                ", h=" + h +
                ", f=" + f +
                '}';
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Node node = (Node) o;
        return this.x == node.x && this.y == node.y;
    }

    @Override
    public int hashCode() {
        return Objects.hash(x, y);
    }

    @Override public int compareTo(Node other) {
        double other_f = ((Node)other).getF();


        return  (int) (- other_f + this.getF());

    }
}
