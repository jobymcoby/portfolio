package s3.ai.pathfinding;


import java.util.ArrayList;
import java.util.Collections;
import java.util.List;


import s3.base.S3;
import s3.entities.S3Entity;
import s3.entities.S3PhysicalEntity;
import s3.util.Pair;


public class AStar {
	Scroller looking = new Scroller();
	Node path_node;
	
	
	public static int pathDistance(double start_x, double start_y, double goal_x, double goal_y,
			S3PhysicalEntity i_entity, S3 the_game) {
		AStar a = new AStar(start_x,start_y,goal_x,goal_y,i_entity,the_game);
		List<Pair<Double, Double>> path = a.computePath();
		if (path!=null) return path.size();
		return -1;
	}

	public AStar(double start_x, double start_y, double goal_x, double goal_y,
			S3PhysicalEntity i_entity, S3 the_game) {


		//System.out.println(i_entity.entityID);
		int max_x = the_game.m_map.getWidth();
		int max_y = the_game.m_map.getHeight();

		//if(i_entity.entityID ==7) {
		//	System.out.printf("Entity #%d, at x: %d at y: %d \n", i_entity.entityID, i_entity.getX(), i_entity.getY());
		//	System.out.printf("wants to move at x: %f at y: %f \n", goal_x, goal_y);
		//}



		Node end = new Node((int) goal_x, (int) goal_y, null, null, max_x, max_y);
		Node start = new Node(i_entity.getX(), i_entity.getY(), null, end, max_x, max_y);

		List<Node> visited_nodes = new ArrayList<>(
				List.of(start)
		);

		List<Node> neighbors_1 = start.getNeighbors();
		List<Node> walls_1 = new ArrayList<>();
		for (Node neighbor : neighbors_1) {
			looking.setX(neighbor.getX());
			looking.setY(neighbor.getY());
			S3Entity game_space = the_game.anyLevelCollision(looking);
			if (game_space != null) {
				walls_1.add(neighbor);
			}
		}
		if (!walls_1.isEmpty()) {
			// don't look at these nodes we have already been there.
			for (Node node : walls_1) {
				neighbors_1.remove(node);
			}
		}

		List<Node> new_nodes = neighbors_1;
		new_nodes.sort(Node::compareTo);
		int count = 0;
		while (!new_nodes.isEmpty()) {

			// node with smallest F score
			Node current_node = new_nodes.get(0);

			//Check goal
			if (current_node.equals(end)) {
				path_node = current_node;
				break;
			}

			// Visited nodes
			new_nodes.remove(current_node);
			visited_nodes.add(current_node);

			List<Node> neighbors = current_node.getNeighbors();

			// Here we go through the neighboring nodes and see if they are in our list already
			List<Node> been_there = new ArrayList<>();
			for (Node neighbor : neighbors) {

				if (visited_nodes.contains(neighbor)) {
					been_there.add(neighbor);

				}
				if (new_nodes.contains(neighbor)) {
					been_there.add(neighbor);
				}
			}
			if (!been_there.isEmpty()) {
				//System.out.printf("%d overlap\n", been_there.size());
				//System.out.printf("%d old one\n", neighbors.size());
				// don't look at these nodes we have already been there.
				for (Node node : been_there) {
					neighbors.remove(node);
				}
				//System.out.printf("%d now in neighbors\n", neighbors.size());

			}



			//Now we move our scroller to see if we hit a collision
			List<Node> walls = new ArrayList<>();
			for (Node neighbor : neighbors) {
				looking.setX(neighbor.getX());
				looking.setY(neighbor.getY());
				S3Entity game_space = the_game.anyLevelCollision(looking);
				if (game_space != null) {
					walls.add(neighbor);
				}
			}
			if (!walls.isEmpty()) {
				// don't look at these nodes we have already been there.
				for (Node node : walls) {
					neighbors.remove(node);
				}
			}


			// add neighbors to new node
			new_nodes.addAll(neighbors);
			new_nodes.sort(Node::compareTo);
			count++;
		}
	}

	public List<Pair<Double, Double>> computePath() {
		List<Pair<Double, Double>> path =  new ArrayList<>();
		Node step = path_node;

		if (step != null){
			while(step.parent != null)	{
				path.add(new Pair<>((double)step.getX(), (double)step.getY()));
				step = step.parent;
			}
			Collections.reverse(path);
		}

		return path;
	}
}
