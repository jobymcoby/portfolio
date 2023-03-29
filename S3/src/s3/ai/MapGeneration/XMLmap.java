package s3.ai.MapGeneration;

import org.jdom.Attribute;
import org.jdom.Document;
import org.jdom.Element;
import org.jdom.output.Format;
import org.jdom.output.XMLOutputter;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.List;


public class XMLmap {



    public XMLmap(){

    }


    public static void write_mapfile(int x, int y, MapNode[] nodes, List<String> rows, int[] player_start, int[] enemy_start){
        // extract entities from nodes here
        // gold mines and peasent start locations

        try{
            //root element
            int i = 0;
            Element root = new Element("gamestate");
            Document doc = new Document(root);

            Element entity = new Element("entity");
            entity.setAttribute(new Attribute("id", Integer.toString(i)));

            Element type = new Element("type");
            type.setText("map");

            Element width = new Element("width");
            width.setText(Integer.toString(x));

            Element height = new Element("height");
            height.setText(Integer.toString(y));

            entity.addContent(type);
            entity.addContent(width);
            entity.addContent(height);

            Element background = new Element("background");

            for (String s_row: rows) {
                Element row = new Element("row");
                row.setText(s_row);
                background.addContent(row);
            }
            entity.addContent(background);

            doc.getRootElement().addContent(entity);


            // Add player attribute
            for (int j = 1; j < 3; j++) {
                String player;
                if(j == 1)
                    player = "player1";
                else
                    player = "player2";


                entity = new Element("entity");
                entity.setAttribute(new Attribute("id", Integer.toString(j)));
                type = new Element("type");
                type.setText("WPlayer");

                Element gold = new Element("gold");
                gold.setText(Integer.toString(2000));

                Element wood = new Element("wood");
                wood.setText(Integer.toString(1500));

                Element owner = new Element("owner");
                owner.setText(player);

                entity.addContent(type);
                entity.addContent(gold);
                entity.addContent(wood);
                entity.addContent(owner);

                doc.getRootElement().addContent(entity);

            }
            // add unit and resources

            for (int j = 3; j < 7; j++) {
                String player;
                int[] position;
                if(j == 3 || j == 4) {
                    player = "player1";
                    position = player_start;
                }
                else {
                    player = "player2";
                    position = enemy_start;
                }

                entity = new Element("entity");
                entity.setAttribute(new Attribute("id", Integer.toString(j)));
                type = new Element("type");
                type.setText("WPeasant");

                Element x_pos = new Element("x");
                x_pos.setText(Integer.toString(position[0]));

                Element y_pos = new Element("y");
                y_pos.setText(Integer.toString(position[1]));

                Element current_hitpoints = new Element("current_hitpoints");
                current_hitpoints.setText(Integer.toString(30));

                Element owner = new Element("owner");
                owner.setText(player);

                entity.addContent(type);
                entity.addContent(x_pos);
                entity.addContent(y_pos);
                entity.addContent(owner);
                entity.addContent(current_hitpoints);
                doc.getRootElement().addContent(entity);

                j++;

                entity = new Element("entity");
                entity.setAttribute(new Attribute("id", Integer.toString(j)));
                type = new Element("type");
                type.setText("WGoldMine");

                x_pos = new Element("x");
                x_pos.setText(Integer.toString(position[2]));

                y_pos = new Element("y");
                y_pos.setText(Integer.toString(position[3]));

                Element remaining_gold = new Element("remaining_gold");
                remaining_gold.setText(Integer.toString(50000));


                current_hitpoints = new Element("current_hitpoints");
                current_hitpoints.setText(Integer.toString(25500));


                entity.addContent(type);
                entity.addContent(x_pos);
                entity.addContent(y_pos);
                entity.addContent(remaining_gold);
                entity.addContent(current_hitpoints);

                doc.getRootElement().addContent(entity);

            }

            XMLOutputter xmlOutput = new XMLOutputter();

            // display ml
            xmlOutput.setFormat(Format.getPrettyFormat());
            // xmlOutput.output(doc, System.out);

            OutputStream out = new FileOutputStream("maps/chris.xml");
            Writer writer = new OutputStreamWriter(out, StandardCharsets.UTF_8);
            xmlOutput.output(doc,writer);

        } catch(IOException e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {


    }
}