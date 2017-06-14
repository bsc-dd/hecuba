// package storage;
import com.datastax.driver.core.*;
import java.util.*;
import java.util.function.Function;
import static java.util.stream.Collectors.*;
import java.net.InetAddress;
import java.net.UnknownHostException;
import java.util.regex.Pattern;

public class StorageItf {

    private static Cluster cluster = null;
    private static Session session = null;

    /**
     * This function returns a order list of the hosts that have the higher number of local tokens.
     *
     * @param objectID the block identifier
     * @return
     * @throws storage.StorageException
     */
    public static List<String> getLocations(String objectID){
        UUID uuid = UUID.fromString(objectID.replace(" ", ""));
	    List<String> resultSet = Collections.<String>emptyList();
        checkCassandra();
        Row storage_info = session.execute("SELECT * FROM hecuba.istorage WHERE storage_id = ?", uuid).one();
        String class_name = storage_info.getString("class_name");
        System.out.println("class_name:  " + class_name);
        if(class_name.equals("hecuba.hdict.StorageDict")){
	        Metadata metadata = cluster.getMetadata();
	        String name = storage_info.getString("name");
	        int pposition = name.indexOf('.');
	        if (pposition == -1) {
	            System.out.println("Error calculation pposition");
	        }
	        final String nodeKp = name.substring(0, pposition);

	        Set<Map.Entry<Host, Long>> hostsTkns = storage_info.getList("tokens", TupleValue.class).stream()
	                .map(tok -> metadata.newToken(tok.getLong(0) + ""))
	                .flatMap(token ->
	                        metadata.getReplicas(Metadata.quote(nodeKp), metadata.newTokenRange(token, token)).stream())
	                .collect(groupingBy(Function.identity(), counting())).entrySet();

	        ArrayList<Map.Entry<Host, Long>> result = new ArrayList<>(hostsTkns);
	        Collections.sort(result, Comparator.comparing(o -> (o.getValue())));
	        List<String> toReturn;
	        toReturn = result.stream().map(a -> a.getKey().getAddress().toString().replaceAll("^.*/", "")).collect(toList());
	        List<String> toReturnHN = new ArrayList<String>();
	        for (String ip : toReturn){
	            try{
	                InetAddress addr = InetAddress.getByName(ip);
	                String host = addr.getHostName();
	                String[] HNsplitted = host.split("-");   //prev Pattern.quote(".")
	                toReturnHN.add(HNsplitted[0]);
	            }catch(UnknownHostException e){
                    System.out.println("Problem obtaining hostaddress:" + e);
	            }
	        }
	        closeCassandra();
	        System.out.println("Result for objectID " + objectID + ":" + toReturnHN.get(0));
	        return toReturnHN;
        }
        if(class_name.equals("hecuba.hdict.StorageObj")){
	        System.out.println("Result for objectID " + objectID + ": []");
	        closeCassandra();
        }
	    return resultSet;
    }

    private static void checkCassandra() {
        if (cluster == null) {
            String[] nodeIP = System.getenv("CONTACT_NAMES").split(",");
            int nodePort = Integer.parseInt(System.getenv("NODE_PORT"));
	        System.out.println("nodeIP:  " + nodeIP[0]);
	        System.out.println("nodePort:" + nodePort);
            cluster = new Cluster.Builder().addContactPoints(nodeIP).withPort(nodePort).build();
            session = cluster.connect();
        }
    }

    private static void closeCassandra() {
        if (cluster != null) {
            session.close();
            session = null;
            cluster.close();
            cluster = null;
        }

    }

    public static void newReplica(String objectID, String node){
    }

    public static String newVersion(String objectID, String node){
        return objectID;
    }

    public static void consolidateVersion(String objectID){
    }

    public static void delete(String objectID){
    }

    public static void finish(){
        closeCassandra();

    }

    public static java.lang.Object getByID(String objectID){
        return null;
    }

    public static void init(){

    }

    public static void main(String[] args){

        StorageItf client = new StorageItf();

        client.init();

        client.getLocations("25c80de1-0e4d-3706-8dfd-6e8c3bea901f");
        System.out.println("Application ended");
    }
}
