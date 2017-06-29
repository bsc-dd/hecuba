package storage;
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
    public static List<String> getLocations(String objectID) throws storage.StorageException {
        UUID uuid = UUID.fromString(objectID.replace(" ", ""));
	    List<String> resultSet = Collections.<String>emptyList();
        checkCassandra();
        Row storage_info = session.execute("SELECT class_name FROM hecuba.istorage WHERE storage_id = ?", uuid).one();
        String class_name = storage_info.getString("class_name");
        if(class_name == "hecuba.hdict.StorageDict"){
	        Metadata metadata = cluster.getMetadata();
	        String name = storage_info.getString("name");
	        int pposition = name.indexOf('.');
	        if (pposition == -1) {
	            throw new StorageException("I cannot detect the keyspace name from " + name);
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
                        String[] HNsplitted = host.split("-");
                        HNsplitted = HNsplitted[0].split(Pattern.quote("."));
                        toReturnHN.add(HNsplitted[0]);
                    }catch(UnknownHostException e){
                    throw new storage.StorageException("Problem obtaining hostaddress:" + e);
	            }
	        }
	        System.out.println("Result for objectID " + objectID + ":" + toReturnHN.get(0));
	        return toReturnHN;
        }
        if(class_name == "hecuba.hdict.StorageObj"){
	        System.out.println("Result for objectID " + objectID + ": []");
        }
	return resultSet;
    }


    private static void checkCassandra() {
        if (cluster == null) {
            String[] nodeIP = System.getenv("CONTACT_NAMES").split(",");
            int nodePort = Integer.parseInt(System.getenv("NODE_PORT"));
            cluster = new Cluster.Builder()
                    .addContactPoints(nodeIP)
                    .withPort(nodePort)
                    .build();
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


    public static void newReplica(String objectID, String node) throws StorageException {
    }

    public static String newVersion(String objectID, String node) throws StorageException {
        //return "";
        return objectID;
    }

    public static void consolidateVersion(String objectID) throws StorageException {
    }

    public static void delete(String objectID) throws StorageException {
    }

    public static void finish() throws StorageException {
        closeCassandra();

    }

    public static java.lang.Object getByID(String objectID) throws StorageException {
        return null;
    }

    public static void init(String configFile) throws storage.StorageException {

    }

    public static void main(String[] args) throws StorageException {

        StorageItf client = new StorageItf();

        try {
            client.init(null);
        } catch (StorageException e) {
            e.printStackTrace();
        }

        client.getLocations(args[0]).forEach(System.out::println);
        System.out.println("Application ended");
    }
}
